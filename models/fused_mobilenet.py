import torch
import torch.nn as nn

from .layers import *
from .quantization_utils import *

from functools import partial
from torch import Tensor
from torch.nn import functional as F
from typing import Any, Callable, Dict, List, Optional, Sequence

from torchvision.models.mobilenetv2 import _make_divisible, ConvBNActivation
from .mobilenet import SqueezeExcitation, InvertedResidualConfig

class FusedSqueezeExcitation(nn.Module):
    # Implemented as described at Figure 4 of the MobileNetV3 paper
    def __init__(self, input_channels: int, squeeze_factor: int = 4, smooth: float = 0.999, bit: int = 32):
        super().__init__()
        self.ema_init = False
        self.act_range = nn.Parameter(torch.zeros(2), requires_grad=False)
        self.smooth = smooth
        self.bit = bit
        self.q_max = 2 ** self.bit - 1

        squeeze_channels = _make_divisible(input_channels // squeeze_factor, 8)
        self.fc1 = FusedConv2d(input_channels, squeeze_channels, kernel_size=1, bias=True, activation_layer=nn.ReLU, smooth=smooth, bit=bit)
        self.fc2 = FusedConv2d(squeeze_channels, input_channels, kernel_size=1, bias=True, smooth=smooth, bit=bit)
        self.QAct = QActivation(activation=nn.Hardsigmoid, bit=bit, smooth=smooth)

    def _scale(self, input: Tensor, inplace: bool) -> Tensor:
        scale = F.adaptive_avg_pool2d(input, 1)
        scale = self.fc1(scale)
        scale = self.fc2(scale)
        scale = self.QAct(scale)
        # return F.hardsigmoid(scale, inplace=inplace)
        return scale

    def forward(self, input: Tensor) -> Tensor:
        scale = self._scale(input, True)
        x = scale * input
        if self.training:
            if self.ema_init:
                self.act_range[0], self.act_range[1] = ema(x, self.act_range, self.smooth)
                s, z = calc_qparams(self.act_range[0], self.act_range[1], self.q_max)
                x = fake_quantize(x, s, z)
            else:
                self.act_range[0] = torch.min(x).item()
                self.act_range[1] = torch.max(x).item()
                self.ema_init = True
        return x


class InvertedResidual(nn.Module):
    # Implemented as described at section 5 of MobileNetV3 paper
    def __init__(self, cnf: InvertedResidualConfig, norm_layer: Callable[..., nn.Module], smooth: float = 0.999, bit: int = 32):
        super().__init__()
        if not (1 <= cnf.stride <= 2):
            raise ValueError('illegal stride value')

        self.smooth = smooth
        self.bit = bit
        self.q_max = 2 ** self.bit - 1
        self.ema_init = False
        self.act_range = nn.Parameter(torch.zeros(2), requires_grad=False)
        self.use_res_connect = cnf.stride == 1 and cnf.input_channels == cnf.out_channels

        layers: List[nn.Module] = []
        activation_layer = nn.ReLU if not cnf.use_hs else None

        # expand
        if cnf.expanded_channels != cnf.input_channels:
            layers.append(FusedConv2d(cnf.input_channels, cnf.expanded_channels, kernel_size=1,
                                      norm_layer=norm_layer, activation_layer=activation_layer, smooth=smooth, bit=bit))
            if cnf.use_hs:
                layers.append(QActivation(activation=nn.Hardswish, smooth=smooth, bit=bit))

        # depthwise
        stride = 1 if cnf.dilation > 1 else cnf.stride
        # groups = input_channel -> input_channel / input_channel = 1 -> operates on single channel
        layers.append(FusedConv2d(cnf.expanded_channels, cnf.expanded_channels, kernel_size=cnf.kernel,
                                  padding=(cnf.kernel-1)//2, stride=stride, dilation=cnf.dilation, groups=cnf.expanded_channels,
                                  norm_layer=norm_layer, activation_layer=activation_layer, smooth=smooth, bit=bit))
        if cnf.use_hs:
            layers.append(QActivation(activation=nn.Hardswish, smooth=smooth, bit=bit))

        if cnf.use_se:
            layers.append(FusedSqueezeExcitation(cnf.expanded_channels, bit=bit, smooth=smooth))

        # project
        layers.append(FusedConv2d(cnf.expanded_channels, cnf.out_channels, kernel_size=1, norm_layer=norm_layer, smooth=smooth, bit=bit))

        self.block = nn.Sequential(*layers)
        self.out_channels = cnf.out_channels
        self._is_cn = cnf.stride > 1

    def forward(self, input: Tensor) -> Tensor:
        result = self.block(input)
        if self.use_res_connect:
            result += input

        if self.training:
            if self.ema_init:
                self.act_range[0], self.act_range[1] = ema(result, self.act_range, self.smooth)
                s, z = calc_qparams(self.act_range[0], self.act_range[1], self.q_max)
                result = fake_quantize(result, s, z)
            else:
                self.act_range[0] = torch.min(result).item()
                self.act_range[1] = torch.max(result).item()
                self.ema_init = True
        return result


class FusedMobileNet(nn.Module):
    def __init__(
            self,
            inverted_residual_setting: List[InvertedResidualConfig],
            last_channel: int,
            num_classes: int = 1000,
            block: Optional[Callable[..., nn.Module]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            smooth: float = 0.999,
            bit: int = 32,
            dilation: int = 1,
            **kwargs: Any
    ) -> None:
        super().__init__()
        self.bit = bit
        self.in_range = nn.Parameter(torch.zeros(2), requires_grad=False)
        self.ema_init = False
        self.smooth = smooth
        self.q_max = 2 ** self.bit - 1

        if not inverted_residual_setting:
            raise ValueError("The inverted_residual_setting should not be empty")
        elif not (isinstance(inverted_residual_setting, Sequence) and
                  all([isinstance(s, InvertedResidualConfig) for s in inverted_residual_setting])):
            raise TypeError("The inverted_residual_setting should be List[InvertedResidualConfig]")

        if block is None:
            block = InvertedResidual

        if norm_layer is None:
            norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)

        layers: List[nn.Module] = []

        # building first layer
        firstconv_output_channels = inverted_residual_setting[0].input_channels
        layers.append(FusedConv2d(3, firstconv_output_channels, kernel_size=3, padding=1, stride=2,
                                  norm_layer=norm_layer, smooth=smooth, bit=bit))
        layers.append(QActivation(activation=nn.Hardswish, smooth=smooth, bit=bit))

        # building inverted residual blocks
        for cnf in inverted_residual_setting:
            layers.append(block(cnf, norm_layer, bit=bit, smooth=smooth))

        # building last several layers
        lastconv_input_channels = inverted_residual_setting[-1].out_channels
        lastconv_output_channels = 6 * lastconv_input_channels
        layers.append(FusedConv2d(lastconv_input_channels, lastconv_output_channels, kernel_size=1,
                                  norm_layer=norm_layer, smooth=smooth, bit=bit))
        layers.append(QActivation(activation=nn.Hardswish, smooth=smooth, bit=bit))

        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            FusedLinear(lastconv_output_channels, last_channel, bit=bit, smooth=smooth),
            QActivation(activation=nn.Hardswish, smooth=smooth, bit=bit),
            nn.Dropout(p=0.2, inplace=True),
            FusedLinear(last_channel, num_classes, bit=bit, smooth=smooth)
        )

        for m in self.modules():
            if isinstance(m, FusedConv2d):
                nn.init.kaiming_normal_(m.conv.weight, mode='fan_out')
                if m.conv.bias is not None:
                    nn.init.zeros_(m.conv.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, FusedLinear):
                nn.init.normal_(m.fc.weight, 0, 0.01)
                nn.init.zeros_(m.fc.bias)

    def _forward_impl(self, x: Tensor) -> Tensor:
        if self.training:
            if self.ema_init:
                self.in_range[0], self.in_range[1] = ema(x, self.in_range, self.smooth)
                s, z = calc_qparams(self.in_range[0], self.in_range[1], self.q_max)
                x = fake_quantize(x, s, z)
            else:
                self.in_range[0] = torch.min(x).item()
                self.in_range[1] = torch.max(x).item()
                self.ema_init = True

        x = self.features(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.classifier(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

    def show_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.show_params()

    def set_quantization_params(self):
        self.scale, self.zero_point = calc_qparams(self.in_range[0], self.in_range[1], self.q_max)
        prev_s, prev_z = self.features[0].set_qparams(self.scale, self.zero_point)
        prev_s, prev_z = self.features[1].set_qparams(prev_s, prev_z)

        for feature_idx in range(2, len(self.features)-2):
            for block_idx in range(len(self.features[feature_idx].block)):
                fused_module = self.features[feature_idx].block[block_idx]
                if isinstance(fused_module, FusedSqueezeExcitation):
                    prev_s, prev_z = fused_module.fc1.set_qparams(prev_s, prev_z)
                    prev_s, prev_z = fused_module.fc2.set_qparams(prev_s, prev_z)
                else:
                    prev_s, prev_z = fused_module.set_qparams(prev_s, prev_z)

        prev_s, prev_z = self.features[-2].set_qparams(prev_s, prev_z)
        prev_s, prev_z = self.features[-1].set_qparams(prev_s, prev_z)

        prev_s, prev_z = self.classifier[0].set_qparams(prev_s, prev_z)
        prev_s, prev_z = self.classifier[1].set_qparams(prev_s, prev_z)
        _, _ = self.classifier[3].set_qparams(prev_s, prev_z)


def _mobilenet_v3_conf(width_mult: float = 1.0, reduced_tail: bool = False, dilated: bool = False, **kwargs: Any):
    reduce_divider = 2 if reduced_tail else 1
    dilation = 2 if dilated else 1

    bneck_conf = partial(InvertedResidualConfig, width_mult=width_mult)
    adjust_channels = partial(InvertedResidualConfig.adjust_channels, width_mult=width_mult)

    inverted_residual_setting = [
        # inc, kernel, expanded channel, out_c, use_se, use_hs, stride, dilation
        bneck_conf(16, 3, 16, 16, True, "RE", 2, 1),  # C1
        bneck_conf(16, 3, 72, 24, False, "RE", 2, 1),  # C2
        bneck_conf(24, 3, 88, 24, False, "RE", 1, 1),
        bneck_conf(24, 5, 96, 40, True, "HS", 2, 1),  # C3
        bneck_conf(40, 5, 240, 40, True, "HS", 1, 1),
        bneck_conf(40, 5, 240, 40, True, "HS", 1, 1),
        bneck_conf(40, 5, 120, 48, True, "HS", 1, 1),
        bneck_conf(48, 5, 144, 48, True, "HS", 1, 1),
        bneck_conf(48, 5, 288, 96 // reduce_divider, True, "HS", 2, dilation),  # C4
        bneck_conf(96 // reduce_divider, 5, 576 // reduce_divider, 96 // reduce_divider, True, "HS", 1, dilation),
        bneck_conf(96 // reduce_divider, 5, 576 // reduce_divider, 96 // reduce_divider, True, "HS", 1, dilation),
    ]
    last_channel = adjust_channels(1024 // reduce_divider)  # C5
    return inverted_residual_setting, last_channel


def fused_mobilenet(smooth: float = 0.999, bit: int = 32, num_classes: int = 1000, **kwargs: Any) -> FusedMobileNet:
    inverted_residual_setting, last_channel = _mobilenet_v3_conf(**kwargs)
    return FusedMobileNet(smooth=smooth, bit=bit, num_classes=num_classes, inverted_residual_setting=inverted_residual_setting,
                          last_channel=last_channel, **kwargs)


def set_fused_mobilenet(fused, pre):
    """
        Copy pre model's params & set fused layers.
        Use fused architecture, but not really fused (use CONV & BN seperately)
    """
    # First layer
    fused.features[0] = copy_from_pretrained(fused.features[0], pre.features[0][0], pre.features[0][1])

    # InvertedResidual
    fused_feature_idx = 2
    for pre_feature_idx in range(1, len(pre.features)-1):
        fused_block_idx = 0
        for pre_block_idx in range(len(pre.features[pre_feature_idx].block)):
            if isinstance(fused.features[fused_feature_idx].block[fused_block_idx], QActivation):
                fused_block_idx += 1
            fused_module = fused.features[fused_feature_idx].block[fused_block_idx]
            pre_module = pre.features[pre_feature_idx].block[pre_block_idx]
            if isinstance(pre_module, ConvBNActivation):
                fused_module = copy_from_pretrained(fused_module, pre_module[0], pre_module[1])
            else:   # SqueezeExcitation
                fused_module.fc1 = copy_from_pretrained(fused_module.fc1, pre_module.fc1, None)
                fused_module.fc2 = copy_from_pretrained(fused_module.fc2, pre_module.fc2, None)
            fused_block_idx += 1
        fused_feature_idx += 1

    # Last conv
    fused.features[-2] = copy_from_pretrained(fused.features[-2], pre.features[-1][0], pre.features[-1][1])

    # Fully Connected
    fused.classifier[0] = copy_from_pretrained(fused.classifier[0], pre.classifier[0], None)
    fused.classifier[3] = copy_from_pretrained(fused.classifier[3], pre.classifier[3], None)
    return fused


def fold_mobilenet(model):
    # first layer
    model.features[0].fuse_conv_and_bn()

    # InvertedResidual
    for feature_idx in range(1, len(model.features)):
        if isinstance(model.features[feature_idx], InvertedResidual):
            for block_idx in range(len(model.features[feature_idx].block)):
                fused_module = model.features[feature_idx].block[block_idx]
                if isinstance(fused_module, FusedConv2d):
                    fused_module.fuse_conv_and_bn()
                elif isinstance(fused_module, FusedSqueezeExcitation):
                    fused_module.fc1.fuse_conv_and_bn()
                    fused_module.fc2.fuse_conv_and_bn()
        else:
            break

    # Last conv
    model.features[-2].fuse_conv_and_bn()

    return model


from operator import itemgetter

import torch
import torch.nn as nn

from .layers import *
from .quantization_utils import *

from functools import partial
from torch import Tensor
from torch.nn import functional as F
from typing import Any, Callable, Dict, List, Optional, Sequence

from torchvision.models.mobilenetv2 import _make_divisible
from .mobilenet import InvertedResidualConfig


class PCQSqueezeExcitation(nn.Module):
    # Implemented as described at Figure 4 of the MobileNetV3 paper
    def __init__(self, input_channels: int, squeeze_factor: int = 4, arg_dict: dict = None):
        super().__init__()
        self.arg_dict = arg_dict
        self.bit, self.smooth, self.num_clusters, self.runtime_helper, self.use_ste, self.quant_noise, self.qn_prob \
            = itemgetter('bit', 'smooth', 'cluster', 'runtime_helper', 'ste', 'quant_noise', 'qn_prob')(arg_dict)
        self.q_max = 2 ** self.bit - 1
        self.act_range = nn.Parameter(torch.zeros(self.num_clusters, 2), requires_grad=False)

        self.apply_ema = np.zeros(self.num_clusters, dtype=bool)

        squeeze_channels = _make_divisible(input_channels // squeeze_factor, 8)
        # self.fc1 = PCQConv2d(input_channels, squeeze_channels, kernel_size=1, bias=True,
        #                      activation=nn.ReLU6, arg_dict=arg_dict)
        self.fc1 = PCQConv2d(input_channels, squeeze_channels, kernel_size=1, bias=True,
                             activation=nn.ReLU, arg_dict=self.arg_dict)
        self.fc2 = PCQConv2d(squeeze_channels, input_channels, kernel_size=1, bias=True, arg_dict=self.arg_dict)
        self.QAct = PCQActivation(activation=nn.Hardsigmoid, arg_dict=self.arg_dict)

    def _scale(self, x):
        scale = F.adaptive_avg_pool2d(x, 1)
        scale = self.fc1(scale)
        scale = self.fc2(scale)
        scale = self.QAct(scale)
        return scale

    def forward(self, x):
        scale = self._scale(x)
        out = scale * x

        if not self.training:
            return out

        if self.runtime_helper.apply_fake_quantization and self.use_ste:
            _out = torch.zeros(out.shape).cuda()
        else:
            _out = out

        done = 0
        for i in range(self.runtime_helper.batch_cluster.shape[0]):
            c = self.runtime_helper.batch_cluster[i][0].item()
            n = self.runtime_helper.batch_cluster[i][1].item()
            if self.apply_ema[c]:
                self.act_range[c][0], self.act_range[c][1] = ema(out[done:done + n], self.act_range[c], self.smooth)
                if self.runtime_helper.apply_fake_quantization:
                    s, z = calc_qparams(self.act_range[c][0], self.act_range[c][1], self.q_max)
                    _out[done:done + n] = fake_quantize(out[done:done + n], s, z, self.q_max, use_ste=self.use_ste)
            else:
                self.act_range[c][0] = torch.min(out).item()
                self.act_range[c][1] = torch.max(out).item()
                self.apply_ema[c] = True
            done += n
        return _out

    def set_squeeze_qparams(self, s1, z1):
        prev_s, prev_z = self.fc1.set_qparams(s1, z1)
        prev_s, prev_z = self.fc2.set_qparams(prev_s, prev_z)
        _, _ = self.QAct.set_qparams(prev_s, prev_z)

        self.s3 = nn.Parameter(torch.zeros(self.num_clusters, dtype=torch.float32), requires_grad=False)
        self.z3 = nn.Parameter(torch.zeros(self.num_clusters, dtype=torch.int32), requires_grad=False)
        for c in range(self.num_clusters):
            self.s3[c], self.z3[c] = calc_qparams(self.act_range[c][0], self.act_range[c][1], self.q_max)
        return self.s3, self.z3


class PCQInvertedResidual(nn.Module):
    # Implemented as described at section 5 of MobileNetV3 paper
    def __init__(self, cnf: InvertedResidualConfig, norm_layer: Callable[..., nn.Module], arg_dict=None):
        super().__init__()
        if not (1 <= cnf.stride <= 2):
            raise ValueError('illegal stride value')
        self.arg_dict = arg_dict
        self.bit, self.smooth, self.num_clusters, self.runtime_helper, self.quant_noise, self.qn_prob \
            = itemgetter('bit', 'smooth', 'cluster', 'runtime_helper', 'quant_noise', 'qn_prob')(arg_dict)
        self.q_max = 2 ** self.bit - 1
        self.act_range = nn.Parameter(torch.zeros(self.num_clusters, 2), requires_grad=False)

        self.apply_ema = np.zeros(self.num_clusters, dtype=bool)

        self.use_res_connect = cnf.stride == 1 and cnf.input_channels == cnf.out_channels

        layers: List[nn.Module] = []
        self.activation = nn.ReLU if not cnf.use_hs else None

        # expand
        if cnf.expanded_channels != cnf.input_channels:
            layers.append(PCQConv2d(cnf.input_channels, cnf.expanded_channels, kernel_size=1,
                                      norm_layer=norm_layer, activation=self.activation, arg_dict=self.arg_dict))
            if cnf.use_hs:
                layers.append(PCQActivation(activation=nn.Hardswish, arg_dict=self.arg_dict))
        # depthwise
        stride = 1 if cnf.dilation > 1 else cnf.stride
        layers.append(PCQConv2d(cnf.expanded_channels, cnf.expanded_channels, kernel_size=cnf.kernel,
                                padding=(cnf.kernel-1)//2*cnf.dilation, stride=stride, dilation=cnf.dilation,
                                groups=cnf.expanded_channels, norm_layer=norm_layer,
                                activation=self.activation, arg_dict=self.arg_dict))
        if cnf.use_hs:
            layers.append(PCQActivation(activation=nn.Hardswish, arg_dict=self.arg_dict))
        if cnf.use_se:
            layers.append(PCQSqueezeExcitation(cnf.expanded_channels, arg_dict=self.arg_dict))

        # project
        layers.append(PCQConv2d(cnf.expanded_channels, cnf.out_channels, kernel_size=1,
                                norm_layer=norm_layer, arg_dict=self.arg_dict))

        self.block = nn.Sequential(*layers)
        self.out_channels = cnf.out_channels
        self._is_cn = cnf.stride > 1

    def forward(self, x):
        identity = x
        out = self.block(x)
        if self.use_res_connect:
            out += identity

        if not self.training:
            return out

        if self.runtime_helper.apply_fake_quantization and self.use_ste:
            _out = torch.zeros(out.shape).cuda()
        else:
            _out = out

        done = 0
        for i in range(self.runtime_helper.batch_cluster.shape[0]):
            c = self.runtime_helper.batch_cluster[i][0].item()
            n = self.runtime_helper.batch_cluster[i][1].item()
            if self.apply_ema[c]:
                self.act_range[c][0], self.act_range[c][1] = ema(out[done:done + n], self.act_range[c], self.smooth)
                if self.runtime_helper.apply_fake_quantization:
                    s, z = calc_qparams(self.act_range[c][0], self.act_range[c][1], self.q_max)
                    _out[done:done + n] = fake_quantize(out[done:done + n], s, z, self.q_max, use_ste=self.use_ste)
            else:
                self.act_range[c][0] = torch.min(out).item()
                self.act_range[c][1] = torch.max(out).item()
                self.apply_ema[c] = True
            done += n
        return _out

    def set_block_qparams(self, s1, z1):
        prev_s, prev_z = self.block[0].set_qparams(s1, z1)
        for i in range(1, len(self.block)):
            if isinstance(self.block[i], PCQSqueezeExcitation):
                prev_s, prev_z = self.block[i].set_squeeze_qparams(prev_s, prev_z)
            else:
                prev_s, prev_z = self.block[i].set_qparams(prev_s, prev_z)
        self.s3 = nn.Parameter(torch.zeros(self.num_clusters, dtype=torch.float32), requires_grad=False)
        self.z3 = nn.Parameter(torch.zeros(self.num_clusters, dtype=torch.int32), requires_grad=False)
        for c in range(self.num_clusters):
            self.s3[c], self.z3[c] = calc_qparams(self.act_range[c][0], self.act_range[c][1], self.q_max)
        return self.s3, self.z3


class PCQMobileNet(nn.Module):
    def __init__(
            self,
            inverted_residual_setting: List[InvertedResidualConfig],
            last_channel: int,
            arg_dict: dict = None,
            num_classes: int = 1000,
            block: Optional[Callable[..., nn.Module]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            dilation: int = 1,
            **kwargs: Any
    ) -> None:
        super(PCQMobileNet, self).__init__()
        self.arg_dict = arg_dict
        self.bit, self.smooth, self.num_clusters, self.runtime_helper, self.quant_noise, self.qn_prob \
            = itemgetter('bit', 'smooth', 'cluster', 'runtime_helper', 'quant_noise', 'qn_prob')(arg_dict)
        self.dilation = dilation
        self.q_max = 2 ** self.bit - 1
        self.in_range = nn.Parameter(torch.zeros(self.num_clusters, 2), requires_grad=False)

        self.apply_ema = np.zeros(self.num_clusters, dtype=bool)

        if not inverted_residual_setting:
            raise ValueError("The inverted_residual_setting should not be empty")
        elif not (isinstance(inverted_residual_setting, Sequence) and
                  all([isinstance(s, InvertedResidualConfig) for s in inverted_residual_setting])):
            raise TypeError("The inverted_residual_setting should be List[InvertedResidualConfig]")

        if block is None:
            block = PCQInvertedResidual

        if norm_layer is None:
            norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)

        layers: List[nn.Module] = []

        # building first layer
        firstconv_output_channels = inverted_residual_setting[0].input_channels
        layers.append(PCQConv2d(3, firstconv_output_channels, kernel_size=3, padding=self.dilation, stride=2,
                                  norm_layer=norm_layer, arg_dict=self.arg_dict))
        layers.append(PCQActivation(activation=nn.Hardswish, arg_dict=self.arg_dict))

        # building inverted residual blocks
        for cnf in inverted_residual_setting:
            layers.append(block(cnf, norm_layer, arg_dict=self.arg_dict))

        # building last several layers
        lastconv_input_channels = inverted_residual_setting[-1].out_channels
        lastconv_output_channels = 6 * lastconv_input_channels
        layers.append(PCQConv2d(lastconv_input_channels, lastconv_output_channels, kernel_size=1,
                                norm_layer=norm_layer, arg_dict=self.arg_dict))
        layers.append(PCQActivation(activation=nn.Hardswish, arg_dict=self.arg_dict))

        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            PCQLinear(lastconv_output_channels, last_channel, arg_dict=self.arg_dict),
            PCQActivation(activation=nn.Hardswish, arg_dict=self.arg_dict),
            PCQLinear(last_channel, num_classes, arg_dict=self.arg_dict)
        )

        for m in self.modules():
            if isinstance(m, PCQConv2d):
                nn.init.kaiming_normal_(m.conv.weight, mode='fan_out')
                if m.conv.bias is not None:
                    nn.init.zeros_(m.conv.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, PCQLinear):
                nn.init.normal_(m.fc.weight, 0, 0.01)
                nn.init.zeros_(m.fc.bias)

    def _forward_impl(self, x: Tensor) -> Tensor:
        if self.training:
            done = 0
            for i in range(self.runtime_helper.batch_cluster.shape[0]):
                c = self.runtime_helper.batch_cluster[i][0].item()
                n = self.runtime_helper.batch_cluster[i][1].item()
                if self.apply_ema[c]:
                    self.in_range[c][0], self.in_range[c][1] = ema(x[done:done + n], self.in_range[c], self.smooth)
                    if self.runtime_helper.apply_fake_quantization:
                        s, z = calc_qparams(self.in_range[c][0], self.in_range[c][1], self.q_max)
                        x[done:done + n] = fake_quantize(x[done:done + n], s, z)
                else:
                    self.in_range[c][0] = torch.min(x).item()
                    self.in_range[c][1] = torch.max(x).item()
                    self.apply_ema[c] = True
                done += n

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
        self.scale = nn.Parameter(torch.zeros(self.num_clusters, dtype=torch.float32), requires_grad=False)
        self.zero_point = nn.Parameter(torch.zeros(self.num_clusters, dtype=torch.int32), requires_grad=False)
        for c in range(self.num_clusters):
            self.scale[c], self.zero_point[c] = calc_qparams(self.in_range[c][0], self.in_range[c][1], self.q_max)
        prev_s, prev_z = self.features[0].set_qparams(self.scale, self.zero_point)
        prev_s, prev_z = self.features[1].set_qparams(prev_s, prev_z)

        for feature_idx in range(2, len(self.features)-2):
            prev_s, prev_z = self.features[feature_idx].set_block_qparams(prev_s, prev_z)

        prev_s, prev_z = self.features[-2].set_qparams(prev_s, prev_z)
        prev_s, prev_z = self.features[-1].set_qparams(prev_s, prev_z)

        prev_s, prev_z = self.classifier[0].set_qparams(prev_s, prev_z)
        prev_s, prev_z = self.classifier[1].set_qparams(prev_s, prev_z)
        _, _ = self.classifier[2].set_qparams(prev_s, prev_z)


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


def pcq_mobilenet(arg_dict: dict, **kwargs: Any) -> PCQMobileNet:
    inverted_residual_setting, last_channel = _mobilenet_v3_conf(**kwargs)
    return PCQMobileNet(inverted_residual_setting, last_channel, arg_dict, **kwargs)

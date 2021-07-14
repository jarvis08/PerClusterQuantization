import torch
import torch.nn as nn
from torch import nn, Tensor

from .layers import *
from .quantization_utils import *
from .mobilenet import _mobilenet_v3_conf, InvertedResidualConfig
from .fused_mobilenet import FusedSqueezeExcitation
from torchvision.models.mobilenetv2 import _make_divisible
import torch.nn.quantized.functional


class QuantizedSqueezeExcitation(nn.Module):
    # Implemented as described at Figure 4 of the MobileNetV3 paper
    def __init__(self, input_channels: int, squeeze_factor: int = 4, smooth: float = 0.999, bit: int = 32):
        super().__init__()
        squeeze_channels = _make_divisible(input_channels // squeeze_factor, 8)
        self.fc1 = QuantizedConv2d(input_channels, squeeze_channels, 1, bias=True, bit=bit)
        self.fc2 = QuantizedConv2d(squeeze_channels, input_channels, 1, activation_layer=F.hardsigmoid(inplace=True), bias=True, bit=bit)

    def _scale(self, input: Tensor, inplace: bool) -> Tensor:
        scale = F.adaptive_avg_pool2d(input, 1)
        scale = self.fc1(scale)
        scale = self.fc2(scale)
        return scale

    def forward(self, input: Tensor) -> Tensor:
        scale = self._scale(input, True)
        return scale * input


class InvertedResidual(nn.Module):
    # Implemented as described at section 5 of MobileNetV3 paper
    def __init__(self, cnf: InvertedResidualConfig, norm_layer: Callable[..., nn.Module], smooth: float = 0.999,
                 num_clusters: int = 1, bit: int = 8):
        super().__init__()
        if not (1 <= cnf.stride <= 2):
            raise ValueError('illegal stride value')

        self.use_res_connect = cnf.stride == 1 and cnf.input_channels == cnf.out_channels

        layers: List[nn.Module] = []
        activation_layer = nn.Hardswish() if cnf.use_hs else None

        # expand
        if cnf.expanded_channels != cnf.input_channels:
            layers.append(QuantizedConv2d(cnf.input_channels, cnf.expanded_channels, kernel_size=1, activation_layer=activation_layer, bit=bit))

        # depthwise
        stride = 1 if cnf.dilation > 1 else cnf.stride
        # groups = input_channel -> input_channel / input_channel = 1 -> operates on single channel
        layers.append(QuantizedConv2d(cnf.expanded_channels, cnf.expanded_channels, kernel_size=cnf.kernel,
                                      padding=(cnf.kernel - 1) // 2 * cnf.dilation, stride=stride, dilation=cnf.dilation,
                                      groups=cnf.expanded_channels, activation_layer=activation_layer, bit=bit))

        if cnf.use_se:
            layers.append(QuantizedSqueezeExcitation(cnf.expanded_channels, bit=bit))

        # project
        layers.append(QuantizedConv2d(cnf.expanded_channels, cnf.out_channels, kernel_size=1, bit=bit))

        # shortcut
        if self.use_res_connect:
            self.shortcut = QuantizedShortcut(bit=bit, num_clusters=num_clusters)
        self.block = nn.Sequential(*layers)
        self.out_channels = cnf.out_channels
        self._is_cn = cnf.stride > 1

    def forward(self, input: Tensor) -> Tensor:
        result = self.block(input)
        if self.use_res_connect:
            result = self.shortcut(input, result, None)
        return result


class QuantizedMobileNet(nn.Module):
    def __init__(
            self,
            inverted_residual_setting: List[InvertedResidualConfig],
            last_channel: int,
            num_classes: int = 1000,
            block: Optional[Callable[..., nn.Module]] = None,
            bit: int = 8,
            num_clusters=1,
            **kwargs: Any
    ) -> None:
        super().__init__()
        self.bit = bit
        self.in_range = nn.Parameter(torch.zeros(2), requires_grad=False)
        self.ema_init = False

        self.num_clusters = num_clusters
        t_init = list(range(num_clusters)) if num_clusters > 1 else 0
        self.scale = nn.Parameter(torch.tensor(t_init, dtype=torch.float32), requires_grad=False)
        self.zero_point = nn.Parameter(torch.tensor(t_init, dtype=torch.int32), requires_grad=False)

        if not inverted_residual_setting:
            raise ValueError("The inverted_residual_setting should not be empty")
        elif not (isinstance(inverted_residual_setting, Sequence) and
                  all([isinstance(s, InvertedResidualConfig) for s in inverted_residual_setting])):
            raise TypeError("The inverted_residual_setting should be List[InvertedResidualConfig]")

        if block is None:
            block = InvertedResidual

        layers: List[nn.Module] = []

        # building first layer
        firstconv_output_channels = inverted_residual_setting[0].input_channels
        layers.append(QuantizedConv2d(3, firstconv_output_channels, kernel_size=3, padding=1, stride=2, activation_layer=nn.Hardswish(), bit=bit))

        # building inverted residual blocks
        for cnf in inverted_residual_setting:
            layers.append(block(cnf, bit=self.bit))

        # building last several layers
        lastconv_input_channels = inverted_residual_setting[-1].out_channels
        lastconv_output_channels = 6 * lastconv_input_channels
        layers.append(QuantizedConv2d(lastconv_input_channels, lastconv_output_channels, kernel_size=1, activation_layer=nn.Hardswish(), bit=bit))

        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            QuantizedLinear(lastconv_output_channels, last_channel, activation_layer=nn.Hardswish(), bit=bit),
            nn.Dropout(p=0.2, inplace=True),
            QuantizedLinear(last_channel, num_classes, bit=bit)
        )

        for m in self.modules():
            if isinstance(m, QuantizedConv2d):
                nn.init.kaiming_normal_(m.conv.weight, mode='fan_out')
                if m.conv.bias is not None:
                    nn.init.zeros_(m.conv.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, QuantizedLinear):
                nn.init.normal_(m.fc.weight, 0, 0.01)
                nn.init.zeros_(m.fc.bias)

    def _forward_impl(self, x: Tensor) -> Tensor:
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


def quantized_mobilenet(bit: int = 32, num_classes: int = 1000, **kwargs: Any) -> QuantizedMobileNet:
    inverted_residual_setting, last_channel = _mobilenet_v3_conf(**kwargs)
    return QuantizedMobileNet(bit=bit, num_classes=num_classes, inverted_residual_setting=inverted_residual_setting,
                          last_channel=last_channel, **kwargs)

def set_shortcut_qparams(m, s_bypass, z_bypass, s_prev, z_prev, s3, z3):
    m.s_bypass = nn.Parameter(s_bypass, requires_grad=False)
    m.z_bypass = nn.Parameter(z_bypass, requires_grad=False)
    m.s_prev = nn.Parameter(s_prev, requires_grad=False)
    m.z_prev = nn.Parameter(z_prev, requires_grad=False)
    m.s3 = nn.Parameter(s3, requires_grad=False)
    m.z3 = nn.Parameter(z3, requires_grad=False)

    if m.num_clusters > 1:
        for c in range(m.num_clusters):
            m.M0_bypass[c], m.shift_bypass[c] = quantize_shortcut_M(s_bypass[c] / s3[c])
            m.M0_prev[c], m.shift_prev[c] = quantize_shortcut_M(s_prev[c] / s3[c])
    else:
        m.M0_bypass, m.shift_bypass = quantize_shortcut_M(s_bypass / s3)
        m.M0_prev, m.shift_prev = quantize_shortcut_M(s_prev / s3)
    return m

def quantize_block(_fp, _int):
    for i in range(len(_int)):
        _int[i].conv1 = quantize(_fp[i].conv1, _int[i].conv1)
        _int[i].conv2 = quantize(_fp[i].conv2, _int[i].conv2)
        if _int[i].downsample:
            _int[i].downsample = quantize(_fp[i].downsample, _int[i].downsample)
            _int[i].shortcut = set_shortcut_qparams(_int[i].shortcut,
                                                    _int[i].downsample.s3, _int[i].downsample.z3,
                                                    _int[i].conv2.s3, _int[i].conv2.z3,
                                                    _fp[i].s3, _fp[i].z3)
        else:
            _int[i].shortcut = set_shortcut_qparams(_int[i].shortcut,
                                                    _int[i].conv1.s1, _int[i].conv1.z1,
                                                    _int[i].conv2.s3, _int[i].conv2.z3,
                                                    _fp[i].s3, _fp[i].z3)
    return _int


def quantize_mobilenet(fp_model, int_model):
    int_model.scale = torch.nn.Parameter(fp_model.scale, requires_grad=False)
    int_model.zero_point = torch.nn.Parameter(fp_model.zero_point, requires_grad=False)
    int_model.features[0] = quantize(fp_model.features[0], int_model.features[0])

    for feature_idx in range(1, len(fp_model.features)):
        if isinstance(fp_model.features[feature_idx], InvertedResidual):
            for block_idx in range(len(fp_model.features[feature_idx].block)):
                fused_module = fp_model.features[feature_idx].block[block_idx]
                int_module = int_model.features[feature_idx].block[block_idx]
                if isinstance(fused_module, FusedConv2d):
                    int_module = quantize(fused_module, int_module)
                else:  # SqueezeExcitation
                    int_module.fc1 = quantize(fused_module.fc1, int_module.fc1)
                    int_module.fc2 = quantize(fused_module.fc2, int_module.fc2)
        else:
            break

    int_model.features[-1] = quantize(fp_model.features[-1], int_model.features[-1])

    int_model.classifer[0] = quantize(fp_model.classifer[0], int_model.classifer[0])
    int_model.classifer[2] = quantize(fp_model.classifer[2], int_model.classifer[2])


def calculate_Hardswish(module):
    lookup_table = nn.Parameter(torch.zeros(2 ** module.bit))
    output_min, output_max = module.activation.layer.act_range[0].type(torch.cuda.IntTensor), module.activation.layer.act_range[1].type(torch.cuda.IntTensor)
    inverse_output_scale = 1.0 / module.s3
    for i in range(module.q_max + 1):
        if module.bit == 8:
            y = module.s2 * (i - 128 - module.z2)
        else:
            y = module.s2 * (i - module.z2)
        y2 = y + 3.0
        y2 = y2 if y2 > 0 else 0
        y2 = y2 if y2 < 6.0 else 6.0
        y2 = y * y2 / 6.0

        scaled_hardswish = inverse_output_scale * y2 + module.z3
        if scaled_hardswish < output_min:
            scaled_hardswish = output_min
        if scaled_hardswish > output_max:
            scaled_hardswish = output_max
        # lookup_table[i] = int(round(scaled_hardswish)).type(torch.cuda.ByteTensor)
        lookup_table[i] = int(round(scaled_hardswish))

    return lookup_table


def calculate_Hardsigmoid(module):
    lookup_table = nn.Parameter(torch.zeros(2 ** module.bit))
    output_min, output_max = module.activation.layer.act_range[0].type(torch.cuda.IntTensor), module.activation.layer.act_range[1].type(torch.cuda.IntTensor)
    inverse_output_scale = 1.0 / module.s3
    for i in range(module.q_max + 1):
        if module.bit == 8:
            y = module.s2 * (i - 128 - module.z2)
        else:
            y = module.s2 * (i - module.z2)
        y2 = y + 3.0
        y2 = y2 if y2 > 0 else 0
        y2 = y2 if y2 < 6.0 else 6.0
        y2 = y2 / 6.0

        scaled_hardswish = inverse_output_scale * y2 + module.z3
        if scaled_hardswish < output_min:
            scaled_hardswish = output_min
        if scaled_hardswish > output_max:
            scaled_hardswish = output_max
        # lookup_table[i] = int(round(scaled_hardswish)).type(torch.cuda.ByteTensor)
        lookup_table[i] = int(round(scaled_hardswish))

    return lookup_table


def make_lookup_table_Hardswish(fp_model, int_model):
    assert fp_model.bit != 4 and fp_model.bit != 8, "Bit should be 4 or 8"

    if isinstance(fp_model.features[0].activation_layer, QActivation):
        int_model.features[0].lookup_table = calculate_Hardswish(fp_model.features[0])

    for feature_idx in range(1, len(fp_model.features)):
        if isinstance(fp_model.features[feature_idx], InvertedResidual):
            for block_idx in range(len(fp_model.features[feature_idx].block)):
                cur_module = fp_model.features[feature_idx].block[block_idx]
                if isinstance(cur_module, FusedSqueezeExcitation):
                    if isinstance(cur_module.fc1.activation_layer, QActivation):
                        int_model.cur_module.fc1.lookup_table = calculate_Hardswish(cur_module.fc1)
                    if isinstance(cur_module.fc2.activation_layer, QActivation):
                        int_model.cur_module.fc2.lookup_table = calculate_Hardsigmoid(cur_module.fc1)
                else:
                    if isinstance(cur_module, QActivation):
                        int_model.cur_module.lookup_table = calculate_Hardswish(cur_module)
        else:
            break

    # Last conv
    if isinstance(fp_model.features[-1].activation_layer, QActivation):
        int_model.features[-1].lookup_table = calculate_Hardswish(fp_model.features[-1])

    # Fully Connected
    if isinstance(fp_model.classifier[0].activation_layer, QActivation):
        int_model.classfier[0].lookuup_table = calculate_Hardswish(fp_model.classifier[0])
    if isinstance(fp_model.classifier[2].activation_layer, QActivation):
        int_model.classfier[2].lookuup_table = calculate_Hardswish(fp_model.classifier[2])

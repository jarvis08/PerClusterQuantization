from operator import itemgetter

import torch
import torch.nn as nn
from torch import nn, Tensor

from .layers import *
from .quantization_utils import *
from .mobilenet import _mobilenet_v3_conf, InvertedResidualConfig
from .fused_mobilenet import FusedSqueezeExcitation
from torchvision.models.mobilenetv2 import _make_divisible
import torch.nn.quantized.functional
from typing import Any, Callable, Dict, List, Optional, Sequence

from typing import Any, Callable, Dict, List, Optional, Sequence


class QuantizedSqueezeExcitation(nn.Module):
    # Implemented as described at Figure 4 of the MobileNetV3 paper
    def __init__(self, input_channels: int, squeeze_factor: int = 4, arg_dict: dict = None):
        super().__init__()
        self.bit, self.num_clusters, self.runtime_helper = itemgetter('bit', 'cluster', 'runtime_helper')(arg_dict)
        squeeze_channels = _make_divisible(input_channels // squeeze_factor, 8)
        self.fc1 = QuantizedConv2d(input_channels, squeeze_channels, kernel_size=1, arg_dict=arg_dict)
        self.fc2 = QuantizedConv2d(squeeze_channels, input_channels, kernel_size=1, activation='Hardsigmoid', arg_dict=arg_dict)
        self.mul = QuantizedMul(arg_dict=arg_dict)

    def _scale(self, x: Tensor) -> Tensor:
        identity = x
        scale = F.adaptive_avg_pool2d(x, 1)
        scale = self.fc1(scale)
        scale = self.fc2(scale)
        scale = self.mul(scale, identity)
        return scale

    def forward(self, x: Tensor) -> Tensor:
        out = self._scale(x)
        # out = scale * identity
        return out


class InvertedResidual(nn.Module):
    # Implemented as described at section 5 of MobileNetV3 paper
    def __init__(self, cnf: InvertedResidualConfig,  arg_dict: dict = None):
        super().__init__()
        if not (1 <= cnf.stride <= 2):
            raise ValueError('illegal stride value')

        self.bit, self.num_clusters, self.runtime_helper = itemgetter('bit', 'cluster', 'runtime_helper')(arg_dict)
        self.q_max = 2 ** self.bit - 1

        self.use_res_connect = cnf.stride == 1 and cnf.input_channels == cnf.out_channels
        self.activation = 'Hardswish' if cnf.use_hs else None

        layers: List[nn.Module] = []

        # expand
        if cnf.expanded_channels != cnf.input_channels:
            layers.append(QuantizedConv2d(cnf.input_channels, cnf.expanded_channels, activation=self.activation,
                                          kernel_size=1, arg_dict=arg_dict))

        # depthwise
        stride = 1 if cnf.dilation > 1 else cnf.stride
        # groups = input_channel -> input_channel / input_channel = 1 -> operates on single channel
        layers.append(QuantizedConv2d(cnf.expanded_channels, cnf.expanded_channels, kernel_size=cnf.kernel,
                                      padding=(cnf.kernel - 1) // 2 * cnf.dilation, stride=stride, dilation=cnf.dilation,
                                      activation=self.activation, groups=cnf.expanded_channels, arg_dict=arg_dict))

        if cnf.use_se:
            layers.append(QuantizedSqueezeExcitation(cnf.expanded_channels, arg_dict=arg_dict))

        # project
        layers.append(QuantizedConv2d(cnf.expanded_channels, cnf.out_channels, kernel_size=1, arg_dict=arg_dict))

        self.block = nn.Sequential(*layers)
        # shortcut
        if self.use_res_connect:
            self.shortcut = QuantizedAdd(arg_dict=arg_dict)
        self.out_channels = cnf.out_channels
        self._is_cn = cnf.stride > 1

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.block(x)
        if self.use_res_connect:
            out = self.shortcut(identity, out)
        return out


class QuantizedMobileNet(nn.Module):
    def __init__(
            self,
            inverted_residual_setting: List[InvertedResidualConfig],
            last_channel: int,
            arg_dict,
            num_classes: int = 1000,
            block: Optional[Callable[..., nn.Module]] = None,
            dilation: int = 1,
            **kwargs: Any
    ) -> None:
        super().__init__()
        self.bit, self.num_clusters, self.runtime_helper = itemgetter('bit', 'cluster', 'runtime_helper')(arg_dict)
        self.dilation = dilation
        self.q_max = 2 ** self.bit - 1

        t_init = list(range(self.num_clusters)) if self.num_clusters > 1 else 0
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
        layers.append(QuantizedConv2d(3, firstconv_output_channels, kernel_size=3, padding=self.dilation, stride=2,
                                      activation='Hardswish', arg_dict=arg_dict))

        # building inverted residual blocks
        for cnf in inverted_residual_setting:
            layers.append(block(cnf, arg_dict=arg_dict))

        # building last several layers
        lastconv_input_channels = inverted_residual_setting[-1].out_channels
        lastconv_output_channels = 6 * lastconv_input_channels
        layers.append(QuantizedConv2d(lastconv_input_channels, lastconv_output_channels, kernel_size=1,
                                      activation='Hardswish', arg_dict=arg_dict))

        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            QuantizedLinear(lastconv_output_channels, last_channel, activation='Hardswish', arg_dict=arg_dict),
            QuantizedLinear(last_channel, num_classes, arg_dict=arg_dict)
        )

    def _forward_impl(self, x: Tensor) -> Tensor:
        if self.runtime_helper.batch_cluster is not None:
            x = quantize_matrix_4d(x, self.scale, self.zero_point, self.runtime_helper.batch_cluster, self.q_max)
        else:
            x = quantize_matrix(x, self.scale, self.zero_point, self.q_max)

        x = self.features(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.classifier(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def quantized_mobilenet(arg_dict: dict, **kwargs: Any) -> QuantizedMobileNet:
    inverted_residual_setting, last_channel = _mobilenet_v3_conf(**kwargs)
    return QuantizedMobileNet(inverted_residual_setting, last_channel, arg_dict, **kwargs)

def set_shortcut_qparams(m, s_bypass, z_bypass, s_prev, z_prev, s3, z3):
    m.s_bypass = nn.Parameter(s_bypass, requires_grad=False)
    m.z_bypass = nn.Parameter(z_bypass, requires_grad=False)
    m.s_prev = nn.Parameter(s_prev, requires_grad=False)
    m.z_prev = nn.Parameter(z_prev, requires_grad=False)
    m.s3 = nn.Parameter(s3, requires_grad=False)
    m.z3 = nn.Parameter(z3, requires_grad=False)

    if m.num_clusters > 1:
        for c in range(m.num_clusters):
            m.M0_bypass[c], m.shift_bypass[c] = quantize_M(s_bypass[c] / s3[c])
            m.M0_prev[c], m.shift_prev[c] = quantize_M(s_prev[c] / s3[c])
    else:
        m.M0_bypass, m.shift_bypass = quantize_M(s_bypass / s3)
        m.M0_prev, m.shift_prev = quantize_M(s_prev / s3)
    return m

def set_mul_qparams(_int, s_bypass, z_bypass, s_prev, z_prev, s3, z3):
    _int.s_bypass = nn.Parameter(s_bypass, requires_grad=False)
    _int.z_bypass = nn.Parameter(z_bypass, requires_grad=False)
    _int.s_prev = nn.Parameter(s_prev, requires_grad=False)
    _int.z_prev = nn.Parameter(z_prev, requires_grad=False)
    _int.s3 = nn.Parameter(s3, requires_grad=False)
    _int.z3 = nn.Parameter(z3, requires_grad=False)

    if _int.num_clusters > 1:
        for c in range(_int.num_clusters):
            _int.M0[c], _int.shift[c] = quantize_M(s_bypass[c] * s_prev[c] / _int.s3[c])
    else:
        _int.M0, _int.shift = quantize_M(s_bypass * s_prev / _int.s3)
    return _int

def set_activation(_fp, _int):
    if _int.num_clusters > 1:
        for i in range(_int.num_clusters):
            _int.s_activation[i] = torch.nn.Parameter(_fp.s3[i], requires_grad=False)
            _int.z_activation[i] = torch.nn.Parameter(_fp.z3[i], requires_grad=False)
            _int.hardswish_6[i].copy_(torch.round(6 / _int.s_activation[i] + _int.z_activation[i]))
            _int.hardswish_3[i].copy_(torch.round(3 / _int.s_activation[i] + _int.z_activation[i]))
    else:
        _int.s_activation = torch.nn.Parameter(_fp.s3, requires_grad=False)
        _int.z_activation = torch.nn.Parameter(_fp.z3, requires_grad=False)
        _int.hardswish_6.copy_(nn.Parameter(torch.round(6 / _int.s_activation + _int.z_activation)))
        _int.hardswish_3.copy_(nn.Parameter(torch.round(3 / _int.s_activation + _int.z_activation)))


def quantize_mobilenet(fp_model, int_model):
    int_model.scale = torch.nn.Parameter(fp_model.scale, requires_grad=False)
    int_model.zero_point = torch.nn.Parameter(fp_model.zero_point, requires_grad=False)
    int_model.features[0] = quantize(fp_model.features[0], int_model.features[0])
    set_activation(fp_model.features[1], int_model.features[0])

    fp_feature_idx = 2
    for int_feature_idx in range(1, len(int_model.features)-1):
        fp_block_idx = 0
        for block_idx in range(len(int_model.features[int_feature_idx].block)):
            if isinstance(fp_model.features[fp_feature_idx].block[fp_block_idx], QActivation):
                set_activation(fp_model.features[fp_feature_idx].block[fp_block_idx], int_model.features[int_feature_idx].block[block_idx-1])
                fp_block_idx += 1
            fp_module = fp_model.features[fp_feature_idx].block[fp_block_idx]
            int_module = int_model.features[int_feature_idx].block[block_idx]
            if isinstance(fp_module, FusedConv2d):
                int_module = quantize(fp_module, int_module)
            elif isinstance(fp_module, FusedSqueezeExcitation):
                int_module.fc1 = quantize(fp_module.fc1, int_module.fc1)
                int_module.fc2 = quantize(fp_module.fc2, int_module.fc2)
                set_activation(fp_module.QAct, int_module.fc2)
                int_module.mul = set_mul_qparams(int_module.mul, fp_module.QAct.s3, fp_module.QAct.z3, fp_module.s1,
                                                 fp_module.z1, fp_module.s3, fp_module.z3)
            fp_block_idx += 1
        if int_model.features[int_feature_idx].use_res_connect:
            int_model.features[int_feature_idx].shortcut = set_shortcut_qparams(int_model.features[int_feature_idx].shortcut,
                                                    int_model.features[int_feature_idx].block[0].s1,
                                                    int_model.features[int_feature_idx].block[0].z1,
                                                    int_model.features[int_feature_idx].block[-1].s3,
                                                    int_model.features[int_feature_idx].block[-1].z3,
                                                    fp_model.features[fp_feature_idx].s3,
                                                    fp_model.features[fp_feature_idx].z3)
        fp_feature_idx += 1

    int_model.features[-1] = quantize(fp_model.features[-2], int_model.features[-1])
    set_activation(fp_model.features[-1], int_model.features[-1])

    int_model.classifier[0] = quantize(fp_model.classifier[0], int_model.classifier[0])
    set_activation(fp_model.classifier[1], int_model.classifier[0])
    int_model.classifier[1] = quantize(fp_model.classifier[2], int_model.classifier[1])
    return int_model


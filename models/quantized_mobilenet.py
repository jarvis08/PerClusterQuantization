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
    def __init__(self, input_channels: int, squeeze_factor: int = 4, smooth: float = 0.999, bit: int = 32, num_clusters=1):
        super().__init__()
        self.bit = bit
        self.num_clusters =num_clusters
        squeeze_channels = _make_divisible(input_channels // squeeze_factor, 8)
        self.fc1 = QuantizedConv2d(input_channels, squeeze_channels, kernel_size=1, bias=True, bit=bit, num_clusters=num_clusters)
        self.fc2 = QuantizedConv2d(squeeze_channels, input_channels, kernel_size=1, bias=True, activation='Hardsigmoid', bit=bit,
                                    num_clusters=num_clusters)

    def _scale(self, x: Tensor, cluster_info=None) -> Tensor:
        scale = F.adaptive_avg_pool2d(x, 1)
        scale = self.fc1((scale, cluster_info))
        scale = self.fc2((scale, cluster_info))
        return scale

    def forward(self, x: Tensor) -> Tensor:
        _x = x[0]
        cluster_info = x[1]

        scale = self._scale(_x, cluster_info)
        out = scale * _x
        return (out, cluster_info)


class InvertedResidual(nn.Module):
    # Implemented as described at section 5 of MobileNetV3 paper
    def __init__(self, cnf: InvertedResidualConfig, num_clusters: int = 1, bit: int = 8):
        super().__init__()
        if not (1 <= cnf.stride <= 2):
            raise ValueError('illegal stride value')

        self.bit = bit 
        self.q_max = 2 ** self.bit - 1
        self.num_clusters = num_clusters

        self.use_res_connect = cnf.stride == 1 and cnf.input_channels == cnf.out_channels
        self.activation = 'Hardswish' if cnf.use_hs else None

        layers: List[nn.Module] = []

        # expand
        if cnf.expanded_channels != cnf.input_channels:
            layers.append(QuantizedConv2d(cnf.input_channels, cnf.expanded_channels, activation=self.activation,
                                          kernel_size=1, bit=bit, num_clusters=num_clusters))

        # depthwise
        stride = 1 if cnf.dilation > 1 else cnf.stride
        # groups = input_channel -> input_channel / input_channel = 1 -> operates on single channel
        layers.append(QuantizedConv2d(cnf.expanded_channels, cnf.expanded_channels, kernel_size=cnf.kernel,
                                      padding=(cnf.kernel - 1) // 2 * cnf.dilation, stride=stride, dilation=cnf.dilation,
                                      activation=self.activation, groups=cnf.expanded_channels, bit=bit,
                                      num_clusters=num_clusters))

        if cnf.use_se:
            layers.append(QuantizedSqueezeExcitation(cnf.expanded_channels, bit=bit, num_clusters=num_clusters))

        # project
        layers.append(QuantizedConv2d(cnf.expanded_channels, cnf.out_channels, kernel_size=1, bit=bit, num_clusters=num_clusters))

        self.block = nn.Sequential(*layers)
        # shortcut
        if self.use_res_connect:
            self.shortcut = QuantizedShortcut(bit=bit, num_clusters=num_clusters)
        self.out_channels = cnf.out_channels
        self._is_cn = cnf.stride > 1

    def forward(self, x: Tensor) -> Tensor:
        _x = x[0]
        cluster_info = x[1]
        out = self.block((_x, cluster_info))
        if self.use_res_connect:
            out = self.shortcut(_x, out, cluster_info)
        return (out, cluster_info)


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
        self.q_max = 2 ** self.bit - 1
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
        layers.append(QuantizedConv2d(3, firstconv_output_channels, kernel_size=3, padding=1, stride=2,
                                      activation='Hardswish', bit=bit, num_clusters=num_clusters))

        # building inverted residual blocks
        for cnf in inverted_residual_setting:
            layers.append(block(cnf, bit=self.bit, num_clusters=num_clusters))

        # building last several layers
        lastconv_input_channels = inverted_residual_setting[-1].out_channels
        lastconv_output_channels = 6 * lastconv_input_channels
        layers.append(QuantizedConv2d(lastconv_input_channels, lastconv_output_channels, kernel_size=1,
                                      activation='Hardswish', bit=bit, num_clusters=num_clusters))

        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            QuantizedLinear(lastconv_output_channels, last_channel, activation='Hardswish', bit=bit, num_clusters=num_clusters),
            QuantizedLinear(last_channel, num_classes, bit=bit, num_clusters=num_clusters)
        )

    def _forward_impl(self, x: Tensor, cluster_info=None) -> Tensor:
        if cluster_info is not None:
            done = 0
            for i in range(cluster_info.shape[0]):
                c = cluster_info[i][0].item()
                n = cluster_info[i][1].item()
                x[done:done + n] = quantize_matrix(x[done:done + n], self.scale[c], self.zero_point[c], self.q_max)
                done += n
        else:
            x = quantize_matrix(x, self.scale, self.zero_point, self.q_max)

        x = self.features((x, cluster_info))

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.classifier((x, cluster_info))

        return x

    def forward(self, x: Tensor, cluster_info=None) -> Tensor:
        return self._forward_impl(x, cluster_info)

    def show_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.show_params()


def quantized_mobilenet(bit: int = 8, num_classes: int = 1000, num_clusters=1, **kwargs: Any) -> QuantizedMobileNet:
    inverted_residual_setting, last_channel = _mobilenet_v3_conf(**kwargs)
    return QuantizedMobileNet(bit=bit, num_classes=num_classes, inverted_residual_setting=inverted_residual_setting,
                          last_channel=last_channel, num_clusters=num_clusters, **kwargs)

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
            fp_block_idx += 1
        if  int_model.features[int_feature_idx].use_res_connect:
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


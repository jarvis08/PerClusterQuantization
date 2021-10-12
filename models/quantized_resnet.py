from operator import itemgetter

import torch
import torch.nn as nn

from .layers import *
from .quantization_utils import *


def quantized_conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1, bias=False, arg_dict=None):
    """3x3 convolution with padding"""
    return QuantizedConv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation,
                           groups=groups, dilation=dilation, bias=bias, arg_dict=arg_dict)


def quantized_conv1x1(in_planes, out_planes, stride=1, bias=False, arg_dict=None):
    """1x1 convolution"""
    return QuantizedConv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=bias, arg_dict=arg_dict)


class QuantizedBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1,
                 arg_dict:dict = None):
        super(QuantizedBasicBlock, self).__init__()
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.downsample = downsample
        self.stride = stride

        self.num_clusters = itemgetter('cluster')(arg_dict)

        self.conv1 = quantized_conv3x3(inplanes, planes, stride, arg_dict=arg_dict)
        self.conv2 = quantized_conv3x3(planes, planes, arg_dict=arg_dict)
        self.shortcut = QuantizedAdd(arg_dict=arg_dict)

        if self.downsample is not None:
            self.bn_down = QuantizedBn2d(planes, arg_dict=arg_dict)
        self.bn1 = QuantizedBn2d(planes, arg_dict=arg_dict)
        self.bn2 = QuantizedBn2d(planes, arg_dict=arg_dict)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
            identity = self.bn_down(identity)

        out = self.shortcut(identity, out)
        return out


class QuantizedBottleneck(nn.Module):
    expansion: int = 4
    batch_cluster = None
    def __init__(self, inplane: int, planes: int, stride: int = 1, downsample=None,
                 groups: int = 1, base_width: int = 64, dilation: int = 1,
                 arg_dict:dict=None) -> None:
        super(QuantizedBottleneck, self).__init__()

        self.downsample = downsample
        self.stride = stride
        self.arg_dict = arg_dict

        self.num_clusters = itemgetter('cluster')(arg_dict)

        width = int(planes * (base_width/64.)) * groups
        self.conv1 = quantized_conv1x1(in_planes=inplane, out_planes=width, arg_dict=arg_dict)
        self.conv2 = quantized_conv3x3(in_planes=width, out_planes=width, stride=stride, groups=groups, dilation=dilation,
                                       arg_dict=arg_dict)
        self.conv3 = quantized_conv1x1(in_planes=width, out_planes=planes * self.expansion, arg_dict=arg_dict)
        self.shortcut = QuantizedAdd(arg_dict=arg_dict)

        if self.downsample is not None:
            self.bn_down = QuantizedBn2d(planes * self.expansion, arg_dict=arg_dict)
        self.bn1 = QuantizedBn2d(width, arg_dict=arg_dict)
        self.bn2 = QuantizedBn2d(width, arg_dict=arg_dict)
        self.bn3 = QuantizedBn2d(planes * self.expansion, arg_dict=arg_dict)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)
            identity = self.bn_down(identity)
        out = self.shortcut(identity, out)
        return out


class QuantizedResNet(nn.Module):
    def __init__(self, block, layers, arg_dict, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None):
        super(QuantizedResNet, self).__init__()
        self.bit, self.num_clusters, self.runtime_helper = \
            itemgetter('bit', 'cluster', 'runtime_helper')(arg_dict)
        self.q_max = 2 ** self.bit - 1
        self.arg_dict = arg_dict

        t_init = list(range(self.num_clusters)) if self.num_clusters > 1 else 0
        self.scale = nn.Parameter(torch.tensor(t_init, dtype=torch.float32), requires_grad=False)
        self.zero_point = nn.Parameter(torch.tensor(t_init, dtype=torch.int32), requires_grad=False)

        self.num_blocks = 4
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.first_conv = QuantizedConv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, arg_dict=arg_dict)
        self.bn1 = QuantizedBn2d(self.inplanes, arg_dict=arg_dict)
        self.maxpool = QuantizedMaxPool2d(kernel_size=3, stride=2, padding=1, arg_dict=arg_dict)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = QuantizedLinear(512 * block.expansion, num_classes, arg_dict=arg_dict)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        # Planes : n_channel_output
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = quantized_conv1x1(self.inplanes, planes * block.expansion, stride, arg_dict=self.arg_dict)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, arg_dict=self.arg_dict))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                arg_dict=self.arg_dict))
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.runtime_helper.batch_cluster is not None:
            x = quantize_matrix_4d(x, self.scale, self.zero_point, self.runtime_helper.batch_cluster, self.q_max)
        else:
            x = quantize_matrix(x, self.scale, self.zero_point, self.q_max)

        x = self.first_conv(x)
        x = self.bn1(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.type(torch.cuda.IntTensor)
        x = x.type(torch.cuda.FloatTensor)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class QuantizedResNet20(nn.Module):
    def __init__(self, block, layers, arg_dict, num_classes=10):
        super(QuantizedResNet20, self).__init__()
        self.num_clusters, self.runtime_helper = itemgetter('cluster', 'runtime_helper')(arg_dict)

        self.target_bit = nn.Parameter(torch.tensor(0, dtype=torch.int8), requires_grad=False)
        self.in_bit = nn.Parameter(torch.tensor(0, dtype=torch.int8), requires_grad=False)
        self.arg_dict = arg_dict

        t_init = list(range(self.num_clusters)) if self.num_clusters > 1 else 0
        self.scale = nn.Parameter(torch.tensor(t_init, dtype=torch.float32), requires_grad=False)
        self.zero_point = nn.Parameter(torch.tensor(t_init, dtype=torch.int32), requires_grad=False)

        self.inplanes = 16
        self.dilation = 1
        self.num_blocks = 3

        self.first_conv = QuantizedConv2d(3, 16, kernel_size=3, stride=1, padding=1, arg_dict=arg_dict)
        self.bn1 = QuantizedBn2d(16, arg_dict=arg_dict)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.fc = QuantizedLinear(64 * block.expansion, num_classes, arg_dict=arg_dict)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = quantized_conv1x1(self.inplanes, planes * block.expansion, stride, arg_dict=self.arg_dict)
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, arg_dict=self.arg_dict))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, arg_dict=self.arg_dict))

        return nn.Sequential(*layers)

    def forward(self, x):
        print()
        print("Input Bit: {}".format(self.in_bit.data))
        if self.runtime_helper.batch_cluster is not None:
            x = quantize_matrix_4d(x, self.scale, self.zero_point, self.runtime_helper.batch_cluster, self.in_bit)
        else:
            x = quantize_matrix(x, self.scale, self.zero_point, self.in_bit)

        x = self.first_conv(x)
        x = self.bn1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.type(torch.cuda.IntTensor)
        x = x.type(torch.cuda.FloatTensor)

        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# def quantized_resnet18(arg_dict, **kwargs):
#     return QuantizedResNet18(QuantizedBasicBlock, [2, 2, 2, 2], arg_dict, **kwargs)


def quantized_resnet20(arg_dict, num_classes=10):
    return QuantizedResNet20(QuantizedBasicBlock, [3, 3, 3], arg_dict, num_classes=num_classes)


def quantized_resnet50(arg_dict, **kwargs):
    return QuantizedResNet(QuantizedBottleneck, [3, 4, 6, 3], arg_dict, **kwargs)


def set_shortcut_qparams(m, bit, s_bypass, z_bypass, s_prev, z_prev, s3, z3):
    m.bit.data = bit
    m.s_bypass.data = s_bypass
    m.z_bypass.data = z_bypass
    m.s_prev.data = s_prev
    m.z_prev.data = z_prev
    m.s3.data = s3
    m.z3.data = z3

    if m.num_clusters > 1:
        for c in range(m.num_clusters):
            m.M0_bypass[c], m.shift_bypass[c] = quantize_M(s_bypass[c] / s3[c])
            m.M0_prev[c], m.shift_prev[c] = quantize_M(s_prev[c] / s3[c])
    else:
        m.M0_bypass.data, m.shift_bypass.data = quantize_M(s_bypass / s3)
        m.M0_prev.data, m.shift_prev.data = quantize_M(s_prev / s3)
    return m


def quantize_block(_fp, _int):
    for i in range(len(_int)):
        _int[i].conv1 = quantize(_fp[i].conv1, _int[i].conv1)
        _int[i].conv2 = quantize(_fp[i].conv2, _int[i].conv2)
        if _fp[i].downsample:
            _int[i].downsample = quantize(_fp[i].downsample, _int[i].downsample)
            if type(_int[i]) == QuantizedBottleneck:
                _int[i].shortcut = set_shortcut_qparams(_int[i].shortcut, _fp[i].bit.data,
                                                        _int[i].downsample.s3, _int[i].downsample.z3,
                                                        _int[i].bn3.s3, _int[i].bn3.z3,
                                                        _fp[i].s3, _fp[i].z3)
            else:
                _int[i].shortcut = set_shortcut_qparams(_int[i].shortcut, _fp[i].bit.data,
                                                        _int[i].downsample.s3, _int[i].downsample.z3,
                                                        _int[i].bn2.s3, _int[i].bn2.z3,
                                                        _fp[i].s3, _fp[i].z3)
        else:
            if type(_int[i]) == QuantizedBottleneck:
                _int[i].shortcut = set_shortcut_qparams(_int[i].shortcut, _fp[i].bit.data,
                                                        _int[i].conv1.s1, _int[i].conv1.z1,
                                                        _int[i].bn3.s3, _int[i].bn3.z3,
                                                        _fp[i].s3, _fp[i].z3)
            else:
                _int[i].shortcut = set_shortcut_qparams(_int[i].shortcut, _fp[i].bit.data,
                                                        _int[i].conv1.s1, _int[i].conv1.z1,
                                                        _int[i].bn2.s3, _int[i].bn2.z3,
                                                        _fp[i].s3, _fp[i].z3)
    return _int


def quantize_pcq_block(_fp, _int):
    for i in range(len(_int)):
        _int[i].conv1 = quantize(_fp[i].conv1, _int[i].conv1)
        _int[i].bn1 = quantize(_fp[i].bn1, _int[i].bn1)
        _int[i].conv2 = quantize(_fp[i].conv2, _int[i].conv2)
        _int[i].bn2 = quantize(_fp[i].bn2, _int[i].bn2)
        if type(_int[i]) == QuantizedBottleneck:
            _int[i].conv3 = quantize(_fp[i].conv3, _int[i].conv3)
            _int[i].bn3 = quantize(_fp[i].bn3, _int[i].bn3)
        if _int[i].downsample:
            _int[i].downsample = quantize(_fp[i].downsample, _int[i].downsample)
            _int[i].bn_down = quantize(_fp[i].bn_down, _int[i].bn_down)
            if type(_int[i]) == QuantizedBottleneck:
                _int[i].shortcut = set_shortcut_qparams(_int[i].shortcut, _fp[i].bit.data,
                                                        _int[i].bn_down.s3, _int[i].bn_down.z3,
                                                        _int[i].bn3.s3, _int[i].bn3.z3,
                                                        _fp[i].s3, _fp[i].z3)
            else:
                _int[i].shortcut = set_shortcut_qparams(_int[i].shortcut, _fp[i].bit.data,
                                                        _int[i].bn_down.s3, _int[i].bn_down.z3,
                                                        _int[i].bn2.s3, _int[i].bn2.z3,
                                                        _fp[i].s3, _fp[i].z3)
        else:
            if type(_int[i]) == QuantizedBottleneck:
                _int[i].shortcut = set_shortcut_qparams(_int[i].shortcut, _fp[i].bit.data,
                                                        _int[i].conv1.s1, _int[i].conv1.z1,
                                                        _int[i].bn3.s3, _int[i].bn3.z3,
                                                        _fp[i].s3, _fp[i].z3)
            else:
                _int[i].shortcut = set_shortcut_qparams(_int[i].shortcut, _fp[i].bit.data,
                                                        _int[i].conv1.s1, _int[i].conv1.z1,
                                                        _int[i].bn2.s3, _int[i].bn2.z3,
                                                        _fp[i].s3, _fp[i].z3)
    return _int


def quantize_resnet(fp_model, int_model):
    int_model.scale = torch.nn.Parameter(fp_model.scale, requires_grad=False)
    int_model.zero_point = torch.nn.Parameter(fp_model.zero_point, requires_grad=False)
    int_model.first_conv = quantize(fp_model.first_conv, int_model.first_conv)
    int_model.bn1 = quantize(fp_model.bn1, int_model.bn1)
    int_model.layer1 = quantize_block(fp_model.layer1, int_model.layer1)
    int_model.layer2 = quantize_block(fp_model.layer2, int_model.layer2)
    int_model.layer3 = quantize_block(fp_model.layer3, int_model.layer3)
    if int_model.num_blocks == 4:
        int_model.layer4 = quantize_block(fp_model.layer4, int_model.layer4)
    int_model.fc = quantize(fp_model.fc, int_model.fc)
    return int_model


def quantize_pcq_resnet(fp_model, int_model):
    int_model.target_bit.data = fp_model.target_bit
    int_model.in_bit.data = fp_model.in_bit
    int_model.scale.data = fp_model.scale
    int_model.zero_point.data = fp_model.zero_point
    int_model.first_conv = quantize(fp_model.first_conv, int_model.first_conv)
    int_model.bn1 = quantize(fp_model.bn1, int_model.bn1)
    int_model.layer1 = quantize_pcq_block(fp_model.layer1, int_model.layer1)
    int_model.layer2 = quantize_pcq_block(fp_model.layer2, int_model.layer2)
    int_model.layer3 = quantize_pcq_block(fp_model.layer3, int_model.layer3)
    if int_model.num_blocks == 4:
        int_model.layer4 = quantize_pcq_block(fp_model.layer4, int_model.layer4)
    int_model.fc = quantize(fp_model.fc, int_model.fc)
    return int_model


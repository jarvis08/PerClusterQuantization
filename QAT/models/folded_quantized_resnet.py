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


class FoldedQuantizedBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1,
                 arg_dict:dict = None):
        super(FoldedQuantizedBasicBlock, self).__init__()
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.downsample = downsample
        self.stride = stride

        self.num_clusters, self.runtime_helper = itemgetter('cluster', 'runtime_helper')(arg_dict)
        self.target_bit = nn.Parameter(torch.tensor(0, dtype=torch.int8), requires_grad=False)
        self.a_bit = nn.Parameter(torch.tensor(0, dtype=torch.int8), requires_grad=False)

        t_init = list(range(self.num_clusters)) if self.num_clusters > 1 else 0
        self.s1 = nn.Parameter(torch.tensor(t_init, dtype=torch.float32), requires_grad=False)
        self.s_target = nn.Parameter(torch.tensor(t_init, dtype=torch.float32), requires_grad=False)
        self.z1 = nn.Parameter(torch.tensor(t_init, dtype=torch.int32), requires_grad=False)
        self.z_target = nn.Parameter(torch.tensor(t_init, dtype=torch.int32), requires_grad=False)
        self.M0 = nn.Parameter(torch.tensor(t_init, dtype=torch.int32), requires_grad=False)
        self.shift = nn.Parameter(torch.tensor(t_init, dtype=torch.int32), requires_grad=False)

        self.conv1 = quantized_conv3x3(inplanes, planes, stride, arg_dict=arg_dict)
        self.conv2 = quantized_conv3x3(planes, planes, arg_dict=arg_dict)
        self.shortcut = QuantizedAdd(arg_dict=arg_dict)

    def forward(self, x):
        identity = x

        conv_x = x
        if self.a_bit > self.target_bit:
            conv_x = rescale_matrix(x, self.z1, self.z_target, self.M0, self.shift,
                                    self.target_bit, self.runtime_helper)
        conv_x = conv_x.type(torch.cuda.FloatTensor)
        out = self.conv1(conv_x)
        out = self.conv2(out.type(torch.cuda.FloatTensor))

        if self.downsample is not None:
            identity = self.downsample(conv_x)

        out = self.shortcut(identity, out)
        return out


class FoldedQuantizedBottleneck(nn.Module):
    expansion: int = 4
    def __init__(self, inplane: int, planes: int, stride: int = 1, downsample=None,
                 groups: int = 1, base_width: int = 64, dilation: int = 1,
                 arg_dict:dict=None) -> None:
        super(FoldedQuantizedBottleneck, self).__init__()

        self.downsample = downsample
        self.stride = stride

        self.num_clusters, self.runtime_helper = itemgetter('cluster', 'runtime_helper')(arg_dict)
        self.target_bit = nn.Parameter(torch.tensor(0, dtype=torch.int8), requires_grad=False)
        self.a_bit = nn.Parameter(torch.tensor(0, dtype=torch.int8), requires_grad=False)

        t_init = list(range(self.num_clusters)) if self.num_clusters > 1 else 0
        self.s1 = nn.Parameter(torch.tensor(t_init, dtype=torch.float32), requires_grad=False)
        self.s_target = nn.Parameter(torch.tensor(t_init, dtype=torch.float32), requires_grad=False)
        self.z1 = nn.Parameter(torch.tensor(t_init, dtype=torch.int32), requires_grad=False)
        self.z_target = nn.Parameter(torch.tensor(t_init, dtype=torch.int32), requires_grad=False)
        self.M0 = nn.Parameter(torch.tensor(t_init, dtype=torch.int32), requires_grad=False)
        self.shift = nn.Parameter(torch.tensor(t_init, dtype=torch.int32), requires_grad=False)

        width = int(planes * (base_width/64.)) * groups
        self.conv1 = quantized_conv1x1(in_planes=inplane, out_planes=width, arg_dict=arg_dict)
        self.conv2 = quantized_conv3x3(in_planes=width, out_planes=width, stride=stride, groups=groups,
                                       dilation=dilation, arg_dict=arg_dict)
        self.conv3 = quantized_conv1x1(in_planes=width, out_planes=planes * self.expansion, arg_dict=arg_dict)
        self.shortcut = QuantizedAdd(arg_dict=arg_dict)

    def forward(self, x):
        identity = x

        conv_x = x
        if self.a_bit > self.target_bit:
            conv_x = rescale_matrix(x, self.z1, self.z_target, self.M0, self.shift,
                                    self.target_bit, self.runtime_helper)
        conv_x = conv_x.type(torch.cuda.FloatTensor)

        out = self.conv1(conv_x)
        out = self.conv2(out.type(torch.cuda.FloatTensor))
        out = self.conv3(out.type(torch.cuda.FloatTensor))

        if self.downsample is not None:
            identity = self.downsample(conv_x)

        out = self.shortcut(identity, out)
        return out


class FoldedQuantizedResNet(nn.Module):
    def __init__(self, block, layers, arg_dict, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None):
        super(FoldedQuantizedResNet, self).__init__()
        self.num_clusters, self.bit_first, self.runtime_helper = itemgetter('cluster', 'bit_first', 'runtime_helper')(arg_dict)

        self.target_bit = nn.Parameter(torch.tensor(0, dtype=torch.int8), requires_grad=False)
        self.a_bit = nn.Parameter(torch.tensor(0, dtype=torch.int8), requires_grad=False)
        self.in_bit = nn.Parameter(torch.tensor(0, dtype=torch.int8), requires_grad=False)
        self.arg_dict = arg_dict

        t_init = list(range(self.num_clusters)) if self.num_clusters > 1 else 0
        self.scale = nn.Parameter(torch.tensor(t_init, dtype=torch.float32), requires_grad=False)
        self.zero_point = nn.Parameter(torch.tensor(t_init, dtype=torch.int32), requires_grad=False)
        t_init = list(range(self.num_clusters)) if self.num_clusters > 1 else 0
        self.s1 = nn.Parameter(torch.tensor(t_init, dtype=torch.float32), requires_grad=False)
        self.s_target = nn.Parameter(torch.tensor(t_init, dtype=torch.float32), requires_grad=False)
        self.z1 = nn.Parameter(torch.tensor(t_init, dtype=torch.int32), requires_grad=False)
        self.z_target = nn.Parameter(torch.tensor(t_init, dtype=torch.int32), requires_grad=False)
        self.M0 = nn.Parameter(torch.tensor(t_init, dtype=torch.int32), requires_grad=False)
        self.shift = nn.Parameter(torch.tensor(t_init, dtype=torch.int32), requires_grad=False)

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
        self.first_conv = QuantizedConv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, is_first=True,
                                          arg_dict=arg_dict)
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
        if self.runtime_helper.qat_batch_cluster is not None:
            x = quantize_matrix_4d(x, self.scale, self.zero_point, self.runtime_helper.qat_batch_cluster, self.in_bit)
        else:
            x = quantize_matrix(x, self.scale, self.zero_point, self.in_bit)

        x = self.first_conv(x.type(torch.cuda.FloatTensor))
        x = self.maxpool(x.type(torch.cuda.FloatTensor))

        x = self.layer1(x.type(torch.cuda.LongTensor))
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x.type(torch.cuda.FloatTensor))
        # x = x.floor()
        x = x.trunc()

        x = torch.flatten(x, 1)
        if self.a_bit > self.target_bit:
            x = rescale_matrix(x.type(torch.cuda.LongTensor), self.z1, self.z_target, self.M0,
                               self.shift, self.target_bit, self.runtime_helper)
            x = self.fc(x.type(torch.cuda.FloatTensor))
        else:
            x = self.fc(x)
        return x.type(torch.cuda.FloatTensor)


class FoldedQuantizedResNet20(nn.Module):
    def __init__(self, block, layers, arg_dict, num_classes=10):
        super(FoldedQuantizedResNet20, self).__init__()
        self.num_clusters, self.bit_first, self.runtime_helper = itemgetter('cluster', 'bit_first', 'runtime_helper')(arg_dict)

        self.target_bit = nn.Parameter(torch.tensor(0, dtype=torch.int8), requires_grad=False)
        self.a_bit = nn.Parameter(torch.tensor(0, dtype=torch.int8), requires_grad=False)
        self.in_bit = nn.Parameter(torch.tensor(self.bit_first, dtype=torch.int8), requires_grad=False)
        self.arg_dict = arg_dict

        t_init = list(range(self.num_clusters)) if self.num_clusters > 1 else 0
        self.scale = nn.Parameter(torch.tensor(t_init, dtype=torch.float32), requires_grad=False)
        self.zero_point = nn.Parameter(torch.tensor(t_init, dtype=torch.int32), requires_grad=False)
        t_init = list(range(self.num_clusters)) if self.num_clusters > 1 else 0
        self.s1 = nn.Parameter(torch.tensor(t_init, dtype=torch.float32), requires_grad=False)
        self.s_target = nn.Parameter(torch.tensor(t_init, dtype=torch.float32), requires_grad=False)
        self.z1 = nn.Parameter(torch.tensor(t_init, dtype=torch.int32), requires_grad=False)
        self.z_target = nn.Parameter(torch.tensor(t_init, dtype=torch.int32), requires_grad=False)
        self.M0 = nn.Parameter(torch.tensor(t_init, dtype=torch.int32), requires_grad=False)
        self.shift = nn.Parameter(torch.tensor(t_init, dtype=torch.int32), requires_grad=False)

        self.inplanes = 16
        self.dilation = 1
        self.num_blocks = 3

        self.first_conv = QuantizedConv2d(3, 16, kernel_size=3, stride=1, padding=1, is_first=True, arg_dict=arg_dict)
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
        if self.runtime_helper.qat_batch_cluster is not None:
            x = quantize_matrix_4d(x, self.scale, self.zero_point, self.runtime_helper.qat_batch_cluster, self.in_bit)
        else:
            x = quantize_matrix(x, self.scale, self.zero_point, self.in_bit)

        x = self.first_conv(x.type(torch.cuda.FloatTensor))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x.type(torch.cuda.FloatTensor))
        # x = x.floor()
        # x = x.trunc()

        x = torch.flatten(x, 1)
        if self.a_bit > self.target_bit:
            x = rescale_matrix(x.type(torch.cuda.LongTensor), self.z1, self.z_target, self.M0,
                               self.shift, self.target_bit, self.runtime_helper)
            x = self.fc(x.type(torch.cuda.FloatTensor))
        else:
            x = self.fc(x)
        return x.type(torch.cuda.FloatTensor)


def quantized_resnet18_folded(arg_dict, **kwargs):
    return FoldedQuantizedResNet(FoldedQuantizedBasicBlock, [2, 2, 2, 2], arg_dict, **kwargs)


def quantized_resnet20_folded(arg_dict, num_classes=10):
    return FoldedQuantizedResNet20(FoldedQuantizedBasicBlock, [3, 3, 3], arg_dict, num_classes=num_classes)


def quantized_resnet50_folded(arg_dict, **kwargs):
    return FoldedQuantizedResNet(FoldedQuantizedBottleneck, [3, 4, 6, 3], arg_dict, **kwargs)


def set_shortcut_qparams(m, bit, s_bypass, z_bypass, s_prev, z_prev, s3, z3):
    m.a_bit.data = bit
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

        bypass_neg_values = (m.shift_bypass < 0).nonzero(as_tuple=True)[0]
        prev_neg_values = (m.shift_prev < 0).nonzero(as_tuple=True)[0]
        if len(bypass_neg_values):
            m.is_bypass_shift_neg.data = torch.tensor(True, dtype=torch.bool)
        if len(prev_neg_values):
            m.is_prev_shift_neg.data = torch.tensor(True, dtype=torch.bool)
    else:
        m.M0_bypass.data, m.shift_bypass.data = quantize_M(s_bypass / s3)
        m.M0_prev.data, m.shift_prev.data = quantize_M(s_prev / s3)
    return m


# def quantize_block(_fp, _int):
#     for i in range(len(_int)):
#         _int[i].conv1 = quantize(_fp[i].conv1, _int[i].conv1)
#         _int[i].conv2 = quantize(_fp[i].conv2, _int[i].conv2)
#         if _fp[i].downsample:
#             _int[i].downsample = quantize(_fp[i].downsample, _int[i].downsample)
#             if type(_int[i]) == FoldedQuantizedBottleneck:
#                 _int[i].shortcut = set_shortcut_qparams(_int[i].shortcut, _fp[i].a_bit.data,
#                                                         _int[i].downsample.s3, _int[i].downsample.z3,
#                                                         _int[i].conv3.s3, _int[i].conv3.z3,
#                                                         _fp[i].s3, _fp[i].z3)
#             else:
#                 _int[i].shortcut = set_shortcut_qparams(_int[i].shortcut, _fp[i].a_bit.data,
#                                                         _int[i].downsample.s3, _int[i].downsample.z3,
#                                                         _int[i].conv2.s3, _int[i].conv2.z3,
#                                                         _fp[i].s3, _fp[i].z3)
#         else:
#             if type(_int[i]) == FoldedQuantizedBottleneck:
#                 _int[i].shortcut = set_shortcut_qparams(_int[i].shortcut, _fp[i].a_bit.data,
#                                                         _int[i].conv1.s1, _int[i].conv1.z1,
#                                                         _int[i].conv3.s3, _int[i].conv3.z3,
#                                                         _fp[i].s3, _fp[i].z3)
#             else:
#                 _int[i].shortcut = set_shortcut_qparams(_int[i].shortcut, _fp[i].a_bit.data,
#                                                         _int[i].conv1.s1, _int[i].conv1.z1,
#                                                         _int[i].conv2.s3, _int[i].conv2.z3,
#                                                         _fp[i].s3, _fp[i].z3)
#     return _int


def folded_quantize_pcq_block(_fp, _int):
    for i in range(len(_int)):
        _int[i].target_bit.data = _fp[i].target_bit
        _int[i].a_bit.data = _fp[i].a_bit
        _int[i].s1.data = _fp[i].s1              # S, Z of 8/16/32 bit
        _int[i].z1.data = _fp[i].z1
        _int[i].s_target.data = _fp[i].s_target  # S, Z of 4/8 bit
        _int[i].z_target.data = _fp[i].z_target
        _int[i].M0.data = _fp[i].M0
        _int[i].shift.data = _fp[i].shift

        _int[i].conv1 = quantize(_fp[i].conv1, _int[i].conv1)
        _int[i].conv2 = quantize(_fp[i].conv2, _int[i].conv2)
        if type(_int[i]) == FoldedQuantizedBottleneck:
            _int[i].conv3 = quantize(_fp[i].conv3, _int[i].conv3)

        if _int[i].downsample:
            _int[i].downsample = quantize(_fp[i].downsample, _int[i].downsample)
            bypass = _int[i].downsample.s3, _int[i].downsample.z3
        else:
            bypass = _int[i].s1, _int[i].z1

        if type(_int[i]) == FoldedQuantizedBottleneck:
            prev = _int[i].conv3.s3, _int[i].conv3.z3
        else:
            prev = _int[i].conv2.s3, _int[i].conv2.z3

        _int[i].shortcut = set_shortcut_qparams(_int[i].shortcut, _fp[i].a_bit.data,
                                                bypass[0], bypass[1],
                                                prev[0], prev[1],
                                                _fp[i].s3, _fp[i].z3)
    return _int


# def quantize_resnet(fp_model, int_model):
#     int_model.scale = torch.nn.Parameter(fp_model.scale, requires_grad=False)
#     int_model.zero_point = torch.nn.Parameter(fp_model.zero_point, requires_grad=False)
#     int_model.first_conv = quantize(fp_model.first_conv, int_model.first_conv)
#     int_model.bn1 = quantize(fp_model.bn1, int_model.bn1)
#     int_model.layer1 = quantize_block(fp_model.layer1, int_model.layer1)
#     int_model.layer2 = quantize_block(fp_model.layer2, int_model.layer2)
#     int_model.layer3 = quantize_block(fp_model.layer3, int_model.layer3)
#     if int_model.num_blocks == 4:
#         int_model.layer4 = quantize_block(fp_model.layer4, int_model.layer4)
#     int_model.fc = quantize(fp_model.fc, int_model.fc)
#     return int_model


def folded_quantize_pcq_resnet(fp_model, int_model):
    int_model.target_bit.data = fp_model.target_bit
    int_model.in_bit.data = fp_model.in_bit
    int_model.scale.data = fp_model.scale
    int_model.zero_point.data = fp_model.zero_point
    int_model.first_conv = quantize(fp_model.first_conv, int_model.first_conv)
    int_model.layer1 = folded_quantize_pcq_block(fp_model.layer1, int_model.layer1)
    int_model.layer2 = folded_quantize_pcq_block(fp_model.layer2, int_model.layer2)
    int_model.layer3 = folded_quantize_pcq_block(fp_model.layer3, int_model.layer3)
    if int_model.num_blocks == 4:
        int_model.layer4 = folded_quantize_pcq_block(fp_model.layer4, int_model.layer4)
        int_model.maxpool.bit.data = int_model.first_conv.a_bit.data
        int_model.maxpool.zero_point.data = int_model.first_conv.z3.data

    int_model.a_bit.data = fp_model.a_bit
    int_model.s1.data = fp_model.s1  # S, Z of 8/16/32 bit
    int_model.z1.data = fp_model.z1
    int_model.s_target.data = fp_model.s_target  # S, Z of 4/8 bit
    int_model.z_target.data = fp_model.z_target
    int_model.M0.data = fp_model.M0
    int_model.shift.data = fp_model.shift

    int_model.fc = quantize(fp_model.fc, int_model.fc)
    return int_model


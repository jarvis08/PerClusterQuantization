import torch
import torch.nn as nn

from .layers import *
from .quantization_utils import *


def quantized_conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1, bias=False, bit=8, num_clusters=1):
    """3x3 convolution with padding"""
    return QuantizedConv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                       padding=dilation, groups=groups, dilation=dilation, bias=bias, bit=bit, num_clusters=num_clusters)


def quantized_conv1x1(in_planes, out_planes, stride=1, bias=False, bit=8, num_clusters=1):
    """1x1 convolution"""
    return QuantizedConv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=bias, bit=bit, num_clusters=num_clusters)


class QuantizedBasicBlock(nn.Module):
    expansion = 1
    batch_cluster = None

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1,
                 bit=8, num_clusters=1):
        super(QuantizedBasicBlock, self).__init__()
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.downsample = downsample
        self.stride = stride

        self.bit = bit

        self.num_clusters = num_clusters
        self.batch_cluster = None

        self.conv1 = quantized_conv3x3(inplanes, planes, stride, bias=False, bit=bit, num_clusters=num_clusters)
        self.conv2 = quantized_conv3x3(planes, planes, bias=False, bit=bit, num_clusters=num_clusters)
        self.shortcut = QuantizedAdd(bit=bit, num_clusters=num_clusters)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.shortcut(identity, out)
        return out


class QuantizedResNet18(nn.Module):
    batch_cluster = None

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None, bit=8, num_clusters=1):
        super(QuantizedResNet18, self).__init__()
        self.bit = bit
        self.num_clusters = num_clusters
        t_init = list(range(num_clusters)) if num_clusters > 1 else 0
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
        self.first_conv = QuantizedConv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False,
                                          bit=bit, num_clusters=num_clusters)
        self.maxpool = QuantizedMaxPool2d(kernel_size=3, stride=2, padding=1, num_clusters=num_clusters)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = QuantizedLinear(512 * block.expansion, num_classes, bias=False, bit=bit, num_clusters=num_clusters)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        # Planes : n_channel_output
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = quantized_conv1x1(self.inplanes, planes * block.expansion, stride, bias=False,
                                           bit=self.bit, num_clusters=self.num_clusters)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, bit=self.bit, num_clusters=self.num_clusters))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                bit=self.bit, num_clusters=self.num_clusters))
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.batch_cluster is not None:
            done = 0
            for i in range(self.batch_cluster.shape[0]):
                c = self.batch_cluster[i][0].item()
                n = self.batch_cluster[i][1].item()
                x[done:done + n] = quantize_matrix(x[done:done + n], self.scale[c], self.zero_point[c], self.q_max)
                done += n
        else:
            x = quantize_matrix(x, self.scale, self.zero_point, self.q_max)

        x = self.first_conv(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    @classmethod
    def set_cluster_information_of_batch(cls, info):
        cls.batch_cluster = info
        QuantizedBasicBlock.batch_cluster = info
        # QuantizedBottleneck.batch_cluster = info
        QuantizedConv2d.batch_cluster = info
        QuantizedLinear.batch_cluster = info
        QuantizedMaxPool2d.batch_cluster = info
        QuantizedAdd.batch_cluster = info


class QuantizedResNet20(nn.Module):
    batch_cluster = None

    def __init__(self, block, layers, num_classes=10, bit=8, num_clusters=1):
        super(QuantizedResNet20, self).__init__()
        self.bit = bit
        self.q_max = 2 ** bit - 1
        self.num_clusters = num_clusters
        t_init = list(range(num_clusters)) if num_clusters > 1 else 0
        self.scale = nn.Parameter(torch.tensor(t_init, dtype=torch.float32), requires_grad=False)
        self.zero_point = nn.Parameter(torch.tensor(t_init, dtype=torch.int32), requires_grad=False)

        self.batch_cluster = None

        self.inplanes = 16
        self.dilation = 1
        self.num_blocks = 3

        self.first_conv = QuantizedConv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False,
                                          bit=bit, num_clusters=num_clusters)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.fc = QuantizedLinear(64 * block.expansion, num_classes, bias=False, bit=bit, num_clusters=num_clusters)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = quantized_conv1x1(self.inplanes, planes * block.expansion, stride, bias=False,
                                           bit=self.bit, num_clusters=self.num_clusters)
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, bit=self.bit, num_clusters=self.num_clusters))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, bit=self.bit, num_clusters=self.num_clusters))

        return nn.Sequential(*layers)

    def forward(self, x):
        if self.batch_cluster is not None:
            done = 0
            for i in range(self.batch_cluster.shape[0]):
                c = self.batch_cluster[i][0].item()
                n = self.batch_cluster[i][1].item()
                x[done:done + n] = quantize_matrix(x[done:done + n], self.scale[c], self.zero_point[c], self.q_max)
                done += n
        else:
            x = quantize_matrix(x, self.scale, self.zero_point, self.q_max)

        x = self.first_conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def show_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.show_params()

    @classmethod
    def set_cluster_information_of_batch(cls, info):
        cls.batch_cluster = info
        QuantizedBasicBlock.batch_cluster = info
        QuantizedConv2d.batch_cluster = info
        QuantizedLinear.batch_cluster = info
        QuantizedMaxPool2d.batch_cluster = info
        QuantizedAdd.batch_cluster = info


def quantized_resnet18(bit=8, num_clusters=1, **kwargs):
    return QuantizedResNet18(QuantizedBasicBlock, [2, 2, 2, 2], bit=bit, num_clusters=num_clusters, **kwargs)


def quantized_resnet20(bit=8, num_clusters=1):
    return QuantizedResNet20(QuantizedBasicBlock, [3, 3, 3], bit=bit, num_clusters=num_clusters)


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


def quantize_resnet(fp_model, int_model):
    int_model.scale = torch.nn.Parameter(fp_model.scale, requires_grad=False)
    int_model.zero_point = torch.nn.Parameter(fp_model.zero_point, requires_grad=False)
    int_model.first_conv = quantize(fp_model.first_conv, int_model.first_conv)
    int_model.layer1 = quantize_block(fp_model.layer1, int_model.layer1)
    int_model.layer2 = quantize_block(fp_model.layer2, int_model.layer2)
    int_model.layer3 = quantize_block(fp_model.layer3, int_model.layer3)
    if int_model.num_blocks == 4:
        int_model.layer4 = quantize_block(fp_model.layer4, int_model.layer4)
    int_model.fc = quantize(fp_model.fc, int_model.fc)
    return int_model

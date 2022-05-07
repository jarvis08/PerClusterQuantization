from operator import itemgetter 
import torch
import torch.nn as nn

from .layers.conv2d import *
from .layers.linear import *
from .layers.norm import *
from .quant_noise import _quant_noise
from .quantization_utils import *


def pcq_conv3x3(in_planes, out_planes, stride=1, dilation=1, bias=False, norm_layer=None, activation=None, a_bit=None, arg_dict=None):
    return PCQConv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation,
                     bias=bias, norm_layer=norm_layer, activation=activation, a_bit=a_bit, arg_dict=arg_dict)


def pcq_conv1x1(in_planes, out_planes, stride=1, bias=False, norm_layer=None, activation=None, a_bit=None, arg_dict=None):
    return PCQConv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=bias, norm_layer=norm_layer,
                     activation=activation, a_bit=a_bit, arg_dict=arg_dict)


class FoldedPCQBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1, arg_dict=None):
        super(FoldedPCQBasicBlock, self).__init__()
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.downsample = downsample
        self.stride = stride

        target_bit, bit_conv_act, bit_addcat, self.smooth, self.use_ste, self.num_clusters, self.runtime_helper \
            = itemgetter('bit', 'bit_conv_act', 'bit_addcat', 'smooth', 'ste', 'cluster', 'runtime_helper')(arg_dict)
        self.a_bit = torch.nn.Parameter(torch.tensor(bit_addcat, dtype=torch.int8), requires_grad=False)
        self.target_bit = torch.nn.Parameter(torch.tensor(target_bit, dtype=torch.int8), requires_grad=False)

        self.act_range = nn.Parameter(torch.zeros(self.num_clusters, 2), requires_grad=False)
        self.apply_ema = nn.Parameter(torch.zeros(self.num_clusters, dtype=torch.bool), requires_grad=False)

        self.conv1 = pcq_conv3x3(inplanes, planes, stride, norm_layer=self._norm_layer, activation=nn.ReLU, arg_dict=arg_dict, a_bit=target_bit)
        self.conv2 = pcq_conv3x3(planes, planes, norm_layer=self._norm_layer, arg_dict=arg_dict, a_bit=bit_addcat)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        if self.training:
            self._update_activation_ranges(out)
            if self.runtime_helper.apply_fake_quantization:
                out = self._fake_quantize_activation(out)
        return out

    @torch.no_grad()
    def _update_activation_ranges(self, x):
        cluster = self.runtime_helper.qat_batch_cluster
        if self.runtime_helper.undo_gema:
            _max = x.max.item()
        else:
            data = x.view(x.size(0), -1)
            _max = data.max(dim=1).values.mean()

        if self.apply_ema[cluster]:
            self.act_range[cluster][1] = self.act_range[cluster][1] * self.smooth + _max * (1 - self.smooth)
        else:
            self.act_range[cluster][1] = _max
            self.apply_ema[cluster] = True


    def _fake_quantize_activation(self, x):
        cluster = self.runtime_helper.qat_batch_cluster
        s, z = calc_qparams(self.act_range[cluster][0], self.act_range[cluster][1], self.a_bit)
        return fake_quantize(x, s, z, self.target_bit, use_ste=self.use_ste)

    @torch.no_grad()
    def set_block_qparams(self, s1, z1, s_target, z_target):
        self.s1, self.z1 = s1, z1                          # S, Z of 8/16/32 bit
        self.s_target, self.z_target = s_target, z_target  # S, Z of 4/8 bit
        self.M0 = torch.zeros(self.num_clusters, dtype=torch.int32)
        self.shift = torch.zeros(self.num_clusters, dtype=torch.int32)
        for c in range(self.num_clusters):
            self.M0[c], self.shift[c] = quantize_M(self.s1[c] / self.s_target[c])

        if self.downsample:
            self.downsample.set_qparams(s_target, z_target)

        prev_s, prev_z = self.conv1.set_qparams(s_target, z_target)
        self.conv2.set_qparams(prev_s, prev_z)

        zero = self.runtime_helper.fzero
        self.s3, self.z3 = calc_qparams_per_cluster(self.act_range, self.a_bit, zero)
        nxt_s_target, nxt_z_target = calc_qparams_per_cluster(self.act_range, self.target_bit, zero)
        return self.s3, self.z3, nxt_s_target, nxt_z_target


class FoldedPCQBottleneck(nn.Module):
    expansion: int = 4

    def __init__(
            self, inplanes: int, planes: int, stride: int = 1, downsample = None,
            groups: int = 1, base_width: int = 64, dilation: int = 1, arg_dict = None
    ) -> None:
        super(FoldedPCQBottleneck, self).__init__()

        self.downsample = downsample
        self.stride = stride

        target_bit, bit_conv_act, bit_addcat, self.smooth, self.use_ste, self.num_clusters, self.runtime_helper \
            = itemgetter('bit', 'bit_conv_act', 'bit_addcat', 'smooth', 'ste', 'cluster', 'runtime_helper')(arg_dict)
        self.a_bit = torch.nn.Parameter(torch.tensor(bit_addcat, dtype=torch.int8), requires_grad=False)
        self.target_bit = torch.nn.Parameter(torch.tensor(target_bit, dtype=torch.int8), requires_grad=False)

        self.act_range = nn.Parameter(torch.zeros(self.num_clusters, 2), requires_grad=False)
        self.apply_ema = nn.Parameter(torch.zeros(self.num_clusters, dtype=torch.bool), requires_grad=False)

        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = pcq_conv1x1(in_planes=inplanes, norm_layer=self._norm_layer, activation=nn.ReLU, a_bit=target_bit, arg_dict=arg_dict)
        self.conv2 = pcq_conv3x3(in_planes=width, out_planes=width, stride=stride, dilation=dilation, norm_layer=self._norm_layer, activation=nn.ReLU,
                                   a_bit=target_bit, arg_dict=arg_dict)
        self.conv3 = pcq_conv1x1(in_planes=width, out_planes=planes * self.expansion, norm_layer=self._norm_layer,
                                 a_bit=bit_addcat, arg_dict=arg_dict)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        if self.training:
            self._update_activation_ranges(out)
            if self.runtime_helper.apply_fake_quantization:
                out = self._fake_quantize_activation(out)
        return out

    @torch.no_grad()
    def _update_activation_ranges(self, x):
        cluster = self.runtime_helper.qat_batch_cluster
        if self.runtime_helper.undo_gema:
            _max = x.max.item()
        else:
            data = x.view(x.size(0), -1)
            _max = data.max(dim=1).values.mean()

        if self.apply_ema[cluster]:
            self.act_range[cluster][1] = self.act_range[cluster][1] * self.smooth + _max * (1 - self.smooth)
        else:
            self.act_range[cluster][1] = _max
            self.apply_ema[cluster] = True

    def _fake_quantize_activation(self, x):
        cluster = self.runtime_helper.qat_batch_cluster

        s, z = calc_qparams(self.act_range[cluster][0], self.act_range[cluster][1], self.a_bit)
        return fake_quantize(x, s, z, self.a_bit, use_ste=self.use_ste)

    def set_block_qparams(self, s1, z1, s_target, z_target):
        self.s1, self.z1 = s1, z1                          # S, Z of 8/16/32 bit
        self.s_target, self.z_target = s_target, z_target  # S, Z of 4/8 bit
        self.M0 = torch.zeros(self.num_clusters, dtype=torch.int32)
        self.shift = torch.zeros(self.num_clusters, dtype=torch.int32)
        for c in range(self.num_clusters):
            self.M0[c], self.shift[c] = quantize_M(self.s1[c] / self.s_target[c])

        if self.downsample:
            self.downsample.set_qparams(s_target, z_target)

        prev_s, prev_z = self.conv1.set_qparams(s_target, z_target)
        prev_s, prev_z = self.conv2.set_qparams(prev_s, prev_z)
        self.conv3.set_qparams(prev_s, prev_z)

        zero = self.runtime_helper.fzero
        self.s3, self.z3 = calc_qparams_per_cluster(self.act_range, self.a_bit, zero)
        nxt_s_target, nxt_z_target = calc_qparams_per_cluster(self.act_range, self.target_bit, zero)
        return self.s3, self.z3, nxt_s_target, nxt_z_target


class FoldedPCQResNet(nn.Module):
    def __init__(self, block, layers, arg_dict, num_classes=1000, groups=1, \
                 width_per_group=64, replace_stride_with_dilation=None):
        super(FoldedPCQResNet, self).__init__()
        self.arg_dict = arg_dict
        target_bit, self.bit_conv_act, bit_addcat, bit_first, bit_classifier, self.smooth, self.num_clusters, self.runtime_helper \
            = itemgetter('bit', 'bit_conv_act', 'bit_addcat', 'bit_first', 'bit_classifier', 'smooth', 'cluster', 'runtime_helper')(arg_dict)
        self.target_bit = torch.nn.Parameter(torch.tensor(target_bit, dtype=torch.int8), requires_grad=False)
        self.a_bit = torch.nn.Parameter(torch.tensor(bit_addcat, dtype=torch.int8), requires_grad=False)
        self.in_bit = torch.nn.Parameter(torch.tensor(bit_first, dtype=torch.int8), requires_grad=False)

        self.in_range = nn.Parameter(torch.zeros(self.num_clusters, 2), requires_grad=False)
        self.apply_ema = nn.Parameter(torch.zeros(self.num_clusters, dtype=torch.bool), requires_grad=False)

        self._norm_layer = nn.BatchNorm2d
        self.inplanes = 64
        self.dilation = 1
        self.num_blocks = 4
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.first_conv = PCQConv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, w_bit=bit_first, norm_layer=self._norm_layer, activation=nn.ReLU, a_bit=self.bit_addcat, arg_dict=arg_dict)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = PCQLinear(512 * block.expansion, num_classes, is_classifier=True,
                            w_bit=bit_classifier, a_bit=bit_classifier, arg_dict=self.arg_dict)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        # Planes : n_channel_output
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = pcq_conv1x1(self.inplanes, planes * block.expansion, stride, arg_dict=self.arg_dict, norm_layer=self._norm_layer, a_bit=self.bit_addcat)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups, self.base_width,
                            previous_dilation, arg_dict=self.arg_dict, norm_layer=self._norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups, base_width=self.base_width,
                                dilation=self.dilation, arg_dict=self.arg_dict, norm_layer=self._norm_layer))
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.training:
            self._update_input_ranges(x)
            if self.runtime_helper.apply_fake_quantization:
                x = self._fake_quantize_input(x)

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

    @torch.no_grad()
    def _update_input_ranges(self, x):
        cluster = self.runtime_helper.qat_batch_cluster

        if self.runtime_helper.undo_gema:
            _min = x.min().item()
            _max = x.max().item()
        else:
            data = x.view(x.size(0), -1)
            _min = data.min(dim=1).values.mean()
            _max = data.max(dim=1).values.mean()
        if self.apply_ema[cluster]:
            self.in_range[cluster][0] = self.in_range[cluster][0] * self.smooth + _min * (1 - self.smooth)
            self.in_range[cluster][1] = self.in_range[cluster][1] * self.smooth + _max * (1 - self.smooth)
        else:
            self.in_range[cluster][0], self.in_range[cluster][1] = _min, _max
            self.apply_ema[cluster] = True

    def _fake_quantize_input(self, x):
        cluster = self.runtime_helper.qat_batch_cluster
        s, z = calc_qparams(self.in_range[cluster][0], self.in_range[cluster][1], self.in_bit,
                            self.runtime_helper.fzero)
        return fake_quantize(x, s, z, self.in_bit)

    @torch.no_grad()
    def set_quantization_params(self):
        zero = self.runtime_helper.fzero
        self.scale, self.zero_point = calc_qparams_per_cluster(self.in_range, self.in_bit, zero)
        prev_s, prev_z = self.first_conv.set_qparams(self.scale, self.zero_point)
        s1, z1 = self.bn1.set_qparams(prev_s, prev_z)
        s_target, z_target = calc_qparams_per_cluster(self.bn1.act_range, self.target_bit, zero)

        blocks = [self.layer1, self.layer2, self.layer3, self.layer4]
        for block in blocks:
            for b in range(len(block)):
                s1, z1, s_target, z_target = block[b].set_block_qparams(s1, z1, s_target, z_target)

        self.s1, self.z1 = s1, z1                          # S, Z of 8/16/32 bit
        self.s_target, self.z_target = s_target, z_target  # S, Z of 4/8 bit
        self.M0 = torch.zeros(self.num_clusters, dtype=torch.int32)
        self.shift = torch.zeros(self.num_clusters, dtype=torch.int32)
        for c in range(self.num_clusters):
            self.M0[c], self.shift[c] = quantize_M(self.s1[c] / self.s_target[c])
        self.fc.set_qparams(self.s_target, self.z_target)


class FoldedPCQResNet20(nn.Module):
    def __init__(self, block, layers, arg_dict, num_classes=10):
        super(FoldedPCQResNet20, self).__init__()
        self.arg_dict = arg_dict
        target_bit, self.bit_conv_act, self.bit_addcat, bit_first, bit_classifier, self.smooth, self.num_clusters, self.runtime_helper \
            = itemgetter('bit', 'bit_conv_act', 'bit_addcat', 'bit_first', 'bit_classifier', 'smooth', 'cluster', 'runtime_helper')(arg_dict)
        self.target_bit = torch.nn.Parameter(torch.tensor(target_bit, dtype=torch.int8), requires_grad=False)
        self.a_bit = torch.nn.Parameter(torch.tensor(self.bit_addcat, dtype=torch.int8), requires_grad=False)
        self.in_bit = torch.nn.Parameter(torch.tensor(bit_first, dtype=torch.int8), requires_grad=False)

        self.in_range = nn.Parameter(torch.zeros(self.num_clusters, 2), requires_grad=False)
        self.apply_ema = nn.Parameter(torch.zeros(self.num_clusters, dtype=torch.bool), requires_grad=False)

        self._norm_layer = nn.BatchNorm2d
        self.inplanes = 16
        self.dilation = 1
        self.num_blocks = 3

        self.first_conv = PCQConv2d(3, 16, kernel_size=3, stride=1, padding=1, w_bit=bit_first, norm_layer=self._norm_layer, activation=nn.ReLU, a_bit=self.bit_addcat, arg_dict=self.arg_dict)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.fc = PCQLinear(64 * block.expansion, num_classes, is_classifier=True,
                            w_bit=bit_classifier, a_bit=bit_classifier, arg_dict=self.arg_dict)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = pcq_conv1x1(self.inplanes, planes * block.expansion, stride, norm_layer=self._norm_layer,
                                     a_bit=self.bit_addcat, arg_dict=self.arg_dict)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, norm_layer=self._norm_layer, arg_dict=self.arg_dict))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_layer=self._norm_layer, arg_dict=self.arg_dict))
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.training:
            self._update_input_ranges(x)
            if self.runtime_helper.apply_fake_quantization:
                x = self._fake_quantize_input(x)

        x = self.first_conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    @torch.no_grad()
    def _update_input_ranges(self, x):
        cluster = self.runtime_helper.qat_batch_cluster

        if self.runtime_helper.undo_gema:
            _min = x.min().item()
            _max = x.max().item()
        else:
            data = x.view(x.size(0), -1)
            _min = data.min(dim=1).values.mean()
            _max = data.max(dim=1).values.mean()

        if self.apply_ema[cluster]:
            self.in_range[cluster][0] = self.in_range[cluster][0] * self.smooth + _min * (1 - self.smooth)
            self.in_range[cluster][1] = self.in_range[cluster][1] * self.smooth + _max * (1 - self.smooth)
        else:
            self.in_range[cluster][0], self.in_range[cluster][1] = _min, _max
            self.apply_ema[cluster] = True


    def _fake_quantize_input(self, x):
        cluster = self.runtime_helper.qat_batch_cluster
        s, z = calc_qparams(self.in_range[cluster][0], self.in_range[cluster][1], self.in_bit)
        return fake_quantize(x, s, z, self.in_bit)

    @torch.no_grad()
    def set_quantization_params(self):
        zero = self.runtime_helper.fzero
        self.scale, self.zero_point = calc_qparams_per_cluster(self.in_range, self.in_bit, zero)
        prev_s, prev_z = self.first_conv.set_qparams(self.scale, self.zero_point)

        s1, z1 = self.bn1.set_qparams(prev_s, prev_z)
        s_target, z_target = calc_qparams_per_cluster(self.bn1.act_range, self.target_bit, zero)

        blocks = [self.layer1, self.layer2, self.layer3]
        for block in blocks:
            for b in range(len(block)):
                s1, z1, s_target, z_target = block[b].set_block_qparams(s1, z1, s_target, z_target)

        self.s1, self.z1 = s1, z1                          # S, Z of 8/16/32 bit
        self.s_target, self.z_target = s_target, z_target  # S, Z of 4/8 bit
        self.M0 = torch.zeros(self.num_clusters, dtype=torch.int32)
        self.shift = torch.zeros(self.num_clusters, dtype=torch.int32)
        for c in range(self.num_clusters):
            self.M0[c], self.shift[c] = quantize_M(self.s1[c] / self.s_target[c])
        self.fc.set_qparams(self.s_target, self.z_target)


def pcq_resnet18_folded(arg_dict, **kwargs):
    return FoldedPCQResNet(FoldedPCQBottleneck, [2, 2, 2, 2], arg_dict, **kwargs)


def pcq_resnet50_folded(arg_dict, **kwargs):
    return FoldedPCQResNet(FoldedPCQBottleneck, [3, 4, 6, 3], arg_dict=arg_dict, **kwargs)


def pcq_resnet20_folded(arg_dict, num_classes=10):
    return FoldedPCQResNet20(FoldedPCQBasicBlock, [3, 3, 3], arg_dict, num_classes=num_classes)


def set_folded_pcq_resnet(fused, pre):
    """
        Copy from pre model's params to fused layers.
        Use fused architecture, but not really fused (use CONV & BN seperately)
    """
    bn_momentum = fused.arg_dict['bn_momentum']
    num_clusters = fused.arg_dict['cluster']
    num_norms = num_clusters if fused.arg_dict['multi_bn'] else 1
    # First layer
    fused.first_conv = folded_pcq_copy_from_pretrained(fused.first_conv, pre.conv1, pre.bn1, bn_momentum, num_norms)

    # Block 1
    block = fused.layer1
    if block[0].downsample is not None:
        block[0].downsample = folded_pcq_copy_from_pretrained(block[0].downsample, pre.layer1[0].downsample[0], pre.layer1[0].downsample[1], bn_momentum, num_norms)
    for i in range(len(block)):
        block[i].conv1 = folded_pcq_copy_from_pretrained(block[i].conv1, pre.layer1[i].conv1, pre.layer1[i].bn1, bn_momentum, num_norms)
        block[i].conv2 = folded_pcq_copy_from_pretrained(block[i].conv2, pre.layer1[i].conv2, pre.layer1[i].bn2, bn_momentum, num_norms)
        if type(block[i]) == FoldedPCQBottleneck:
            block[i].conv3 = folded_pcq_copy_from_pretrained(block[i].conv3, pre.layer1[i].conv3, pre.layer1[i].bn3, bn_momentum, num_norms)

    # Block 2
    block = fused.layer2
    if block[0].downsample is not None:
        block[0].downsample = folded_pcq_copy_from_pretrained(block[0].downsample, pre.layer2[0].downsample[0],
                                                   pre.layer2[0].downsample[1], bn_momentum, num_norms)
    for i in range(len(block)):
        block[i].conv1 = folded_pcq_copy_from_pretrained(block[i].conv1, pre.layer2[i].conv1, pre.layer2[i].bn1, bn_momentum, num_norms)
        block[i].conv2 = folded_pcq_copy_from_pretrained(block[i].conv2, pre.layer2[i].conv2, pre.layer2[i].bn2, bn_momentum, num_norms)
        if type(block[i]) == FoldedPCQBottleneck:
            block[i].conv3 = folded_pcq_copy_from_pretrained(block[i].conv3, pre.layer2[i].conv3, pre.layer2[i].bn3, bn_momentum, num_norms)

    # Block 3
    block = fused.layer3
    if block[0].downsample is not None:
        block[0].downsample = folded_pcq_copy_from_pretrained(block[0].downsample, pre.layer3[0].downsample[0],
                                                   pre.layer3[0].downsample[1], bn_momentum, num_norms)
    for i in range(len(block)):
        block[i].conv1 = folded_pcq_copy_from_pretrained(block[i].conv1, pre.layer3[i].conv1, pre.layer3[i].bn1, bn_momentum, num_norms)
        block[i].conv2 = folded_pcq_copy_from_pretrained(block[i].conv2, pre.layer3[i].conv2, pre.layer3[i].bn2, bn_momentum, num_norms)
        if type(block[i]) == FoldedPCQBottleneck:
            block[i].conv3 = folded_pcq_copy_from_pretrained(block[i].conv3, pre.layer3[i].conv3, pre.layer3[i].bn3, bn_momentum, num_norms)

    # Block 4
    if fused.num_blocks == 4:
        block = fused.layer4
        if block[0].downsample is not None:
            block[0].downsample = folded_pcq_copy_from_pretrained(block[0].downsample, pre.layer4[0].downsample[0],
                                                       pre.layer4[0].downsample[1], bn_momentum, num_norms)
        for i in range(len(block)):
            block[i].conv1 = folded_pcq_copy_from_pretrained(block[i].conv1, pre.layer4[i].conv1, pre.layer4[i].bn1, bn_momentum, num_norms)
            block[i].conv2 = folded_pcq_copy_from_pretrained(block[i].conv2, pre.layer4[i].conv2, pre.layer4[i].bn2, bn_momentum, num_norms)
            block[i].conv3 = folded_pcq_copy_from_pretrained(block[i].conv3, pre.layer4[i].conv3, pre.layer4[i].bn3, bn_momentum, num_norms)

    # Classifier
    fused.fc = copy_from_pretrained(fused.fc, pre.fc)
    return fused


def fold_pcq_resnet(model):
    # First layer
    model.first_conv.fold_conv_and_bn()

    # Block 1
    fp_block = model.layer1
    for i in range(len(fp_block)):
        fp_block[i].conv1.fold_conv_and_bn()
        fp_block[i].conv2.fold_conv_and_bn()
        if type(fp_block[i]) == FoldedPCQBottleneck:
            fp_block[i].conv3.fold_conv_and_bn()

    # Block 2
    fp_block = model.layer2
    fp_block[0].downsample.fold_conv_and_bn()
    for i in range(len(fp_block)):
        fp_block[i].conv1.fold_conv_and_bn()
        fp_block[i].conv2.fold_conv_and_bn()
        if type(fp_block[i]) == FoldedPCQBottleneck:
            fp_block[i].conv3.fold_conv_and_bn()

    # Block 3
    fp_block = model.layer3
    fp_block[0].downsample.fold_conv_and_bn()
    for i in range(len(fp_block)):
        fp_block[i].conv1.fold_conv_and_bn()
        fp_block[i].conv2.fold_conv_and_bn()
        if type(fp_block[i]) == FoldedPCQBottleneck:
            fp_block[i].conv3.fold_conv_and_bn()

    # Block 4
    if model.num_blocks == 4:
        fp_block = model.layer4
        fp_block[0].downsample.fold_conv_and_bn()
        for i in range(len(fp_block)):
            fp_block[i].conv1.fold_conv_and_bn()
            fp_block[i].conv2.fold_conv_and_bn()
            if type(fp_block[i]) == FoldedPCQBottleneck:
                fp_block[i].conv3.fold_conv_and_bn()



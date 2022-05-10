from operator import itemgetter 
import torch
import torch.nn as nn

from .layers.conv2d import *
from .layers.linear import *
from .layers.norm import *
from .quant_noise import _quant_noise
from .quantization_utils import *


def pcq_conv3x3(in_planes, out_planes, stride=1, dilation=1, bias=False, activation=None, a_bit=None, arg_dict=None):
    return PCQConv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation,
                     bias=bias, activation=activation, a_bit=a_bit, arg_dict=arg_dict)


def pcq_conv1x1(in_planes, out_planes, stride=1, bias=False, activation=None, a_bit=None, arg_dict=None):
    return PCQConv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=bias,
                     activation=activation, a_bit=a_bit, arg_dict=arg_dict)


class PCQBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1, out_bit=None, arg_dict=None):
        super(PCQBasicBlock, self).__init__()
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
        self.in_bit = torch.nn.Parameter(torch.tensor(bit_addcat, dtype=torch.int8), requires_grad=False)
        self.target_bit = torch.nn.Parameter(torch.tensor(target_bit, dtype=torch.int8), requires_grad=False)

        if out_bit is not None and (self.a_bit == self.target_bit):
            self.a_bit.data = torch.tensor(out_bit, dtype=torch.int8)

        self.act_range = nn.Parameter(torch.zeros(self.num_clusters, 2), requires_grad=False)
        self.apply_ema = nn.Parameter(torch.zeros(self.num_clusters, dtype=torch.bool), requires_grad=False)

        if self.downsample is not None:
            self.bn_down = PCQBnReLU(planes, a_bit=self.target_bit, arg_dict=arg_dict)
        self.conv1 = pcq_conv3x3(inplanes, planes, stride, arg_dict=arg_dict, a_bit=bit_conv_act)
        self.bn1 = PCQBnReLU(planes, activation=nn.ReLU, a_bit=self.target_bit, arg_dict=arg_dict)
        self.conv2 = pcq_conv3x3(planes, planes, arg_dict=arg_dict, a_bit=bit_conv_act)
        self.bn2 = PCQBnReLU(planes, a_bit=self.target_bit, arg_dict=arg_dict)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
            identity = self.bn_down(identity)

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
            _max = x.max().item()
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
        s, z = calc_qparams(self.act_range[cluster][0], self.act_range[cluster][1], self.a_bit, symmetric=False)
        return fake_quantize(x, s, z, self.a_bit, symmetric=False, use_ste=self.use_ste)


    @torch.no_grad()
    def set_block_qparams(self, s1, z1, s_target, z_target):
        self.s1, self.z1 = s1, z1                          # S, Z of 8/16/32 bit
        self.s_target, self.z_target = s_target, z_target  # S, Z of 4/8 bit
        self.M0 = torch.zeros(self.num_clusters, dtype=torch.int32)
        self.shift = torch.zeros(self.num_clusters, dtype=torch.int32)
        for c in range(self.num_clusters):
            self.M0[c], self.shift[c] = quantize_M(self.s1[c] / self.s_target[c])

        if self.downsample:
            prev_s, prev_z = self.downsample.set_qparams(s_target, z_target)
            self.bn_down.set_qparams(prev_s, prev_z)

        prev_s, prev_z = self.conv1.set_qparams(s_target, z_target)
        prev_s, prev_z = self.bn1.set_qparams(prev_s, prev_z)
        prev_s, prev_z = self.conv2.set_qparams(prev_s, prev_z)
        self.bn2.set_qparams(prev_s, prev_z)

        self.s3, self.z3 = calc_qparams_per_cluster(self.act_range, self.a_bit, symmetric=False)
        nxt_s_target, nxt_z_target = calc_qparams_per_cluster(self.act_range, self.target_bit, symmetric=False)
        return self.s3, self.z3, nxt_s_target, nxt_z_target


class PCQBottleneck(nn.Module):
    expansion: int = 4

    def __init__(
            self, inplanes: int, planes: int, stride: int = 1, downsample = None, out_bit=None,
            groups: int = 1, base_width: int = 64, dilation: int = 1, arg_dict = None
    ) -> None:
        super(PCQBottleneck, self).__init__()

        self.downsample = downsample
        self.stride = stride

        target_bit, bit_conv_act, bit_addcat, self.smooth, self.use_ste, self.num_clusters, self.runtime_helper \
            = itemgetter('bit', 'bit_conv_act', 'bit_addcat', 'smooth', 'ste', 'cluster', 'runtime_helper')(arg_dict)
        self.a_bit = torch.nn.Parameter(torch.tensor(bit_addcat, dtype=torch.int8), requires_grad=False)
        self.in_bit = torch.nn.Parameter(torch.tensor(bit_addcat, dtype=torch.int8), requires_grad=False)
        self.target_bit = torch.nn.Parameter(torch.tensor(target_bit, dtype=torch.int8), requires_grad=False)

        if out_bit is not None and (self.a_bit == self.target_bit):
            self.a_bit.data = torch.tensor(out_bit, dtype=torch.int8)

        self.act_range = nn.Parameter(torch.zeros(self.num_clusters, 2), requires_grad=False)
        self.apply_ema = nn.Parameter(torch.zeros(self.num_clusters, dtype=torch.bool), requires_grad=False)

        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        if self.downsample is not None:
            self.bn_down = PCQBnReLU(planes * self.expansion, a_bit=self.target_bit, arg_dict=arg_dict)
        self.conv1 = pcq_conv1x1(in_planes=inplanes, out_planes=width, a_bit=bit_conv_act, arg_dict=arg_dict)
        self.bn1 = PCQBnReLU(width, activation=nn.ReLU, a_bit=self.target_bit, arg_dict=arg_dict)
        self.conv2 = pcq_conv3x3(in_planes=width, out_planes=width, stride=stride, dilation=dilation, 
                                 a_bit=bit_conv_act, arg_dict=arg_dict)
        self.bn2 = PCQBnReLU(width, activation=nn.ReLU, a_bit=self.target_bit, arg_dict=arg_dict)
        self.conv3 = pcq_conv1x1(in_planes=width, out_planes=planes * self.expansion,
                                 a_bit=bit_conv_act, arg_dict=arg_dict)
        self.bn3 = PCQBnReLU(planes * self.expansion, a_bit=self.target_bit, arg_dict=arg_dict)
        self.relu = nn.ReLU(inplace=True)

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

        out += identity

        out = self.relu(out)

        if self.training:
            self._update_activation_ranges(out)
            if self.runtime_helper.apply_fake_quantization:
                out = self._fake_quantize_activation(out)

        return out

    def _update_activation_ranges(self, x):
        with torch.no_grad():
            cluster = self.runtime_helper.qat_batch_cluster
            if self.runtime_helper.undo_gema:
                _max = x.max().item()
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
        s, z = calc_qparams(self.act_range[cluster][0], self.act_range[cluster][1], self.a_bit, symmetric=False)
        return fake_quantize(x, s, z, self.a_bit, symmetric=False, use_ste=self.use_ste)

    def set_block_qparams(self, s1, z1, s_target, z_target):
        self.s1, self.z1 = s1, z1                          # S, Z of 8/16/32 bit
        self.s_target, self.z_target = s_target, z_target  # S, Z of 4/8 bit
        self.M0 = torch.zeros(self.num_clusters, dtype=torch.int32)
        self.shift = torch.zeros(self.num_clusters, dtype=torch.int32)
        for c in range(self.num_clusters):
            self.M0[c], self.shift[c] = quantize_M(self.s1[c] / self.s_target[c])

        if self.downsample:
            prev_s, prev_z = self.downsample.set_qparams(s_target, z_target)
            self.bn_down.set_qparams(prev_s, prev_z)

        prev_s, prev_z = self.conv1.set_qparams(s_target, z_target)
        prev_s, prev_z = self.bn1.set_qparams(prev_s, prev_z)
        prev_s, prev_z = self.conv2.set_qparams(prev_s, prev_z)
        prev_s, prev_z = self.bn2.set_qparams(prev_s, prev_z)
        prev_s, prev_z = self.conv3.set_qparams(prev_s, prev_z)
        self.bn3.set_qparams(prev_s, prev_z)

        self.s3, self.z3 = calc_qparams_per_cluster(self.act_range, self.a_bit, symmetric=False)
        nxt_s_target, nxt_z_target = calc_qparams_per_cluster(self.act_range, self.target_bit, symmetric=False)
        return self.s3, self.z3, nxt_s_target, nxt_z_target


class PCQResNet(nn.Module):
    def __init__(self, block, layers, arg_dict, num_classes=1000, groups=1, \
                 width_per_group=64, replace_stride_with_dilation=None):
        super(PCQResNet, self).__init__()
        self.arg_dict = arg_dict
        target_bit, self.bit_conv_act, bit_addcat, bit_first, bit_classifier, self.smooth, self.num_clusters, self.runtime_helper \
            = itemgetter('bit', 'bit_conv_act', 'bit_addcat', 'bit_first', 'bit_classifier', 'smooth', 'cluster', 'runtime_helper')(arg_dict)
        self.target_bit = torch.nn.Parameter(torch.tensor(target_bit, dtype=torch.int8), requires_grad=False)
        self.a_bit = torch.nn.Parameter(torch.tensor(bit_addcat, dtype=torch.int8), requires_grad=False)
        self.in_bit = torch.nn.Parameter(torch.tensor(bit_first, dtype=torch.int8), requires_grad=False)

        self.in_range = nn.Parameter(torch.zeros(self.num_clusters, 2), requires_grad=False)
        self.apply_ema = nn.Parameter(torch.zeros(self.num_clusters, dtype=torch.bool), requires_grad=False)

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
        self.first_conv = PCQConv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                                    w_bit=bit_first, a_bit=self.bit_conv_act, arg_dict=arg_dict)
        self.bn1 = PCQBnReLU(self.inplanes, activation=nn.ReLU, a_bit=self.a_bit, arg_dict=self.arg_dict)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], out_bit=self.a_bit)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0], out_bit=self.a_bit)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1], out_bit=self.a_bit)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2], out_bit=bit_classifier)
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1, padding=0)
        self.fc = PCQLinear(512 * block.expansion, num_classes, is_classifier=True,
                            w_bit=bit_classifier, a_bit=bit_classifier, arg_dict=self.arg_dict)

        # if bit_first > target_bit:
        #     self.layer4[layers[3]-1].bn3.change_a_bit(bit_first)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, out_bit=None):
        # Planes : n_channel_output
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = pcq_conv1x1(self.inplanes, planes * block.expansion, stride, arg_dict=self.arg_dict,
                                     a_bit=self.bit_conv_act)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups, self.base_width,
                            previous_dilation, arg_dict=self.arg_dict))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks-1):
            layers.append(block(self.inplanes, planes, groups=self.groups, base_width=self.base_width,
                                dilation=self.dilation, arg_dict=self.arg_dict))
        layers.append(block(self.inplanes, planes, groups=self.groups, base_width=self.base_width,
                            dilation=self.dilation, out_bit=out_bit, arg_dict=self.arg_dict))
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.training:
            self._update_input_ranges(x)
            if self.runtime_helper.apply_fake_quantization:
                x = self._fake_quantize_input(x)

        x = self.first_conv(x)
        x = self.bn1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


    def _update_input_ranges(self, x):
        cluster = self.runtime_helper.qat_batch_cluster
        with torch.no_grad():
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
        s, z = calc_qparams(self.in_range[cluster][0], self.in_range[cluster][1], self.in_bit, symmetric=True)
        return fake_quantize(x, s, z, self.in_bit, symmetric=True, use_ste=self.use_ste)


    def set_quantization_params(self):
        with torch.no_grad():
            self.scale, self.zero_point = calc_qparams_per_cluster(self.in_range, self.in_bit, symmetric=True)
            prev_s, prev_z = self.first_conv.set_qparams(self.scale, self.zero_point)
            s1, z1 = self.bn1.set_qparams(prev_s, prev_z)
            s_target, z_target = calc_qparams_per_cluster(self.bn1.act_range, self.target_bit, symmetric=False)

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



class PCQResNet20(nn.Module):
    def __init__(self, block, layers, arg_dict, num_classes=10):
        super(PCQResNet20, self).__init__()
        self.arg_dict = arg_dict
        target_bit, self.bit_conv_act, bit_addcat, bit_first, bit_classifier, self.smooth, self.num_clusters, self.runtime_helper \
            = itemgetter('bit', 'bit_conv_act', 'bit_addcat', 'bit_first', 'bit_classifier', 'smooth', 'cluster', 'runtime_helper')(arg_dict)
        self.target_bit = torch.nn.Parameter(torch.tensor(target_bit, dtype=torch.int8), requires_grad=False)
        self.a_bit = torch.nn.Parameter(torch.tensor(bit_addcat, dtype=torch.int8), requires_grad=False)
        self.in_bit = torch.nn.Parameter(torch.tensor(bit_first, dtype=torch.int8), requires_grad=False)

        self.in_range = nn.Parameter(torch.zeros(self.num_clusters, 2), requires_grad=False)
        self.apply_ema = nn.Parameter(torch.zeros(self.num_clusters, dtype=torch.bool), requires_grad=False)

        self.inplanes = 16
        self.dilation = 1
        self.num_blocks = 3

        self.first_conv = PCQConv2d(3, 16, kernel_size=3, stride=1, padding=1,
                                    w_bit=bit_first, a_bit=self.bit_conv_act, arg_dict=self.arg_dict)
        self.bn1 = PCQBnReLU(16, activation=nn.ReLU, a_bit=self.a_bit, arg_dict=arg_dict)
        self.layer1 = self._make_layer(block, 16, layers[0], out_bit=self.a_bit)
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2, out_bit=self.a_bit)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2, out_bit=bit_classifier)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.fc = PCQLinear(64 * block.expansion, num_classes, is_classifier=True,
                            w_bit=bit_classifier, a_bit=bit_classifier, arg_dict=self.arg_dict)


    def _make_layer(self, block, planes, blocks, stride=1, out_bit=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = pcq_conv1x1(self.inplanes, planes * block.expansion, stride,
                                     a_bit=self.bit_conv_act, arg_dict=self.arg_dict)
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, arg_dict=self.arg_dict))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks-1):
            layers.append(block(self.inplanes, planes, arg_dict=self.arg_dict))
        layers.append(block(self.inplanes, planes, out_bit=out_bit, arg_dict=self.arg_dict))
        return nn.Sequential(*layers)


    def forward(self, x):
        if self.training:
            self._update_input_ranges(x)
            if self.runtime_helper.apply_fake_quantization:
                x = self._fake_quantize_input(x)

        x = self.first_conv(x)
        x = self.bn1(x)
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
        s, z = calc_qparams(self.in_range[cluster][0], self.in_range[cluster][1], self.in_bit, symmetric=True)
        return fake_quantize(x, s, z, self.in_bit, symmetric=True)


    @torch.no_grad()
    def set_quantization_params(self):
        zero = self.runtime_helper.fzero
        self.scale, self.zero_point = calc_qparams_per_cluster(self.in_range, self.in_bit, symmetric=True)
        prev_s, prev_z = self.first_conv.set_qparams(self.scale, self.zero_point)

        s1, z1 = self.bn1.set_qparams(prev_s, prev_z)
        s_target, z_target = calc_qparams_per_cluster(self.bn1.act_range, self.target_bit, symmetric=False)

        blocks = [self.layer1, self.layer2, self.layer3]
        for block in blocks:
            for b in range(len(block)):
                s1, z1, s_target, z_target = block[b].set_block_qparams(s1, z1, s_target, z_target)

        self.s1, self.z1 = s1, z1
        self.fc.set_qparams(self.s1, self.z1)


def pcq_resnet18(arg_dict, **kwargs):
    return PCQResNet(PCQBasicBlock, [2, 2, 2, 2], arg_dict, **kwargs)


def pcq_resnet50(arg_dict, **kwargs):
    return PCQResNet(PCQBottleneck, [3, 4, 6, 3], arg_dict=arg_dict, **kwargs)


def pcq_resnet20(arg_dict, num_classes=10):
    return PCQResNet20(PCQBasicBlock, [3, 3, 3], arg_dict, num_classes=num_classes)


def set_pcq_resnet(fused, pre):
    """
        Copy from pre model's params to fused layers.
        Use fused architecture, but not really fused (use CONV & BN seperately)
    """
    n = fused.arg_dict['cluster']
    momentum = fused.arg_dict['bn_momentum']
    # First layer
    fused.first_conv = copy_weight_from_pretrained(fused.first_conv, pre.conv1)
    fused.bn1 = copy_pcq_bn_from_pretrained(fused.bn1, pre.bn1, n, momentum)

    # Block 1
    block = fused.layer1
    if block[0].downsample is not None:
        block[0].downsample = copy_weight_from_pretrained(block[0].downsample, pre.layer1[0].downsample[0])
        block[0].bn_down = copy_pcq_bn_from_pretrained(block[0].bn_down, pre.layer1[0].downsample[1], n, momentum)
    for i in range(len(block)):
        block[i].conv1 = copy_weight_from_pretrained(block[i].conv1, pre.layer1[i].conv1)
        block[i].conv2 = copy_weight_from_pretrained(block[i].conv2, pre.layer1[i].conv2)
        block[i].bn1 = copy_pcq_bn_from_pretrained(block[i].bn1, pre.layer1[i].bn1, n, momentum)
        block[i].bn2 = copy_pcq_bn_from_pretrained(block[i].bn2, pre.layer1[i].bn2, n, momentum)
        if type(block[i]) == PCQBottleneck:
            block[i].conv3 = copy_weight_from_pretrained(block[i].conv3, pre.layer1[i].conv3)
            block[i].bn3 = copy_pcq_bn_from_pretrained(block[i].bn3, pre.layer1[i].bn3, n, momentum)

    # Block 2
    block = fused.layer2
    block[0].downsample = copy_weight_from_pretrained(block[0].downsample, pre.layer2[0].downsample[0])
    block[0].bn_down = copy_pcq_bn_from_pretrained(block[0].bn_down, pre.layer2[0].downsample[1], n, momentum)
    for i in range(len(block)):
        block[i].conv1 = copy_weight_from_pretrained(block[i].conv1, pre.layer2[i].conv1)
        block[i].conv2 = copy_weight_from_pretrained(block[i].conv2, pre.layer2[i].conv2)
        block[i].bn1 = copy_pcq_bn_from_pretrained(block[i].bn1, pre.layer2[i].bn1, n, momentum)
        block[i].bn2 = copy_pcq_bn_from_pretrained(block[i].bn2, pre.layer2[i].bn2, n, momentum)
        if type(block[i]) == PCQBottleneck:
            block[i].conv3 = copy_weight_from_pretrained(block[i].conv3, pre.layer2[i].conv3)
            block[i].bn3 = copy_pcq_bn_from_pretrained(block[i].bn3, pre.layer2[i].bn3, n, momentum)

    # Block 3
    block = fused.layer3
    block[0].downsample = copy_weight_from_pretrained(block[0].downsample, pre.layer3[0].downsample[0])
    block[0].bn_down = copy_pcq_bn_from_pretrained(block[0].bn_down, pre.layer3[0].downsample[1], n, momentum)
    for i in range(len(block)):
        block[i].conv1 = copy_weight_from_pretrained(block[i].conv1, pre.layer3[i].conv1)
        block[i].conv2 = copy_weight_from_pretrained(block[i].conv2, pre.layer3[i].conv2)
        block[i].bn1 = copy_pcq_bn_from_pretrained(block[i].bn1, pre.layer3[i].bn1, n, momentum)
        block[i].bn2 = copy_pcq_bn_from_pretrained(block[i].bn2, pre.layer3[i].bn2, n, momentum)
        if type(block[i]) == PCQBottleneck:
            block[i].conv3 = copy_weight_from_pretrained(block[i].conv3, pre.layer3[i].conv3)
            block[i].bn3 = copy_pcq_bn_from_pretrained(block[i].bn3, pre.layer3[i].bn3, n, momentum)

    # Block 4
    if fused.num_blocks == 4:
        block = fused.layer4
        block[0].downsample = copy_weight_from_pretrained(block[0].downsample, pre.layer4[0].downsample[0])
        block[0].bn_down = copy_pcq_bn_from_pretrained(block[0].bn_down, pre.layer4[0].downsample[1], n, momentum)
        for i in range(len(block)):
            block[i].conv1 = copy_weight_from_pretrained(block[i].conv1, pre.layer4[i].conv1)
            block[i].conv2 = copy_weight_from_pretrained(block[i].conv2, pre.layer4[i].conv2)
            block[i].conv3 = copy_weight_from_pretrained(block[i].conv3, pre.layer4[i].conv3)
            block[i].bn1 = copy_pcq_bn_from_pretrained(block[i].bn1, pre.layer4[i].bn1, n, momentum)
            block[i].bn2 = copy_pcq_bn_from_pretrained(block[i].bn2, pre.layer4[i].bn2, n, momentum)
            block[i].bn3 = copy_pcq_bn_from_pretrained(block[i].bn3, pre.layer4[i].bn3, n, momentum)

    # Classifier
    fused.fc = copy_from_pretrained(fused.fc, pre.fc)
    return fused



import torch
import torch.nn as nn

from .layers import *
from .quant_noise import _quant_noise
from .quantization_utils import *


def fused_conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1, bias=False,
                  norm_layer=None, activation=None, a_bit=None, arg_dict=None):
    """3x3 convolution with padding"""
    return FusedConv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                       padding=dilation, groups=groups, dilation=dilation, bias=bias,
                       norm_layer=norm_layer, activation=activation, a_bit=a_bit, arg_dict=arg_dict)


def fused_conv1x1(in_planes, out_planes, stride=1, bias=False, norm_layer=None, activation=None, a_bit=None, arg_dict=None):
    """1x1 convolution"""
    return FusedConv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=bias,
                       norm_layer=norm_layer, activation=activation, a_bit=a_bit, arg_dict=arg_dict)


class FusedBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1, out_bit=None,
                 norm_layer=None, arg_dict=None):
        super(FusedBasicBlock, self).__init__()
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

        if out_bit is not None:
            self.a_bit.data = torch.tensor(out_bit, dtype=torch.int8)

        self.act_range = nn.Parameter(torch.zeros(2), requires_grad=False)
        self.apply_ema = nn.Parameter(torch.tensor(0, dtype=torch.bool), requires_grad=False)

        if self.downsample is not None:
            self.bn_down = FusedBnReLU(planes, a_bit=self.a_bit, arg_dict=arg_dict)
        self.conv1 = fused_conv3x3(inplanes, planes, stride, a_bit=bit_conv_act, arg_dict=arg_dict)
        self.bn1 = FusedBnReLU(planes, activation=nn.ReLU, a_bit=self.target_bit, arg_dict=arg_dict)

        self.conv2 = fused_conv3x3(planes, planes, a_bit=bit_conv_act, arg_dict=arg_dict)
        self.bn2 = FusedBnReLU(planes, a_bit=self.a_bit, arg_dict=arg_dict)
        self.relu = nn.ReLU(inplace=True)


    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(identity)
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
            if self.runtime_helper.undo_gema:
                _max = x.max().item()
            else:
                data = x.view(x.size(0), -1)
                _max = data.max(dim=1).values.mean()

            if self.apply_ema:
                self.act_range[1] = self.act_range[1] * self.smooth + _max * (1 - self.smooth)
            else:
                self.act_range[1] = _max
                self.apply_ema.data = torch.tensor(True, dtype=torch.bool)


    def _fake_quantize_activation(self, x):
        s, z = calc_qparams(self.act_range[0], self.act_range[1], self.a_bit, symmetric=False)
        return fake_quantize(x, s, z, self.a_bit, symmetric=False, use_ste=self.use_ste)


    def set_block_qparams(self, s1, z1):
        with torch.no_grad():
            self.s1, self.z1 = s1, z1

            if self.downsample:
                prev_s, prev_z = self.downsample.set_qparams(self.s1, self.z1)
                self.bn_down.set_qparams(prev_s, prev_z)

            prev_s, prev_z = self.conv1.set_qparams(self.s1, self.z1)
            prev_s, prev_z = self.bn1.set_qparams(prev_s, prev_z)
            prev_s, prev_z = self.conv2.set_qparams(prev_s, prev_z)
            self.bn2.set_qparams(prev_s, prev_z)

            self.s3, self.z3 = calc_qparams(self.act_range[0], self.act_range[1], self.a_bit, symmetric=False)
            return self.s3, self.z3


class FusedBottleneck(nn.Module):
    expansion: int = 4

    def __init__(self, inplane:int, planes:int, stride:int=1, downsample=None, out_bit=None,
                 groups: int = 1, base_width:int =64, dilation:int =1, a_bit=None, arg_dict=None) -> None:
        super(FusedBottleneck, self).__init__()

        self.stride = stride
        self.downsample = downsample

        target_bit, bit_conv_act, bit_addcat, self.smooth, self.use_ste, self.num_clusters, self.runtime_helper \
            = itemgetter('bit', 'bit_conv_act', 'bit_addcat', 'smooth', 'ste', 'cluster', 'runtime_helper')(arg_dict)
        self.a_bit = torch.nn.Parameter(torch.tensor(bit_addcat, dtype=torch.int8), requires_grad=False)
        self.in_bit = torch.nn.Parameter(torch.tensor(bit_addcat, dtype=torch.int8), requires_grad=False)
        self.target_bit = torch.nn.Parameter(torch.tensor(target_bit, dtype=torch.int8), requires_grad=False)

        if out_bit is not None:
            self.a_bit.data = torch.tensor(out_bit, dtype=torch.int8)

        self.act_range = nn.Parameter(torch.zeros(2), requires_grad=False)
        self.apply_ema = nn.Parameter(torch.tensor(0, dtype=torch.bool), requires_grad=False)

        width = int(planes * (base_width/64.)) * groups

        if downsample is not None:
            self.downsample = fused_conv1x1(inplane, planes * self.expansion, stride, a_bit=bit_conv_act, arg_dict=arg_dict)
            self.bn_down = FusedBnReLU(planes * self.expansion, a_bit=self.a_bit, arg_dict=arg_dict)

        self.conv1 = fused_conv1x1(inplane, width, a_bit=bit_conv_act, arg_dict=arg_dict)
        self.bn1 = FusedBnReLU(width, activation=nn.ReLU, a_bit=self.target_bit, arg_dict=arg_dict)

        self.conv2 = fused_conv3x3(width, width, stride=stride, groups=groups, dilation=dilation,
                                   a_bit=bit_conv_act, arg_dict=arg_dict)
        self.bn2 = FusedBnReLU(width, activation=nn.ReLU, a_bit=self.target_bit, arg_dict=arg_dict)

        self.conv3 = fused_conv1x1(width, planes * self.expansion, a_bit=bit_conv_act, arg_dict=arg_dict)
        self.bn3 = FusedBnReLU(planes * self.expansion, a_bit=self.a_bit, arg_dict=arg_dict)
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
            identity = self.downsample(identity)
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
            if self.runtime_helper.undo_gema:
                _max = x.max().item()
            else:
                data = x.view(x.size(0), -1)
                _max = data.max(dim=1).values.mean().item()

            if self.apply_ema:
                self.act_range[1] = self.act_range[1] * self.smooth + _max * (1 - self.smooth)
            else:
                self.act_range[1] = _max
                self.apply_ema.data = torch.tensor(True, dtype=torch.bool)


    def _fake_quantize_activation(self, x):
        s, z = calc_qparams(self.act_range[0], self.act_range[1], self.a_bit, symmetric=False)
        return fake_quantize(x, s, z, self.a_bit, symmetric=False, use_ste=self.use_ste)


    def set_block_qparams(self, s1, z1):
        with torch.no_grad():
            self.s1, self.z1 = s1, z1

            if self.downsample:
                prev_s, prev_z = self.downsample.set_qparams(self.s1, self.z1)
                self.bn_down.set_qparams(prev_s, prev_z)

            prev_s, prev_z = self.conv1.set_qparams(self.s1, self.z1)
            prev_s, prev_z = self.bn1.set_qparams(prev_s, prev_z)
            prev_s, prev_z = self.conv2.set_qparams(prev_s, prev_z)
            prev_s, prev_z = self.bn2.set_qparams(prev_s, prev_z)
            prev_s, prev_z = self.conv3.set_qparams(prev_s, prev_z)
            self.bn3.set_qparams(prev_s, prev_z)

            self.s3, self.z3 = calc_qparams(self.act_range[0], self.act_range[1], self.a_bit, symmetric=False)
            return self.s3, self.z3


class FusedResNet(nn.Module):
    def __init__(self, block, layers, arg_dict, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None):
        super(FusedResNet, self).__init__()
        self.arg_dict = arg_dict
        target_bit, self.bit_conv_act, bit_addcat, bit_first, bit_classifier, self.smooth, self.num_clusters, self.runtime_helper \
            = itemgetter('bit', 'bit_conv_act', 'bit_addcat', 'bit_first', 'bit_classifier', 'smooth', 'cluster', 'runtime_helper')(arg_dict)
        self.target_bit = torch.nn.Parameter(torch.tensor(target_bit, dtype=torch.int8), requires_grad=False)
        self.a_bit = torch.nn.Parameter(torch.tensor(bit_addcat, dtype=torch.int8), requires_grad=False)
        self.in_bit = torch.nn.Parameter(torch.tensor(bit_first, dtype=torch.int8), requires_grad=False)

        self.in_range = nn.Parameter(torch.zeros(2), requires_grad=False)
        self.apply_ema = nn.Parameter(torch.tensor(0, dtype=torch.bool), requires_grad=False)

        self.inplanes = 64
        self.dilation = 1
        self.num_blocks = 4
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        self.first_conv = FusedConv2d(3, 64, kernel_size=7, stride=2, padding=3,
                                      w_bit=self.in_bit, a_bit=self.bit_conv_act, arg_dict=arg_dict)
        self.bn1 = FusedBnReLU(64, activation=nn.ReLU, a_bit=self.a_bit, arg_dict=self.arg_dict)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], out_bit=self.a_bit)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0], out_bit=self.a_bit)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1], out_bit=self.a_bit)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2], out_bit=bit_classifier)
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1, padding=0)
        self.fc = FusedLinear(512 * block.expansion, num_classes, is_classifier=True,
                              w_bit=bit_classifier, a_bit=bit_classifier, arg_dict=self.arg_dict)


    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, out_bit=None):
        # Planes : n_channel_output
        downsample = None
        previous_dilation = self.dilation

        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = True

        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, downsample=downsample, groups=self.groups,
                            base_width=self.base_width, dilation=previous_dilation, arg_dict=self.arg_dict))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks - 1):
            layers.append(block(self.inplanes, planes, groups=self.groups, base_width=self.base_width,
                                dilation=self.dilation, arg_dict=self.arg_dict))
        layers.append(block(self.inplanes, planes, groups=self.groups, base_width=self.base_width,
                                dilation=self.dilation, out_bit=out_bit, arg_dict=self.arg_dict))
        return nn.Sequential(*layers)


    def forward(self, x:torch.Tensor) -> torch.Tensor:
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
        with torch.no_grad():
            if self.runtime_helper.undo_gema:
                _min = x.min().item()
                _max = x.max().item()
            else:
                data = x.view(x.size(0), -1)
                _min = data.min(dim=1).values.mean()
                _max = data.max(dim=1).values.mean()

            if self.apply_ema:
                self.in_range[0] = self.in_range[0] * self.smooth + _min * (1 - self.smooth)
                self.in_range[1] = self.in_range[1] * self.smooth + _max * (1 - self.smooth)
            else:
                self.in_range[0], self.in_range[1] = _min, _max
                self.apply_ema.data = torch.tensor(True, dtype=torch.bool)


    def _fake_quantize_input(self, x):
        s, z = calc_qparams(self.in_range[0], self.in_range[1], self.in_bit, symmetric=True)
        return fake_quantize(x, s, z, self.in_bit, symmetric=True, use_ste=self.use_ste)


    def set_quantization_params(self):
        self.scale, self.zero_point = calc_qparams(self.in_range[0], self.in_range[1], self.in_bit, symmetric=True)
        prev_s, prev_z = self.first_conv.set_qparams(self.scale, self.zero_point)

        s1, z1 = self.bn1.set_qparams(prev_s, prev_z)

        blocks = [self.layer1, self.layer2, self.layer3, self.layer4]
        for block in blocks:
            for b in range(len(block)):
                s1, z1 = block[b].set_block_qparams(s1, z1)

        self.s1, self.z1 = s1, z1
        self.fc.set_qparams(self.s1, self.z1)



class FusedResNet20(nn.Module):
    def __init__(self, block, layers, arg_dict, num_classes=10):
        super(FusedResNet20, self).__init__()
        self.arg_dict = arg_dict
        target_bit, self.bit_conv_act, bit_addcat, bit_first, bit_classifier, self.smooth, self.use_ste, self.num_clusters, self.runtime_helper \
            = itemgetter('bit', 'bit_conv_act', 'bit_addcat', 'bit_first', 'bit_classifier', 'smooth', 'ste', 'cluster', 'runtime_helper')(arg_dict)
        self.target_bit = torch.nn.Parameter(torch.tensor(target_bit, dtype=torch.int8), requires_grad=False)
        self.a_bit = torch.nn.Parameter(torch.tensor(bit_addcat, dtype=torch.int8), requires_grad=False)
        self.in_bit = torch.nn.Parameter(torch.tensor(bit_first, dtype=torch.int8), requires_grad=False)

        self.in_range = nn.Parameter(torch.zeros(2), requires_grad=False)
        self.apply_ema = nn.Parameter(torch.tensor(0, dtype=torch.bool), requires_grad=False)

        self._norm_layer = nn.BatchNorm2d
        self.inplanes = 16
        self.dilation = 1
        self.num_blocks = 3

        self.first_conv = FusedConv2d(3, 16, kernel_size=3, stride=1, padding=1,
                                      w_bit=self.in_bit, a_bit=self.bit_conv_act, arg_dict=arg_dict)
        self.bn1 = FusedBnReLU(16, activation=nn.ReLU, a_bit=self.a_bit, arg_dict=arg_dict)
        self.layer1 = self._make_layer(block, 16, layers[0], out_bit=self.a_bit)
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2, out_bit=self.a_bit)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2, out_bit=bit_classifier)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.fc = FusedLinear(64 * block.expansion, num_classes, is_classifier=True,
                              w_bit=bit_classifier, a_bit=bit_classifier, arg_dict=arg_dict)


    def _make_layer(self, block, planes, blocks, stride=1, out_bit=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = fused_conv1x1(self.inplanes, planes * block.expansion, stride,
                                       a_bit=self.bit_conv_act, arg_dict=self.arg_dict)
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample,
                            norm_layer=self._norm_layer, arg_dict=self.arg_dict))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks-1):
            layers.append(block(self.inplanes, planes, norm_layer=self._norm_layer, arg_dict=self.arg_dict))
        layers.append(block(self.inplanes, planes, out_bit=out_bit, norm_layer=self._norm_layer, arg_dict=self.arg_dict))
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
        if self.runtime_helper.undo_gema:
            _min = x.min().item()
            _max = x.max().item()
        else:
            data = x.view(x.size(0), -1)
            _min = data.min(dim=1).values.mean()
            _max = data.max(dim=1).values.mean()

        if self.apply_ema:
            self.in_range[0] = self.in_range[0] * self.smooth + _min * (1 - self.smooth)
            self.in_range[1] = self.in_range[1] * self.smooth + _max * (1 - self.smooth)
        else:
            self.in_range[0], self.in_range[1] = _min, _max
            self.apply_ema.data = torch.tensor(True, dtype=torch.bool)


    def _fake_quantize_input(self, x):
        s, z = calc_qparams(self.in_range[0], self.in_range[1], self.in_bit, symmetric=True)
        return fake_quantize(x, s, z, self.in_bit, symmetric=True, use_ste=self.use_ste)


    def set_quantization_params(self):
        self.scale, self.zero_point = calc_qparams(self.in_range[0], self.in_range[1], self.in_bit, symmetric=True)
        prev_s, prev_z = self.first_conv.set_qparams(self.scale, self.zero_point)

        s1, z1 = self.bn1.set_qparams(prev_s, prev_z)

        blocks = [self.layer1, self.layer2, self.layer3]
        for block in blocks:
            for b in range(len(block)):
                s1, z1 = block[b].set_block_qparams(s1, z1)

        self.s1, self.z1 = s1, z1
        self.fc.set_qparams(self.s1, self.z1)



def fused_resnet18(arg_dict, **kwargs):
    return FusedResNet(FusedBasicBlock, [2, 2, 2, 2], arg_dict, **kwargs)


def fused_resnet50(arg_dict, **kwargs):
    return FusedResNet(FusedBottleneck, [3, 4, 6, 3], arg_dict=arg_dict, **kwargs)


def fused_resnet20(arg_dict, num_classes=10):
    return FusedResNet20(FusedBasicBlock, [3, 3, 3], arg_dict, num_classes=num_classes)


def set_fused_resnet(fused, pre):
    """
        Copy from pre model's params to fused layers.
        Use fused architecture, but not really fused (use CONV & BN seperately)
    """
    bn_momentum = fused.arg_dict['bn_momentum']
    # First layer
    fused.first_conv = copy_weight_from_pretrained(fused.first_conv, pre.conv1)
    fused.bn1 = copy_bn_from_pretrained(fused.bn1, pre.bn1)
    fused.bn1.bn.momentum = bn_momentum

    # Block 1
    block = fused.layer1
    if block[0].downsample is not None:
        block[0].downsample = copy_weight_from_pretrained(block[0].downsample, pre.layer1[0].downsample[0])
        block[0].bn_down = copy_bn_from_pretrained(block[0].bn_down, pre.layer1[0].downsample[1])
        block[0].bn_down.bn.momentum = bn_momentum
    for i in range(len(block)):
        block[i].conv1 = copy_weight_from_pretrained(block[i].conv1, pre.layer1[i].conv1)
        block[i].conv2 = copy_weight_from_pretrained(block[i].conv2, pre.layer1[i].conv2)
        block[i].bn1 = copy_bn_from_pretrained(block[i].bn1, pre.layer1[i].bn1)
        block[i].bn2 = copy_bn_from_pretrained(block[i].bn2, pre.layer1[i].bn2)
        block[i].bn1.bn.momentum = bn_momentum
        block[i].bn2.bn.momentum = bn_momentum
        if type(block[i]) == FusedBottleneck:
            block[i].conv3 = copy_weight_from_pretrained(block[i].conv3, pre.layer1[i].conv3)
            block[i].bn3 = copy_bn_from_pretrained(block[i].bn3, pre.layer1[i].bn3)
            block[i].bn3.bn.momentum = bn_momentum

    # Block 2
    block = fused.layer2
    block[0].downsample = copy_weight_from_pretrained(block[0].downsample, pre.layer2[0].downsample[0])
    block[0].bn_down = copy_bn_from_pretrained(block[0].bn_down, pre.layer2[0].downsample[1])
    block[0].bn_down.bn.momentum = bn_momentum
    for i in range(len(block)):
        block[i].conv1 = copy_weight_from_pretrained(block[i].conv1, pre.layer2[i].conv1)
        block[i].conv2 = copy_weight_from_pretrained(block[i].conv2, pre.layer2[i].conv2)
        block[i].bn1 = copy_bn_from_pretrained(block[i].bn1, pre.layer2[i].bn1)
        block[i].bn2 = copy_bn_from_pretrained(block[i].bn2, pre.layer2[i].bn2)
        block[i].bn1.bn.momentum = bn_momentum
        block[i].bn2.bn.momentum = bn_momentum
        if type(block[i]) == FusedBottleneck:
            block[i].conv3 = copy_weight_from_pretrained(block[i].conv3, pre.layer2[i].conv3)
            block[i].bn3 = copy_bn_from_pretrained(block[i].bn3, pre.layer2[i].bn3)
            block[i].bn3.bn.momentum = bn_momentum

    # Block 3
    block = fused.layer3
    block[0].downsample = copy_weight_from_pretrained(block[0].downsample, pre.layer3[0].downsample[0])
    block[0].bn_down = copy_bn_from_pretrained(block[0].bn_down, pre.layer3[0].downsample[1])
    block[0].bn_down.bn.momentum = bn_momentum
    for i in range(len(block)):
        block[i].conv1 = copy_weight_from_pretrained(block[i].conv1, pre.layer3[i].conv1)
        block[i].conv2 = copy_weight_from_pretrained(block[i].conv2, pre.layer3[i].conv2)
        block[i].bn1 = copy_bn_from_pretrained(block[i].bn1, pre.layer3[i].bn1)
        block[i].bn2 = copy_bn_from_pretrained(block[i].bn2, pre.layer3[i].bn2)
        block[i].bn1.bn.momentum = bn_momentum
        block[i].bn2.bn.momentum = bn_momentum
        if type(block[i]) == FusedBottleneck:
            block[i].conv3 = copy_weight_from_pretrained(block[i].conv3, pre.layer3[i].conv3)
            block[i].bn3 = copy_bn_from_pretrained(block[i].bn3, pre.layer3[i].bn3)
            block[i].bn3.bn.momentum = bn_momentum

    # Block 4
    if fused.num_blocks == 4:
        block = fused.layer4
        block[0].downsample = copy_weight_from_pretrained(block[0].downsample, pre.layer4[0].downsample[0])
        block[0].bn_down = copy_bn_from_pretrained(block[0].bn_down, pre.layer4[0].downsample[1])
        block[0].bn_down.bn.momentum = bn_momentum
        for i in range(len(block)):
            block[i].conv1 = copy_weight_from_pretrained(block[i].conv1, pre.layer4[i].conv1)
            block[i].conv2 = copy_weight_from_pretrained(block[i].conv2, pre.layer4[i].conv2)
            block[i].conv3 = copy_weight_from_pretrained(block[i].conv3, pre.layer4[i].conv3)
            block[i].bn1 = copy_bn_from_pretrained(block[i].bn1, pre.layer4[i].bn1)
            block[i].bn2 = copy_bn_from_pretrained(block[i].bn2, pre.layer4[i].bn2)
            block[i].bn3 = copy_bn_from_pretrained(block[i].bn3, pre.layer4[i].bn3)
            block[i].bn1.bn.momentum = bn_momentum
            block[i].bn2.bn.momentum = bn_momentum
            block[i].bn3.bn.momentum = bn_momentum

    # Classifier
    fused.fc = copy_from_pretrained(fused.fc, pre.fc)
    return fused


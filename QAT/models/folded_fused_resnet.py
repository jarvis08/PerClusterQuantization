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


def fused_conv1x1(in_planes, out_planes, stride=1, bias=False, norm_layer=None, activation=None, a_bit=None,
                  arg_dict=None):
    """1x1 convolution"""
    return FusedConv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=bias,
                       norm_layer=norm_layer, activation=activation, a_bit=a_bit, arg_dict=arg_dict)


class FoldedFusedBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1,
                 norm_layer=None, arg_dict=None):
        super(FoldedFusedBasicBlock, self).__init__()
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.downsample = downsample
        self.stride = stride
        self._norm_layer = norm_layer

        target_bit, bit_conv_act, bit_addcat, self.smooth, self.use_ste, self.num_clusters, self.runtime_helper \
            = itemgetter('bit', 'bit_conv_act', 'bit_addcat', 'smooth', 'ste', 'cluster', 'runtime_helper')(arg_dict)
        self.a_bit = torch.nn.Parameter(torch.tensor(bit_addcat, dtype=torch.int8), requires_grad=False)
        self.target_bit = torch.nn.Parameter(torch.tensor(target_bit, dtype=torch.int8), requires_grad=False)

        self.act_range = nn.Parameter(torch.zeros(2), requires_grad=False)
        self.apply_ema = nn.Parameter(torch.tensor(0, dtype=torch.bool), requires_grad=False)

        self.conv1 = fused_conv3x3(inplanes, planes, stride, norm_layer=self._norm_layer, activation=nn.ReLU, arg_dict=arg_dict, a_bit=target_bit)
        self.conv2 = fused_conv3x3(planes, planes, norm_layer=self._norm_layer, arg_dict=arg_dict, a_bit=bit_addcat)
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
        s, z = calc_qparams(self.act_range[0], self.act_range[1], self.target_bit, self.runtime_helper.fzero)
        return fake_quantize(x, s, z, self.target_bit, use_ste=self.use_ste)

    @torch.no_grad()
    def set_block_qparams(self, s1, z1, s_target, z_target):
        self.s1, self.z1 = s1, z1  # S, Z of 8/16/32 bit
        self.s_target, self.z_target = s_target, z_target  # S, Z of 4/8 bit
        self.M0, self.shift = quantize_M(self.s1 / self.s_target)

        if self.downsample:
            self.downsample.set_qparams(s_target, z_target)

        prev_s, prev_z = self.conv1.set_qparams(s_target, z_target)
        self.conv2.set_qparams(prev_s, prev_z)

        self.s3, self.z3 = calc_qparams(self.act_range[0], self.act_range[1], self.a_bit)
        nxt_s_target, nxt_z_target = calc_qparams(self.act_range[0], self.act_range[1], self.target_bit)
        return self.s3, self.z3, nxt_s_target, nxt_z_target


class FoldedFusedBottleneck(nn.Module):
    expansion: int = 4

    def __init__(self, inplane: int, planes: int, stride: int = 1, downsample=None,
                 groups: int = 1, base_width: int = 64, dilation: int = 1, norm_layer=None, arg_dict=None) -> None:
        super(FoldedFusedBottleneck, self).__init__()

        self.downsample = downsample
        self.stride = stride
        self._norm_layer = norm_layer

        target_bit, bit_conv_act, bit_addcat, self.smooth, self.use_ste, self.num_clusters, self.runtime_helper \
            = itemgetter('bit', 'bit_conv_act', 'bit_addcat', 'smooth', 'ste', 'cluster', 'runtime_helper')(arg_dict)
        self.a_bit = torch.nn.Parameter(torch.tensor(bit_addcat, dtype=torch.int8), requires_grad=False)
        self.target_bit = torch.nn.Parameter(torch.tensor(target_bit, dtype=torch.int8), requires_grad=False)

        self.act_range = nn.Parameter(torch.zeros(2), requires_grad=False)
        self.apply_ema = nn.Parameter(torch.tensor(0, dtype=torch.bool), requires_grad=False)

        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1

        self.conv1 = fused_conv1x1(in_planes=inplane, out_planes=width, norm_layer=self._norm_layer, activation=nn.ReLU, a_bit=target_bit, arg_dict=arg_dict)
        self.conv2 = fused_conv3x3(in_planes=width, out_planes=width, stride=stride, groups=groups, dilation=dilation, norm_layer=self._norm_layer, activation=nn.ReLU,
                                   a_bit=target_bit, arg_dict=arg_dict)
        self.conv3 = fused_conv1x1(in_planes=width, out_planes=planes * self.expansion, norm_layer=self._norm_layer,
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

        if not self.training:
            return out

        if self.apply_ema:
            self.act_range[0], self.act_range[1] = ema(out, self.act_range, self.smooth)
            if self.runtime_helper.apply_fake_quantization:
                s, z = calc_qparams(self.act_range[0], self.act_range[1], self.target_bit)
                out = fake_quantize(out, s, z, self.target_bit, use_ste=self.use_ste)
        else:
            self.act_range[0], self.act_range[1] = get_range(out)
            self.apply_ema.data = torch.tensor(True, dtype=torch.bool)
        return out

    def set_block_qparams(self, s1, z1, s_target, z_target):
        self.s1, self.z1 = s1, z1  # S, Z of 8/16/32 bit
        self.s_target, self.z_target = s_target, z_target  # S, Z of 4/8 bit
        self.M0, self.shift = quantize_M(self.s1 / self.s_target)

        if self.downsample:
            self.downsample.set_qparams(s_target, z_target)

        prev_s, prev_z = self.conv1.set_qparams(s_target, z_target)
        prev_s, prev_z = self.conv2.set_qparams(prev_s, prev_z)
        self.conv3.set_qparams(prev_s, prev_z)

        self.s3, self.z3 = calc_qparams(self.act_range[0], self.act_range[1], self.a_bit)
        nxt_s_target, nxt_z_target = calc_qparams(self.act_range[0], self.act_range[1], self.target_bit)
        return self.s3, self.z3, nxt_s_target, nxt_z_target


class FoldedFusedResNet(nn.Module):
    def __init__(self, block, layers, arg_dict, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None):
        super(FoldedFusedResNet, self).__init__()
        self.arg_dict = arg_dict
        target_bit, bit_conv_act, self.bit_addcat, bit_first, bit_classifier, self.smooth, self.num_clusters, self.runtime_helper \
            = itemgetter('bit', 'bit_conv_act', 'bit_addcat', 'bit_first', 'bit_classifier', 'smooth', 'cluster',
                         'runtime_helper')(arg_dict)
        self.target_bit = torch.nn.Parameter(torch.tensor(target_bit, dtype=torch.int8), requires_grad=False)
        self.a_bit = torch.nn.Parameter(torch.tensor(self.bit_addcat, dtype=torch.int8), requires_grad=False)
        self.in_bit = torch.nn.Parameter(torch.tensor(bit_first, dtype=torch.int8), requires_grad=False)

        self.in_range = nn.Parameter(torch.zeros(2), requires_grad=False)
        self.apply_ema = nn.Parameter(torch.tensor(0, dtype=torch.bool), requires_grad=False)

        self._norm_layer = nn.BatchNorm2d
        self.inplanes = 64
        self.dilation = 1
        self.num_blocks = 4

        # self.qn_incre_check = self.quant_noise + self.runtime_helper.qn_prob_increment

        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        self.first_conv = FusedConv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                                      w_bit=bit_first, norm_layer=self._norm_layer, activation=nn.ReLU, a_bit=self.bit_addcat, arg_dict=arg_dict)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = FusedLinear(512 * block.expansion, num_classes, is_classifier=True,
                              w_bit=bit_classifier, a_bit=bit_classifier, arg_dict=self.arg_dict)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        # Planes : n_channel_output
        downsample = None
        previous_dilation = self.dilation

        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = fused_conv1x1(self.inplanes, planes * block.expansion, stride, arg_dict=self.arg_dict,
                                       norm_layer=self._norm_layer, a_bit=self.bit_addcat)

        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, downsample=downsample, groups=self.groups,
                            base_width=self.base_width, dilation=previous_dilation, norm_layer=self._norm_layer, arg_dict=self.arg_dict))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups, base_width=self.base_width, norm_layer=self._norm_layer,
                                dilation=self.dilation, arg_dict=self.arg_dict))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            if self.apply_ema:
                self.in_range[0], self.in_range[1] = ema(x, self.in_range, self.smooth)
                if self.runtime_helper.apply_fake_quantization:
                    s, z = calc_qparams(self.in_range[0], self.in_range[1], self.in_bit)
                    x = fake_quantize(x, s, z, self.in_bit)
            else:
                self.in_range[0], self.in_range[1] = get_range(x)
                self.apply_ema.data = torch.tensor(True, dtype=torch.bool)

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

    def set_quantization_params(self):
        self.scale, self.zero_point = calc_qparams(self.in_range[0], self.in_range[1], self.in_bit)
        s1, z1 = self.first_conv.set_qparams(self.scale, self.zero_point)
        s_target, z_target = calc_qparams(self.first_conv.act_range[0], self.first_conv.act_range[1], self.target_bit)

        blocks = [self.layer1, self.layer2, self.layer3, self.layer4]
        for block in blocks:
            for b in range(len(block)):
                s1, z1, s_target, z_target = block[b].set_block_qparams(s1, z1, s_target, z_target)

        self.s1, self.z1 = s1, z1  # S, Z of 8/16/32 bit
        self.s_target, self.z_target = s_target, z_target  # S, Z of 4/8 bit
        self.M0, self.shift = quantize_M(self.s1 / self.s_target)
        self.fc.set_qparams(self.s_target, self.z_target)


class FoldedFusedResNet20(nn.Module):
    def __init__(self, block, layers, arg_dict, num_classes=10):
        super(FoldedFusedResNet20, self).__init__()
        self.arg_dict = arg_dict
        target_bit, bit_conv_act, self.bit_addcat, bit_first, bit_classifier, self.smooth, self.num_clusters, self.runtime_helper \
            = itemgetter('bit', 'bit_conv_act', 'bit_addcat', 'bit_first', 'bit_classifier', 'smooth', 'cluster',
                         'runtime_helper')(arg_dict)
        self.target_bit = torch.nn.Parameter(torch.tensor(target_bit, dtype=torch.int8), requires_grad=False)
        self.a_bit = torch.nn.Parameter(torch.tensor(self.bit_addcat, dtype=torch.int8), requires_grad=False)
        self.in_bit = torch.nn.Parameter(torch.tensor(bit_first, dtype=torch.int8), requires_grad=False)

        self.in_range = nn.Parameter(torch.zeros(2), requires_grad=False)
        self.apply_ema = nn.Parameter(torch.tensor(0, dtype=torch.bool), requires_grad=False)

        self._norm_layer = nn.BatchNorm2d
        self.inplanes = 16
        self.dilation = 1
        self.num_blocks = 3

        self.arg_dict = arg_dict

        self.first_conv = FusedConv2d(3, 16, kernel_size=3, stride=1, padding=1,
                                      w_bit=bit_first, norm_layer=self._norm_layer, activation=nn.ReLU, a_bit=self.bit_addcat, arg_dict=arg_dict)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.fc = FusedLinear(64 * block.expansion, num_classes, is_classifier=True,
                              w_bit=bit_classifier, a_bit=bit_classifier, arg_dict=arg_dict)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = fused_conv1x1(self.inplanes, planes * block.expansion, stride, norm_layer=self._norm_layer,
                                       a_bit=self.bit_addcat, arg_dict=self.arg_dict)
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample,
                            norm_layer=self._norm_layer, arg_dict=self.arg_dict))
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
    def _update_activation_ranges(self, x):
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
        s, z = calc_qparams(self.in_range[0], self.in_range[1], self.in_bit, self.runtime_helper.fzero)
        return fake_quantize(x, s, z, self.in_bit)


    def set_quantization_params(self):
        self.scale, self.zero_point = calc_qparams(self.in_range[0], self.in_range[1], self.in_bit)
        s1, z1 = self.first_conv.set_qparams(self.scale, self.zero_point)
        s_target, z_target = calc_qparams(self.first_conv.act_range[0], self.first_conv.act_range[1], self.target_bit)

        blocks = [self.layer1, self.layer2, self.layer3]
        for block in blocks:
            for b in range(len(block)):
                s1, z1, s_target, z_target = block[b].set_block_qparams(s1, z1, s_target, z_target)

        self.s1, self.z1 = s1, z1  # S, Z of 8/16/32 bit
        self.s_target, self.z_target = s_target, z_target  # S, Z of 4/8 bit
        self.M0, self.shift = quantize_M(self.s1 / self.s_target)
        self.fc.set_qparams(self.s_target, self.z_target)


def fused_resnet18(arg_dict, **kwargs):
    return FoldedFusedResNet(FoldedFusedBasicBlock, [2, 2, 2, 2], arg_dict, **kwargs)


def fused_resnet50_folded(arg_dict, **kwargs):
    return FoldedFusedResNet(FoldedFusedBottleneck, [3, 4, 6, 3], arg_dict=arg_dict, **kwargs)


def fused_resnet20_folded(arg_dict, num_classes=10):
    return FoldedFusedResNet20(FoldedFusedBasicBlock, [3, 3, 3], arg_dict, num_classes=num_classes)


def set_folded_fused_resnet(fused, pre):
    """
        Copy from pre model's params to fused layers.
        Use fused architecture, but not really fused (use CONV & BN seperately)
    """
    bn_momentum = fused.arg_dict['bn_momentum']
    # First layer
    fused.first_conv = copy_from_pretrained(fused.first_conv, pre.conv1, pre.bn1)
    fused.first_conv._norm_layer.bn_momentum = bn_momentum

    # Block 1
    block = fused.layer1
    if block[0].downsample is not None:
        block[0].downsample = copy_from_pretrained(block[0].downsample, pre.layer1[0].downsample[0], pre.layer1[0].downsample[1])
        block[0].downsample._norm_layer.bn_momentum = bn_momentum
    for i in range(len(block)):
        block[i].conv1 = copy_from_pretrained(block[i].conv1, pre.layer1[i].conv1, pre.layer1[i].bn1)
        block[i].conv2 = copy_from_pretrained(block[i].conv2, pre.layer1[i].conv2, pre.layer1[i].bn2)
        block[i].conv1._norm_layer.bn_momentum = bn_momentum
        block[i].conv2._norm_layer.bn_momentum = bn_momentum
        if type(block[i]) == FoldedFusedBottleneck:
            block[i].conv3 = copy_from_pretrained(block[i].conv3, pre.layer1[i].conv3, pre.layer1[i].bn3)
            block[i].conv3._norm_layer.bn_momentum = bn_momentum

    # Block 2
    block = fused.layer2
    if block[0].downsample is not None:
        block[0].downsample = copy_from_pretrained(block[0].downsample, pre.layer2[0].downsample[0],
                                                   pre.layer2[0].downsample[1])
        block[0].downsample._norm_layer.bn_momentum = bn_momentum
    for i in range(len(block)):
        block[i].conv1 = copy_from_pretrained(block[i].conv1, pre.layer2[i].conv1, pre.layer2[i].bn1)
        block[i].conv2 = copy_from_pretrained(block[i].conv2, pre.layer2[i].conv2, pre.layer2[i].bn2)
        block[i].conv1._norm_layer.bn_momentum = bn_momentum
        block[i].conv2._norm_layer.bn_momentum = bn_momentum
        if type(block[i]) == FoldedFusedBottleneck:
            block[i].conv3 = copy_from_pretrained(block[i].conv3, pre.layer2[i].conv3, pre.layer2[i].bn3)
            block[i].conv3._norm_layer.bn_momentum = bn_momentum

    # Block 3
    block = fused.layer3
    if block[0].downsample is not None:
        block[0].downsample = copy_from_pretrained(block[0].downsample, pre.layer3[0].downsample[0],
                                                   pre.layer3[0].downsample[1])
        block[0].downsample._norm_layer.bn_momentum = bn_momentum
    for i in range(len(block)):
        block[i].conv1 = copy_from_pretrained(block[i].conv1, pre.layer3[i].conv1, pre.layer3[i].bn1)
        block[i].conv2 = copy_from_pretrained(block[i].conv2, pre.layer3[i].conv2, pre.layer3[i].bn2)
        block[i].conv1._norm_layer.bn_momentum = bn_momentum
        block[i].conv2._norm_layer.bn_momentum = bn_momentum
        if type(block[i]) == FoldedFusedBottleneck:
            block[i].conv3 = copy_from_pretrained(block[i].conv3, pre.layer3[i].conv3, pre.layer3[i].bn3)
            block[i].conv3._norm_layer.bn_momentum = bn_momentum

    # Block 4
    if fused.num_blocks == 4:
        block = fused.layer4
        if block[0].downsample is not None:
            block[0].downsample = copy_from_pretrained(block[0].downsample, pre.layer4[0].downsample[0],
                                                       pre.layer4[0].downsample[1])
            block[0].downsample._norm_layer.bn_momentum = bn_momentum
        for i in range(len(block)):
            block[i].conv1 = copy_from_pretrained(block[i].conv1, pre.layer4[i].conv1, pre.layer4[i].bn1)
            block[i].conv2 = copy_from_pretrained(block[i].conv2, pre.layer4[i].conv2, pre.layer4[i].bn2)
            block[i].conv3 = copy_from_pretrained(block[i].conv3, pre.layer4[i].conv3, pre.layer4[i].bn3)
            block[i].conv1._norm_layer.bn_momentum = bn_momentum
            block[i].conv2._norm_layer.bn_momentum = bn_momentum
            block[i].conv3._norm_layer.bn_momentum = bn_momentum

    # Classifier
    fused.fc = copy_from_pretrained(fused.fc, pre.fc)
    return fused


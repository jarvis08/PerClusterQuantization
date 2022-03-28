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

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1,
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
        self.target_bit = torch.nn.Parameter(torch.tensor(target_bit, dtype=torch.int8), requires_grad=False)

        self.act_range = nn.Parameter(torch.zeros(2), requires_grad=False)
        self.apply_ema = nn.Parameter(torch.tensor(0, dtype=torch.bool), requires_grad=False)

        if self.downsample is not None:
            self.bn_down = FusedBnReLU(planes, a_bit=bit_addcat, arg_dict=arg_dict)
        self.conv1 = fused_conv3x3(inplanes, planes, stride, arg_dict=arg_dict, a_bit=bit_conv_act)
        self.bn1 = FusedBnReLU(planes, activation=nn.ReLU, a_bit=target_bit, arg_dict=arg_dict)
        self.conv2 = fused_conv3x3(planes, planes, arg_dict=arg_dict, a_bit=bit_conv_act)
        self.bn2 = FusedBnReLU(planes, a_bit=bit_addcat, arg_dict=arg_dict)
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

    @torch.no_grad()
    def set_block_qparams(self, s1, z1, s_target, z_target):
        self.s1, self.z1 = s1, z1  # S, Z of 8/16/32 bit
        self.s_target, self.z_target = s_target, z_target  # S, Z of 4/8 bit
        self.M0, self.shift = quantize_M(self.s1 / self.s_target)

        if self.downsample:
            prev_s, prev_z = self.downsample.set_qparams(s_target, z_target)
            self.bn_down.set_qparams(prev_s, prev_z)

        prev_s, prev_z = self.conv1.set_qparams(s_target, z_target)
        prev_s, prev_z = self.bn1.set_qparams(prev_s, prev_z)
        prev_s, prev_z = self.conv2.set_qparams(prev_s, prev_z)
        self.bn2.set_qparams(prev_s, prev_z)

        self.s3, self.z3 = calc_qparams(self.act_range[0], self.act_range[1], self.a_bit)
        nxt_s_target, nxt_z_target = calc_qparams(self.act_range[0], self.act_range[1], self.target_bit)
        return self.s3, self.z3, nxt_s_target, nxt_z_target


class FusedBottleneck(nn.Module):
    expansion: int = 4

    def __init__(self, inplane:int, planes:int, stride:int=1, downsample=None,
                 groups: int = 1, base_width:int =64, dilation:int =1, a_bit=None, arg_dict=None) -> None:
        super(FusedBottleneck, self).__init__()

        self.downsample = downsample
        self.stride = stride

        target_bit, bit_conv_act, bit_addcat, self.smooth, self.use_ste, self.num_clusters, self.runtime_helper, self.inference_bit \
            = itemgetter('bit', 'bit_conv_act', 'bit_addcat', 'smooth', 'ste', 'cluster', 'runtime_helper', 'inference_bit')(arg_dict)
        self.a_bit = torch.nn.Parameter(torch.tensor(bit_addcat, dtype=torch.int8), requires_grad=False)
        self.target_bit = torch.nn.Parameter(torch.tensor(target_bit, dtype=torch.int8), requires_grad=False)

        self.act_range = nn.Parameter(torch.zeros(2), requires_grad=False)
        self.apply_ema = nn.Parameter(torch.tensor(0, dtype=torch.bool), requires_grad=False)

        width = int(planes * (base_width/64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1

        if self.downsample is not None:
            self.bn_down = FusedBnReLU(planes * self.expansion, a_bit=bit_addcat, arg_dict=arg_dict)
        self.conv1 = fused_conv1x1(in_planes=inplane, out_planes=width, a_bit=bit_conv_act, arg_dict=arg_dict)
        self.bn1 = FusedBnReLU(width, activation=nn.ReLU, a_bit=target_bit, arg_dict=arg_dict)
        self.conv2 = fused_conv3x3(in_planes=width, out_planes=width, stride=stride, groups=groups, dilation=dilation,
                                   a_bit=bit_conv_act, arg_dict=arg_dict)
        self.bn2 = FusedBnReLU(width, activation=nn.ReLU, a_bit=target_bit, arg_dict=arg_dict)
        self.conv3 = fused_conv1x1(in_planes=width, out_planes=planes * self.expansion,
                                   a_bit=bit_conv_act, arg_dict=arg_dict)
        self.bn3 = FusedBnReLU(planes * self.expansion, a_bit=bit_addcat, arg_dict=arg_dict)
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

        # if not self.training:
        #     return out
        #
        # if self.apply_ema:
        #     self.act_range[0], self.act_range[1] = ema(out, self.act_range, self.smooth)
        #     if self.runtime_helper.apply_fake_quantization:
        #         s, z = calc_qparams(self.act_range[0], self.act_range[1], self.target_bit)
        #         out = fake_quantize(out, s, z, self.target_bit, use_ste=self.use_ste)
        # else:
        #     self.act_range[0], self.act_range[1] = get_range(out)
        #     self.apply_ema.data = torch.tensor(True, dtype=torch.bool)

        if self.apply_ema:
            self.act_range[0], self.act_range[1] = ema(out, self.act_range, self.smooth)
        else:
            self.act_range[0], self.act_range[1] = get_range(out)
            self.apply_ema.data = torch.tensor(True, dtype=torch.bool)
        return out

    def set_block_qparams(self, s1, z1, s_target, z_target):
        self.a_bit.data = self.inference_bit
        self.target_bit.data = self.inference_bit
        self.s1, self.z1 = s1, z1                          # S, Z of 8/16/32 bit
        self.s_target, self.z_target = s_target, z_target  # S, Z of 4/8 bit
        self.M0, self.shift = quantize_M(self.s1 / self.s_target)

        if self.downsample:
            prev_s, prev_z = self.downsample.set_qparams(s_target, z_target)
            self.bn_down.set_qparams(prev_s, prev_z)

        prev_s, prev_z = self.conv1.set_qparams(s_target, z_target)
        prev_s, prev_z = self.bn1.set_qparams(prev_s, prev_z)
        prev_s, prev_z = self.conv2.set_qparams(prev_s, prev_z)
        prev_s, prev_z = self.bn2.set_qparams(prev_s, prev_z)
        prev_s, prev_z = self.conv3.set_qparams(prev_s, prev_z)
        self.bn3.set_qparams(prev_s, prev_z)

        self.s3, self.z3 = calc_qparams(self.act_range[0], self.act_range[1], self.a_bit)
        nxt_s_target, nxt_z_target = calc_qparams(self.act_range[0], self.act_range[1], self.target_bit)
        return self.s3, self.z3, nxt_s_target, nxt_z_target


class FusedResNet(nn.Module):
    def __init__(self, block, layers, arg_dict, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None):
        super(FusedResNet, self).__init__()
        self.arg_dict = arg_dict
        target_bit, self.bit_conv_act, bit_addcat, bit_first, bit_classifier, self.smooth, self.num_clusters, self.runtime_helper, self.inference_bit \
            = itemgetter('bit', 'bit_conv_act', 'bit_addcat', 'bit_first', 'bit_classifier', 'smooth', 'cluster', 'runtime_helper', 'inference_bit')(arg_dict)
        self.target_bit = torch.nn.Parameter(torch.tensor(target_bit, dtype=torch.int8), requires_grad=False)
        self.a_bit = torch.nn.Parameter(torch.tensor(bit_addcat, dtype=torch.int8), requires_grad=False)
        self.in_bit = torch.nn.Parameter(torch.tensor(bit_first, dtype=torch.int8), requires_grad=False)

        self.in_range = nn.Parameter(torch.zeros(2), requires_grad=False)
        self.apply_ema = nn.Parameter(torch.tensor(0, dtype=torch.bool), requires_grad=False)

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
                                      w_bit=bit_first, a_bit=self.bit_conv_act, arg_dict=arg_dict)
        self.bn1 = FusedBnReLU(self.inplanes, activation=nn.ReLU, a_bit=bit_addcat, arg_dict=self.arg_dict)
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
                                       a_bit=self.bit_conv_act)

        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, downsample=downsample, groups=self.groups,
                            base_width=self.base_width, dilation=previous_dilation, arg_dict=self.arg_dict))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups, base_width=self.base_width,
                                dilation=self.dilation, arg_dict=self.arg_dict))
        return nn.Sequential(*layers)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        # if self.training:
        #     if self.apply_ema:
        #         self.in_range[0], self.in_range[1] = ema(x, self.in_range, self.smooth)
        #         if self.runtime_helper.apply_fake_quantization:
        #             s, z = calc_qparams(self.in_range[0], self.in_range[1], self.in_bit)
        #             x = fake_quantize(x, s, z, self.in_bit)
        #     else:
        #         self.in_range[0], self.in_range[1] = get_range(x)
        #         self.apply_ema.data = torch.tensor(True, dtype=torch.bool)

        if self.apply_ema:
            self.in_range[0], self.in_range[1] = ema(x, self.in_range, self.smooth)
        else:
            self.in_range[0], self.in_range[1] = get_range(x)
            self.apply_ema.data = torch.tensor(True, dtype=torch.bool)

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

    def set_quantization_params(self):
        self.in_bit.data = self.inference_bit
        self.target_bit.data = self.inference_bit
        self.scale, self.zero_point = calc_qparams(self.in_range[0], self.in_range[1], self.in_bit)
        prev_s, prev_z = self.first_conv.set_qparams(self.scale, self.zero_point)

        s1, z1 = self.bn1.set_qparams(prev_s, prev_z)
        s_target, z_target = calc_qparams(self.bn1.act_range[0], self.bn1.act_range[1], self.target_bit)

        blocks = [self.layer1, self.layer2, self.layer3, self.layer4]
        for block in blocks:
            for b in range(len(block)):
                s1, z1, s_target, z_target = block[b].set_block_qparams(s1, z1, s_target, z_target)

        self.s1, self.z1 = s1, z1  # S, Z of 8/16/32 bit
        self.s_target, self.z_target = s_target, z_target  # S, Z of 4/8 bit
        self.M0, self.shift = quantize_M(self.s1 / self.s_target)
        self.fc.set_qparams(self.s_target, self.z_target)


class FusedResNet20(nn.Module):
    def __init__(self, block, layers, arg_dict, num_classes=10):
        super(FusedResNet20, self).__init__()
        self.arg_dict = arg_dict
        target_bit, self.bit_conv_act, bit_addcat, bit_first, bit_classifier, self.smooth, self.num_clusters, self.runtime_helper \
            = itemgetter('bit', 'bit_conv_act', 'bit_addcat', 'bit_first', 'bit_classifier', 'smooth', 'cluster', 'runtime_helper')(arg_dict)
        self.target_bit = torch.nn.Parameter(torch.tensor(target_bit, dtype=torch.int8), requires_grad=False)
        self.a_bit = torch.nn.Parameter(torch.tensor(bit_addcat, dtype=torch.int8), requires_grad=False)
        self.in_bit = torch.nn.Parameter(torch.tensor(bit_first, dtype=torch.int8), requires_grad=False)

        self.in_range = nn.Parameter(torch.zeros(2), requires_grad=False)
        self.apply_ema = nn.Parameter(torch.tensor(0, dtype=torch.bool), requires_grad=False)

        self._norm_layer = nn.BatchNorm2d
        self.inplanes = 16
        self.dilation = 1
        self.num_blocks = 3

        self.arg_dict = arg_dict

        self.first_conv = FusedConv2d(3, 16, kernel_size=3, stride=1, padding=1,
                                      w_bit=bit_first, a_bit=self.bit_conv_act, arg_dict=arg_dict)
        self.bn1 = FusedBnReLU(16, activation=nn.ReLU, a_bit=bit_addcat, arg_dict=arg_dict)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.fc = FusedLinear(64 * block.expansion, num_classes, is_classifier=True,
                              w_bit=bit_classifier, a_bit=bit_classifier, arg_dict=arg_dict)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = fused_conv1x1(self.inplanes, planes * block.expansion, stride,
                                       a_bit=self.bit_conv_act, arg_dict=self.arg_dict)
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample,
                            norm_layer=self._norm_layer, arg_dict=self.arg_dict))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_layer=self._norm_layer, arg_dict=self.arg_dict))
        return nn.Sequential(*layers)

    def forward(self, x):
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
        x = self.bn1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def set_quantization_params(self):
        self.scale, self.zero_point = calc_qparams(self.in_range[0], self.in_range[1], self.in_bit)
        prev_s, prev_z = self.first_conv.set_qparams(self.scale, self.zero_point)

        s1, z1 = self.bn1.set_qparams(prev_s, prev_z)
        s_target, z_target = calc_qparams(self.bn1.act_range[0], self.bn1.act_range[1], self.target_bit)

        blocks = [self.layer1, self.layer2, self.layer3]
        for block in blocks:
            for b in range(len(block)):
                s1, z1, s_target, z_target = block[b].set_block_qparams(s1, z1, s_target, z_target)

        self.s1, self.z1 = s1, z1  # S, Z of 8/16/32 bit
        self.s_target, self.z_target = s_target, z_target  # S, Z of 4/8 bit
        self.M0, self.shift = quantize_M(self.s1 / self.s_target)
        self.fc.set_qparams(self.s_target, self.z_target)


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


def fold_resnet(model):
    # First layer
    model.first_conv.fold_conv_and_bn()

    # Block 1
    fp_block = model.layer1
    for i in range(len(fp_block)):
        fp_block[i].conv1.fold_conv_and_bn()
        fp_block[i].conv2.fold_conv_and_bn()
        if type(fp_block[i]) == FusedBottleneck:
            fp_block[i].conv3.fold_conv_and_bn()

    # Block 2
    fp_block = model.layer2
    fp_block[0].downsample.fold_conv_and_bn()
    for i in range(len(fp_block)):
        fp_block[i].conv1.fold_conv_and_bn()
        fp_block[i].conv2.fold_conv_and_bn()
        if type(fp_block[i]) == FusedBottleneck:
            fp_block[i].conv3.fold_conv_and_bn()

    # Block 3
    fp_block = model.layer3
    fp_block[0].downsample.fold_conv_and_bn()
    for i in range(len(fp_block)):
        fp_block[i].conv1.fold_conv_and_bn()
        fp_block[i].conv2.fold_conv_and_bn()
        if type(fp_block[i]) == FusedBottleneck:
            fp_block[i].conv3.fold_conv_and_bn()

    # Block 4
    if model.num_blocks == 4:
        fp_block = model.layer4
        fp_block[0].downsample.fold_conv_and_bn()
        for i in range(len(fp_block)):
            fp_block[i].conv1.fold_conv_and_bn()
            fp_block[i].conv2.fold_conv_and_bn()
            if type(fp_block[i]) == FusedBottleneck:
                fp_block[i].conv3.fold_conv_and_bn()
    return model
def modify_fused_resnet_qn_pre_hook(model):
    """ 
        Copy from pre model's params to fused layers.
        Use fused architecture, but not really fused (use CONV & BN seperately)
    """
    # model.first_conv.quant_noise = False
    model.first_conv.conv = _quant_noise(model.first_conv.conv, model.qn_prob, 1, bit=model.bit)
    # Block 1
    block = model.layer1
    if len(model.layer3) == 6: #ResNet50 일땐 첫번째 블록에도 downsample이 있음.
        m = block[0].downsample
        m.conv = _quant_noise(m.conv, m.runtime_helper.qn_prob, 4, bit=m.bit)
    for i in range(len(block)):
        if type(block[i]) == FusedBottleneck:
            block[i].conv1.conv = _quant_noise(block[i].conv1.conv, block[i].conv1.runtime_helper.qn_prob, 4, bit=block[i].bit)
            block[i].conv2.conv = _quant_noise(block[i].conv2.conv, block[i].conv2.runtime_helper.qn_prob, 9, bit=block[i].bit)
            block[i].conv3.conv = _quant_noise(block[i].conv3.conv, block[i].conv3.runtime_helper.qn_prob, 4, bit=block[i].bit)
        elif type(block[i]) == FusedBasicBlock:
            block[i].conv1.conv = _quant_noise(block[i].conv1.conv, block[i].conv1.runtime_helper.qn_prob, 9, bit=block[i].bit)
            block[i].conv2.conv = _quant_noise(block[i].conv2.conv, block[i].conv2.runtime_helper.qn_prob, 9, bit=block[i].bit)

    # Block 2
    block = model.layer2
    block[0].downsample.conv = _quant_noise(block[0].downsample.conv, block[0].downsample.runtime_helper.qn_prob,
                                            4, bit=block[0].downsample.bit)
    for i in range(len(block)):
        if type(block[i]) == FusedBottleneck:
            block[i].conv1.conv = _quant_noise(block[i].conv1.conv, block[i].conv1.runtime_helper.qn_prob, 4, bit=block[i].bit)
            block[i].conv2.conv = _quant_noise(block[i].conv2.conv, block[i].conv2.runtime_helper.qn_prob, 9, bit=block[i].bit)
            block[i].conv3.conv = _quant_noise(block[i].conv3.conv, block[i].conv3.runtime_helper.qn_prob, 4, bit=block[i].bit)
        elif type(block[i]) == FusedBasicBlock:
            block[i].conv1.conv = _quant_noise(block[i].conv1.conv, block[i].conv1.runtime_helper.qn_prob, 9, bit=block[i].bit)
            block[i].conv2.conv = _quant_noise(block[i].conv2.conv, block[i].conv2.runtime_helper.qn_prob, 9, bit=block[i].bit)

    # Block 3
    block = model.layer3
    block[0].downsample.conv = _quant_noise(block[0].downsample.conv, block[0].downsample.runtime_helper.qn_prob,
                                            4, bit=block[0].downsample.bit)
    for i in range(len(block)):
        if type(block[i]) == FusedBottleneck:
            block[i].conv1.conv = _quant_noise(block[i].conv1.conv, block[i].conv1.runtime_helper.qn_prob, 4, bit=block[i].bit)
            block[i].conv2.conv = _quant_noise(block[i].conv2.conv, block[i].conv2.runtime_helper.qn_prob, 9, bit=block[i].bit)
            block[i].conv3.conv = _quant_noise(block[i].conv3.conv, block[i].conv3.runtime_helper.qn_prob, 4, bit=block[i].bit)
        elif type(block[i]) == FusedBasicBlock:
            block[i].conv1.conv = _quant_noise(block[i].conv1.conv, block[i].conv1.runtime_helper.qn_prob, 9, bit=block[i].bit)
            block[i].conv2.conv = _quant_noise(block[i].conv2.conv, block[i].conv2.runtime_helper.qn_prob, 9, bit=block[i].bit)
    # Block 4
    if model.num_blocks == 4:
        block = model.layer4
        block[0].downsample.conv = _quant_noise(block[0].downsample.conv, block[0].downsample.runtime_helper.qn_prob,
                                                4, bit=block[0].downsample.bit)
        for i in range(len(block)):
            if type(block[i]) == FusedBottleneck:
                block[i].conv1.conv = _quant_noise(block[i].conv1.conv, block[i].conv1.runtime_helper.qn_prob, 4, bit=block[i].bit)
                block[i].conv2.conv = _quant_noise(block[i].conv2.conv, block[i].conv2.runtime_helper.qn_prob, 9, bit=block[i].bit)
                block[i].conv3.conv = _quant_noise(block[i].conv3.conv, block[i].conv3.runtime_helper.qn_prob, 4, bit=block[i].bit)
            elif type(block[i]) == FusedBasicBlock:
                block[i].conv1.conv = _quant_noise(block[i].conv1.conv, block[i].conv1.runtime_helper.qn_prob, 9, bit=block[i].bit)
                block[i].conv2.conv = _quant_noise(block[i].conv2.conv, block[i].conv2.runtime_helper.qn_prob, 9, bit=block[i].bit)

    # Classifier
    model.fc.quant_noise = False
    model.qn_prob = model.runtime_helper.qn_prob
    # return model


def set_hawq_resnet(hawq, pre, bn_momentum=0.99):
    """
        Copy from pre model's params to fused layers.
        Use fused architecture, but not really fused (use CONV & BN seperately)
    """
    # First layer
    hawq.quant_init_convbn.conv = copy_weight_from_pretrained(hawq.quant_init_convbn.conv, pre.conv1)
    hawq.quant_init_convbn.bn = copy_bn_from_pretrained(hawq.quant_init_convbn.bn, pre.bn1)
    channel = hawq.channel
    # Block 1
    for stage_num in range(0, 4):
        pre_layer = getattr(pre, f'layer{stage_num+1}')
        for channel_num in range(channel[stage_num]):
            unit = getattr(hawq, f'stage{stage_num+1}.' + f'unit{channel_num+1}')
            pre_unit = pre_layer[channel_num]
            unit.quant_convbn1.conv = copy_weight_from_pretrained(unit.quant_convbn1.conv, pre_unit.conv1)
            unit.quant_convbn1.bn = copy_bn_from_pretrained(unit.quant_convbn1.bn, pre_unit.bn1)
            unit.quant_convbn1.bn.momentum = bn_momentum
            unit.quant_convbn2.conv = copy_weight_from_pretrained(unit.quant_convbn2.conv, pre_unit.conv2)
            unit.quant_convbn2.bn = copy_bn_from_pretrained(unit.quant_convbn2.bn, pre_unit.bn2)
            unit.quant_convbn2.bn.momentum = bn_momentum
            unit.quant_convbn3.conv = copy_weight_from_pretrained(unit.quant_convbn3.conv, pre_unit.conv3)
            unit.quant_convbn3.bn = copy_bn_from_pretrained(unit.quant_convbn3.bn, pre_unit.bn3)
            unit.quant_convbn3.bn.momentum = bn_momentum
            try:
                if unit.quant_identity_convbn is not None:
                    unit.quant_identity_convbn.conv = copy_weight_from_pretrained(unit.quant_identity_convbn.conv,
                                                                                  pre_unit.downsample[0])
                    unit.quant_identity_convbn.bn = copy_bn_from_pretrained(unit.quant_identity_convbn.bn,
                                                                                  pre_unit.downsample[1])
                    unit.quant_identity_convbn.bn.momentum = bn_momentum
            except:
                pass
    hawq.quant_output = copy_weight_from_pretrained(hawq.quant_output, pre.fc)
    return hawq

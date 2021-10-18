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
        norm_layer = norm_layer if norm_layer else torch.nn.BatchNorm2d

        arg_bit, self.smooth, self.use_ste, self.runtime_helper, self.quant_noise, self.qn_prob\
            = itemgetter('bit', 'smooth', 'ste', 'runtime_helper', 'quant_noise', 'qn_prob')(arg_dict)
        self.bit = torch.nn.Parameter(torch.tensor(arg_bit, dtype=torch.int8), requires_grad=False)

        self.act_range = nn.Parameter(torch.zeros(2), requires_grad=False)
        self.apply_ema = False

        self.conv1 = fused_conv3x3(inplanes, planes, stride, norm_layer=norm_layer, activation=nn.ReLU, arg_dict=arg_dict)
        self.conv2 = fused_conv3x3(planes, planes, norm_layer=norm_layer, arg_dict=arg_dict)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        if not self.training:
            if self.runtime_helper.check2mix:
                if self.apply_ema:
                    self.act_range[0], self.act_range[1] = ema(out, self.act_range, self.smooth)
                else:
                    self.act_range[0], self.act_range[1] = get_range(out)
                    self.apply_ema = True
            return out
        
        if self.apply_ema:
            self.act_range[0], self.act_range[1] = ema(out, self.act_range, self.smooth)
            if self.runtime_helper.apply_fake_quantization:
                s, z = calc_qparams(self.act_range[0], self.act_range[1], self.bit)
                out = fake_quantize(out, s, z, self.bit, self.use_ste)
        else:
            self.act_range[0], self.act_range[1] = get_range(out)
            self.apply_ema = True
        return out

    def set_block_qparams(self, s1, z1):
        if self.downsample:
            self.downsample.set_qparams(s1, z1)
        prev_s, prev_z = self.conv1.set_qparams(s1, z1)
        self.conv2.set_qparams(prev_s, prev_z)
        self.s3, self.z3 = calc_qparams(self.act_range[0], self.act_range[1], self.bit)
        return self.s3, self.z3

    def find_block_longest_range(self, longest, found_layers):
        layers = [self.conv1, self.conv2]
        if self.downsample:
            layers.append(self.downsample)
        for i in range(len(layers)):
            layer = layers[i]
            if len(found_layers) and id(found_layers[0]) == id(layer):
                pass
            else:
                len_range = torch.abs(layer.act_range[1] - layer.act_range[0])
                longest_range = torch.abs(longest.act_range[1] - longest.act_range[0])
                if len_range > longest_range:
                    longest = layer
        return longest

    def find_block_biggest_layer(self, biggest, found_layers):
        layers = [self.conv1, self.conv2]
        if self.downsample:
            layers.append(self.downsample)
        for i in range(len(layers)):
            layer = layers[i]
            if len(found_layers) and id(found_layers[0]) == id(layer):
                pass
            else:
                number_of_params = layer.conv.weight.numel()
                if isinstance(biggest, FusedConv2d):
                    biggest_number = biggest.conv.weight.numel()
                else:
                    biggest_number = biggest.fc.weight.numel()
                if number_of_params > biggest_number:
                    biggest = layer
        return biggest

    def reset_block_ranges(self):
        if self.downsample:
            self.downsample.reset_activation_range()
        self.conv1.reset_activation_range()
        self.conv2.reset_activation_range()
        self.act_range[0], self.act_range[1] = 0.0, 0.0
        self.apply_ema = False
    

class FusedResNet20(nn.Module):
    def __init__(self, block, layers, arg_dict, num_classes=10):
        super(FusedResNet20, self).__init__()
        self.arg_dict = arg_dict
        target_bit, first_bit, classifier_bit, self.smooth, self.runtime_helper \
            = itemgetter('bit', 'first_bit', 'classifier_bit', 'smooth', 'runtime_helper')(arg_dict)
        self.target_bit = torch.nn.Parameter(torch.tensor(target_bit, dtype=torch.int8), requires_grad=False)
        self.in_bit = torch.nn.Parameter(torch.tensor(first_bit, dtype=torch.int8), requires_grad=False)

        self.in_range = nn.Parameter(torch.zeros(2), requires_grad=False)
        self.apply_ema = False

        self._norm_layer = nn.BatchNorm2d
        self.inplanes = 16
        self.dilation = 1
        self.num_blocks = 3

        self.arg_dict = arg_dict

        self.first_conv = FusedConv2d(3, 16, kernel_size=3, stride=1, padding=1, norm_layer=self._norm_layer, activation=nn.ReLU, arg_dict=arg_dict)
        self.bn1 = FusedBnReLU(16, activation=nn.ReLU, arg_dict=arg_dict)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.fc = FusedLinear(64 * block.expansion, num_classes, arg_dict=arg_dict)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = fused_conv1x1(self.inplanes, planes * block.expansion, stride, norm_layer=self._norm_layer, arg_dict=self.arg_dict)
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
                self.apply_ema = True

        x = self.first_conv(x)
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
        blocks = [self.layer1, self.layer2, self.layer3]
        for block in blocks:
            for i in range(len(block)):
                prev_s, prev_z = block[i].set_block_qparams(prev_s, prev_z)
        self.fc.set_qparams(prev_s, prev_z)

    def find_longest_range(self, found_layers):
        longest = None
        if len(found_layers) and id(found_layers[0]) == id(self.first_conv):
            pass
        else:
            longest = self.first_conv

        blocks = [self.layer1, self.layer2, self.layer3]
        for block in blocks:
            for i in range(len(block)):
                longest = block[i].find_block_longest_range(longest, found_layers)
                if len(found_layers) and id(found_layers[0]) == id(block[i]):
                    pass
                else:
                    len_range = torch.abs(block[i].act_range[1] - block[i].act_range[0])
                    longest_range = torch.abs(longest.act_range[1] - longest.act_range[0])
                    if len_range > longest_range:
                        longest = block[i]

        if len(found_layers) and id(found_layers[0]) == id(self.fc):
            pass
        else:
            len_range = torch.abs(self.fc.act_range[1] - self.fc.act_range[0])
            longest_range = torch.abs(longest.act_range[1] - longest.act_range[0])
            if len_range > longest_range:
                longest = self.fc
        print('Longest layer: {}, Length of Range: {}'.format(longest.layer_type, torch.abs(longest.act_range[1] - longest.act_range[0])))
        return longest

    def find_biggest_layer(self, found_layers):
        biggest = None
        if len(found_layers) and id(found_layers[0]) == id(self.first_conv):
            pass
        else:
            biggest = self.first_conv

        blocks = [self.layer1, self.layer2, self.layer3]
        for block in blocks:
            for i in range(len(block)):
                biggest = block[i].find_block_biggest_layer(biggest, found_layers)

        if len(found_layers) and id(found_layers[0]) == id(self.fc):
            pass
        else:
            number_of_params = self.fc.fc.weight.numel()
            biggest_number = biggest.conv.weight.numel()
            if number_of_params > biggest_number:
                biggest = self.fc

        if isinstance(biggest, FusedConv2d):
            n_params = biggest.conv.weight.numel()
        else:
            n_params = biggest.fc.weight.numel()
        print('Biggest layer: {}, # of Params: {}'.format(biggest.layer_type, n_params))
        return biggest

    def reset_ranges(self):
        self.first_conv.reset_activation_range()
        blocks = [self.layer1, self.layer2, self.layer3]
        for block in blocks:
            for i in range(len(block)):
                block[i].reset_block_ranges()
        self.fc.reset_activation_range()


def check2mix_fused_resnet20(arg_dict, num_classes=10):
    return FusedResNet20(FusedBasicBlock, [3, 3, 3], arg_dict, num_classes=num_classes)


def set_fused_resnet_with_fold_method(fused, pre):
    """
        Copy from pre model's params to fused layers.
        Use fused architecture, but not really fused (use CONV & BN seperately)
    """
    # First layer
    fused.first_conv = copy_from_pretrained(fused.first_conv, pre.conv1, pre.bn1)

    # Block 1
    block = fused.layer1
    for i in range(len(block)):
        block[i].conv1 = copy_from_pretrained(block[i].conv1, pre.layer1[i].conv1, pre.layer1[i].bn1)
        block[i].conv2 = copy_from_pretrained(block[i].conv2, pre.layer1[i].conv2, pre.layer1[i].bn2)

    # Block 2
    block = fused.layer2
    block[0].downsample = copy_from_pretrained(block[0].downsample, pre.layer2[0].downsample[0],
                                               pre.layer2[0].downsample[1])
    for i in range(len(block)):
        block[i].conv1 = copy_from_pretrained(block[i].conv1, pre.layer2[i].conv1, pre.layer2[i].bn1)
        block[i].conv2 = copy_from_pretrained(block[i].conv2, pre.layer2[i].conv2, pre.layer2[i].bn2)

    # Block 3
    block = fused.layer3
    block[0].downsample = copy_from_pretrained(block[0].downsample, pre.layer3[0].downsample[0],
                                               pre.layer3[0].downsample[1])
    for i in range(len(block)):
        block[i].conv1 = copy_from_pretrained(block[i].conv1, pre.layer3[i].conv1, pre.layer3[i].bn1)
        block[i].conv2 = copy_from_pretrained(block[i].conv2, pre.layer3[i].conv2, pre.layer3[i].bn1)

    fused.fc = copy_from_pretrained(fused.fc, pre.fc)
    return fused


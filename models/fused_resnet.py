import torch
import torch.nn as nn

from .layers.conv2d import *
from .layers.linear import *
from .quantization_utils import *


def fused_conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1, bias=False, norm_layer=None, activation=None, bit=8, smooth=0.999, quant_noise=False, q_prob=0.1):
    """3x3 convolution with padding"""
    return FusedConv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                       padding=dilation, groups=groups, dilation=dilation, bias=bias,
                       norm_layer=norm_layer, activation=activation, bit=bit, smooth=smooth, quant_noise=quant_noise, q_prob=q_prob)


def fused_conv1x1(in_planes, out_planes, stride=1, bias=False, norm_layer=None, activation=None, bit=8, smooth=0.999, quant_noise=False, q_prob=0.1):
    """1x1 convolution"""
    return FusedConv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=bias,
                       norm_layer=norm_layer, activation=activation, bit=bit, smooth=smooth, quant_noise=quant_noise, q_prob=q_prob)


class FusedBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1,
                 norm_layer=None, bit=8, smooth=0.999, quant_noise=False, q_prob=0.1):
        super(FusedBasicBlock, self).__init__()
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.downsample = downsample
        self.stride = stride
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.bit = bit
        self.q_max = 2 ** self.bit - 1
        self.smooth = smooth
        self.flag_ema_init = False
        self.flag_fake_quantization = False
        self.act_range = nn.Parameter(torch.zeros(2), requires_grad=False)

        self.quant_noise = quant_noise
        if self.quant_noise:
            self.q_prob = q_prob
            self.conv1 = fused_conv3x3(inplanes, planes, stride, norm_layer=self._norm_layer,
                                   activation=nn.ReLU6, bit=bit, smooth=smooth, quant_noise=self.quant_noise, q_prob=self.q_prob)
            self.conv2 = fused_conv3x3(planes, planes, norm_layer=self._norm_layer, bit=bit, smooth=smooth, quant_noise=quant_noise, q_prob=self.q_prob)
        else:
            self.conv1 = fused_conv3x3(inplanes, planes, stride, norm_layer=self._norm_layer,activation=nn.ReLU6, bit=bit, smooth=smooth)
            self.conv2 = fused_conv3x3(planes, planes, norm_layer=self._norm_layer, bit=bit, smooth=smooth)
        self.relu = nn.ReLU6(inplace=False)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)

        if self.training:
            if self.flag_ema_init:
                self.act_range[0], self.act_range[1] = ema(out, self.act_range, self.smooth)
                if self.flag_fake_quantization:
                    s, z = calc_qparams(self.act_range[0], self.act_range[1], self.q_max)
                    with torch.no_grad():
                        out.copy_(fake_quantize(out, s.cuda(), z.cuda(), self.q_max))
            else:
                self.act_range[0] = torch.min(out).item()
                self.act_range[1] = torch.max(out).item()
                self.flag_ema_init = True
        return out

    def set_block_fq_flag(self):
        self.flag_fake_quantization = True
        if self.downsample:
            self.downsample.flag_fake_quantization = True
        self.conv1.flag_fake_quantization = True
        self.conv2.flag_fake_quantization = True

    def set_block_qparams(self, s1, z1):
        if self.downsample:
            self.downsample.set_qparams(s1, z1)
        prev_s, prev_z = self.conv1.set_qparams(s1, z1)
        _, _ = self.conv2.set_qparams(prev_s, prev_z)
        self.s3, self.z3 = calc_qparams(self.act_range[0], self.act_range[1], self.q_max)
        return self.s3, self.z3


class FusedResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None, norm_layer=None,
                 bit=8, smooth=0.999, quant_noise=False, q_prob=0.1):
        super(FusedResNet, self).__init__()
        self.bit = bit
        self.in_range = nn.Parameter(torch.zeros(2), requires_grad=False)
        self.flag_ema_init = False
        self.flag_fake_quantization = False
        self.smooth = smooth
        self.q_max = 2 ** self.bit - 1

        self.quant_noise = quant_noise
        self.q_prob = q_prob

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
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
        self.first_conv = FusedConv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False,
                                      norm_layer=self._norm_layer, activation=nn.ReLU6, bit=bit, smooth=smooth, quant_noise=self.quant_noise, q_prob=self.q_prob)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = FusedLinear(512 * block.expansion, num_classes, bias=True, bit=bit, smooth=smooth, quant_noise=self.quant_noise, q_prob=self.q_prob)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, FusedBasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        # Planes : n_channel_output
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = fused_conv1x1(self.inplanes, planes * block.expansion, stride, bias=False,
                                       norm_layer=self._norm_layer, bit=self.bit, smooth=self.smooth, quant_noise=self.quant_noise, q_prob=self.q_prob)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer=self._norm_layer, bit=self.bit, smooth=self.smooth, quant_noise=self.quant_noise, q_prob=self.q_prob))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation, norm_layer=self._norm_layer,
                                bit=self.bit, smooth=self.smooth, quant_noise=self.quant_noise, q_prob=self.q_prob))
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.training:
            if self.flag_ema_init:
                self.in_range[0], self.in_range[1] = ema(x, self.in_range, self.smooth)
                if self.flag_fake_quantization:
                    s, z = calc_qparams(self.in_range[0], self.in_range[1], self.q_max)
                    x = fake_quantize(x.detach(), s, z, self.q_max)
            else:
                self.in_range[0] = torch.min(x).item()
                self.in_range[1] = torch.max(x).item()
                self.flag_ema_init = True

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

    def show_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.show_params()

    def start_fake_quantization(self):
        self.flag_fake_quantization = True
        self.first_conv.flag_fake_quantization = True
        for i in range(len(self.layer1)):
            self.layer1[i].set_block_fq_flag()
        for i in range(len(self.layer2)):
            self.layer2[i].set_block_fq_flag()
        for i in range(len(self.layer3)):
            self.layer3[i].set_block_fq_flag()
        for i in range(len(self.layer4)):
            self.layer4[i].set_block_fq_flag()
        self.fc.flag_fake_quantization = True

    def set_quantization_params(self):
        self.scale, self.zero_point = calc_qparams(self.in_range[0], self.in_range[1], self.q_max)
        prev_s, prev_z = self.first_conv.set_qparams(self.scale, self.zero_point)
        for i in range(len(self.layer1)):
            prev_s, prev_z = self.layer1[i].set_block_qparams(prev_s, prev_z)
        for i in range(len(self.layer2)):
            prev_s, prev_z = self.layer2[i].set_block_qparams(prev_s, prev_z)
        for i in range(len(self.layer3)):
            prev_s, prev_z = self.layer3[i].set_block_qparams(prev_s, prev_z)
        for i in range(len(self.layer4)):
            prev_s, prev_z = self.layer4[i].set_block_qparams(prev_s, prev_z)
        _, _ = self.fc.set_qparams(prev_s, prev_z)


class FusedResNet20(nn.Module):
    def __init__(self, block, layers, num_classes=10, bit=8, smooth=0.999, quant_noise=False, q_prob=0.1):
        super(FusedResNet20, self).__init__()
        self.bit = bit
        self.in_range = nn.Parameter(torch.zeros(2), requires_grad=False)
        self.smooth = smooth
        self.flag_ema_init = False
        self.flag_fake_quantization = False
        self.q_max = 2 ** self.bit - 1

        self._norm_layer = nn.BatchNorm2d
        self.inplanes = 16
        self.dilation = 1
        self.num_blocks = 3

        self.quant_noise = quant_noise
        self.q_prob = q_prob
        self.first_conv = FusedConv2d(3, 16, kernel_size=3, stride=1, padding=1,
                                      bias=False, norm_layer=self._norm_layer, activation=nn.ReLU6, bit=bit, smooth=smooth, quant_noise=self.quant_noise, q_prob=self.q_prob)

        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.fc = FusedLinear(64 * block.expansion, num_classes, bit=bit, smooth=smooth, quant_noise=self.quant_noise, q_prob=self.q_prob)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
                downsample = fused_conv1x1(self.inplanes, planes * block.expansion, stride, bias=False,
                                       norm_layer=self._norm_layer, bit=self.bit, smooth=self.smooth, quant_noise=self.quant_noise, q_prob=self.q_prob)
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, norm_layer=self._norm_layer,
                            bit=self.bit, smooth=self.smooth, quant_noise=self.quant_noise, q_prob=self.q_prob))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_layer=self._norm_layer, bit=self.bit, smooth=self.smooth, quant_noise=self.quant_noise, q_prob=self.q_prob))
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.training:
            if self.flag_ema_init:
                self.in_range[0], self.in_range[1] = ema(x, self.in_range, self.smooth)
                if self.flag_fake_quantization:
                    s, z = calc_qparams(self.in_range[0], self.in_range[1], self.q_max)
                    x = fake_quantize(x.detach(), s, z, self.q_max)
            else:
                self.in_range[0] = torch.min(x).item()
                self.in_range[1] = torch.max(x).item()
                self.flag_ema_init = True

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

    def start_fake_quantization(self):
        self.flag_fake_quantization = True
        self.first_conv.flag_fake_quantization = True
        for i in range(len(self.layer1)):
            self.layer1[i].set_block_fq_flag()
        for i in range(len(self.layer2)):
            self.layer2[i].set_block_fq_flag()
        for i in range(len(self.layer3)):
            self.layer3[i].set_block_fq_flag()
        self.fc.flag_fake_quantization = True

    def set_quantization_params(self):
        self.scale, self.zero_point = calc_qparams(self.in_range[0], self.in_range[1], self.q_max)
        prev_s, prev_z = self.first_conv.set_qparams(self.scale, self.zero_point)
        for i in range(len(self.layer1)):
            prev_s, prev_z = self.layer1[i].set_block_qparams(prev_s, prev_z)
        for i in range(len(self.layer2)):
            prev_s, prev_z = self.layer2[i].set_block_qparams(prev_s, prev_z)
        for i in range(len(self.layer3)):
            prev_s, prev_z = self.layer3[i].set_block_qparams(prev_s, prev_z)
        _, _ = self.fc.set_qparams(prev_s, prev_z)


def fused_resnet18(bit=8, smooth=0.999, num_classes=1000, quant_noise=False, q_prob=0.1, **kwargs):
    return FusedResNet(FusedBasicBlock, [2, 2, 2, 2], bit=bit, smooth=smooth, num_classes=num_classes, quant_noise=quant_noise, q_prob=q_prob, **kwargs)


def fused_resnet20(bit=8, smooth=0.999, quant_noise=False, q_prob=0.1):
    return FusedResNet20(FusedBasicBlock, [3, 3, 3], bit=bit, smooth=smooth, quant_noise=quant_noise, q_prob=q_prob)


def set_fused_resnet(fused, pre):
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
    block[0].downsample = copy_from_pretrained(block[0].downsample, pre.layer2[0].downsample[0], pre.layer2[0].downsample[1])
    for i in range(len(block)):
        block[i].conv1 = copy_from_pretrained(block[i].conv1, pre.layer2[i].conv1, pre.layer2[i].bn1)
        block[i].conv2 = copy_from_pretrained(block[i].conv2, pre.layer2[i].conv2, pre.layer2[i].bn2)

    # Block 3
    block = fused.layer3
    block[0].downsample = copy_from_pretrained(block[0].downsample, pre.layer3[0].downsample[0], pre.layer3[0].downsample[1])
    for i in range(len(block)):
        block[i].conv1 = copy_from_pretrained(block[i].conv1, pre.layer3[i].conv1, pre.layer3[i].bn1)
        block[i].conv2 = copy_from_pretrained(block[i].conv2, pre.layer3[i].conv2, pre.layer3[i].bn2)

    # Block 4
    if fused.num_blocks == 4:
        block = fused.layer4
        block[0].downsample = copy_from_pretrained(block[0].downsample, pre.layer4[0].downsample[0], pre.layer4[0].downsample[1])
        for i in range(len(block)):
            block[i].conv1 = copy_from_pretrained(block[i].conv1, pre.layer4[i].conv1, pre.layer4[i].bn1)
            block[i].conv2 = copy_from_pretrained(block[i].conv2, pre.layer4[i].conv2, pre.layer4[i].bn2)

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

    # Block 2
    fp_block = model.layer2
    fp_block[0].downsample.fold_conv_and_bn()
    for i in range(len(fp_block)):
        fp_block[i].conv1.fold_conv_and_bn()
        fp_block[i].conv2.fold_conv_and_bn()

    # Block 3
    fp_block = model.layer3
    fp_block[0].downsample.fold_conv_and_bn()
    for i in range(len(fp_block)):
        fp_block[i].conv1.fold_conv_and_bn()
        fp_block[i].conv2.fold_conv_and_bn()

    # Block 4
    if model.num_blocks == 4:
        fp_block = model.layer4
        fp_block[0].downsample.fold_conv_and_bn()
        for i in range(len(fp_block)):
            fp_block[i].conv1.fold_conv_and_bn()
            fp_block[i].conv2.fold_conv_and_bn()
    return model

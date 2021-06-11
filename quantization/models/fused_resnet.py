import torch
import torch.nn as nn

from ..layers.fused_conv import *
from ..layers.fused_linear import *

# import pandas as pd


def fused_conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1, bit=32, smooth=0.995, bn=False, relu=True):
    """3x3 convolution with padding"""
    return FusedConv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                       padding=dilation, groups=groups, dilation=dilation, bit=bit, smooth=smooth, bn=bn, relu=relu)


def fused_conv1x1(in_planes, out_planes, stride=1, bit=32, smooth=0.995, bn=False, relu=False):
    """1x1 convolution"""
    return FusedConv2d(in_planes, out_planes, kernel_size=1, stride=stride, bit=bit, smooth=smooth, bn=bn, relu=relu)


class FusedBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1, norm_layer=None, bit=32, smooth=0.995):
        super(FusedBasicBlock, self).__init__()
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.downsample = downsample
        self.stride = stride
        self.act_range = np.zeros(2, dtype=np.float32)
        self.bit = bit
        self.q_max = 2 ** self.bit - 1
        self.ema_init = False
        # self.ema_init = 5000
        self.smooth = smooth

        self.conv1 = fused_conv3x3(inplanes, planes, stride, bit=bit, smooth=smooth, bn=True, relu=True)
        self.conv2 = fused_conv3x3(planes, planes, bit=bit, smooth=smooth, bn=True, relu=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        # print(pd.Series(out.cpu().detach().numpy().flatten()).describe())

        if self.bit == 32 or not self.training:
            return out

        #self.ema(out)
        #if not self.ema_init:
        if self.ema_init:
            self.ema(out)
            out = self.fake_quantize_activation(out)
        else:
            self.act_range[0] = torch.min(out).item()
            self.act_range[1] = torch.max(out).item()
            self.ema_init = True
            #self.ema_init -= 1
        return out

    def ema(self, x):
        _min = torch.min(x).item()
        _max = torch.max(x).item()
        self.act_range[0] = self.act_range[0] * self.smooth + _min * (1 - self.smooth)
        self.act_range[1] = self.act_range[1] * self.smooth + _max * (1 - self.smooth)

    def get_activation_qparams(self):
        return calc_qprams(self.act_range[0], self.act_range[1], self.q_max)

    def fake_quantize_activation(self, x):
        s, z = self.get_activation_qparams()
        x = torch.round(x.div(s).add(z)).sub(z).mul(s)
        return x


class FusedResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, bit=32, smooth=0.995):
        super(FusedResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.bit = bit
        self.in_range = np.zeros(2, dtype=np.float32)
        self.ema_init = False
        #self.ema_init = 5000
        self.smooth = smooth
        self.q_max = 2 ** self.bit - 1

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
        self.first_conv = FusedConv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bit=self.bit, bn=True, relu=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = FusedLinear(512 * block.expansion, num_classes, smooth=self.smooth, bit=self.bit, relu=False)

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
                # if isinstance(m, Bottleneck):
                #     nn.init.constant_(m.bn3.weight, 0)
                # elif isinstance(m, FusedBasicBlock):
                if isinstance(m, FusedBasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        # Planes : n_channel_output
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = fused_conv1x1(self.inplanes, planes * block.expansion, stride, bit=self.bit, smooth=self.smooth, relu=False)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, bit=self.bit, smooth=self.smooth))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, bit=self.bit, smooth=self.smooth))
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.bit != 32 and self.training:
            #self.ema(x)
            #if not self.ema_init:
            if self.ema_init:
                self.ema(x)
                x = self.fake_quantize_input(x)
            else:
                self.in_range[0] = torch.min(x).item()
                self.in_range[1] = torch.max(x).item()
                self.ema_init = True
                #self.ema_init -= 1

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

    def ema(self, x):
        _min = torch.min(x).item()
        _max = torch.max(x).item()
        self.in_range[0] = self.in_range[0] * self.smooth + _min * (1 - self.smooth)
        self.in_range[1] = self.in_range[1] * self.smooth + _max * (1 - self.smooth)

    def get_input_qparams(self):
        return calc_qprams(self.in_range[0], self.in_range[1], self.q_max)

    def fake_quantize_input(self, x):
        s, z = self.get_input_qparams()
        x = torch.round(x.div(s).add(z)).sub(z).mul(s)
        return x


def fused_resnet(block, layers, bit=32, num_classes=1000, **kwargs):
    model = FusedResNet(block, layers, bit=bit, num_classes=num_classes, **kwargs)
    return model


def set_fused_resnet18_params(fused, pre):
    """
        Copy pre model's params & set fused layers.
        Use fused architecture, but not really fused (use CONV & BN seperately)
    """
    # First layer
    fused.first_conv.copy_from_pretrained(pre.conv1, pre.bn1)

    # Block 1
    fused.layer1[0].conv1.copy_from_pretrained(pre.layer1[0].conv1, pre.layer1[0].bn1)
    fused.layer1[0].conv2.copy_from_pretrained(pre.layer1[0].conv2, pre.layer1[0].bn2)
    fused.layer1[1].conv1.copy_from_pretrained(pre.layer1[1].conv1, pre.layer1[1].bn1)
    fused.layer1[1].conv2.copy_from_pretrained(pre.layer1[1].conv2, pre.layer1[1].bn2)

    # Block 2
    fused.layer2[0].conv1.copy_from_pretrained(pre.layer2[0].conv1, pre.layer2[0].bn1)
    fused.layer2[0].conv2.copy_from_pretrained(pre.layer2[0].conv2, pre.layer2[0].bn2)
    fused.layer2[0].downsample.copy_from_pretrained(pre.layer2[0].downsample[0], pre.layer2[0].downsample[1])
    fused.layer2[1].conv1.copy_from_pretrained(pre.layer2[1].conv1, pre.layer2[1].bn1)
    fused.layer2[1].conv2.copy_from_pretrained(pre.layer2[1].conv2, pre.layer2[1].bn2)

    # Block 3
    fused.layer3[0].conv1.copy_from_pretrained(pre.layer3[0].conv1, pre.layer3[0].bn1)
    fused.layer3[0].conv2.copy_from_pretrained(pre.layer3[0].conv2, pre.layer3[0].bn2)
    fused.layer3[0].downsample.copy_from_pretrained(pre.layer3[0].downsample[0], pre.layer3[0].downsample[1])
    fused.layer3[1].conv1.copy_from_pretrained(pre.layer3[1].conv1, pre.layer3[1].bn1)
    fused.layer3[1].conv2.copy_from_pretrained(pre.layer3[1].conv2, pre.layer3[1].bn2)

    # Block 4
    fused.layer4[0].conv1.copy_from_pretrained(pre.layer4[0].conv1, pre.layer4[0].bn1)
    fused.layer4[0].conv2.copy_from_pretrained(pre.layer4[0].conv2, pre.layer4[0].bn2)
    fused.layer4[0].downsample.copy_from_pretrained(pre.layer4[0].downsample[0], pre.layer4[0].downsample[1])
    fused.layer4[1].conv1.copy_from_pretrained(pre.layer4[1].conv1, pre.layer4[1].bn1)
    fused.layer4[1].conv2.copy_from_pretrained(pre.layer4[1].conv2, pre.layer4[1].bn2)

    # Classifier
    fused.fc.fc.weight = torch.nn.Parameter(pre.fc.weight)
    fused.fc.fc.bias = torch.nn.Parameter(pre.fc.bias)
    return fused


def create_fused_resnet18(bit=32, num_classes=1000, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>'_
    """
    model = fused_resnet(FusedBasicBlock, [2, 2, 2, 2], bit=bit, num_classes=num_classes, **kwargs)
    return model


def fuse_resnet18(model):
    # First layer
    model.first_conv.fuse_conv_and_bn()

    # Block 1
    model.layer1[0].conv1.fuse_conv_and_bn()
    model.layer1[0].conv2.fuse_conv_and_bn()
    model.layer1[1].conv1.fuse_conv_and_bn()
    model.layer1[1].conv2.fuse_conv_and_bn()

    # Block 2
    model.layer2[0].conv1.fuse_conv_and_bn()
    model.layer2[0].conv2.fuse_conv_and_bn()
    model.layer2[0].downsample.fuse_conv_and_bn()
    model.layer2[1].conv1.fuse_conv_and_bn()
    model.layer2[1].conv2.fuse_conv_and_bn()

    # Block 3
    model.layer3[0].conv1.fuse_conv_and_bn()
    model.layer3[0].conv2.fuse_conv_and_bn()
    model.layer3[0].downsample.fuse_conv_and_bn()
    model.layer3[1].conv1.fuse_conv_and_bn()
    model.layer3[1].conv2.fuse_conv_and_bn()

    # Block 4
    model.layer4[0].conv1.fuse_conv_and_bn()
    model.layer4[0].conv2.fuse_conv_and_bn()
    model.layer4[0].downsample.fuse_conv_and_bn()
    model.layer4[1].conv1.fuse_conv_and_bn()
    model.layer4[1].conv2.fuse_conv_and_bn()
    return model

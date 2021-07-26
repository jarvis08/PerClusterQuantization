import torch
import torch.nn as nn

from .layers.conv2d import *
from .layers.linear import *
from .quantization_utils import *


def pcq_conv3x3(in_planes, out_planes, stride=1, dilation=1, bias=False, norm_layer=None, activation=None,
                bit=32, smooth=0.995, num_clusters=10):
    """3x3 convolution with padding"""
    return PCQConv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, bias=bias, norm_layer=norm_layer, activation=activation,
                     bit=bit, smooth=smooth, num_clusters=num_clusters)


def pcq_conv1x1(in_planes, out_planes, stride=1, bias=False, norm_layer=None, activation=None,
                bit=32, smooth=0.995, num_clusters=10):
    """1x1 convolution"""
    return PCQConv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=bias, norm_layer=norm_layer,
                     activation=activation, bit=bit, smooth=smooth, num_clusters=num_clusters)


class PCQBasicBlock(nn.Module):
    expansion = 1
    batch_cluster = None

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1,
                 norm_layer=None, bit=32, smooth=0.995, num_clusters=10):
        super(PCQBasicBlock, self).__init__()
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
        self.act_range = nn.Parameter(torch.zeros(num_clusters, 2), requires_grad=False)
        self.flag_fake_quantization = False
        self.flag_ema_init = np.zeros(num_clusters, dtype=bool)
        self.smooth = smooth
        self.num_clusters = num_clusters

        self.conv1 = pcq_conv3x3(inplanes, planes, stride, norm_layer=self._norm_layer, activation=nn.ReLU,
                                 bit=bit, smooth=smooth, num_clusters=num_clusters)
        self.conv2 = pcq_conv3x3(planes, planes, norm_layer=self._norm_layer,
                                 bit=bit, smooth=smooth, num_clusters=num_clusters)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        if self.training:
            done = 0
            for i in range(self.batch_cluster.shape[0]):
                c = self.batch_cluster[i][0].item()
                n = self.batch_cluster[i][1].item()
                if self.flag_ema_init[c]:
                    self.act_range[c][0], self.act_range[c][1] = ema(out[done:done + n], self.act_range[c], self.smooth)
                    if self.flag_fake_quantization:
                        s, z = calc_qparams(self.act_range[c][0], self.act_range[c][1], self.q_max)
                        out[done:done + n] = fake_quantize(out[done:done + n], s, z, self.q_max)
                else:
                    self.act_range[c][0] = torch.min(out).item()
                    self.act_range[c][1] = torch.max(out).item()
                    self.flag_ema_init[c] = True
                done += n
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

        self.s3 = nn.Parameter(torch.zeros(self.num_clusters, dtype=torch.float32), requires_grad=False)
        self.z3 = nn.Parameter(torch.zeros(self.num_clusters, dtype=torch.int32), requires_grad=False)
        for c in range(self.num_clusters):
            self.s3[c], self.z3[c] = calc_qparams(self.act_range[c][0], self.act_range[c][1], self.q_max)
        return self.s3, self.z3


class PCQResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, groups=1, width_per_group=64,
                 replace_stride_with_dilation=None, norm_layer=None, bit=8, smooth=0.999, num_clusters=10):
        super(PCQResNet, self).__init__()
        self.bit = bit
        self.q_max = 2 ** self.bit - 1
        self.in_range = nn.Parameter(torch.zeros(num_clusters, 2), requires_grad=False)
        self.flag_ema_init = np.zeros(num_clusters, dtype=bool)
        self.flag_fake_quantization = False
        self.smooth = smooth

        self.num_clusters = num_clusters
        self.batch_cluster = None

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
        self.first_conv = PCQConv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                                    bias=False, norm_layer=self._norm_layer, activation=nn.ReLU,
                                    #bias=False, norm_layer=self._norm_layer, activation=nn.ReLU6,
                                    bit=bit, smooth=smooth, num_clusters=num_clusters)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = PCQLinear(512 * block.expansion, num_classes, bias=True,
                            bit=bit, smooth=smooth, num_clusters=num_clusters)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        # Planes : n_channel_output
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = pcq_conv1x1(self.inplanes, planes * block.expansion, stride,
                                     bias=False, norm_layer=self._norm_layer,
                                     bit=self.bit, smooth=self.smooth, num_clusters=self.num_clusters)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer=self._norm_layer,
                            bit=self.bit, smooth=self.smooth, num_clusters=self.num_clusters))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation, norm_layer=self._norm_layer,
                                bit=self.bit, smooth=self.smooth, num_clusters=self.num_clusters))
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.training:
            done = 0
            for i in range(self.batch_cluster.shape[0]):
                c = self.batch_cluster[i][0].item()
                n = self.batch_cluster[i][1].item()
                if self.flag_ema_init[c]:
                    self.in_range[c][0], self.in_range[c][1] = ema(x[done:done + n], self.in_range[c], self.smooth)
                    if self.flag_fake_quantization:
                        s, z = calc_qparams(self.in_range[c][0], self.in_range[c][1], self.q_max)
                        x[done:done + n] = fake_quantize(x[done:done + n], s, z)
                else:
                    self.in_range[c][0] = torch.min(x).item()
                    self.in_range[c][1] = torch.max(x).item()
                    self.flag_ema_init[c] = True
                done += n

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

    @classmethod
    def set_cluster_information_of_batch(cls, info):
        cls.batch_cluster = info
        PCQBasicBlock.batch_cluster = info
        # PCQBottleneck.batch_cluster = info
        PCQConv2d.batch_cluster = info
        PCQLinear.batch_cluster = info

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
        self.scale = nn.Parameter(torch.zeros(self.num_clusters, dtype=torch.float32), requires_grad=False)
        self.zero_point = nn.Parameter(torch.zeros(self.num_clusters, dtype=torch.int32), requires_grad=False)
        for c in range(self.num_clusters):
            self.scale[c], self.zero_point[c] = calc_qparams(self.in_range[c][0], self.in_range[c][1], self.q_max)
        prev_s, prev_z = self.first_conv.set_qparams(self.scale, self.zero_point)
        prev_s, prev_z = self.layer1[0].set_block_qparams(prev_s, prev_z)
        prev_s, prev_z = self.layer1[1].set_block_qparams(prev_s, prev_z)
        prev_s, prev_z = self.layer2[0].set_block_qparams(prev_s, prev_z)
        prev_s, prev_z = self.layer2[1].set_block_qparams(prev_s, prev_z)
        prev_s, prev_z = self.layer3[0].set_block_qparams(prev_s, prev_z)
        prev_s, prev_z = self.layer3[1].set_block_qparams(prev_s, prev_z)
        prev_s, prev_z = self.layer4[0].set_block_qparams(prev_s, prev_z)
        prev_s, prev_z = self.layer4[1].set_block_qparams(prev_s, prev_z)
        _, _ = self.fc.set_qparams(prev_s, prev_z)


class PCQResNet20(nn.Module):
    batch_cluster = None

    def __init__(self, block, layers, norm_layer=None, num_classes=10, bit=8, smooth=0.999, num_clusters=10):
        super(PCQResNet20, self).__init__()
        self.bit = bit
        self.q_max = 2 ** self.bit - 1
        self.in_range = nn.Parameter(torch.zeros(num_clusters, 2), requires_grad=False)
        self.flag_ema_init = np.zeros(num_clusters, dtype=bool)
        self.flag_fake_quantization = False
        self.smooth = smooth

        self.num_clusters = num_clusters

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 16
        self.dilation = 1
        self.num_blocks = 3

        self.first_conv = PCQConv2d(3, 16, kernel_size=3, stride=1, padding=1, norm_layer=self._norm_layer,
                                    activation=nn.ReLU6, bit=bit, smooth=smooth, num_clusters=num_clusters)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.fc = PCQLinear(64 * block.expansion, num_classes,
                            bit=self.bit, smooth=self.smooth, num_clusters=num_clusters)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = pcq_conv1x1(self.inplanes, planes * block.expansion, stride, norm_layer=self._norm_layer,
                                     bit=self.bit, smooth=self.smooth, num_clusters=self.num_clusters)
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, norm_layer=self._norm_layer,
                            bit=self.bit, smooth=self.smooth, num_clusters=self.num_clusters))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_layer=self._norm_layer,
                                bit=self.bit, smooth=self.smooth, num_clusters=self.num_clusters))
        return nn.Sequential(*layers)

    def forward(self, x):
        x_fq = None
        if self.training:
            x_fq = torch.zeros(x.shape).cuda()
            done = 0
            for i in range(self.batch_cluster.shape[0]):
                c = self.batch_cluster[i][0].item()
                n = self.batch_cluster[i][1].item()
                if self.flag_ema_init[c]:
                    self.in_range[c][0], self.in_range[c][1] = ema(x[done:done + n], self.in_range[c], self.smooth)
                    if self.flag_fake_quantization:
                        s, z = calc_qparams(self.in_range[c][0], self.in_range[c][1], self.q_max)
                        x_fq[done:done + n] = fake_quantize(x[done:done + n], s, z, self.q_max)
                        # x[done:done + n] = fake_quantize(x[done:done + n], s, z, self.q_max)
                else:
                    self.in_range[c][0] = torch.min(x).item()
                    self.in_range[c][1] = torch.max(x).item()
                    self.flag_ema_init[c] = True
                done += n
        if self.flag_fake_quantization and self.training:
            x = self.first_conv(x_fq)
        else:
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
        PCQBasicBlock.batch_cluster = info
        # PCQBottleneck.batch_cluster = info
        PCQConv2d.batch_cluster = info
        PCQLinear.batch_cluster = info

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
        self.scale = nn.Parameter(torch.zeros(self.num_clusters, dtype=torch.float32), requires_grad=False)
        self.zero_point = nn.Parameter(torch.zeros(self.num_clusters, dtype=torch.int32), requires_grad=False)
        for c in range(self.num_clusters):
            self.scale[c], self.zero_point[c] = calc_qparams(self.in_range[c][0], self.in_range[c][1], self.q_max)
        prev_s, prev_z = self.first_conv.set_qparams(self.scale, self.zero_point)
        prev_s, prev_z = self.layer1[0].set_block_qparams(prev_s, prev_z)
        prev_s, prev_z = self.layer1[1].set_block_qparams(prev_s, prev_z)
        prev_s, prev_z = self.layer1[2].set_block_qparams(prev_s, prev_z)
        prev_s, prev_z = self.layer2[0].set_block_qparams(prev_s, prev_z)
        prev_s, prev_z = self.layer2[1].set_block_qparams(prev_s, prev_z)
        prev_s, prev_z = self.layer2[2].set_block_qparams(prev_s, prev_z)
        prev_s, prev_z = self.layer3[0].set_block_qparams(prev_s, prev_z)
        prev_s, prev_z = self.layer3[1].set_block_qparams(prev_s, prev_z)
        prev_s, prev_z = self.layer3[2].set_block_qparams(prev_s, prev_z)
        _, _ = self.fc.set_qparams(prev_s, prev_z)


def pcq_resnet18(bit=8, smooth=0.999, num_clusters=10, **kwargs):
    return PCQResNet(PCQBasicBlock, [2, 2, 2, 2], bit=bit, smooth=smooth, num_clusters=num_clusters, **kwargs)


def pcq_resnet20(bit=8, smooth=0.999, num_clusters=10):
    return PCQResNet20(PCQBasicBlock, [3, 3, 3], bit=bit, smooth=smooth, num_clusters=num_clusters)

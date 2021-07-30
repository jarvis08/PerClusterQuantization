from operator import itemgetter 
import torch
import torch.nn as nn

from .layers.conv2d import *
from .layers.linear import *
from .quantization_utils import *


def pcq_conv3x3(in_planes, out_planes, stride=1, dilation=1, bias=False, norm_layer=None, activation=None, arg_dict=None):
    """3x3 convolution with padding"""
    return PCQConv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, bias=bias, norm_layer=norm_layer, activation=activation, arg_dict=arg_dict)


def pcq_conv1x1(in_planes, out_planes, stride=1, bias=False, norm_layer=None, activation=None, arg_dict=None):
    """1x1 convolution"""
    return PCQConv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=bias,
                     norm_layer=norm_layer, activation=activation, arg_dict=arg_dict)


class PCQBasicBlock(nn.Module):
    expansion = 1
    batch_cluster = None

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1,
                 norm_layer=None, arg_dict=None):
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

        self.arg_dict = arg_dict
        self.bit, self.smooth, self.num_clusters, self.use_ste, self.quant_noise, self.qn_prob\
            = itemgetter('bit', 'smooth', 'cluster', 'ste', 'quant_noise', 'qn_prob')(arg_dict)
        self.q_max = 2 ** self.bit - 1
        self.act_range = nn.Parameter(torch.zeros(self.num_clusters, 2), requires_grad=False)

        self.flag_ema_init = np.zeros(self.num_clusters, dtype=bool)
        self.flag_fake_quantization = False

        self.conv1 = pcq_conv3x3(inplanes, planes, stride, norm_layer=self._norm_layer,
                                 activation=nn.ReLU, arg_dict=arg_dict)
        self.conv2 = pcq_conv3x3(planes, planes, norm_layer=self._norm_layer, arg_dict=arg_dict)
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
            return out

        if self.flag_fake_quantization and self.use_ste:
            _out = torch.zeros(out.shape).cuda()
        else:
            _out = out

        done = 0
        for i in range(PCQBasicBlock.batch_cluster.shape[0]):
            c = PCQBasicBlock.batch_cluster[i][0].item()
            n = PCQBasicBlock.batch_cluster[i][1].item()
            if self.flag_ema_init[c]:
                self.act_range[c][0], self.act_range[c][1] = ema(out[done:done + n], self.act_range[c], self.smooth)
                if self.flag_fake_quantization:
                    s, z = calc_qparams(self.act_range[c][0], self.act_range[c][1], self.q_max)
                    _out[done:done + n] = fake_quantize(out[done:done + n], s, z, self.q_max, self.use_ste)
            else:
                self.act_range[c][0] = torch.min(out).item()
                self.act_range[c][1] = torch.max(out).item()
                self.flag_ema_init[c] = True
            done += n
        return _out

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



class PCQBottleneck(nn.Module):
    expansion: int = 4
    batch_cluster = None

    def __init__(
            self, inplanes: int, planes: int, stride: int = 1, downsample = None,
            groups: int = 1, base_width: int = 64,dilation: int = 1,
            norm_layer = None, arg_dict = None
    ) -> None:
        super(PCQBottleneck, self).__init__()

        self.downsample = downsample
        self.stride = stride
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.arg_dict = arg_dict

        self.bit, self.smooth, self.num_clusters, self.use_ste, self.quant_noise, self.qn_prob \
            = itemgetter('bit', 'smooth', 'cluster', 'ste', 'quant_noise', 'qn_prob')(arg_dict)
        self.q_max = 2 ** self.bit - 1
        self.act_range = nn.Parameter(torch.zeros(self.num_clusters, 2), requires_grad=False)

        self.flag_ema_init = np.zeros(self.num_clusters, dtype=bool)
        self.flag_fake_quantization = False

        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = pcq_conv1x1(in_planes=inplanes, out_planes=width,
                                 norm_layer=self._norm_layer, activation=nn.ReLU, arg_dict=self.arg_dict)
        self.conv2 = pcq_conv3x3(in_planes=width, out_planes=width, stride=stride, dilation=dilation,
                                 norm_layer=self._norm_layer, activation=nn.ReLU, arg_dict=self.arg_dict)
        self.conv3 = pcq_conv1x1(in_planes=width, out_planes=planes * self.expansion,
                                 norm_layer=self._norm_layer, arg_dict=self.arg_dict)
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

        if self.flag_fake_quantization and self.use_ste:
            _out = torch.zeros(out.shape).cuda()
        else:
            _out = out

        done = 0
        for i in range(PCQBottleneck.batch_cluster.shape[0]):
            c = PCQBottleneck.batch_cluster[i][0].item()
            n = PCQBottleneck.batch_cluster[i][1].item()
            if self.flag_ema_init[c]:
                self.act_range[c][0], self.act_range[c][1] = ema(out[done:done + n], self.act_range[c], self.smooth)
                if self.flag_fake_quantization:
                    s, z = calc_qparams(self.act_range[c][0], self.act_range[c][1], self.q_max)
                    _out[done:done + n] = fake_quantize(out[done:done + n], s, z, self.q_max, self.use_ste)
            else:
                self.act_range[c][0] = torch.min(out).item()
                self.act_range[c][1] = torch.max(out).item()
                self.flag_ema_init[c] = True
            done += n
        return _out

class PCQResNet(nn.Module):
    batch_cluster = None

    def __init__(self, block, layers, num_classes=1000, groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, arg_dict=None):
        super(PCQResNet, self).__init__()
        self.arg_dict = arg_dict
        self.bit, self.smooth, self.num_clusters, self.quant_noise, self.qn_prob\
            = itemgetter('bit', 'smooth', 'cluster', 'quant_noise', 'qn_prob')(arg_dict)
        self.q_max = 2 ** self.bit - 1
        self.in_range = nn.Parameter(torch.zeros(self.num_clusters, 2), requires_grad=False)
        self.flag_ema_init = np.zeros(self.num_clusters, dtype=bool)
        self.flag_fake_quantization = False

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
        self.first_conv = PCQConv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False,
                                    norm_layer=self._norm_layer, activation=nn.ReLU, arg_dict=arg_dict)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = PCQLinear(512 * block.expansion, num_classes, bias=True, arg_dict=arg_dict)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        # Planes : n_channel_output
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = pcq_conv1x1(self.inplanes, planes * block.expansion, stride,
                                     bias=False, norm_layer=self._norm_layer, arg_dict=self.arg_dict)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer=self._norm_layer, arg_dict=self.arg_dict))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups, base_width=self.base_width,
                                dilation=self.dilation, norm_layer=self._norm_layer, arg_dict=self.arg_dict))
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.training:
            done = 0
            for i in range(PCQResNet.batch_cluster.shape[0]):
                c = PCQResNet.batch_cluster[i][0].item()
                n = PCQResNet.batch_cluster[i][1].item()
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

    def __init__(self, block, layers, arg_dict, num_classes=10, norm_layer=None):
        super(PCQResNet20, self).__init__()
        self.arg_dict = arg_dict
        self.bit, self.smooth, self.num_clusters, self.quant_noise, self.qn_prob\
            = itemgetter('bit', 'smooth', 'cluster', 'quant_noise', 'qn_prob')(arg_dict)
        self.q_max = 2 ** self.bit - 1
        self.in_range = nn.Parameter(torch.zeros(self.num_clusters, 2), requires_grad=False)
        self.flag_ema_init = np.zeros(self.num_clusters, dtype=bool)
        self.flag_fake_quantization = False

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 16
        self.dilation = 1
        self.num_blocks = 3

        self.first_conv = PCQConv2d(3, 16, kernel_size=3, stride=1, padding=1,
                                    norm_layer=self._norm_layer, activation=nn.ReLU, arg_dict=self.arg_dict)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.fc = PCQLinear(64 * block.expansion, num_classes, arg_dict=self.arg_dict)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = pcq_conv1x1(self.inplanes, planes * block.expansion, stride,
                                     norm_layer=self._norm_layer, arg_dict=self.arg_dict)
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, norm_layer=self._norm_layer, arg_dict=self.arg_dict))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_layer=self._norm_layer, arg_dict=self.arg_dict))
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.training:
            done = 0
            for i in range(PCQResNet20.batch_cluster.shape[0]):
                c = PCQResNet20.batch_cluster[i][0].item()
                n = PCQResNet20.batch_cluster[i][1].item()
                if self.flag_ema_init[c]:
                    self.in_range[c][0], self.in_range[c][1] = ema(x[done:done + n], self.in_range[c], self.smooth)
                    if self.flag_fake_quantization:
                        s, z = calc_qparams(self.in_range[c][0], self.in_range[c][1], self.q_max)
                        x[done:done + n] = fake_quantize(x[done:done + n], s, z, self.q_max)
                else:
                    self.in_range[c][0] = torch.min(x).item()
                    self.in_range[c][1] = torch.max(x).item()
                    self.flag_ema_init[c] = True
                done += n

        x = self.first_conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    @classmethod
    def set_cluster_information_of_batch(cls, info):
        cls.batch_cluster = info
        PCQBasicBlock.batch_cluster = info
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


def pcq_resnet18(arg_dict, **kwargs):
    return PCQResNet(PCQBasicBlock, [2, 2, 2, 2], arg_dict, **kwargs)

def pcq_resnet50(arg_dict, **kwargs):
    return PCQResNet(PCQBottleneck, [3, 4, 6, 3], arg_dict, **kwargs)


def pcq_resnet20(arg_dict):
    return PCQResNet20(PCQBasicBlock, [3, 3, 3], arg_dict)

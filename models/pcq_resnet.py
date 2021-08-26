from operator import itemgetter 
import torch
import torch.nn as nn

from .layers.conv2d import *
from .layers.linear import *
from .layers.norm import *
from .quantization_utils import *


def pcq_conv3x3(in_planes, out_planes, stride=1, dilation=1, bias=False, activation=None, act_qmax=None, arg_dict=None):
    return PCQConv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation,\
                     bias=bias, activation=activation, act_qmax=act_qmax, arg_dict=arg_dict)


def pcq_conv1x1(in_planes, out_planes, stride=1, bias=False, activation=None, act_qmax=None, arg_dict=None):
    return PCQConv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=bias,
                     activation=activation, act_qmax=act_qmax, arg_dict=arg_dict)


class PCQBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1, arg_dict=None):
        super(PCQBasicBlock, self).__init__()
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.downsample = downsample
        self.stride = stride

        self.bit, self.smooth, self.num_clusters, self.runtime_helper, self.use_ste, self.quant_noise, self.qn_prob\
            = itemgetter('bit', 'smooth', 'cluster', 'runtime_helper', 'ste', 'quant_noise', 'qn_prob')(arg_dict)
        self.q_max = 2 ** self.bit - 1
        activation_qmax = 2 ** 32 - 1
        self.act_range = nn.Parameter(torch.zeros(self.num_clusters, 2), requires_grad=False)
        self.apply_ema = np.zeros(self.num_clusters, dtype=bool)

        if self.downsample is not None:
            self.bn_down = PCQBnReLU(planes, arg_dict=arg_dict)
        self.conv1 = pcq_conv3x3(inplanes, planes, stride, arg_dict=arg_dict, act_qmax=activation_qmax)
        self.bn1 = PCQBnReLU(planes, activation=nn.ReLU, arg_dict=arg_dict)
        self.conv2 = pcq_conv3x3(planes, planes, arg_dict=arg_dict, act_qmax=activation_qmax)
        self.bn2 = PCQBnReLU(planes, arg_dict=arg_dict)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out, self.conv1.act_range)

        out = self.conv2(out)
        out = self.bn2(out, self.conv2.act_range)

        if self.downsample is not None:
            identity = self.downsample(x)
            identity = self.bn_down(identity, self.downsample.act_range)

        out += identity
        out = self.relu(out)

        if not self.runtime_helper.pcq_initialized:
            done = 0
            for c in range(self.num_clusters):
                if self.apply_ema[c]:
                    self.act_range[c][0], self.act_range[c][1] = ema(out[done:done + 8], self.act_range[c], self.smooth)
                else:
                    self.act_range[c][0] = torch.min(out[done:done + 8]).item()
                    self.act_range[c][1] = torch.max(out[done:done + 8]).item()
                    self.apply_ema[c] = True
                done += 8
            return out

        if not self.training:
            return out

        if self.runtime_helper.apply_fake_quantization:
            _out = torch.zeros(out.shape).cuda()
        else:
            _out = out

        done = 0
        for i in range(self.runtime_helper.batch_cluster.shape[0]):
            c = self.runtime_helper.batch_cluster[i][0]
            n = self.runtime_helper.batch_cluster[i][1]
            if self.apply_ema[c]:
                self.act_range[c][0], self.act_range[c][1] = ema(out[done:done + n], self.act_range[c], self.smooth)
                if self.runtime_helper.apply_fake_quantization:
                    s, z = calc_qparams(self.act_range[c][0], self.act_range[c][1], self.q_max)
                    _out[done:done + n] = fake_quantize(out[done:done + n], s, z, self.q_max, self.use_ste)
            else:
                self.act_range[c][0] = torch.min(out[done:done + n]).item()
                self.act_range[c][1] = torch.max(out[done:done + n]).item()
                self.apply_ema[c] = True
            done += n
        return _out

    def set_block_qparams(self, s1, z1):
        if self.downsample:
            prev_s, prev_z = self.downsample.set_qparams(s1, z1)
            self.bn_down.set_qparams(prev_s, prev_z)
        prev_s, prev_z = self.conv1.set_qparams(s1, z1)
        prev_s, prev_z = self.bn1.set_qparams(prev_s, prev_z)
        prev_s, prev_z = self.conv2.set_qparams(prev_s, prev_z)
        _, _ = self.bn2.set_qparams(prev_s, prev_z)

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
            groups: int = 1, base_width: int = 64,dilation: int = 1, arg_dict = None
    ) -> None:
        super(PCQBottleneck, self).__init__()

        self.downsample = downsample
        self.stride = stride

        self.arg_dict = arg_dict

        self.bit, self.smooth, self.num_clusters, self.runtime_helper, self.use_ste, self.quant_noise, self.qn_prob \
            = itemgetter('bit', 'smooth', 'cluster', 'runtime_helper', 'ste', 'quant_noise', 'qn_prob')(arg_dict)
        self.q_max = 2 ** self.bit - 1
        self.activation_qmax = 2 ** 32 - 1
        self.act_range = nn.Parameter(torch.zeros(self.num_clusters, 2), requires_grad=False)
        self.apply_ema = np.zeros(self.num_clusters, dtype=bool)

        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = pcq_conv1x1(in_planes=inplanes, out_planes=width, arg_dict=self.arg_dict, act_qmax=self.activation_qmax)
        self.bn1 = PCQBnReLU(width, nn.ReLU, arg_dict)
        self.conv2 = pcq_conv3x3(in_planes=width, out_planes=width, stride=stride, dilation=dilation,
                                 arg_dict=self.arg_dict, act_qmax=self.activation_qmax)
        self.bn2 = PCQBnReLU(width, nn.ReLU, arg_dict)
        self.conv3 = pcq_conv1x1(in_planes=width, out_planes=planes * self.expansion, arg_dict=self.arg_dict, act_qmax=self.activation_qmax)
        self.bn3 = PCQBnReLU(planes * self.expansion, arg_dict=self.arg_dict)
        self.relu = nn.ReLU(inplace=True)
        if self.downsample is not None:
            self.bn_down = PCQBnReLU(planes * self.expansion, arg_dict=arg_dict)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.b1(out, self.conv1.act_range)
        out = self.conv2(out)
        out = self.bn2(out, self.conv2.act_range)
        out = self.conv3(out)
        out = self.bn3(out, self.conv3.act_range)

        if self.downsample is not None:
            identity = self.downsample(x)
            identity = self.bn_down(identity, self.downsample.act_range)

        out += identity
        out = self.relu(out)

        if not self.runtime_helper.pcq_initialized:
            done = 0
            for c in range(self.num_clusters):
                if self.apply_ema[c]:
                    self.act_range[c][0], self.act_range[c][1] = ema(out[done:done + 8], self.act_range[c], self.smooth)
                else:
                    self.act_range[c][0] = torch.min(out[done:done + 8]).item()
                    self.act_range[c][1] = torch.max(out[done:done + 8]).item()
                    self.apply_ema[c] = True
                done += 8
            return out

        if not self.training:
            return out

        if self.runtime_helper.apply_fake_quantization and self.use_ste:
            _out = torch.zeros(out.shape).cuda()
        else:
            _out = out

        done = 0
        for i in range(self.runtime_helper.batch_cluster.shape[0]):
            c = self.runtime_helper.batch_cluster[i][0].item()
            n = self.runtime_helper.batch_cluster[i][1].item()
            if self.apply_ema[c]:
                self.act_range[c][0], self.act_range[c][1] = ema(out[done:done + n], self.act_range[c], self.smooth)
                if self.runtime_helper.apply_fake_quantization:
                    s, z = calc_qparams(self.act_range[c][0], self.act_range[c][1], self.q_max)
                    _out[done:done + n] = fake_quantize(out[done:done + n], s, z, self.q_max, self.use_ste)
            else:
                self.act_range[c][0] = torch.min(out[done:done + n]).item()
                self.act_range[c][1] = torch.max(out[done:done + n]).item()
                self.apply_ema[c] = True
            done += n
        return _out

    def set_block_qparams(self, s1, z1):
        if self.downsample:
            prev_s, prev_z = self.downsample.set_qparams(s1, z1)
            self.bn_down.set_qparams(prev_s, prev_z)
        prev_s, prev_z = self.conv1.set_qparams(s1, z1)
        prev_s, prev_z = self.bn1.set_qparams(prev_s, prev_z)
        prev_s, prev_z = self.conv2.set_qparams(prev_s, prev_z)
        prev_s, prev_z = self.bn2.set_qparams(prev_s, prev_z)
        prev_s, prev_z = self.conv3.set_qparams(prev_s, prev_z)
        _, _ = self.bn3.set_qparams(prev_s, prev_z)

        self.s3 = nn.Parameter(torch.zeros(self.num_clusters, dtype=torch.float32), requires_grad=False)
        self.z3 = nn.Parameter(torch.zeros(self.num_clusters, dtype=torch.int32), requires_grad=False)
        for c in range(self.num_clusters):
            self.s3[c], self.z3[c] = calc_qparams(self.act_range[c][0], self.act_range[c][1], self.q_max)
        return self.s3, self.z3


class PCQResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 arg_dict=None):
        super(PCQResNet, self).__init__()
        self.arg_dict = arg_dict
        self.bit, self.smooth, self.num_clusters, self.runtime_helper, self.quant_noise, self.qn_prob\
            = itemgetter('bit', 'smooth', 'cluster', 'runtime_helper', 'quant_noise', 'qn_prob')(arg_dict)
        self.q_max = 2 ** self.bit - 1
        self.activation_qmax = 2 ** 32 - 1
        self.in_range = nn.Parameter(torch.zeros(self.num_clusters, 2), requires_grad=False)
        self.apply_ema = np.zeros(self.num_clusters, dtype=bool)

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
                                    arg_dict=arg_dict, act_qmax=self.activation_qmax)
        self.bn1 = PCQBnReLU(self.inplanes, nn.ReLU, arg_dict=self.arg_dict)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = PCQLinear(512 * block.expansion, num_classes, arg_dict=arg_dict)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        # Planes : n_channel_output
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = pcq_conv1x1(self.inplanes, planes * block.expansion, stride, arg_dict=self.arg_dict,
                                     act_qmax=self.activation_qmax)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, arg_dict=self.arg_dict))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups, base_width=self.base_width,
                                dilation=self.dilation, arg_dict=self.arg_dict))
        return nn.Sequential(*layers)

    def forward(self, x):
        if not self.runtime_helper.pcq_initialized:
            done = 0
            for c in range(self.num_clusters):
                if self.apply_ema[c]:
                    self.in_range[c][0], self.in_range[c][1] = ema(x[done:done + 8], self.in_range[c], self.smooth)
                else:
                    self.in_range[c][0] = torch.min(x[done:done + 8]).item()
                    self.in_range[c][1] = torch.max(x[done:done + 8]).item()
                    self.apply_ema[c] = True
                done += 8

        elif self.training:
            done = 0
            for i in range(self.runtime_helper.batch_cluster.shape[0]):
                c = self.runtime_helper.batch_cluster[i][0]
                n = self.runtime_helper.batch_cluster[i][1]
                if self.apply_ema[c]:
                    self.in_range[c][0], self.in_range[c][1] = ema(x[done:done + n], self.in_range[c], self.smooth)
                    if self.runtime_helper.apply_fake_quantization:
                        s, z = calc_qparams(self.in_range[c][0], self.in_range[c][1], self.q_max)
                        x[done:done + n] = fake_quantize(x[done:done + n], s, z, self.q_max)
                else:
                    self.in_range[c][0] = torch.min(x[done:done + n]).item()
                    self.in_range[c][1] = torch.max(x[done:done + n]).item()
                    self.apply_ema[c] = True
                done += n

        x = self.first_conv(x)
        x = self.bn1(x, self.first_conv.act_range)
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
        self.scale = nn.Parameter(torch.zeros(self.num_clusters, dtype=torch.float32), requires_grad=False)
        self.zero_point = nn.Parameter(torch.zeros(self.num_clusters, dtype=torch.int32), requires_grad=False)
        for c in range(self.num_clusters):
            self.scale[c], self.zero_point[c] = calc_qparams(self.in_range[c][0], self.in_range[c][1], self.q_max)
        prev_s, prev_z = self.first_conv.set_qparams(self.scale, self.zero_point)
        prev_s, prev_z = self.bn1.set_qparams(prev_s, prev_z)
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
    def __init__(self, block, layers, arg_dict, num_classes=10):
        super(PCQResNet20, self).__init__()
        self.arg_dict = arg_dict
        self.bit, self.smooth, self.num_clusters, self.runtime_helper, self.quant_noise, self.qn_prob\
            = itemgetter('bit', 'smooth', 'cluster', 'runtime_helper', 'quant_noise', 'qn_prob')(arg_dict)
        self.q_max = 2 ** self.bit - 1
        self.activation_qmax = 2 ** 16 - 1
        self.in_range = nn.Parameter(torch.zeros(self.num_clusters, 2), requires_grad=False)
        self.apply_ema = np.zeros(self.num_clusters, dtype=bool)

        self.inplanes = 16
        self.dilation = 1
        self.num_blocks = 3

        self.first_conv = PCQConv2d(3, 16, kernel_size=3, stride=1, padding=1, arg_dict=self.arg_dict, act_qmax=self.activation_qmax)
        self.bn1 = PCQBnReLU(16, activation=nn.ReLU, arg_dict=arg_dict)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.fc = PCQLinear(64 * block.expansion, num_classes, arg_dict=self.arg_dict)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = pcq_conv1x1(self.inplanes, planes * block.expansion, stride, arg_dict=self.arg_dict, act_qmax=self.activation_qmax)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, arg_dict=self.arg_dict))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, arg_dict=self.arg_dict))
        return nn.Sequential(*layers)

    def forward(self, x):
        if not self.runtime_helper.pcq_initialized:
            done = 0
            for c in range(self.num_clusters):
                if self.apply_ema[c]:
                    self.in_range[c][0], self.in_range[c][1] = ema(x[done:done + 8], self.in_range[c], self.smooth)
                else:
                    self.in_range[c][0] = torch.min(x[done:done + 8]).item()
                    self.in_range[c][1] = torch.max(x[done:done + 8]).item()
                    self.apply_ema[c] = True
                done += 8

        elif self.training:
            done = 0
            for i in range(self.runtime_helper.batch_cluster.shape[0]):
                c = self.runtime_helper.batch_cluster[i][0]
                n = self.runtime_helper.batch_cluster[i][1]
                if self.apply_ema[c]:
                    self.in_range[c][0], self.in_range[c][1] = ema(x[done:done + n], self.in_range[c], self.smooth)
                    if self.runtime_helper.apply_fake_quantization:
                        s, z = calc_qparams(self.in_range[c][0], self.in_range[c][1], self.q_max)
                        x[done:done + n] = fake_quantize(x[done:done + n], s, z, self.q_max)
                else:
                    self.in_range[c][0] = torch.min(x[done:done + n]).item()
                    self.in_range[c][1] = torch.max(x[done:done + n]).item()
                    self.apply_ema[c] = True
                done += n

        x = self.first_conv(x)
        x = self.bn1(x, self.first_conv.act_range)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def set_quantization_params(self):
        self.scale = nn.Parameter(torch.zeros(self.num_clusters, dtype=torch.float32), requires_grad=False)
        self.zero_point = nn.Parameter(torch.zeros(self.num_clusters, dtype=torch.int32), requires_grad=False)
        for c in range(self.num_clusters):
            self.scale[c], self.zero_point[c] = calc_qparams(self.in_range[c][0], self.in_range[c][1], self.q_max)
        prev_s, prev_z = self.first_conv.set_qparams(self.scale, self.zero_point)
        prev_s, prev_z = self.bn1.set_qparams(prev_s, prev_z)
        prev_s, prev_z = self.layer1[0].set_block_qparams(prev_s, prev_z)
        prev_s, prev_z = self.layer1[1].set_block_qparams(prev_s, prev_z)
        prev_s, prev_z = self.layer1[2].set_block_qparams(prev_s, prev_z)
        prev_s, prev_z = self.layer2[0].set_block_qparams(prev_s, prev_z)
        prev_s, prev_z = self.layer2[1].set_block_qparams(prev_s, prev_z)
        prev_s, prev_z = self.layer2[2].set_block_qparams(prev_s, prev_z)
        prev_s, prev_z = self.layer3[0].set_block_qparams(prev_s, prev_z)
        prev_s, prev_z = self.layer3[1].set_block_qparams(prev_s, prev_z)
        prev_s, prev_z = self.layer3[2].set_block_qparams(prev_s, prev_z)
        self.fc.set_qparams(prev_s, prev_z)


def pcq_resnet18(arg_dict, **kwargs):
    return PCQResNet(PCQBasicBlock, [2, 2, 2, 2], arg_dict, **kwargs)


def pcq_resnet50(arg_dict, **kwargs):
    return PCQResNet(PCQBottleneck, [3, 4, 6, 3], arg_dict, **kwargs)


def pcq_resnet20(arg_dict):
    return PCQResNet20(PCQBasicBlock, [3, 3, 3], arg_dict)


def set_pcq_resnet(fused, pre):
    """
        Copy from pre model's params to fused layers.
        Use fused architecture, but not really fused (use CONV & BN seperately)
    """
    n = fused.arg_dict['cluster']
    bn_momentum = fused.arg_dict['bn_momentum']
    # First layer
    fused.first_conv = copy_weight_from_pretrained(fused.first_conv, pre.conv1)
    for c in range(n):
        fused.bn1.norms[c] = copy_bn_from_pretrained(fused.bn1.norms[c], pre.bn1)
        fused.bn1.norms[c].bn.momentum = bn_momentum

    # Block 1
    block = fused.layer1
    if block[0].downsample is not None:
        block[0].downsample = copy_weight_from_pretrained(block[0].downsample, pre.layer1[0].downsample[0])
        for c in range(n):
            block[0].bn_down.norms[c] = copy_bn_from_pretrained(block[0].bn_down.norms[c], pre.layer1[0].downsample[1])
            block[0].bn_down.norms[c].bn.momentum = bn_momentum
    for i in range(len(block)):
        block[i].conv1 = copy_weight_from_pretrained(block[i].conv1, pre.layer1[i].conv1)
        block[i].conv2 = copy_weight_from_pretrained(block[i].conv2, pre.layer1[i].conv2)
        for c in range(n):
            block[i].bn1.norms[c] = copy_bn_from_pretrained(block[i].bn1.norms[c], pre.layer1[i].bn1)
            block[i].bn2.norms[c] = copy_bn_from_pretrained(block[i].bn2.norms[c], pre.layer1[i].bn2)
            block[i].bn1.norms[c].bn.momentum = bn_momentum
            block[i].bn2.norms[c].bn.momentum = bn_momentum
        if type(block[i]) == PCQBottleneck:
            block[i].conv3 = copy_weight_from_pretrained(block[i].conv3, pre.layer1[i].conv3)
            for c in range(n):
                block[i].bn3.norms[c] = copy_bn_from_pretrained(block[i].bn3.norms[c], pre.layer1[i].bn3)
                block[i].bn3.norms[c].bn.momentum = bn_momentum

    # Block 2
    block = fused.layer2
    block[0].downsample = copy_weight_from_pretrained(block[0].downsample, pre.layer2[0].downsample[0])
    for c in range(n):
        block[0].bn_down.norms[c] = copy_bn_from_pretrained(block[0].bn_down.norms[c], pre.layer2[0].downsample[1])
        block[0].bn_down.norms[c].bn.momentum = bn_momentum
    for i in range(len(block)):
        block[i].conv1 = copy_weight_from_pretrained(block[i].conv1, pre.layer2[i].conv1)
        block[i].conv2 = copy_weight_from_pretrained(block[i].conv2, pre.layer2[i].conv2)
        for c in range(n):
            block[i].bn1.norms[c] = copy_bn_from_pretrained(block[i].bn1.norms[c], pre.layer2[i].bn1)
            block[i].bn2.norms[c] = copy_bn_from_pretrained(block[i].bn2.norms[c], pre.layer2[i].bn2)
            block[i].bn1.norms[c].bn.momentum = bn_momentum
            block[i].bn2.norms[c].bn.momentum = bn_momentum
        if type(block[i]) == PCQBottleneck:
            block[i].conv3 = copy_weight_from_pretrained(block[i].conv3, pre.layer2[i].conv3)
            for c in range(n):
                block[i].bn3.norms[c] = copy_bn_from_pretrained(block[i].bn3.norms[c], pre.layer2[i].bn3)
                block[i].bn3.norms[c].bn.momentum = bn_momentum

    # Block 3
    block = fused.layer3
    block[0].downsample = copy_weight_from_pretrained(block[0].downsample, pre.layer3[0].downsample[0])
    for c in range(n):
        block[0].bn_down.norms[c] = copy_bn_from_pretrained(block[0].bn_down.norms[c], pre.layer3[0].downsample[1])
        block[0].bn_down.norms[c].bn.momentum = bn_momentum
    for i in range(len(block)):
        block[i].conv1 = copy_weight_from_pretrained(block[i].conv1, pre.layer3[i].conv1)
        block[i].conv2 = copy_weight_from_pretrained(block[i].conv2, pre.layer3[i].conv2)
        for c in range(n):
            block[i].bn1.norms[c] = copy_bn_from_pretrained(block[i].bn1.norms[c], pre.layer3[i].bn1)
            block[i].bn2.norms[c] = copy_bn_from_pretrained(block[i].bn2.norms[c], pre.layer3[i].bn2)
            block[i].bn1.norms[c].bn.momentum = bn_momentum
            block[i].bn2.norms[c].bn.momentum = bn_momentum
        if type(block[i]) == PCQBottleneck:
            block[i].conv3 = copy_weight_from_pretrained(block[i].conv3, pre.layer3[i].conv3)
            for c in range(n):
                block[i].bn3.norms[c] = copy_bn_from_pretrained(block[i].bn3.norms[c], pre.layer3[i].bn3)
                block[i].bn3.norms[c].bn.momentum = bn_momentum

    # Block 4
    if fused.num_blocks == 4:
        block = fused.layer4
        block[0].downsample = copy_weight_from_pretrained(block[0].downsample, pre.layer4[0].downsample[0])
        for c in range(n):
            block[0].bn_down.norms[c] = copy_bn_from_pretrained(block[0].bn_down.norms[c], pre.layer4[0].downsample[1])
            block[0].bn_down.norms[c].bn.momentum = bn_momentum
        for i in range(len(block)):
            block[i].conv1 = copy_weight_from_pretrained(block[i].conv1, pre.layer4[i].conv1)
            block[i].conv2 = copy_weight_from_pretrained(block[i].conv2, pre.layer4[i].conv2)
            block[i].conv3 = copy_weight_from_pretrained(block[i].conv3, pre.layer4[i].conv3)
            for c in range(n):
                block[i].bn1.norms[c] = copy_bn_from_pretrained(block[i].bn1.norms[c], pre.layer4[i].bn1)
                block[i].bn2.norms[c] = copy_bn_from_pretrained(block[i].bn2.norms[c], pre.layer4[i].bn2)
                block[i].bn3.norms[c] = copy_bn_from_pretrained(block[i].bn3.norms[c], pre.layer4[i].bn3)
                block[i].bn1.norms[c].bn.momentum = bn_momentum
                block[i].bn2.norms[c].bn.momentum = bn_momentum
                block[i].bn3.norms[c].bn.momentum = bn_momentum

    # Classifier
    fused.fc = copy_from_pretrained(fused.fc, pre.fc)
    return fused


def fold_resnet_bn(model):
    # First layer
    model.bn1.fold_bn()

    block = model.layer1  # Block 1
    if block[0].downsample is not None:
        block[0].bn_down.fold_bn()
    for i in range(len(block)):
        block[i].bn1.fold_bn()
        block[i].bn2.fold_bn()

    # Block 2
    block = model.layer2  # Block 2
    block[0].bn_down.fold_bn()
    for i in range(len(block)):
        block[i].bn1.fold_bn()
        block[i].bn2.fold_bn()

    block = model.layer3  # Block 3
    block[0].bn_down.fold_bn()
    for i in range(len(block)):
        block[i].bn1.fold_bn()
        block[i].bn2.fold_bn()

    if model.num_blocks == 4:  # Block 4
        block = model.layer4
        block[0].bn_down.fold_bn()
        for i in range(len(block)):
            block[i].bn1.fold_bn()
            block[i].bn2.fold_bn()
    return model


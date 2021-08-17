import torch
import torch.nn as nn

from .layers.conv2d import *
from .layers.linear import *
from .quant_noise import _quant_noise
from .quantization_utils import *


def fused_conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1, bias=False,
                  norm_layer=None, activation=None, arg_dict=None):
    """3x3 convolution with padding"""
    return FusedConv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                       padding=dilation, groups=groups, dilation=dilation, bias=bias,
                       norm_layer=norm_layer, activation=activation, arg_dict=arg_dict)


def fused_conv1x1(in_planes, out_planes, stride=1, bias=False, norm_layer=None, activation=None, arg_dict=None):
    """1x1 convolution"""
    return FusedConv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=bias,
                       norm_layer=norm_layer, activation=activation, arg_dict=arg_dict)


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
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.bit, self.smooth, self.runtime_helper, self.use_ste, self.quant_noise, self.qn_prob\
            = itemgetter('bit', 'smooth', 'runtime_helper', 'ste', 'quant_noise', 'qn_prob')(arg_dict)
        self.q_max = 2 ** self.bit - 1
        self.act_range = nn.Parameter(torch.zeros(2), requires_grad=False)

        self.apply_ema = False

        self.conv1 = fused_conv3x3(inplanes, planes, stride,
                                   norm_layer=self._norm_layer, activation=nn.ReLU, arg_dict=arg_dict)
        self.conv2 = fused_conv3x3(planes, planes, norm_layer=self._norm_layer, arg_dict=arg_dict)
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
        
        _out = out
        if self.apply_ema:
            self.act_range[0], self.act_range[1] = ema(out, self.act_range, self.smooth)
            if self.runtime_helper.apply_fake_quantization:
                s, z = calc_qparams(self.act_range[0], self.act_range[1], self.q_max)
                _out = fake_quantize(out, s, z, self.q_max, self.use_ste)
        else:
            self.act_range[0] = torch.min(out).item()
            self.act_range[1] = torch.max(out).item()
            self.apply_ema = True
        return _out

    def set_block_qparams(self, s1, z1):
        if self.downsample:
            self.downsample.set_qparams(s1, z1)
        prev_s, prev_z = self.conv1.set_qparams(s1, z1)
        _, _ = self.conv2.set_qparams(prev_s, prev_z)
        self.s3, self.z3 = calc_qparams(self.act_range[0], self.act_range[1], self.q_max)
        return self.s3, self.z3

# Bottleneck, layers >=50
class FusedBottleneck(nn.Module):
    expansion: int = 4
    def __init__(self, inplane:int, planes:int, stride:int=1, downsample=None,
                 groups: int = 1, base_width:int =64, dilation:int =1,
                 norm_layer=None, arg_dict=None) -> None:
        super(FusedBottleneck, self).__init__()

        self.downsample = downsample
        self.stride = stride
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.arg_dict = arg_dict
        self.bit, self.smooth, self.use_ste, self.quant_noise, self.qn_prob = \
            itemgetter('bit', 'smooth', 'ste', 'quant_noise', 'qn_prob')(arg_dict)

        self.q_max = 2 ** self.bit - 1
        self.act_range = nn.Parameter(torch.zeros(2), requires_grad=False)

        self.flag_ema_init = False
        self.flag_fake_quantization = False


        width = int(planes * (base_width/64.)) * groups
        self.conv1 = fused_conv1x1(in_planes=inplane, out_planes=width,
                                   norm_layer=self._norm_layer, activation=nn.ReLU, arg_dict=self.arg_dict)
        self.conv2 = fused_conv3x3(in_planes=width, out_planes=width, stride=stride, groups=groups, dilation=dilation,
                                   norm_layer=self._norm_layer, activation=nn.ReLU, arg_dict=self.arg_dict)
        self.conv3 = fused_conv1x1(in_planes=width, out_planes=planes * self.expansion,
                                   norm_layer=self._norm_layer, arg_dict=self.arg_dict)
        self.relu = nn.ReLU(inplace=False)


    def forward(self, x):
        identity = x

        out = self.conv1(x)             # Fused conv 내에서 norm_layer, activation 사용.
        out = self.conv2(out)           # Fused conv 내에서 norm_layer, activation 사용.
        out = self.conv3(out)           # Fused conv 내에서 norm_layer

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity                 # Residual connection
        out = self.relu(out)

        if not self.training:
            return out

        _out = out
        if self.flag_ema_init:
            self.act_range[0], self.act_range[1] = ema(out, self.act_range, self.smooth)
            if self.flag_fake_quantization:
                s, z = calc_qparams(self.act_range[0], self.act_range[1], self.q_max)
                _out = fake_quantize(out, s, z, self.q_max, self.use_ste)
        else:
            self.act_range[0] = torch.min(out).item()
            self.act_range[1] = torch.max(out).item()
            self.flag_ema_init = True
        return _out

    def set_block_fq_flag(self):
        self.flag_fake_quantization = True
        if self.downsample:
            self.downsample.flag_fake_quantization = True
        self.conv1.flag_fake_quantization = True
        self.conv2.flag_fake_quantization = True
        self.conv3.flag_fake_quantization = True

    def set_block_qparams(self, s1, z1):
        if self.downsample:
            self.downsample.set_qparams(s1, z1)
        prev_s, prev_z = self.conv1.set_qparams(s1, z1)                     #
        prev_s, prev_z = self.conv2.set_qparams(prev_s, prev_z)             #
        _, _ = self.conv3.set_qparams(prev_s, prev_z)                       #
        self.s3, self.z3 = calc_qparams(self.act_range[0], self.act_range[1], self.q_max)

        return self.s3, self.z3


class FusedResNet(nn.Module):
    def __init__(self, block, layers, arg_dict, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None, norm_layer=None):
        super(FusedResNet, self).__init__()

        self.arg_dict = arg_dict
        self.bit, self.smooth, self.runtime_helper, self.quant_noise, self.qn_prob\
            = itemgetter('bit', 'smooth', 'runtime_helper', 'quant_noise', 'qn_prob')(arg_dict)
        self.arg_dict = arg_dict
        self.q_max = 2 ** self.bit - 1
        self.in_range = nn.Parameter(torch.zeros(2), requires_grad=False)

        self.apply_ema = False

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

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

        self.first_conv = FusedConv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False,
                                      norm_layer=self._norm_layer, activation=nn.ReLU, arg_dict=arg_dict, entire=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = FusedLinear(512 * block.expansion, num_classes, bias=True, arg_dict=arg_dict, entire=True)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        # Planes : n_channel_output
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = fused_conv1x1(self.inplanes, planes * block.expansion, stride, bias=False,
                                       norm_layer=self._norm_layer, arg_dict=self.arg_dict)

        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, downsample=downsample, groups=self.groups,
                            base_width=self.base_width, dilation=previous_dilation,
                            norm_layer=self._norm_layer, arg_dict=self.arg_dict))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups, base_width=self.base_width,
                                dilation=self.dilation, norm_layer=self._norm_layer, arg_dict=self.arg_dict))
        return nn.Sequential(*layers)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        if self.training:
            if self.apply_ema:
                self.in_range[0], self.in_range[1] = ema(x, self.in_range, self.smooth)
                if self.runtime_helper.apply_fake_quantization:
                    s, z = calc_qparams(self.in_range[0], self.in_range[1], self.q_max)
                    x = fake_quantize(x, s, z, self.q_max)
            else:
                self.in_range[0] = torch.min(x).item()
                self.in_range[1] = torch.max(x).item()
                self.apply_ema = True

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
    def __init__(self, block, layers, arg_dict, num_classes=10):
        super(FusedResNet20, self).__init__()
        self.bit, self.smooth, self.runtime_helper, self.quant_noise, self.qn_prob\
            = itemgetter('bit', 'smooth', 'runtime_helper', 'quant_noise', 'qn_prob')(arg_dict)
        self.arg_dict = arg_dict
        self.q_max = 2 ** self.bit - 1
        self.in_range = nn.Parameter(torch.zeros(2), requires_grad=False)

        self.apply_ema = False

        self._norm_layer = nn.BatchNorm2d
        self.inplanes = 16
        self.dilation = 1
        self.num_blocks = 3

        self.arg_dict = arg_dict

        self.first_conv = FusedConv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False,
                                  norm_layer=self._norm_layer, activation=nn.ReLU, arg_dict=arg_dict)

        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.fc = FusedLinear(64 * block.expansion, num_classes, arg_dict=arg_dict)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
                downsample = fused_conv1x1(self.inplanes, planes * block.expansion, stride, bias=False,
                                       norm_layer=self._norm_layer, arg_dict=self.arg_dict)
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
                    s, z = calc_qparams(self.in_range[0], self.in_range[1], self.q_max)
                    x = fake_quantize(x, s, z, self.q_max)
            else:
                self.in_range[0] = torch.min(x).item()
                self.in_range[1] = torch.max(x).item()
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
        self.scale, self.zero_point = calc_qparams(self.in_range[0], self.in_range[1], self.q_max)
        prev_s, prev_z = self.first_conv.set_qparams(self.scale, self.zero_point)
        for i in range(len(self.layer1)):
            prev_s, prev_z = self.layer1[i].set_block_qparams(prev_s, prev_z)
        for i in range(len(self.layer2)):
            prev_s, prev_z = self.layer2[i].set_block_qparams(prev_s, prev_z)
        for i in range(len(self.layer3)):
            prev_s, prev_z = self.layer3[i].set_block_qparams(prev_s, prev_z)
        _, _ = self.fc.set_qparams(prev_s, prev_z)


def fused_resnet18(arg_dict, **kwargs):
    return FusedResNet(FusedBasicBlock, [2, 2, 2, 2], arg_dict, **kwargs)

def fused_resnet50(arg_dict, **kwargs):
    return FusedResNet(FusedBottleneck, [3, 4, 6, 3], arg_dict=arg_dict, **kwargs)

def fused_resnet20(arg_dict):
    return FusedResNet20(FusedBasicBlock, [3, 3, 3], arg_dict)


def set_fused_resnet(fused, pre):
    """
        Copy from pre model's params to fused layers.
        Use fused architecture, but not really fused (use CONV & BN seperately)
    """
    # First layer
    fused.first_conv = copy_from_pretrained(fused.first_conv, pre.conv1, pre.bn1)

    # Block 1
    block = fused.layer1
    if len(pre.layer3) == 6: #ResNet50 일땐 첫번째 블록에도 downsample이 있음.
        block[0].downsample = copy_from_pretrained(block[0].downsample, pre.layer1[0].downsample[0], pre.layer1[0].downsample[1])
    for i in range(len(block)):
        block[i].conv1 = copy_from_pretrained(block[i].conv1, pre.layer1[i].conv1, pre.layer1[i].bn1)
        block[i].conv2 = copy_from_pretrained(block[i].conv2, pre.layer1[i].conv2, pre.layer1[i].bn2)
        if type(block[i]) == FusedBottleneck:
            block[i].conv3 = copy_from_pretrained(block[i].conv3, pre.layer1[i].conv3, pre.layer1[i].bn3)

    # Block 2
    block = fused.layer2
    block[0].downsample = copy_from_pretrained(block[0].downsample, pre.layer2[0].downsample[0], pre.layer2[0].downsample[1])
    for i in range(len(block)):
        block[i].conv1 = copy_from_pretrained(block[i].conv1, pre.layer2[i].conv1, pre.layer2[i].bn1)
        block[i].conv2 = copy_from_pretrained(block[i].conv2, pre.layer2[i].conv2, pre.layer2[i].bn2)

        if type(block[i]) == FusedBottleneck:
            block[i].conv3 = copy_from_pretrained(block[i].conv3, pre.layer2[i].conv3, pre.layer2[i].bn3)

    # Block 3
    block = fused.layer3
    block[0].downsample = copy_from_pretrained(block[0].downsample, pre.layer3[0].downsample[0], pre.layer3[0].downsample[1])
    for i in range(len(block)):
        block[i].conv1 = copy_from_pretrained(block[i].conv1, pre.layer3[i].conv1, pre.layer3[i].bn1)
        block[i].conv2 = copy_from_pretrained(block[i].conv2, pre.layer3[i].conv2, pre.layer3[i].bn2)
        if type(block[i]) == FusedBottleneck:
            block[i].conv3 = copy_from_pretrained(block[i].conv3, pre.layer3[i].conv3, pre.layer3[i].bn3)

    # Block 4
    if fused.num_blocks == 4:
        block = fused.layer4
        block[0].downsample = copy_from_pretrained(block[0].downsample, pre.layer4[0].downsample[0], pre.layer4[0].downsample[1])
        for i in range(len(block)):
            block[i].conv1 = copy_from_pretrained(block[i].conv1, pre.layer4[i].conv1, pre.layer4[i].bn1)
            block[i].conv2 = copy_from_pretrained(block[i].conv2, pre.layer4[i].conv2, pre.layer4[i].bn2)
            if type(block[i]) == FusedBottleneck:
                block[i].conv3 = copy_from_pretrained(block[i].conv3, pre.layer4[i].conv3, pre.layer4[i].bn3)

    # Classifier
    fused.fc = copy_from_pretrained(fused.fc, pre.fc)
    return fused


def fold_resnet(model):
    # First layer
    model.first_conv.fold_conv_and_bn()

    # Block 1 - layer1 downsample is exist at more than layer 50 ResNet
    fp_block = model.layer1
    if len(model.layer3) == 6:
        fp_block[0].downsample.fold_conv_and_bn()

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
    #model.first_conv.quant_noise = False
    # Block 1
    block = model.layer1
    if len(model.layer3) == 6: #ResNet50 일땐 첫번째 블록에도 downsample이 있음.
        m = block[0].downsample
        m.conv = _quant_noise(m.conv, m.runtime_helper.qn_prob, 1, q_max=m.q_max)
    for i in range(len(block)):
        block[i].conv1.conv = _quant_noise(block[i].conv1.conv, block[i].conv1.runtime_helper.qn_prob, 1, q_max=block[i].q_max)
        block[i].conv2.conv = _quant_noise(block[i].conv2.conv, block[i].conv2.runtime_helper.qn_prob, 1, q_max=block[i].q_max)
        if type(block[i]) == FusedBottleneck:
            block[i].conv3.conv = _quant_noise(block[i].conv3.conv, block[i].conv3.runtime_helper.qn_prob, 1, q_max=block[i].q_max)

    # Block 2
    block = model.layer2
    block[0].downsample.conv = _quant_noise(block[0].downsample.conv, block[0].downsample.runtime_helper.qn_prob,
                                            1, q_max=block[0].downsample.q_max)
    for i in range(len(block)):
        block[i].conv1.conv = _quant_noise(block[i].conv1.conv, block[i].conv1.runtime_helper.qn_prob, 1, q_max=block[i].q_max)
        block[i].conv2.conv = _quant_noise(block[i].conv2.conv, block[i].conv2.runtime_helper.qn_prob, 1, q_max=block[i].q_max)
        if type(block[i]) == FusedBottleneck:
            block[i].conv3.conv = _quant_noise(block[i].conv3.conv, block[i].conv3.runtime_helper.qn_prob, 1, q_max=block[i].q_max)

    # Block 3
    block = model.layer3
    block[0].downsample.conv = _quant_noise(block[0].downsample.conv, block[0].downsample.runtime_helper.qn_prob,
                                            1, q_max=block[0].downsample.q_max)
    for i in range(len(block)):
        block[i].conv1.conv = _quant_noise(block[i].conv1.conv, block[i].conv1.runtime_helper.qn_prob, 1, q_max=block[i].q_max)
        block[i].conv2.conv = _quant_noise(block[i].conv2.conv, block[i].conv2.runtime_helper.qn_prob, 1, q_max=block[i].q_max)
        if type(block[i]) == FusedBottleneck:
            block[i].conv3.conv = _quant_noise(block[i].conv3.conv, block[i].conv3.runtime_helper.qn_prob, 1, q_max=block[i].q_max)
    # Block 4
    if model.num_blocks == 4:
        block = model.layer4
        block[0].downsample.conv = _quant_noise(block[0].downsample.conv, block[0].downsample.runtime_helper.qn_prob,
                                                1, q_max=block[0].downsample.q_max)
        for i in range(len(block)):
            block[i].conv1.conv = _quant_noise(block[i].conv1.conv, block[i].conv1.runtime_helper.qn_prob, 1, q_max=block[i].q_max)
            block[i].conv2.conv = _quant_noise(block[i].conv2.conv, block[i].conv2.runtime_helper.qn_prob, 1, q_max=block[i].q_max)
            if type(block[i]) == FusedBottleneck:
                block[i].conv3.conv = _quant_noise(block[i].conv3.conv, block[i].conv3.runtime_helper.qn_prob, 1, q_max=block[i].q_max)

    # Classifier
    #model.fc.quant_noise = False

import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

from ..quantization_utils import *


class FakeConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False, bit=32):
        super(FakeConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.layer_type = 'FakeConv2d'
        self.bit = bit
        self.q_max = 2 ** self.bit - 1
        self.scale = nn.Parameter(torch.tensor(0, dtype=torch.float32), requires_grad=False)
        self.zero_point = nn.Parameter(torch.tensor(0, dtype=torch.int32), requires_grad=False)

    def forward(self, x):
        self.set_qparams()
        self.fake_quantize_weight()
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

    def set_qparams(self):
        self.scale, self.zero_point = calc_qprams(torch.min(self.weight), torch.max(self.weight), self.q_max)

    def fake_quantize_weight(self):
        self.weight.data = torch.round(self.weight.div(self.scale).add(self.zero_point)).sub(self.zero_point).mul(self.scale)


class FusedConv2d(nn.Module):
    """
        Fused Layer to calculate Quantization Parameters (S & Z)
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False, bn=False, relu=True, smooth=0.995, bit=32):
        super(FusedConv2d, self).__init__()
        self.layer_type = 'FusedConv2d'
        self.bit = bit
        self.q_max = 2 ** bit - 1
        self.ema_init = False
        self.smooth = smooth
        self.act_range = nn.Parameter(torch.zeros(2), requires_grad=False)
        self.scale = nn.Parameter(torch.tensor(0, dtype=torch.float32), requires_grad=False)
        self.zero_point = nn.Parameter(torch.tensor(0, dtype=torch.int32), requires_grad=False)
        self.fused = False
        self.out_channels = out_channels

        self.conv = FakeConv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=bias, bit=bit)
        self.bn = nn.BatchNorm2d(out_channels) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        out = self.conv(x)
        if self.fused:
            # Validation with actually fused layer : Skip BatchNorm (fused into CONV weight/bias)
            if self.relu:
                out = self.relu(out)
            return out

        if self.bn:
            out = self.bn(out)
        if self.relu:
            out = self.relu(out)

        if self.training:
            if self.ema_init:
                self.ema(out)
                self.set_qparams()
                out = self.fake_quantize_activation(out)
            else:
                self.act_range[0] = torch.min(out).item()
                self.act_range[1] = torch.max(out).item()
                self.ema_init = True
        else:
            out = self.fake_quantize_activation(out)
        return out

    def ema(self, x):
        _min = torch.min(x).item()
        _max = torch.max(x).item()
        self.act_range[0] = self.act_range[0] * self.smooth + _min * (1 - self.smooth)
        self.act_range[1] = self.act_range[1] * self.smooth + _max * (1 - self.smooth)

    def set_qparams(self):
        self.scale, self.zero_point = calc_qprams(self.act_range[0], self.act_range[1], self.q_max)

    def fake_quantize_activation(self, x):
        x = torch.round(x.div(self.scale).add(self.zero_point)).sub(self.zero_point).mul(self.scale)
        return x

    def copy_from_pretrained(self, conv, bn):
        # Copy weights from pretrained FP model
        self.conv.weight.data = torch.nn.Parameter(conv.weight.data)
        if bn:
            self.bn = bn
        else:
            self.conv.bias.data = torch.nn.Parameter(conv.bias.data)

    def fuse_conv_and_bn(self):
        # In case of validation, fuse pretrained Conv&BatchNorm params
        assert self.training == False, "Do not fuse layers while training."
        alpha, beta, mean, var, eps = self.bn.weight, self.bn.bias, self.bn.running_mean, self.bn.running_var, self.bn.eps
        n_channel = self.conv.weight.shape[0]
        self.conv.bias = torch.nn.Parameter(beta)
        for c in range(n_channel):
            self.conv.weight.data[c] = self.conv.weight.data[c].mul(alpha[c]).div(torch.sqrt(var[c].add(eps)))
            self.conv.bias.data[c] = self.conv.bias.data[c].sub(alpha[c].mul(mean[c]).div(torch.sqrt(var[c])))
        self.bn = SkipBN()
        self.fused = True  # Change Flag

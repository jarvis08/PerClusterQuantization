import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from copy import copy


def calc_qprams(_min, _max, q_max):
    s = (_max - _min) / q_max
    z = - _min / s
    return s, np.clip(z, 0, q_max)


class FakeLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, bit=32, smooth=0.995):
        super(FakeLinear, self).__init__(in_features, out_features, bias)
        self.layer_type = 'FakeLinear'
        self.bit = bit
        self.q_max = 2 ** self.bit - 1
        self.ema_init = False
        self.act_range = np.zeros(2, dtype=np.float)
        self.smooth = smooth
        print("Set Fake Quantizing Linear layer")

    def forward(self, x):
        if self.bit == 32 or not self.training:
            return F.linear(x, self.weight, self.bias)

        self.fake_quantize_weight()
        out = F.linear(x, self.weight, self.bias)
        if self.ema_init:
            self.ema_matrix(out)
            out = self.fake_quantize_activation(out)
        else:
            self.set_min_max(out)
            self.ema_init = True
        return out

    def set_min_max(self, x):
        _min = torch.min(x).item()
        _max = torch.max(x).item()
        self.act_range[0] = _min
        self.act_range[1] = _max

    def ema_matrix(self, x):
        _min = torch.min(x).item()
        _max = torch.max(x).item()
        self.act_range[0] = self.act_range[0] * self.smooth + _min * (1 - self.smooth)
        self.act_range[1] = self.act_range[1] * self.smooth + _max * (1 - self.smooth)

    def get_weight_qparams(self):
        _min = torch.min(self.weight).item()
        _max = torch.max(self.weight).item()
        return calc_qprams(_min, _max, self.q_max)

    def fake_quantize_weight(self):
        s, z = self.get_weight_qparams()
        self.weight.data = torch.round(self.weight.div(s).add(z)).sub(z).mul(s)

    def get_activation_qparams(self):
        _min = self.act_range[0]
        _max = self.act_range[1]
        return calc_qprams(_min, _max, self.q_max)

    def fake_quantize_activation(self, x):
        s, z = self.get_activation_qparams()
        x = torch.round(x.div(s).add(z)).sub(z).mul(s)
        return x


class FakeConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False, bit=32):
        super(FakeConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.layer_type = 'FakeConv2d'
        self.bit = bit
        self.q_max = 2 ** self.bit - 1

    def forward(self, x):
        if self.bit != 32 and self.training:
            self.fake_quantize_weight()
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

    def get_weight_qparams(self):
        _min = torch.min(self.weight).item()
        _max = torch.max(self.weight).item()
        return calc_qprams(_min, _max, self.q_max)

    def fake_quantize_weight(self):
        s, z = self.get_weight_qparams()
        self.weight.data = torch.round(self.weight.div(s).add(z)).sub(z).mul(s)


class FusedConv2d(nn.Module):
    """
        Fused Layer to calculate Quantization Parameters (S & Z)
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bit=32, smooth=0.995, fuse_relu=True):
        super(FusedConv2d, self).__init__()
        self.layer_type = 'FusedConvBnReLU2D'
        self.bit = bit
        self.q_max = 2 ** self.bit - 1
        self.ema_init = False
        self.smooth = smooth
        self.act_range = np.zeros(2, dtype=np.float)
        self.fused = False

        self.conv = FakeConv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bit)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = None
        if fuse_relu:
            print("Set [Conv + BatchNorm + ReLU] Fused layer")
            self.relu = nn.ReLU(inplace=True)
        else:
            print("Set [Conv + BatchNorm] Fused layer")

    def forward(self, x):
        out = self.conv(x)
        if self.fused:
            # Fused Validation: Skip BatchNorm (fused into CONV weight/bias)
            if self.relu:
                out = self.relu(out)
            return out

        out = self.bn(out)
        if self.relu:
            out = self.relu(out)

        if self.bit == 32 or not self.training:
            # General Validation
            return out

        if self.relu:
            # QAT
            if self.ema_init:
                self.ema_matrix(out)
                out = self.fake_quantize_activation(out)
            else:
                self.set_min_max(out)
                self.ema_init = True
        return out

    def set_min_max(self, x):
        _min = torch.min(x).item()
        _max = torch.max(x).item()
        self.act_range[0] = _min
        self.act_range[1] = _max

    def ema_matrix(self, x):
        _min = torch.min(x).item()
        _max = torch.max(x).item()
        self.act_range[0] = self.act_range[0] * self.smooth + _min * (1 - self.smooth)
        self.act_range[1] = self.act_range[1] * self.smooth + _max * (1 - self.smooth)

    def get_activation_qparams(self):
        _min = self.act_range[0]
        _max = self.act_range[1]
        return calc_qprams(_min, _max, self.q_max)

    def fake_quantize_activation(self, x):
        s, z = self.get_activation_qparams()
        x = torch.round(x.div(s).add(z)).sub(z).mul(s)
        return x

    def get_weight_qparams(self):
        return self.conv.get_weight_qprams()

    def copy_from_pretrained(self, conv, bn):
        """
            Copy weights from pretrained FP model.
        """
        self.conv.weight.data = torch.nn.Parameter(conv.weight.data)
        self.bn = bn

    def fuse_conv_and_bn(self):
        """
            In case of validation, fuse pretrained Conv&BatchNorm params
        """
        assert self.training == True, "Do not fuse layers while training."
        alpha, beta, mean, var, eps = self.bn.weight, self.bn.bias, self.bn.running_mean, self.bn.running_var, self.bn.eps
        n_channel = self.conv.weight.shape[0]
        self.conv.bias = torch.nn.Parameter(beta)
        for c in range(n_channel):
            self.conv.weight.data[c] = self.conv.weight.data[c].mul(alpha[c]).div(torch.sqrt(var[c].add(eps)))
            self.conv.bias.data[c] = self.conv.bias.data[c].sub(alpha[c].mul(mean[c]).div(torch.sqrt(var[c])))

        self.fused = True  # Change Flag


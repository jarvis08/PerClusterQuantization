import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from ..quantization_utils import *

class FakeLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, bit=32):
        super(FakeLinear, self).__init__(in_features, out_features, bias)
        self.layer_type = 'FakeLinear'
        self.bit = bit
        self.q_max = 2 ** self.bit - 1

    def forward(self, x):
        if self.bit == 32 or not self.training:
            return F.linear(x, self.weight, self.bias)

        self.fake_quantize_weight()
        out = F.linear(x, self.weight, self.bias)
        return out

    def get_weight_qparams(self):
        return calc_qprams(torch.min(self.weight).item(), torch.max(self.weight).item(), self.q_max)

    def fake_quantize_weight(self):
        s, z = self.get_weight_qparams()
        self.weight.data = torch.round(self.weight.div(s).add(z)).sub(z).mul(s)


class FusedLinear(nn.Module):
    """
        Fused Layer to calculate Quantization Parameters (S & Z)
    """
    def __init__(self, in_features, out_features, bias=True, bit=32, smooth=0.995, relu=True):
        super(FusedLinear, self).__init__()
        self.layer_type = 'FusedLinear'
        self.bit = bit
        self.q_max = 2 ** bit - 1
        self.ema_init = False
        # self.ema_init = 5000
        self.smooth = smooth
        self.act_range = np.zeros(2, dtype=np.float32)

        self.fc = FakeLinear(in_features, out_features, bias=bias, bit=bit)
        self.relu = None
        if relu:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.fc(x)
        if self.relu:
            out = self.relu(out)

        if self.bit == 32 or not self.training:
            # General Validation
            return out

        # self.ema(out)
        # if not self.ema_init:
        if self.ema_init:
            self.ema(out)
            out = self.fake_quantize_activation(out)
        else:
            self.act_range[0] = torch.min(out).item()
            self.act_range[1] = torch.max(out).item()
            self.ema_init = True
            # self.ema_init -= 1
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

    def copy_from_pretrained(self, fc):
        # Copy weights from pretrained FP model
        self.fc.weight.data = torch.nn.Parameter(fc.weight.data)
        self.fc.bias.data = torch.nn.Parameter(fc.bias.data)

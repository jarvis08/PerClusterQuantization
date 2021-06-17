import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from quantization.quantization_utils import *

class FakeLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, bit=32):
        super(FakeLinear, self).__init__(in_features, out_features, bias)
        self.layer_type = 'FakeLinear'
        self.bit = bit
        self.q_max = 2 ** self.bit - 1
        self.scale = nn.Parameter(torch.tensor(0, dtype=torch.float32), requires_grad=False)
        self.zero_point = nn.Parameter(torch.tensor(0, dtype=torch.int32), requires_grad=False)

    def forward(self, x):
        self.set_qparams()
        self.fake_quantize_weight()
        out = F.linear(x, self.weight, self.bias)
        return out

    def set_qparams(self):
        self.scale, self.zero_point = calc_qprams(torch.min(self.weight), torch.max(self.weight), self.q_max)

    def fake_quantize_weight(self):
        self.weight.data = torch.round(self.weight.div(self.scale).add(self.zero_point)).sub(self.zero_point).mul(self.scale)


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
        self.smooth = smooth
        self.act_range = nn.Parameter(torch.zeros(2), requires_grad=False)
        self.scale = nn.Parameter(torch.tensor(0, dtype=torch.float32), requires_grad=False)
        self.zero_point = nn.Parameter(torch.tensor(0, dtype=torch.int32), requires_grad=False)

        self.fc = FakeLinear(in_features, out_features, bias=bias, bit=bit)
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        out = self.fc(x)
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

    def copy_from_pretrained(self, fc):
        # Copy weights from pretrained FP model
        self.fc.weight.data = torch.nn.Parameter(fc.weight.data)
        self.fc.bias.data = torch.nn.Parameter(fc.bias.data)

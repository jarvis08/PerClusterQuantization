import torch.nn as nn
import torch
import torch.nn.functional as F

from ..quantization_utils import *

class QuantizedActivation(nn.Module):
    """
        Quantized Activation Layer(Hardswish, gelu ..)
    """
    def __init__(self, activation=None, bit=8):
        super(QuantizedActivation, self).__init__()
        self.layer_type = 'QuantizedActivation'
        self.bit = bit
        self.q_max = 2 ** bit - 1
        self._activation = activation
        self.hardswish = torch.nn.Hardswish().to(dtype=torch.float)
        self.elu = torch.nn.ELU().to(dtype=torch.float)

    def forward(self, x):
        x = self.hardswish(x)
        x = self.elu(x)
        return x


class QActivation(nn.Module):
    """
        Activation Layer(Hardswish, Hardsigmoid, gelu ..)
    """
    def __init__(self, activation=None, smooth=0.999, bit=32):
        super(QActivation, self).__init__()
        self.layer_type = 'QActivation'
        self.bit = bit
        self.q_max = 2 ** bit - 1
        self.ema_init = False
        self.smooth = smooth
        self.act_range = nn.Parameter(torch.zeros(2), requires_grad=False)
        self._activation = activation(inplace=True)

    def forward(self, x):
        x = self._activation(x)

        # if self.training:
        #     if self.ema_init:
        #         self.act_range[0], self.act_range[1] = ema(x, self.act_range, self.smooth)
        #         s, z = calc_qparams(self.act_range[0], self.act_range[1], self.q_max)
        #         x = fake_quantize(x, s, z)
        #     else:
        #         self.act_range[0] = torch.min(x).item()
        #         self.act_range[1] = torch.max(x).item()
        #         self.ema_init = True
        return x

    def set_qparams(self, s1, z1):
        self.s1, self.z1 = nn.Parameter(s1, requires_grad=False), nn.Parameter(z1, requires_grad=False)
        self.s3, self.z3 = calc_qparams(self.act_range[0], self.act_range[1], self.q_max)
        return self.s3, self.z3


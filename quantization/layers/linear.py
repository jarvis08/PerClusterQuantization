import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from quantization.quantization_utils import *


class QuantizedLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, bit=8):
        super(QuantizedLinear, self).__init__(in_features, out_features, bias)
        self.layer_type = 'QuantizedLinear'
        self.bit = bit
        self.q_max = 2 ** self.bit - 1
        self.s1 = nn.Parameter(torch.tensor(0, dtype=torch.float32), requires_grad=False)
        self.s2 = nn.Parameter(torch.tensor(0, dtype=torch.float32), requires_grad=False)
        self.s3 = nn.Parameter(torch.tensor(0, dtype=torch.float32), requires_grad=False)
        self.z1 = nn.Parameter(torch.tensor(0, dtype=torch.int32), requires_grad=False)
        self.z2 = nn.Parameter(torch.tensor(0, dtype=torch.int32), requires_grad=False)
        self.z3 = nn.Parameter(torch.tensor(0, dtype=torch.int32), requires_grad=False)
        self.M0 = nn.Parameter(torch.tensor(0, dtype=torch.int32), requires_grad=False)
        self.shift = nn.Parameter(torch.tensor(0, dtype=torch.int32), requires_grad=False)

        self.max_int = torch.tensor(9223372036854775807, dtype=torch.int64, device='cuda:0')

    def forward(self, x):
        sum_q1q2 = F.linear(x, self.weight, None)
        return self.totalsum(x, sum_q1q2.type(torch.cuda.IntTensor))

    def totalsum(self, x, sum_q1q2):
        input_feature, output_feature = sum_q1q2.shape[0], sum_q1q2.shape[1]
        if self.bias is not None:
            for out_f in range(output_feature):
                sum_q1q2[:, out_f] = sum_q1q2[:, out_f].add(self.bias[out_f])
        N = x.shape[1]
        nz1z2 = N * self.z1 * self.z2

        sum_a1 = torch.zeros(input_feature, dtype=torch.int32)
        sum_a2 = torch.zeros(output_feature, dtype=torch.int32)
        for out_f in range(output_feature):
            sum_a2[out_f] = torch.sum(self.weight[out_f, :]).mul(self.z1)
        for in_f in range(input_feature):
            sum_a1[in_f] = torch.sum(x[in_f, :]).mul(self.z2)

        sub_sum = sum_q1q2.add(nz1z2)
        for in_f in range(input_feature):
            sub_sum[in_f, :] = torch.sub(sub_sum[in_f, :], sum_a1[in_f])
        for out_f in range(output_feature):      
            sub_sum[:, out_f] = torch.sub(sub_sum[:, out_f], sum_a2[out_f])

        sub_sum = sub_sum.type(torch.cuda.LongTensor)
        multiplied = multiply_M(sub_sum, self.M0)
        total = shifting(multiplied, self.shift.item())
        total = total.add(self.z3)
        if self.bit == 4:
            total = torch.clamp(total, 0, 15).type(torch.cuda.FloatTensor)
        else: 
            total = torch.clamp(total, -128, 127).type(torch.cuda.FloatTensor)
        return total


class FakeLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, bit=32):
        super(FakeLinear, self).__init__(in_features, out_features, bias)
        self.layer_type = 'FakeLinear'
        self.bit = bit
        self.q_max = 2 ** self.bit - 1

    def forward(self, x):
        if self.training:
            self.fake_quantize_weight()
        return F.linear(x, self.weight, self.bias)

    def fake_quantize_weight(self):
        s, z = calc_qparams(torch.min(self.weight), torch.max(self.weight), self.q_max)
        self.weight.data = torch.round(self.weight.div(s).add(z)).sub(z).mul(s)


class FusedLinear(nn.Module):
    """
        Fused Layer to calculate Quantization Parameters (S & Z)
    """
    def __init__(self, in_features, out_features, bias=True, bit=32, smooth=0.995, relu=True):
        super(FusedLinear, self).__init__()
        self.layer_type = 'FusedLinear'
        self.quantized = False
        self.bit = bit
        self.q_max = 2 ** bit - 1
        self.ema_init = False
        self.smooth = smooth
        self.act_range = nn.Parameter(torch.zeros(2), requires_grad=False)
        self.s1 = nn.Parameter(torch.tensor(0, dtype=torch.float32), requires_grad=False)
        self.s2 = nn.Parameter(torch.tensor(0, dtype=torch.float32), requires_grad=False)
        self.s3 = nn.Parameter(torch.tensor(0, dtype=torch.float32), requires_grad=False)
        self.z1 = nn.Parameter(torch.tensor(0, dtype=torch.int32), requires_grad=False)
        self.z2 = nn.Parameter(torch.tensor(0, dtype=torch.int32), requires_grad=False)
        self.z3 = nn.Parameter(torch.tensor(0, dtype=torch.int32), requires_grad=False)
        self.M0 = nn.Parameter(torch.tensor(0, dtype=torch.int32), requires_grad=False)
        self.shift = nn.Parameter(torch.tensor(0, dtype=torch.int32), requires_grad=False)

        self.fc = FakeLinear(in_features, out_features, bias=bias, bit=bit)
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        out = self.fc(x)
        if self.relu:
            out = self.relu(out)

        if self.training:
            if self.ema_init:
                self.ema(out)
                out = self.fake_quantize_activation(out)
            else:
                self.act_range[0] = torch.min(out).item()
                self.act_range[1] = torch.max(out).item()
                self.ema_init = True
        return out

    def ema(self, x):
        _min = torch.min(x).item()
        _max = torch.max(x).item()
        self.act_range[0] = self.act_range[0] * self.smooth + _min * (1 - self.smooth)
        self.act_range[1] = self.act_range[1] * self.smooth + _max * (1 - self.smooth)

    def fake_quantize_activation(self, x):
        s, z = calc_qparams(self.act_range[0], self.act_range[1], self.q_max)
        x = torch.round(x.div(s).add(z)).sub(z).mul(s)
        return x

    def copy_from_pretrained(self, fc):
        # Copy weights from pretrained FP model
        self.fc.weight.data = torch.nn.Parameter(fc.weight.data)
        self.fc.bias.data = torch.nn.Parameter(fc.bias.data)

    def set_fc_qparams(self, s1, z1):
        self.s1, self.z1 = torch.nn.Parameter(s1, requires_grad=False), torch.nn.Parameter(z1, requires_grad=False)
        self.s2, self.z2 = calc_qparams(torch.min(self.fc.weight), torch.max(self.fc.weight), self.q_max)
        self.s3, self.z3 = calc_qparams(self.act_range[0], self.act_range[1], self.q_max)
        self.M0, self.shift = quantize_M(self.s1 * self.s2 / self.s3)
        return self.s3, self.z3

import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from ..quantization_utils import *


class QuantizedLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, bit=8):
        super(QuantizedLinear, self).__init__(in_features, out_features, bias)
        self.layer_type = 'QuantizedLinear'
        self.bit = bit
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

        sum_a1 = torch.zeros(input_feature, dtype=torch.int32)
        sum_a2 = torch.zeros(output_feature, dtype=torch.int32)
        for out_f in range(output_feature):
            sum_a2[out_f] = torch.sum(self.weight[out_f, :]).mul(self.z1)
        for in_f in range(input_feature):
            sum_a1[in_f] = torch.sum(x[in_f, :]).mul(self.z2)

        nz1z2 = N * self.z1 * self.z2
        sub_sum = sum_q1q2.add(nz1z2)
        for in_f in range(input_feature):
            sub_sum[in_f, :] = torch.sub(sub_sum[in_f, :], sum_a1[in_f])
        for out_f in range(output_feature):      
            sub_sum[:, out_f] = torch.sub(sub_sum[:, out_f], sum_a2[out_f])

        multiplied = multiply_M(sub_sum.type(torch.cuda.LongTensor), self.M0)
        total = shifting(multiplied, self.shift.item())
        total = total.add(self.z3)
        if self.bit == 4:
            total = torch.clamp(total, 0, 15)
        else: 
            total = torch.clamp(total, -128, 127)
        return total.type(torch.cuda.FloatTensor)


class PCQLinear(nn.Module):
    """
        Fused Layer to calculate Quantization Parameters(S & Z) with multiple clusters
    """
    def __init__(self, in_features, out_features, bias=True, relu=True, bit=8, smooth=0.995, num_clusters=10):
        super(PCQLinear, self).__init__()
        self.layer_type = 'PCQLinear'
        self.bit = bit
        self.q_max = 2 ** bit - 1
        self.smooth = smooth
        self.ema_init = np.zeros(num_clusters, dtype=bool)
        self.act_range = nn.Parameter(torch.zeros(num_clusters, 2), requires_grad=False)
        self.num_clusters = num_clusters

        self.fc = nn.Linear(in_features, out_features, bias=bias)
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x, cluster_info):
        s, z = calc_qparams(torch.min(self.fc.weight), torch.max(self.fc.weight), self.q_max)
        self.fc.weight.data = fake_quantize(self.fc.weight.data, s, z)

        x = self.fc(x)
        if self.relu:
            x = self.relu(x)

        if self.training:
            done = 0
            for i in range(self.num_clusters):
                c = cluster_info[i][0]
                n = cluster_info[i][1]
                if c == -1:
                    break
                if self.ema_init[c]:
                    self.act_range[c][0], self.act_range[c][1] = ema(x[done:done + n], self.act_range[c], self.smooth)
                    s, z = calc_qparams(self.act_range[c][0], self.act_range[c][1], self.q_max)
                    x[done:done + n] = fake_quantize(x[done:done + n], s, z)
                else:
                    self.act_range[c][0] = torch.min(x).item()
                    self.act_range[c][1] = torch.max(x).item()
                    self.ema_init[c] = True
                done += n
        return x

    def set_qparams(self, s1, z1):
        self.s1, self.z1 = torch.nn.Parameter(s1, requires_grad=False), torch.nn.Parameter(z1, requires_grad=False)
        self.s2, self.z2 = calc_qparams(torch.min(self.conv.weight), torch.max(self.conv.weight), self.q_max)

        self.s3 = nn.Parameter(torch.zeros(self.num_clusters, dtype=torch.float32), requires_grad=False)
        self.z3 = nn.Parameter(torch.zeros(self.num_clusters, dtype=torch.int32), requires_grad=False)
        self.M0 = nn.Parameter(torch.zeros(self.num_clusters, dtype=torch.int32), requires_grad=False)
        self.shift = nn.Parameter(torch.zeros(self.num_clusters, dtype=torch.int32), requires_grad=False)
        for c in range(self.num_clusters):
            self.s3[c], self.z3[c] = calc_qparams(self.act_range[c][0], self.act_range[c][1], self.q_max)
            self.M0[c], self.shift[c] = quantize_M(self.s1[c] * self.s2 / self.s3[c])
        return self.s3, self.z3


class FusedLinear(nn.Module):
    """
        Fused Layer to calculate Quantization Parameters (S & Z)
    """
    def __init__(self, in_features, out_features, bias=True, h_swish=False, bit=32, smooth=0.995, relu=True):
        super(FusedLinear, self).__init__()
        self.layer_type = 'FusedLinear'
        self.bit = bit
        self.q_max = 2 ** bit - 1
        self.ema_init = False
        self.smooth = smooth
        self.act_range = nn.Parameter(torch.zeros(2), requires_grad=False)

        self.fc = nn.Linear(in_features, out_features, bias=bias)
        self.h_swish = nn.Hardswish(inplace=True) if h_swish else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        if self.training:
            s, z = calc_qparams(torch.min(self.fc.weight), torch.max(self.fc.weight), self.q_max)
            self.fc.weight.data = fake_quantize(self.fc.weight.data, s, z)

        x = self.fc(x)
        if self.relu:
            x = self.relu(x)
        if self.h_swish:
            x = self.h_swish(x)

        if self.training:
            if self.ema_init:
                self.act_range[0], self.act_range[1] = ema(x, self.act_range, self.smooth)
                s, z = calc_qparams(self.act_range[0], self.act_range[1], self.q_max)
                x = fake_quantize(x, s, z)
            else:
                self.act_range[0] = torch.min(x).item()
                self.act_range[1] = torch.max(x).item()
                self.ema_init = True
        return x

    def set_qparams(self, s1, z1):
        self.s1, self.z1 = torch.nn.Parameter(s1, requires_grad=False), torch.nn.Parameter(z1, requires_grad=False)
        self.s2, self.z2 = calc_qparams(torch.min(self.fc.weight), torch.max(self.fc.weight), self.q_max)
        self.s3, self.z3 = calc_qparams(self.act_range[0], self.act_range[1], self.q_max)
        self.M0, self.shift = quantize_M(self.s1 * self.s2 / self.s3)
        return self.s3, self.z3

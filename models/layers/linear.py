import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from ..quantization_utils import *
from .activation import *


class QuantizedLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=False, activation=None, bit=8, num_clusters=1):
        super(QuantizedLinear, self).__init__(in_features, out_features, bias)
        self.layer_type = 'QuantizedLinear'
        self.bit = bit
        self.q_max = 2 ** bit - 1
        self.num_clusters = num_clusters
        self.quantized_bias = nn.Parameter(torch.zeros((num_clusters, out_features)), requires_grad=False)
        t_init = list(range(num_clusters)) if num_clusters > 1 else 0
        self.s1 = nn.Parameter(torch.tensor(t_init, dtype=torch.float32), requires_grad=False)
        self.s2 = nn.Parameter(torch.tensor(0, dtype=torch.float32), requires_grad=False)
        self.s3 = nn.Parameter(torch.tensor(t_init, dtype=torch.float32), requires_grad=False)
        self.z1 = nn.Parameter(torch.tensor(t_init, dtype=torch.int32), requires_grad=False)
        self.z2 = nn.Parameter(torch.tensor(0, dtype=torch.int32), requires_grad=False)
        self.z3 = nn.Parameter(torch.tensor(t_init, dtype=torch.int32), requires_grad=False)
        self.M0 = nn.Parameter(torch.tensor(t_init, dtype=torch.int32), requires_grad=False)
        self.shift = nn.Parameter(torch.tensor(t_init, dtype=torch.int32), requires_grad=False)
        self.hardswish_6 = nn.Parameter(torch.tensor(t_init, dtype=torch.int32), requires_grad=False)
        self.hardswish_3 = nn.Parameter(torch.tensor(t_init, dtype=torch.int32), requires_grad=False)
        self.s_activation = nn.Parameter(torch.tensor(t_init, dtype=torch.float32), requires_grad=False)
        self.z_activation = nn.Parameter(torch.tensor(t_init, dtype=torch.int32), requires_grad=False)
        self.activation = activation

    def forward(self, x):
        _x = x[0]
        cluster_info = x[1]
        sum_q1q2 = F.linear(_x, self.weight, None)
        if cluster_info is not None:
            return self.pcq_totalsum(_x, sum_q1q2.type(torch.cuda.IntTensor), cluster_info)
        else:
            return self.general_totalsum(_x, sum_q1q2.type(torch.cuda.IntTensor))

    def pcq_totalsum(self, x, sum_q1q2, cluster_info):
        input_feature, output_feature = sum_q1q2.shape[0], sum_q1q2.shape[1]
        N = x.shape[1]
        done = 0
        for i in range(cluster_info.shape[0]):
            c = cluster_info[i][0].item()
            n = cluster_info[i][1].item()
            for out_f in range(output_feature):
                sum_q1q2[done:done+n, out_f] = sum_q1q2[done:done+n, out_f].add(self.quantized_bias[c][out_f])
            done += n

        sum_a1 = torch.zeros(input_feature, dtype=torch.int32)
        sum_a2 = torch.zeros((cluster_info.shape[0], output_feature), dtype=torch.int32)

        for i in range(cluster_info.shape[0]):
            c = cluster_info[i][0].item()
            for out_f in range(output_feature):
                sum_a2[i, out_f] = torch.sum(self.weight[out_f, :]).mul(self.z1[c])

        for in_f in range(input_feature):
            sum_a1[in_f] = torch.sum(x[in_f, :]).mul(self.z2)

        done = 0
        for i in range(cluster_info.shape[0]):
            c = cluster_info[i][0].item()
            n = cluster_info[i][1].item()
            nz1z2 = N * self.z1[c] * self.z2
            sum_q1q2[done:done + n] = sum_q1q2[done:done + n].add(nz1z2)
            done += n

        for in_f in range(input_feature):
            sum_q1q2[in_f, :] = torch.sub(sum_q1q2[in_f, :], sum_a1[in_f])

        done = 0
        for i in range(cluster_info.shape[0]):
            n = cluster_info[i][1].item()
            for out_f in range(output_feature):
                sum_q1q2[done:done + n, out_f] = torch.sub(sum_q1q2[done:done + n, out_f], sum_a2[i, out_f])
            done += n

        done = 0
        total = torch.zeros(sum_q1q2.shape, dtype=torch.int32).cuda()
        for i in range(cluster_info.shape[0]):
            c = cluster_info[i][0].item()
            n = cluster_info[i][1].item()
            multiplied = multiply_M(sum_q1q2[done:done + n].type(torch.cuda.LongTensor), self.M0[c])
            total[done:done + n] = shifting(multiplied, self.shift[c].item())
            total[done:done + n] = total[done:done + n].add(self.z3[c])
            done += n

        if self.bit == 4:
            total = torch.clamp(total, 0, 15)
        else: 
            total = torch.clamp(total, -128, 127)
        return total.type(torch.cuda.FloatTensor)

    def general_totalsum(self, x, sum_q1q2):
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

        if self.activation:
            hs_total = total + self.hardswish_3
            hs_total = torch.clamp(hs_total, self.z3, self.hardswish_6)
            hs_total = hs_total / self.hardswish_6
            if self.activation == 'Hardswish':
                total = total * hs_total
            else:
                total = hs_total

        if self.bit == 4:
            total = torch.clamp(total, 0, 15)
        else:
            total = torch.clamp(total, -128, 127)
        return total.type(torch.cuda.FloatTensor)


class PCQLinear(nn.Module):
    """
        Fused Layer to calculate Quantization Parameters(S & Z) with multiple clusters
    """
    def __init__(self, in_features, out_features, bias=True, activation=None, bit=8, smooth=0.999, num_clusters=10):
        super(PCQLinear, self).__init__()
        self.layer_type = 'PCQLinear'
        self.bit = bit
        self.q_max = 2 ** bit - 1
        self.smooth = smooth
        self.flag_ema_init = np.zeros(num_clusters, dtype=bool)
        self.flag_fake_quantization = False
        self.act_range = nn.Parameter(torch.zeros((num_clusters, 2)), requires_grad=False)
        self.num_clusters = num_clusters

        self.fc = nn.Linear(in_features, out_features, bias=bias)
        self._activation = activation(inplace=False) if activation else None

    def forward(self, x, cluster_info=None):
        if self.training:
            s, z = calc_qparams(torch.min(self.fc.weight), torch.max(self.fc.weight), self.q_max)
            with torch.no_grad():
                self.fc.weight.copy_(fake_quantize(self.fc.weight.detach(), s, z, self.q_max))

        x = self.fc(x)
        if self._activation:
            x = self._activation(x)

        if self.training:
            done = 0
            for i in range(cluster_info.shape[0]):
                c = cluster_info[i][0].item()
                n = cluster_info[i][1].item()
                if self.flag_ema_init[c]:
                    self.act_range[c][0], self.act_range[c][1] = ema(x[done:done + n], self.act_range[c], self.smooth)
                    if self.flag_fake_quantization:
                        s, z = calc_qparams(self.act_range[c][0], self.act_range[c][1], self.q_max)
                        with torch.no_grad():
                            x[done:done + n].copy_(fake_quantize(x[done:done + n].detach(), s, z, self.q_max))
                else:
                    self.act_range[c][0] = torch.min(x[done:done + n]).item()
                    self.act_range[c][1] = torch.max(x[done:done + n]).item()
                    self.flag_ema_init[c] = True
                done += n
        return x

    def set_fake_quantization_flag(self):
        self.flag_fake_quantization = True

    def set_qparams(self, s1, z1):
        self.s1, self.z1 = torch.nn.Parameter(s1, requires_grad=False), torch.nn.Parameter(z1, requires_grad=False)

        self.s2, self.z2 = calc_qparams(torch.min(self.fc.weight), torch.max(self.fc.weight), self.q_max)

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
    def __init__(self, in_features, out_features, bias=True, activation=None, bit=32, smooth=0.995):
        super(FusedLinear, self).__init__()
        self.layer_type = 'FusedLinear'
        self.bit = bit
        self.q_max = 2 ** bit - 1
        self.smooth = smooth
        self.flag_ema_init = False
        self.flag_fake_quantization = False
        self.act_range = nn.Parameter(torch.zeros(2), requires_grad=False)

        self.fc = nn.Linear(in_features, out_features, bias=bias)
        self._activation = activation(inplace=False) if activation else None

    def forward(self, x):
        if self.training:
            s, z = calc_qparams(torch.min(self.fc.weight), torch.max(self.fc.weight), self.q_max)
            with torch.no_grad():
                self.fc.weight.copy_(fake_quantize(self.fc.weight.detach(), s, z, self.q_max))

        x = self.fc(x)
        if self._activation:
            x = self._activation(x)

        if self.training:
            if self.flag_ema_init:
                self.act_range[0], self.act_range[1] = ema(x, self.act_range, self.smooth)
                if self.flag_fake_quantization:
                    s, z = calc_qparams(self.act_range[0], self.act_range[1], self.q_max)
                    with torch.no_grad():
                        x.copy_(fake_quantize(x.detach(), s, z, self.q_max))
            else:
                self.act_range[0] = torch.min(x).item()
                self.act_range[1] = torch.max(x).item()
                self.flag_ema_init = True
        return x

    def set_fake_quantization_flag(self):
        self.flag_fake_quantization = True

    def set_qparams(self, s1, z1):
        self.s1, self.z1 = torch.nn.Parameter(s1, requires_grad=False), torch.nn.Parameter(z1, requires_grad=False)
        self.s2, self.z2 = calc_qparams(torch.min(self.fc.weight), torch.max(self.fc.weight), self.q_max)
        self.s3, self.z3 = calc_qparams(self.act_range[0], self.act_range[1], self.q_max)
        self.M0, self.shift = quantize_M(self.s1 * self.s2 / self.s3)
        return self.s3, self.z3

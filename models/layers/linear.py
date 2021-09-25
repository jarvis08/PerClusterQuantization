from operator import itemgetter

import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

from ..quant_noise import _quant_noise
from ..quantization_utils import *
from .activation import *


class QuantizedLinear(nn.Linear):
    batch_cluster = None

    def __init__(self, in_features, out_features, bias=False, activation=None, arg_dict=None):
        super(QuantizedLinear, self).__init__(in_features, out_features, bias)
        self.layer_type = 'QuantizedLinear'
        self.bit, self.num_clusters, self.runtime_helper =\
                itemgetter('bit', 'cluster', 'runtime_helper')(arg_dict)
        self.q_max = 2 ** self.bit - 1
        self.act_qmax = nn.Parameter(torch.tensor(0, dtype=torch.int32), requires_grad=False)
        self.out_features = out_features

        self.is_bias = nn.Parameter(torch.tensor(False, dtype=torch.bool), requires_grad=False)
        self.quantized_bias = nn.Parameter(torch.zeros((self.num_clusters, out_features)), requires_grad=False)

        t_init = list(range(self.num_clusters)) if self.num_clusters > 1 else 0
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
        sum_q1q2 = F.linear(x, self.weight, None)
        if self.runtime_helper.batch_cluster is not None:
            return self.pcq_totalsum(x, sum_q1q2.type(torch.cuda.IntTensor))
        else:
            return self.general_totalsum(x, sum_q1q2.type(torch.cuda.IntTensor))

    def pcq_totalsum(self, x, sum_q1q2):
        bc = self.runtime_helper.batch_cluster
        z1 = torch.index_select(self.z1, 0, bc)
        z3 = torch.index_select(self.z3, 0, bc).reshape(bc.shape[0], 1)
        M0 = torch.index_select(self.M0, 0, bc).reshape(bc.shape[0], 1)
        shift = torch.index_select(self.shift, 0, bc).reshape(bc.shape[0], 1)

        input_feature, output_feature = sum_q1q2.shape[0], sum_q1q2.shape[1]
        N = x.shape[1]

        if self.is_bias:
            bias = torch.index_select(self.quantized_bias, 0, bc)
            for out_f in range(output_feature):
                sum_q1q2[:, out_f] = sum_q1q2[:, out_f].add_(bias[:, out_f].type(torch.cuda.IntTensor))

        sum_a1 = torch.zeros(input_feature, dtype=torch.int32).cuda()
        sum_a2 = torch.zeros((bc.shape[0], output_feature), dtype=torch.int32).cuda()

        for out_f in range(output_feature):
            sum_a2[:, out_f] = torch.sum(self.weight[out_f, :]).mul(z1)

        for in_f in range(input_feature):
            sum_a1[in_f] = torch.sum(x[in_f, :]).mul(self.z2)

        z1 = z1.reshape(bc.shape[0], 1)
        nz1z2 = N * z1 * self.z2
        sum_q1q2 = sum_q1q2.add_(nz1z2.type(torch.cuda.IntTensor))

        for in_f in range(input_feature):
            sum_q1q2[in_f, :] = torch.sub(sum_q1q2[in_f, :], sum_a1[in_f])

        for out_f in range(output_feature):
            sum_q1q2[:, out_f] = torch.sub(sum_q1q2[:, out_f], sum_a2[:, out_f])

        multiplied = multiply_M(sum_q1q2.type(torch.cuda.LongTensor), M0)
        total = shifting2d(multiplied, shift)
        total = total.add(z3)

        if self.act_qmax == 15:
            total = torch.clamp(total, 0, 15)
        elif self.act_qmax == 255:
            total = torch.clamp(total, -128, 127)
        elif self.act_qmax == 65535:  # INT 16
            total = torch.clamp(total, -32768, 32767)
        elif self.act_qmax == 4294967295:  # INT 32
            total = torch.clamp(total, -2147483648, 2147483647)
        return total.type(torch.cuda.FloatTensor)

    def general_totalsum(self, x, sum_q1q2):
        input_feature, output_feature = sum_q1q2.shape[0], sum_q1q2.shape[1]

        if self.is_bias:
            for out_f in range(output_feature):
                sum_q1q2[:, out_f] = sum_q1q2[:, out_f].add(self.quantized_bias[0][out_f])
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

        if self.shift < 0:
            multiplied = multiply_M((sub_sum.type(torch.cuda.LongTensor) << - self.shift.item()), self.M0)
            total = shifting(multiplied, 0)
        else:
            multiplied = multiply_M(sub_sum.type(torch.cuda.LongTensor), self.M0)
            total = shifting(multiplied, self.shift.item())
        total = total.add(self.z3)

        if self.activation is not None:
            hs_total = total + self.hardswish_3
            hs_total = torch.clamp(hs_total, self.z3.item(), self.hardswish_6.item())
            hs_total = hs_total / self.hardswish_6
            if self.activation == 'Hardswish':
                total = total * hs_total
            else:
                total = hs_total

        if self.act_qmax == 15:
            total = torch.clamp(total, 0, 15)
        elif self.act_qmax == 255:
            total = torch.clamp(total, -128, 127)
        elif self.act_qmax == 65535:  # INT 16
            total = torch.clamp(total, -32768, 32767)
        elif self.act_qmax == 4294967295:  # INT 32
            total = torch.clamp(total, -2147483648, 2147483647)
        return total.type(torch.cuda.FloatTensor)


class PCQLinear(nn.Module):
    """
        Fused Layer to calculate Quantization Parameters(S & Z) with multiple clusters
    """
    def __init__(self, in_features, out_features, bias=True, activation=None, act_qmax=None, arg_dict=None):
        super(PCQLinear, self).__init__()
        self.layer_type = 'PCQLinear'
        
        self.bit, self.smooth, self.num_clusters, self.runtime_helper, self.use_ste, self.quant_noise, self.qn_prob\
            = itemgetter('bit', 'smooth', 'cluster', 'runtime_helper', 'ste', 'quant_noise', 'qn_prob')(arg_dict)
        self.q_max = 2 ** self.bit - 1
        self.act_qmax = act_qmax if act_qmax else self.q_max
        self.act_range = nn.Parameter(torch.zeros((self.num_clusters, 2)), requires_grad=False)
        self.apply_ema = nn.Parameter(torch.zeros(self.num_clusters, dtype=torch.bool), requires_grad=False)

        self.fc = nn.Linear(in_features, out_features, bias=bias)
        self._activation = activation(inplace=False) if activation else None

    def forward(self, x, external_range=None):
        if not self.training:
            return self._forward_impl(x)

        out = self._pcq(x)
        self._update_activation_ranges(out, external_range)
        if self.runtime_helper.apply_fake_quantization:
            out = self._fake_quantize_activation(out, external_range)
        return out

    def _forward_impl(self, x):
        x = self.fc(x)
        if self._activation:
            x = self._activation(x)
        return x

    def _pcq(self, x):
        s, z = calc_qparams(self.fc.weight.min(), self.fc.weight.max(), self.q_max)
        if not self.quant_noise:
            w = fake_quantize(self.fc.weight, s, z, self.q_max, use_ste=self.use_ste)
        else:
            w = apply_qn(self.fc.weight, s, z, self.q_max, qn_prob=self.qn_prob)

        out = F.linear(x, w, self.fc.bias)
        if self._activation:
            out = self._activation(out)
        return out

    def _update_activation_ranges(self, x, external_range=None):
        if external_range is not None:
            return None

        cluster = self.runtime_helper.batch_cluster
        if self.apply_ema[cluster]:
            self.act_range[cluster][0], self.act_range[cluster][1] = ema(x, self.act_range[cluster], self.smooth)
        else:
            self.act_range[cluster][0] = torch.min(x).item()
            self.act_range[cluster][1] = torch.max(x).item()
            self.apply_ema[cluster] = True

    def _fake_quantize_activation(self, x, external_range=None):
        cluster = self.runtime_helper.batch_cluster
        if external_range is not None:
            s, z = calc_qparams(external_range[cluster][0], external_range[cluster][1], self.act_qmax)
        else:
            s, z = calc_qparams(self.act_range[cluster][0], self.act_range[cluster][1], self.act_qmax)
        return fake_quantize(x, s, z, self.act_qmax, use_ste=self.use_ste)

    def set_qparams(self, s1, z1, s_external=None, z_external=None):
        self.s1, self.z1 = torch.nn.Parameter(s1, requires_grad=False), torch.nn.Parameter(z1, requires_grad=False)
        self.s2, self.z2 = calc_qparams(torch.min(self.fc.weight), torch.max(self.fc.weight), self.q_max)

        if s_external:
            self.s3, self.z3 = nn.Parameter(s_external, requires_grad=False),\
                               nn.Parameter(z_external, requires_grad=False)
        else:
            self.s3, self.z3 = calc_qparams_per_cluster(self.act_range, self.act_qmax)

        self.M0 = nn.Parameter(torch.zeros(self.num_clusters, dtype=torch.int32), requires_grad=False)
        self.shift = nn.Parameter(torch.zeros(self.num_clusters, dtype=torch.int32), requires_grad=False)
        for c in range(self.num_clusters):
            self.M0[c], self.shift[c] = quantize_M(self.s1[c] * self.s2 / self.s3[c])
        return self.s3, self.z3


class FusedLinear(nn.Module):
    """
        Fused Layer to calculate Quantization Parameters (S & Z)
    """
    def __init__(self, in_features, out_features, bias=True, activation=None, act_qmax=None, arg_dict=None):
        super(FusedLinear, self).__init__()
        self.layer_type = 'FusedLinear'

        self.arg_dict = arg_dict
        self.bit, self.smooth, self.use_ste, self.runtime_helper, self.quant_noise, self.qn_prob \
            = itemgetter('bit', 'smooth', 'ste', 'runtime_helper', 'quant_noise', 'qn_prob')(arg_dict)
        self.q_max = 2 ** self.bit - 1
        self.act_qmax = act_qmax if act_qmax else 2 ** self.bit - 1
        self.act_range = nn.Parameter(torch.zeros(2), requires_grad=False)
        self.apply_ema = False

        self.fc = nn.Linear(in_features, out_features, bias=bias)
        self._activation = activation(inplace=False) if activation else None

    def forward(self, x):
        if not self.training:
            x = self.fc(x)
            if self._activation:
                x = self._activation(x)
            return x

        w = self.fc.weight
        s, z = calc_qparams(self.fc.weight.min(), self.fc.weight.max(), self.q_max)
        if not self.quant_noise:
            w = fake_quantize(self.fc.weight, s, z, self.q_max, self.use_ste)
        else:
            w = apply_qn(self.fc.weight, s, z, self.q_max, qn_prob=self.qn_prob)

        x = F.linear(x, w, self.fc.bias)
        if self._activation:
            x = self._activation(x)

        out = x
        if self.apply_ema:
            self.act_range[0], self.act_range[1] = ema(x, self.act_range, self.smooth)
            if self.runtime_helper.apply_fake_quantization:
                s, z = calc_qparams(self.act_range[0], self.act_range[1], self.act_qmax)
                out = fake_quantize(x, s, z, self.act_qmax, self.use_ste)
        else:
            self.act_range[0] = torch.min(x).item()
            self.act_range[1] = torch.max(x).item()
            self.apply_ema = True
        return out

    def set_qparams(self, s1, z1, s_external=None, z_external=None):
        self.s1, self.z1 = torch.nn.Parameter(s1, requires_grad=False), torch.nn.Parameter(z1, requires_grad=False)
        self.s2, self.z2 = calc_qparams(self.fc.weight.min(), self.fc.weight.max(), self.q_max)

        if s_external:
            self.s3, self.z3 = nn.Parameter(s_external, requires_grad=False), nn.Parameter(z_external, requires_grad=False)
        else:
            self.s3, self.z3 = calc_qparams(self.act_range[0], self.act_range[1], self.act_qmax)

        self.M0, self.shift = quantize_M(self.s1 * self.s2 / self.s3)
        return self.s3, self.z3

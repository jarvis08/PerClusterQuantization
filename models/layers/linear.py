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
        bit, self.num_clusters, self.runtime_helper = itemgetter('bit', 'cluster', 'runtime_helper')(arg_dict)
        self.w_bit = nn.Parameter(torch.tensor(bit, dtype=torch.int8), requires_grad=False)
        self.a_bit = nn.Parameter(torch.tensor(bit, dtype=torch.int8), requires_grad=False)
        self.out_features = out_features

        self.is_bias = nn.Parameter(torch.tensor(False, dtype=torch.bool), requires_grad=False)
        self.quantized_bias = nn.Parameter(torch.zeros((self.num_clusters, out_features), dtype=torch.int32), requires_grad=False)

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
            return self.pcq_totalsum(x.type(torch.cuda.IntTensor), sum_q1q2.type(torch.cuda.IntTensor))
        else:
            return self.general_totalsum(x.type(torch.cuda.IntTensor), sum_q1q2.type(torch.cuda.IntTensor))

    def pcq_totalsum(self, x, sum_q1q2):
        bc = self.runtime_helper.batch_cluster
        z1 = torch.index_select(self.z1, 0, bc)
        z3 = torch.index_select(self.z3, 0, bc).reshape(bc.shape[0], 1)
        M0 = torch.index_select(self.M0, 0, bc).reshape(bc.shape[0], 1)
        shift = torch.index_select(self.shift, 0, bc).reshape(bc.shape[0], 1)

        if self.is_bias:
            bias = torch.index_select(self.quantized_bias, 0, bc)
            sum_q1q2.add_(bias)

        sum_a1 = torch.sum(x, dim=1).mul(self.z2)
        sum_a2 = torch.sum(self.weight, dim=1).view(1, -1).repeat(x.size(0), 1).mul(z1[:, None])

        nz1z2 = x.size(1) * z1 * self.z2
        sum_q1q2 = sum_q1q2.add(nz1z2[:, None])
        sum_q1q2 = sum_q1q2.sub(sum_a1[:, None])
        sum_q1q2 = sum_q1q2.sub(sum_a2)

        multiplied = multiply_M(sum_q1q2.type(torch.cuda.LongTensor), M0)
        total = shifting2d(multiplied, shift)
        total = total.add(z3)

        if self.a_bit == 4:
            total = torch.clamp(total, 0, 15)
        elif self.a_bit == 8:
            total = torch.clamp(total, -128, 127)
        elif self.a_bit == 16:
            total = torch.clamp(total, -32768, 32767)
        elif self.a_bit == 32:
            total = torch.clamp(total, -2147483648, 2147483647)
        return total.type(torch.cuda.FloatTensor)

    def general_totalsum(self, x, sum_q1q2):
        if self.is_bias:
            sum_q1q2.add_(self.quantized_bias[0][None, :])

        sum_a1 = torch.sum(x, dim=1).mul(self.z2)
        sum_a2 = torch.sum(self.weight, dim=1).mul(self.z1)

        nz1z2 = x.size(1) * self.z1 * self.z2
        sum_q1q2 = sum_q1q2.add(nz1z2)
        sum_q1q2 = sum_q1q2.sub(sum_a1[:, None])
        sum_q1q2 = sum_q1q2.sub(sum_a2[None, :])

        if self.shift < 0:
            multiplied = multiply_M((sum_q1q2.type(torch.cuda.LongTensor) << - self.shift.item()), self.M0)
            total = shifting(multiplied, 0)
        else:
            multiplied = multiply_M(sum_q1q2.type(torch.cuda.LongTensor), self.M0)
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

        if self.a_bit == 4:
            total = torch.clamp(total, 0, 15)
        elif self.a_bit == 8:
            total = torch.clamp(total, -128, 127)
        elif self.a_bit == 16:
            total = torch.clamp(total, -32768, 32767)
        elif self.a_bit == 32:
            total = torch.clamp(total, -2147483648, 2147483647)
        return total.type(torch.cuda.FloatTensor)


class PCQLinear(nn.Module):
    """
        Fused Layer to calculate Quantization Parameters(S & Z) with multiple clusters
    """
    def __init__(self, in_features, out_features, bias=True, activation=None, is_classifier=False,
                 w_bit=None, a_bit=None, arg_dict=None):
        super(PCQLinear, self).__init__()
        self.layer_type = 'PCQLinear'
        
        bit, self.smooth, self.num_clusters, self.runtime_helper, self.use_ste, self.quant_noise, self.qn_prob\
            = itemgetter('bit', 'smooth', 'cluster', 'runtime_helper', 'ste', 'quant_noise', 'qn_prob')(arg_dict)

        w_bit = w_bit if w_bit is not None else bit
        a_bit = a_bit if a_bit is not None else bit
        self.w_bit = torch.nn.Parameter(torch.tensor(w_bit, dtype=torch.int8), requires_grad=False)
        self.a_bit = torch.nn.Parameter(torch.tensor(a_bit, dtype=torch.int8), requires_grad=False)

        self.act_range = nn.Parameter(torch.zeros((self.num_clusters, 2)), requires_grad=False)
        self.apply_ema = nn.Parameter(torch.zeros(self.num_clusters, dtype=torch.bool), requires_grad=False)

        self.fc = nn.Linear(in_features, out_features, bias=bias)
        self._activation = activation(inplace=True) if activation else None

    def forward(self, x, external_range=None):
        if not self.training:
            return self._forward_impl(x)

        out = self._pcq(x)
        if external_range is None:
            self._update_activation_ranges(out)
        if self.runtime_helper.apply_fake_quantization:
            out = self._fake_quantize_activation(out, external_range)
        return out

    def _forward_impl(self, x):
        x = self.fc(x)
        if self._activation:
            x = self._activation(x)
        return x

    def _pcq(self, x):
        s, z = calc_qparams(self.fc.weight.detach().min(), self.fc.weight.detach().max(), self.w_bit)
        if not self.quant_noise:
            w = fake_quantize(self.fc.weight, s, z, self.w_bit, use_ste=self.use_ste)
        else:
            w = apply_qn(self.fc.weight, s, z, self.w_bit, qn_prob=self.qn_prob)

        out = F.linear(x, w, self.fc.bias)
        if self._activation:
            out = self._activation(out)
        return out

    @torch.no_grad()
    def _update_activation_ranges(self, x):
        cluster = self.runtime_helper.batch_cluster
        if self.apply_ema[cluster]:
            self.act_range[cluster][0], self.act_range[cluster][1] = ema(x, self.act_range[cluster], self.smooth)
        else:
            self.act_range[cluster][0], self.act_range[cluster][1] = x.min(), x.max()
            self.apply_ema[cluster] = True

    def _fake_quantize_activation(self, x, external_range=None):
        cluster = self.runtime_helper.batch_cluster
        if external_range is not None:
            s, z = calc_qparams(external_range[cluster][0], external_range[cluster][1], self.a_bit)
        else:
            s, z = calc_qparams(self.act_range[cluster][0], self.act_range[cluster][1], self.a_bit)
        return fake_quantize(x, s, z, self.a_bit, use_ste=self.use_ste)

    @torch.no_grad()
    def set_qparams(self, s1, z1, s_external=None, z_external=None):
        self.s1, self.z1 = s1, z1
        self.s2, self.z2 = calc_qparams(self.fc.weight.min(), self.fc.weight.max(), self.w_bit)

        if s_external is not None:
            self.s3, self.z3 = s_external, z_external
        else:
            self.s3, self.z3 = calc_qparams_per_cluster(self.act_range, self.a_bit)

        self.M0 = torch.zeros(self.num_clusters, dtype=torch.int32)
        self.shift = torch.zeros(self.num_clusters, dtype=torch.int32)
        for c in range(self.num_clusters):
            self.M0[c], self.shift[c] = quantize_M(self.s1[c] * self.s2 / self.s3[c])
        return self.s3, self.z3


class FusedLinear(nn.Module):
    """
        Fused Layer to calculate Quantization Parameters (S & Z)
    """
    def __init__(self, in_features, out_features, bias=True, activation=None, is_classifier=False,
                 w_bit=None, a_bit=None, arg_dict=None):
        super(FusedLinear, self).__init__()
        self.layer_type = 'FusedLinear'

        self.arg_dict = arg_dict
        bit, self.smooth, self.use_ste, self.runtime_helper, self.quant_noise, self.qn_prob \
            = itemgetter('bit', 'smooth', 'ste', 'runtime_helper', 'quant_noise', 'qn_prob')(arg_dict)

        w_bit = w_bit if w_bit is not None else bit
        a_bit = a_bit if a_bit is not None else bit
        self.w_bit = torch.nn.Parameter(torch.tensor(w_bit, dtype=torch.int8), requires_grad=False)
        self.a_bit = torch.nn.Parameter(torch.tensor(a_bit, dtype=torch.int8), requires_grad=False)

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

        s, z = calc_qparams(self.fc.weight.detach().min(), self.fc.weight.detach().max(), self.w_bit)
        if not self.quant_noise:
            w = fake_quantize(self.fc.weight, s, z, self.w_bit, self.use_ste)
        else:
            w = apply_qn(self.fc.weight, s, z, self.w_bit, qn_prob=self.qn_prob)

        out = F.linear(x, w, self.fc.bias)
        if self._activation:
            out = self._activation(out)

        if self.apply_ema:
            self.act_range[0], self.act_range[1] = ema(out, self.act_range, self.smooth)
            if self.runtime_helper.apply_fake_quantization:
                s, z = calc_qparams(self.act_range[0], self.act_range[1], self.a_bit)
                out = fake_quantize(out, s, z, self.a_bit, self.use_ste)
        else:
            self.act_range[0], self.act_range[1] = get_range(out)
            self.apply_ema = True
        return out

    def set_qparams(self, s1, z1, s_external=None, z_external=None):
        self.s1, self.z1 = s1, z1
        self.s2, self.z2 = calc_qparams(self.fc.weight.min(), self.fc.weight.max(), self.w_bit)

        if s_external is not None:
            self.s3, self.z3 = s_external, z_external
        else:
            self.s3, self.z3 = calc_qparams(self.act_range[0], self.act_range[1], self.a_bit)

        self.M0, self.shift = quantize_M(self.s1 * self.s2 / self.s3)
        return self.s3, self.z3

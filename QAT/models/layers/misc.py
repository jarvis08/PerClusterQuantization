from operator import itemgetter

import torch.nn as nn
from ..quantization_utils import *


class QuantizedAdd(nn.Module):
    def __init__(self, arg_dict=None):
        super(QuantizedAdd, self).__init__()
        self.layer_type = 'QuantizedAdd'
        self.num_clusters, self.runtime_helper = itemgetter('cluster', 'runtime_helper')(arg_dict)
        self.a_bit = nn.Parameter(torch.tensor(0, dtype=torch.int8), requires_grad=False)

        t_init = list(range(self.num_clusters)) if self.num_clusters > 1 else 0
        self.z_bypass = nn.Parameter(torch.tensor(t_init, dtype=torch.int32), requires_grad=False)
        self.z_prev = nn.Parameter(torch.tensor(t_init, dtype=torch.int32), requires_grad=False)
        self.M0_bypass = nn.Parameter(torch.tensor(t_init, dtype=torch.int32), requires_grad=False)
        self.M0_prev = nn.Parameter(torch.tensor(t_init, dtype=torch.int32), requires_grad=False)
        self.shift_bypass = nn.Parameter(torch.tensor(t_init, dtype=torch.int32), requires_grad=False)
        self.is_bypass_shift_neg = nn.Parameter(torch.tensor(False, dtype=torch.bool), requires_grad=False)
        self.is_prev_shift_neg = nn.Parameter(torch.tensor(False, dtype=torch.bool), requires_grad=False)
        self.shift_prev = nn.Parameter(torch.tensor(t_init, dtype=torch.int32), requires_grad=False)
        self.s_bypass = nn.Parameter(torch.tensor(t_init, dtype=torch.float32), requires_grad=False)
        self.s_prev = nn.Parameter(torch.tensor(t_init, dtype=torch.float32), requires_grad=False)
        self.s3 = nn.Parameter(torch.tensor(t_init, dtype=torch.float32), requires_grad=False)
        self.z3 = nn.Parameter(torch.tensor(t_init, dtype=torch.int32), requires_grad=False)

    def forward(self, bypass, prev):
        if self.num_clusters > 1:
            out = self._pcq_add(bypass, prev)
        else:
            out = self._general_add(bypass, prev)
        return clamp_matrix(out, self.a_bit)

    def _pcq_add(self, bypass, prev):
        bc = self.runtime_helper.qat_batch_cluster
        z_bypass = torch.index_select(self.z_bypass, 0, bc)[:, None, None, None]
        z_prev = torch.index_select(self.z_prev, 0, bc)[:, None, None, None]
        z3 = torch.index_select(self.z3, 0, bc)[:, None, None, None]
        M0_bypass = torch.index_select(self.M0_bypass, 0, bc)[:, None, None, None]
        M0_prev = torch.index_select(self.M0_prev, 0, bc)[:, None, None, None]
        shift_bypass = torch.index_select(self.shift_bypass, 0, bc)[:, None, None, None]
        shift_prev = torch.index_select(self.shift_prev, 0, bc)[:, None, None, None]
        shape = bypass.shape
        mask = self.runtime_helper.mask_4d[:shape[0]]

        bypass = bypass - z_bypass
        prev = prev - z_prev
        zero = self.runtime_helper.izero

        if not self.is_bypass_shift_neg:
            x1 = mul_and_shift(bypass, M0_bypass, shift_bypass, mask)
        else:
            neg_shift = torch.where(shift_bypass < zero, - shift_bypass, zero)
            shift = torch.where(shift_bypass >= zero, shift_bypass, zero)
            bypass = bypass << neg_shift
            x1 = mul_and_shift(bypass, M0_bypass, shift, mask)

        if not self.is_prev_shift_neg:
            x2 = mul_and_shift(prev, M0_prev, shift_prev, mask)
        else:
            neg_shift = torch.where(shift_prev < zero, - shift_prev, zero)
            shift = torch.where(shift_prev >= zero, shift_prev, zero)
            prev = prev << neg_shift
            x2 = mul_and_shift(prev, M0_prev, shift, mask)
        return (x1 + x2).add(z3)


    def _general_add(self, bypass, prev):
        bypass = bypass - self.z_bypass
        prev = prev - self.z_prev

        if self.shift_bypass < 0:
            x1 = mul_and_shift(bypass << -self.shift_bypass.item(), self.M0_bypass, 0)
        else:
            x1 = mul_and_shift(bypass, self.M0_bypass, self.shift_bypass.item())

        if self.shift_prev < 0:
            x2 = mul_and_shift(prev << -self.shift_prev.item(), self.M0_prev, 0)
        else:
            x2 = mul_and_shift(prev, self.M0_prev, self.shift_prev.item())
        return (x1 + x2).add(self.z3)


class QuantizedMul(nn.Module):
    def __init__(self, arg_dict=None):
        super(QuantizedMul, self).__init__()
        self.layer_type = 'QuantizedMul'
        self.a_bit, self.num_clusters, self.runtime_helper = itemgetter('bit', 'cluster', 'runtime_helper')(arg_dict)

        t_init = list(range(self.num_clusters)) if self.num_clusters > 1 else 0
        self.s_prev = nn.Parameter(torch.tensor(t_init, dtype=torch.float32), requires_grad=False)
        self.z_prev = nn.Parameter(torch.tensor(t_init, dtype=torch.int32), requires_grad=False)
        self.s_bypass = nn.Parameter(torch.tensor(t_init, dtype=torch.float32), requires_grad=False)
        self.z_bypass = nn.Parameter(torch.tensor(t_init, dtype=torch.int32), requires_grad=False)
        self.s3 = nn.Parameter(torch.tensor(t_init, dtype=torch.float32), requires_grad=False)
        self.z3 = nn.Parameter(torch.tensor(t_init, dtype=torch.int32), requires_grad=False)
        self.M0 = nn.Parameter(torch.tensor(t_init, dtype=torch.int32), requires_grad=False)
        self.shift = nn.Parameter(torch.tensor(t_init, dtype=torch.int32), requires_grad=False)

    def forward(self, prev, bypass):
        if self.runtime_helper.qat_batch_cluster is not None:
            return self.pcq_mul(bypass, prev)
        else:
            return self.general_mul(prev, bypass)

    def general_mul(self, prev, bypass):
        mul_q1q2 = torch.mul(prev, bypass)
        z1z2 = self.z_bypass * self.z_prev
        z2q1 = torch.mul(prev, self.z_bypass)
        z1q2 = torch.mul(bypass, self.z_prev)
        mul_q1q2 = torch.sub(mul_q1q2, z2q1)
        mul_q1q2 = torch.sub(mul_q1q2, z1q2)
        mul_q1q2 = mul_q1q2.add(z1z2)

        if self.shift < 0:
            multiplied = multiply_M((mul_q1q2.type(torch.cuda.LongTensor) << - self.shift.item()), self.M0)
            total = shifting(multiplied, 0)
        else:
            multiplied = multiply_M(mul_q1q2.type(torch.cuda.LongTensor), self.M0)
            total = shifting(multiplied, self.shift.item())

        total = total.add(self.z3)

        if self.a_bit == 4:
            total = torch.clamp(total, 0, 15)
        elif self.a_bit == 8:
            total = torch.clamp(total, -128, 127)
        elif self.a_bit == 16:
            total = torch.clamp(total, -32768, 32767)
        elif self.a_bit == 32:
            total = torch.clamp(total, -2147483648, 2147483647)
        return total


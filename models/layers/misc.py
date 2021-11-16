from operator import itemgetter

import torch.nn as nn
from ..quantization_utils import *


class QuantizedAdd(nn.Module):
    def __init__(self, arg_dict=None):
        super(QuantizedAdd, self).__init__()
        self.layer_type = 'QuantizedAdd'
        self.num_clusters, self.runtime_helper = itemgetter('cluster', 'runtime_helper')(arg_dict)
        self.a_bit = nn.Parameter(torch.tensor(0, dtype=torch.int8), requires_grad=False)
        self.total = None  # for faster inference

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
        if self.runtime_helper.batch_cluster is None:
            return self.general_bypass(bypass, prev)

        batch_size = bypass.size(0)
        bc = self.runtime_helper.batch_cluster
        z_bypass = torch.index_select(self.z_bypass, 0, bc)[:, None, None, None]
        z_prev = torch.index_select(self.z_prev, 0, bc)[:, None, None, None]
        z3 = torch.index_select(self.z3, 0, bc)[:, None, None, None]
        M0_bypass = torch.index_select(self.M0_bypass, 0, bc)[:, None, None, None]
        M0_prev = torch.index_select(self.M0_prev, 0, bc)[:, None, None, None]
        shift_bypass = torch.index_select(self.shift_bypass, 0, bc)[:, None, None, None]
        shift_prev = torch.index_select(self.shift_prev, 0, bc)[:, None, None, None]

        if not self.is_bypass_shift_neg:
            x1 = multiply_M((bypass.sub(z_bypass)), M0_bypass)
            x1 = shifting4d_without_cast(x1, shift_bypass,
                                         self.runtime_helper.mask_4d[:batch_size],
                                         self.runtime_helper.zero_4d[:batch_size],
                                         self.runtime_helper.one_4d[:batch_size])
        else:
            x1 = self._pcq_with_negative_value(bypass, z_bypass, M0_bypass, shift_bypass)

        if not self.is_prev_shift_neg:
            x2 = multiply_M((prev.sub(z_prev)), M0_prev)
            x2 = shifting4d_without_cast(x2, shift_prev,
                                         self.runtime_helper.mask_4d[:batch_size],
                                         self.runtime_helper.zero_4d[:batch_size],
                                         self.runtime_helper.one_4d[:batch_size])
        else:
            x2 = self._pcq_with_negative_value(prev, z_prev, M0_prev, shift_prev)

        total = (x1 + x2).add(z3)
        if self.a_bit == 4:
            total = torch.clamp(total, 0, 15)
        elif self.a_bit == 8:
            total = torch.clamp(total, -128, 127)
        elif self.a_bit == 16:
            total = torch.clamp(total, -32768, 32767)
        elif self.a_bit == 32:
            total = torch.clamp(total, -2147483648, 2147483647)
        return total

    def _pcq_with_negative_value(self, x, z, M0, shift):
        _x = torch.zeros(x.shape, dtype=torch.int64, device='cuda')
        under = (shift < 0).nonzero(as_tuple=True)[0]
        over = (shift >= 0).nonzero(as_tuple=True)[0]
        n_over = len(over)
        if len(under) > 0:
            _shift = - shift[under]
            x_under = multiply_M((x[under].sub(z[under]) << _shift), M0[under])
            x_under = shifting_without_cast(x_under, 0)
            _x[under] = x_under
        if n_over > 0:
            _shift = shift[over]
            x_over = multiply_M((x[over].sub(z[over])), M0[over])
            x_over = shifting4d_without_cast(x_over, _shift,
                                             self.runtime_helper.mask_4d[:n_over],
                                             self.runtime_helper.zero_4d[:n_over],
                                             self.runtime_helper.one_4d[:n_over])
            _x[over] = x_over
        return _x

    def general_bypass(self, bypass, prev):
        if self.shift_bypass < 0:
            x1 = multiply_M((bypass.sub(self.z_bypass) << - self.shift_bypass), self.M0_bypass)
            x1 = shifting(x1, 0)
        else:
            x1 = multiply_M(bypass.sub(self.z_bypass), self.M0_bypass)
            x1 = shifting(x1, self.shift_bypass.item())
        
        if self.shift_prev < 0:
            x2 = multiply_M((prev.sub(self.z_prev) << - self.shift_prev), self.M0_prev)
            x2 = shifting(x2, 0)
        else:
            x2 = multiply_M(prev.sub(self.z_prev), self.M0_prev)
            x2 = shifting(x2, self.shift_prev.item())

        total = (x1 + x2).add(self.z3)
        if self.a_bit == 4:
            total = torch.clamp(total, 0, 15)
        elif self.a_bit == 8:
            total = torch.clamp(total, -128, 127)
        elif self.a_bit == 16:
            total = torch.clamp(total, -32768, 32767)
        elif self.a_bit == 32:
            total = torch.clamp(total, -2147483648, 2147483647)
        return total


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
        if self.runtime_helper.batch_cluster is not None:
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


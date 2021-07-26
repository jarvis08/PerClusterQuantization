import torch.nn as nn
import torch
import torch.nn.functional as F

from ..quant_noise import _quant_noise
from ..quantization_utils import *
from .activation import *
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Sequence


class QuantizedMul(nn.Module):
    batch_cluster = None

    def __init__(self, bit=8, num_clusters=1):
        super(QuantizedMul, self).__init__()
        self.layer_type = 'QuantizedMul'
        self.bit = bit
        self.q_max = 2 ** bit - 1
        self.num_clusters = num_clusters
        t_init = list(range(num_clusters)) if num_clusters > 1 else 0
        self.s_prev = nn.Parameter(torch.tensor(t_init, dtype=torch.float32), requires_grad=False)
        self.z_prev = nn.Parameter(torch.tensor(t_init, dtype=torch.int32), requires_grad=False)
        self.s_bypass = nn.Parameter(torch.tensor(t_init, dtype=torch.float32), requires_grad=False)
        self.z_bypass = nn.Parameter(torch.tensor(t_init, dtype=torch.int32), requires_grad=False)
        self.s3 = nn.Parameter(torch.tensor(t_init, dtype=torch.float32), requires_grad=False)
        self.z3 = nn.Parameter(torch.tensor(t_init, dtype=torch.int32), requires_grad=False)
        self.M0 = nn.Parameter(torch.tensor(t_init, dtype=torch.int32), requires_grad=False)
        self.shift = nn.Parameter(torch.tensor(t_init, dtype=torch.int32), requires_grad=False)

    def forward(self, bypass, prev):
        if self.batch_cluster is not None:
            return self.pcq_mul(bypass, prev)
        else:
            return self.general_mul(bypass, prev)

    def general_mul(self, bypass, prev):
        print(prev.shape)
        print(bypass.shape)
        print((bypass*prev).shape)
        # mul_q1q2 = torch.mul(bypass, prev).type(torch.cuda.IntTensor)
        return

    # def general_totalsum(self, x, sum_q1q2):
    #     input_batch, input_ch, input_col, input_row = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
    #     filter_batch, filter_ch, filter_col, filter_row = self.weight.shape[0], self.weight.shape[1], self.weight.shape[
    #         2], self.weight.shape[3]
    #     stride = self.stride[0]
    #
    #     for output_ch in range(filter_batch):
    #         sum_q1q2[:, output_ch, :, :] = sum_q1q2[:, output_ch, :, :].add(self.quantized_bias[0][output_ch])
    #
    #     output_col = sum_q1q2.shape[2]
    #     output_row = sum_q1q2.shape[3]
    #     sum_a1 = torch.zeros((input_batch, output_col, output_row), dtype=torch.int32).cuda()
    #     sum_a2 = torch.zeros(filter_batch, dtype=torch.int32).cuda()
    #
    #     for output_ch in range(0, filter_batch):
    #         sum_a2[output_ch] = torch.sum(self.weight.data[output_ch, :, :, :]).mul(self.z1)
    #
    #     for o_col in range(0, output_col):
    #         for o_row in range(0, output_row):
    #             col_st, col_end = o_col * stride, o_col * stride + filter_col
    #             row_st, row_end = o_row * stride, o_row * stride + filter_row
    #             sum_a1[:, o_col, o_row] = torch.sum(x[:, :, col_st: col_end, row_st: row_end], (1, 2, 3)).mul(self.z2)
    #     nz1z2 = input_ch * filter_col * filter_row * self.z1 * self.z2
    #     sum_q1q2 = sum_q1q2.add(nz1z2)
    #
    #     for i_batch in range(0, input_batch):
    #         sum_q1q2[i_batch, :] = torch.sub(sum_q1q2[i_batch, :], sum_a1[i_batch])
    #     for out_c in range(0, filter_batch):
    #         sum_q1q2[:, out_c] = torch.sub(sum_q1q2[:, out_c], sum_a2[out_c])
    #
    #     if self.shift < 0:
    #         multiplied = multiply_M((sum_q1q2.type(torch.cuda.LongTensor) << - self.shift.item()), self.M0)
    #         total = shifting(multiplied, 0)
    #     else:
    #         multiplied = multiply_M(sum_q1q2.type(torch.cuda.LongTensor), self.M0)
    #         total = shifting(multiplied, self.shift.item())
    #     total = total.add(self.z3)
    #
    #     if self.activation is not None:
    #         hs_total = total + self.hardswish_3
    #         hs_total = torch.clamp(hs_total, self.z3.item(), self.hardswish_6.item())
    #         hs_total = hs_total / self.hardswish_6
    #         if self.activation == 'Hardswish':
    #             total = total * hs_total
    #         else:
    #             total = hs_total
    #
    #     if self.bit == 4:
    #         total = torch.clamp(total, 0, 15)
    #     else:
    #         total = torch.clamp(total, -128, 127)
    #     return total.type(torch.cuda.FloatTensor)
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
        # prev    [128, 16, 56, 56]
        # bypass  [128, 16,  1,  1]
        mul_q1q2 = torch.mul(prev, bypass).type(torch.cuda.IntTensor)
        z1z2 = self.z_bypass * self.z_prev
        z2q1 = torch.mul(prev, self.z_bypass)
        z1q2 = torch.mul(bypass, self.z_prev)
        mul_q1q2 = torch.sub(mul_q1q2, z2q1)
        mul_q1q2 = torch.sub(mul_q1q2, z1q2)

        if self.shift < 0:
            multiplied = multiply_M((mul_q1q2.type(torch.cuda.LongTensor) << - self.shift.item()), self.M0)
            total = shifting(multiplied, 0)
        else:
            multiplied = multiply_M(mul_q1q2.type(torch.cuda.LongTensor), self.M0)
            total = shifting(multiplied, self.shift.item())

        total = total.add(z1z2)

        if self.bit == 4:
            out = torch.clamp(total, 0, 15)
        else:
            out = torch.clamp(total, -128, 127)
        return out.type(torch.cuda.FloatTensor)


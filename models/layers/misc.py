from operator import itemgetter

import torch.nn as nn
from ..quantization_utils import *


class QuantizedAdd(nn.Module):
    def __init__(self, arg_dict=None):
        super(QuantizedAdd, self).__init__()
        self.layer_type = 'QuantizedAdd'
        self.bit, self.num_clusters, self.runtime_helper = itemgetter('bit', 'cluster', 'runtime_helper')(arg_dict)
        self.q_max = 2 ** self.bit - 1

        t_init = list(range(self.num_clusters)) if self.num_clusters > 1 else 0
        self.z_bypass = nn.Parameter(torch.tensor(t_init, dtype=torch.int32), requires_grad=False)
        self.z_prev = nn.Parameter(torch.tensor(t_init, dtype=torch.int32), requires_grad=False)
        self.M0_bypass = nn.Parameter(torch.tensor(t_init, dtype=torch.int32), requires_grad=False)
        self.M0_prev = nn.Parameter(torch.tensor(t_init, dtype=torch.int32), requires_grad=False)
        self.shift_bypass = nn.Parameter(torch.tensor(t_init, dtype=torch.int32), requires_grad=False)
        self.shift_prev = nn.Parameter(torch.tensor(t_init, dtype=torch.int32), requires_grad=False)
        self.s_bypass = nn.Parameter(torch.tensor(t_init, dtype=torch.float32), requires_grad=False)
        self.s_prev = nn.Parameter(torch.tensor(t_init, dtype=torch.float32), requires_grad=False)
        self.s3 = nn.Parameter(torch.tensor(t_init, dtype=torch.float32), requires_grad=False)
        self.z3 = nn.Parameter(torch.tensor(t_init, dtype=torch.int32), requires_grad=False)

    def forward(self, bypass, prev):
        if self.runtime_helper.batch_cluster is not None:
            return self.pcq_bypass(bypass.type(torch.cuda.LongTensor), prev.type(torch.cuda.LongTensor))
        else:
            return self.general_bypass(bypass.type(torch.cuda.LongTensor), prev.type(torch.cuda.LongTensor))

    def pcq_bypass(self, bypass, prev):
        bc = self.runtime_helper.batch_cluster
        z_bypass = torch.index_select(self.z_bypass, 0, bc).reshape(bc.shape[0], 1, 1, 1)
        z_prev = torch.index_select(self.z_prev, 0, bc).reshape(bc.shape[0], 1, 1, 1)
        M0_bypass = torch.index_select(self.M0_bypass, 0, bc).reshape(bc.shape[0], 1, 1, 1)
        M0_prev = torch.index_select(self.M0_prev, 0, bc).reshape(bc.shape[0], 1, 1, 1)
        shift_bypass = torch.index_select(self.shift_bypass, 0, bc)
        shift_prev = torch.index_select(self.shift_prev, 0, bc)
        z3 = torch.index_select(self.z3, 0, bc).reshape(bc.shape[0], 1, 1, 1)

        out = torch.zeros(bypass.shape, dtype=torch.int32).cuda()
        x1_under = (shift_bypass < 0).nonzero(as_tuple=True)[0]
        x1_over = (shift_bypass >= 0).nonzero(as_tuple=True)[0]
        x2_under = (shift_prev < 0).nonzero(as_tuple=True)[0]
        x2_over = (shift_prev >= 0).nonzero(as_tuple=True)[0]
        if len(x1_under) > 0:
            shift = - shift_bypass[x1_under].reshape(x1_under.shape[0], 1, 1, 1)
            x1 = multiply_M((bypass[x1_under].sub(z_bypass[x1_under]) << shift), M0_bypass[x1_under])
            x1 = shifting(x1, 0)
            out[x1_under] = out[x1_under].add(x1)
        if len(x1_over) > 0:
            shift = shift_bypass[x1_over].reshape(x1_over.shape[0], 1, 1, 1)
            x1 = multiply_M((bypass[x1_over].sub(z_bypass[x1_over])), M0_bypass[x1_over])
            x1 = shifting4d(x1, shift)
            out[x1_over] = out[x1_over].add(x1)

        if len(x2_under) > 0:
            shift = - shift_prev[x2_under].reshape(x2_under.shape[0], 1, 1, 1)
            x2 = multiply_M((prev[x2_under].sub(z_prev[x2_under]) << shift), M0_prev[x2_under])
            x2 = shifting(x2, 0)
            out[x2_under] = out[x2_under].add(x2)
        if len(x2_over) > 0:
            shift = shift_prev[x2_over].reshape(x2_over.shape[0], 1, 1, 1)
            x2 = multiply_M((prev[x2_over].sub(z_prev[x2_over])), M0_prev[x2_over])
            x2 = shifting4d(x2, shift)
            out[x2_over] = out[x2_over].add(x2)
        out = out.add(z3)

        if self.bit == 4:
            out = torch.clamp(out, 0, 15)
        else:
            out = torch.clamp(out, -128, 127)
        return out.type(torch.cuda.FloatTensor)

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

        out = (x1 + x2).add(self.z3)
        if self.bit == 4:
            out = torch.clamp(out, 0, 15)
        else:
            out = torch.clamp(out, -128, 127)
        return out.type(torch.cuda.FloatTensor)


class QuantizedMul(nn.Module):
    def __init__(self, arg_dict=None):
        super(QuantizedMul, self).__init__()
        self.layer_type = 'QuantizedMul'
        self.bit, self.num_clusters, self.runtime_helper = itemgetter('bit', 'cluster', 'runtime_helper')(arg_dict)
        self.q_max = 2 ** self.bit - 1

        t_init = list(range(self.num_clusters)) if self.num_clusters > 1 else 0
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


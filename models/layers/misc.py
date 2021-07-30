from operator import itemgetter

import torch.nn as nn
from ..quantization_utils import *


class QuantizedAdd(nn.Module):
    def __init__(self, arg_dict=None):
        super(QuantizedAdd, self).__init__()
        self.layer_type = 'QuantizedAdd'
        self.bit, self.num_clusters, self.runtime_helper = itemgetter('bit', 'cluster', 'runtime_helper')(arg_dict)

        t_init = list(range(self.num_clusters)) if self.num_clusters > 1 else 0
        self.s_bypass = nn.Parameter(torch.tensor(t_init, dtype=torch.float32), requires_grad=False)
        self.z_bypass = nn.Parameter(torch.tensor(t_init, dtype=torch.int32), requires_grad=False)
        self.M0_bypass = nn.Parameter(torch.tensor(t_init, dtype=torch.int32), requires_grad=False)
        self.shift_bypass = nn.Parameter(torch.tensor(t_init, dtype=torch.int32), requires_grad=False)

        self.s_prev = nn.Parameter(torch.tensor(t_init, dtype=torch.float32), requires_grad=False)
        self.z_prev = nn.Parameter(torch.tensor(t_init, dtype=torch.int32), requires_grad=False)
        self.M0_prev = nn.Parameter(torch.tensor(t_init, dtype=torch.int32), requires_grad=False)
        self.shift_prev = nn.Parameter(torch.tensor(t_init, dtype=torch.int32), requires_grad=False)

        self.s3 = nn.Parameter(torch.tensor(t_init, dtype=torch.float32), requires_grad=False)
        self.z3 = nn.Parameter(torch.tensor(t_init, dtype=torch.int32), requires_grad=False)

    def forward(self, bypass, prev):
        """
        print(self.s_bypass, self.z_bypass, self.M0_bypass, self.shift_bypass)
        print(self.s_prev, self.z_prev, self.M0_prev, self.shift_prev)
        print(self.s3, self.z3)

        s_bypass: tensor([0.3055, 0.3989, 0.3090, 0.3779, 0.3845, 0.3871, 0.3393, 0.3533, 0.3569, 0.3536], device='cuda:0')
        z_bypass: tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], device='cuda:0', dtype=torch.int32)
        M0_bypass: tensor([1642439424, 2141534848, 1661240832, 2028621440, 2064485120, 2078014080,
        				   1821662464, 1896790528, 1916008704, 1898526208], device='cuda:0', dtype=torch.int32)
        shift_bypass: tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], device='cuda:0', dtype=torch.int32)
        
        s_prev: tensor([0.7295, 1.0316, 0.7124, 0.9565, 0.8385, 0.9024, 0.8156, 0.8553, 0.7992, 0.8327]
        z_prev: tensor([8, 8, 8, 8, 8, 8, 8, 8, 8, 8], device='cuda:0', dtype=torch.int32)
        M0_prev: tensor([1961123456, 1384570624, 1914815488, 1283744384, 1125567360, 1211163392,
                           1094875136, 1147960576, 2145643008, 1117711104], device='cuda:0', dtype=torch.int32)
        shift_bypass: tensor([-1, -2, -1, -2, -2, -2, -2, -2, -1, -2], device='cuda:0', dtype=torch.int32)

        s3: tensor([0.3994, 0.4000, 0.3995, 0.4000, 0.4000, 0.4000, 0.3999, 0.4000, 0.4000, 0.4000], device='cuda:0')
        z3: tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], device='cuda:0', dtype=torch.int32)
        exit()
        """
        if self.runtime_helper.batch_cluster is not None:
            return self.pcq_bypass(bypass.type(torch.cuda.LongTensor), prev.type(torch.cuda.LongTensor))
        else:
            return self.general_bypass(bypass.type(torch.cuda.LongTensor), prev.type(torch.cuda.LongTensor))

    def pcq_bypass(self, bypass, prev):
        done = 0
        out = torch.zeros(bypass.shape, dtype=torch.int32).cuda()
        for i in range(self.runtime_helper.batch_cluster.shape[0]):
            c = self.runtime_helper.batch_cluster[i][0].item()
            n = self.runtime_helper.batch_cluster[i][1].item()
            if self.shift_bypass[c] < 0:
                x1 = multiply_M((bypass[done:done + n].sub(self.z_bypass[c]) << - self.shift_bypass[c]), self.M0_bypass[c])
                x1 = shifting(x1, 0)
            else:
                x1 = multiply_M(bypass[done:done + n].sub(self.z_bypass[c]), self.M0_bypass[c])
                x1 = shifting(x1, self.shift_bypass[c].item())

            if self.shift_prev[c] < 0:
                x2 = multiply_M((prev[done:done + n].sub(self.z_prev[c]) << - self.shift_prev[c]), self.M0_prev[c])
                x2 = shifting(x2, 0)
            else:
                x2 = multiply_M(prev[done:done + n].sub(self.z_prev[c]), self.M0_prev[c])
                x2 = shifting(x2, self.shift_prev[c].item())

            out[done:done + n] = (x1 + x2).add(self.z3[c])
            done += n

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


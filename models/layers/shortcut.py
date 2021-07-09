import torch.nn as nn
from ..quantization_utils import *


class QuantizedShortcut(nn.Module):
    def __init__(self, bit=8, num_clusters=1):
        super(QuantizedShortcut, self).__init__()
        self.layer_type = 'QuantizedShortcut'
        self.num_clusters = num_clusters
        self.bit = bit
        t_init = list(range(num_clusters)) if num_clusters > 1 else 0
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

    def forward(self, bypass, prev, cluster_info):
        if cluster_info is not None:
            return self.pcq_bypass(bypass.type(torch.cuda.LongTensor), prev.type(torch.cuda.LongTensor), cluster_info)
        else:
            return self.general_bypass(bypass.type(torch.cuda.LongTensor), prev.type(torch.cuda.LongTensor))

    def pcq_bypass(self, bypass, prev, cluster_info):
        done = 0
        out = torch.zeros(bypass.shape, dtype=torch.int32).cuda()
        for i in range(cluster_info.shape[0]):
            c = cluster_info[i][0].item()
            n = cluster_info[i][1].item()
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

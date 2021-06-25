import torch.nn as nn
from quantization.quantization_utils import *


class QuantizedShortcut(nn.Module):
    def __init__(self, bit=8):
        super(QuantizedShortcut, self).__init__()
        self.layer_type = 'QuantizedShortcut'
        self.bit = bit
        self.s_bypass = nn.Parameter(torch.tensor(0, dtype=torch.float32), requires_grad=False)
        self.z_bypass = nn.Parameter(torch.tensor(0, dtype=torch.int32), requires_grad=False)
        self.M0_bypass = nn.Parameter(torch.tensor(0, dtype=torch.int32), requires_grad=False)
        self.shift_bypass = nn.Parameter(torch.tensor(0, dtype=torch.int32), requires_grad=False)

        self.s_prev = nn.Parameter(torch.tensor(0, dtype=torch.float32), requires_grad=False)
        self.z_prev = nn.Parameter(torch.tensor(0, dtype=torch.int32), requires_grad=False)
        self.M0_prev = nn.Parameter(torch.tensor(0, dtype=torch.int32), requires_grad=False)
        self.shift_prev = nn.Parameter(torch.tensor(0, dtype=torch.int32), requires_grad=False)

        self.s3 = nn.Parameter(torch.tensor(0, dtype=torch.float32), requires_grad=False)
        self.z3 = nn.Parameter(torch.tensor(0, dtype=torch.int32), requires_grad=False)

    def forward(self, bypass, prev):
        # bypass = bypass.type(torch.cuda.LongTensor)
        # prev = prev.type(torch.cuda.LongTensor)
        # if self.shift_bypass < 0:
        #     x1 = multiply_M((bypass.sub(self.z_bypass) << (- self.shift_bypass * 2)), self.M0_bypass)
        #     x1 = shifting(x1, - self.shift_bypass.item())
        # else:
        #     x1 = multiply_M(bypass.sub(self.z_bypass), self.M0_bypass)
        #     x1 = shifting(x1, self.shift_bypass.item())
        # x1 = x1.add(self.z3)
        #
        # if self.shift_prev < 0:
        #     x2 = multiply_M((prev.sub(self.z_prev) << (- self.shift_prev * 2)), self.M0_prev)
        #     x2 = shifting(x2, - self.shift_prev.item())
        # else:
        #     x2 = multiply_M(prev.sub(self.z_prev), self.M0_prev)
        #     x2 = shifting(x2, self.shift_prev.item())
        # x2 = x2.add(self.z3)
        # out = x1 + x2
        # if self.bit == 4:
        #     out = torch.clamp(out, 0, 15).type(torch.cuda.FloatTensor)
        # else:
        #     out = torch.clamp(out, -128, 127).type(torch.cuda.FloatTensor)
        # return out

        x1 = bypass.sub(self.z_bypass).mul(self.s_bypass / self.s3)
        x2 = prev.sub(self.z_prev).mul(self.s_prev / self.s3)
        out = (x1 + x2).add(self.z3)
        if self.bit == 4:
            out = torch.clamp(out, 0, 15)
        else:
            out = torch.clamp(out, -128, 127)
        return out


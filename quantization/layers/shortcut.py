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
        x1 = multiply_M(bypass.sub(self.z_bypass), self.M0_bypass)
        x2 = multiply_M(prev.sub(self.z_prev), self.M0_prev)
        x1 = shifting(x1, self.shift_bypass)
        x2 = shifting(x2, self.shift_prev)
        x1 = x1.add(self.z3)
        x2 = x2.add(self.z3)
        if self.bit == 4:
            x1 = torch.clamp(x1, 0, 15).type(torch.cuda.FloatTensor)
            x2 = torch.clamp(x2, 0, 15).type(torch.cuda.FloatTensor)
        else:
            x1 = torch.clamp(x1, -128, 127).type(torch.cuda.FloatTensor)
            x2 = torch.clamp(x2, -128, 127).type(torch.cuda.FloatTensor)
        return x1 + x2

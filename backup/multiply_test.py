import torch.nn as nn
import torch
import torch.nn.functional as F

from quantization_utils import *

s1 = nn.Parameter(torch.tensor(0, dtype=torch.float32), requires_grad=False)
s2 = nn.Parameter(torch.tensor(0, dtype=torch.float32), requires_grad=False)
s3 = nn.Parameter(torch.tensor(0, dtype=torch.float32), requires_grad=False)
z1 = nn.Parameter(torch.tensor(0, dtype=torch.int32), requires_grad=False)
z2 = nn.Parameter(torch.tensor(0, dtype=torch.int32), requires_grad=False)
z3 = nn.Parameter(torch.tensor(0, dtype=torch.int32), requires_grad=False)
M0 = nn.Parameter(torch.tensor(0, dtype=torch.int32), requires_grad=False).cuda()
shift = nn.Parameter(torch.tensor(0, dtype=torch.int32), requires_grad=False).cuda()

prev = torch.rand((3,2,3,3), dtype=torch.float).cuda()
bypass = torch.rand((3,2,1,1), dtype=torch.float).cuda()
out_float = prev * bypass
print(out_float.shape)

s1, z1 = calc_qparams(torch.min(prev), torch.max(prev), 15)
s2, z2 = calc_qparams(torch.min(bypass), torch.max(bypass), 15)
s3, z3 = calc_qparams(torch.min(out_float), torch.max(out_float), 15)

prev_q = quantize_matrix(prev, s1, z1, 15)
bypass_q = quantize_matrix(bypass, s2, z2, 15)
out_float_q = quantize_matrix(out_float, s3, z3, 15)

M0, shift = quantize_M(s1*s2/s3)

print(prev_q)
print(bypass_q)
mul_q1q2 = torch.mul(prev_q, bypass_q).type(torch.cuda.IntTensor)
print(mul_q1q2)
exit()
z1z2 = z2 * z1
z2q1 = torch.mul(prev, z2)
z1q2 = torch.mul(bypass, z1)
mul_q1q2 = torch.sub(mul_q1q2, z2q1)
mul_q1q2 = torch.sub(mul_q1q2, z1q2)
mul_q1q2 = mul_q1q2.add(z1z2)

if shift < 0:
    multiplied = multiply_M((mul_q1q2.type(torch.cuda.LongTensor) << - shift.item()), M0)
    total = shifting(multiplied, 0)
else:
    multiplied = multiply_M(mul_q1q2.type(torch.cuda.LongTensor), M0)
    total = shifting(multiplied, shift.item())

total = total.add(z3)

out = torch.clamp(total, 0, 15)

print(out_float_q)
print(out)

diff = torch.abs(torch.sub(out_float_q, out))
avg = torch.sum(diff) / (total.shape[0] * total.shape[1] * total.shape[2] * total.shape[3])
print(torch.min(diff).item(), torch.max(diff).item())
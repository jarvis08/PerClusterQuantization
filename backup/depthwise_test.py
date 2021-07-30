import torch.nn as nn
import torch
import torch.nn.functional as F
from models.quantization_utils import *

s1 = nn.Parameter(torch.tensor(0, dtype=torch.float32), requires_grad=False)
s2 = nn.Parameter(torch.tensor(0, dtype=torch.float32), requires_grad=False)
s3 = nn.Parameter(torch.tensor(0, dtype=torch.float32), requires_grad=False)
z1 = nn.Parameter(torch.tensor(0, dtype=torch.int32), requires_grad=False)
z2 = nn.Parameter(torch.tensor(0, dtype=torch.int32), requires_grad=False)
z3 = nn.Parameter(torch.tensor(0, dtype=torch.int32), requires_grad=False)
M0 = nn.Parameter(torch.tensor(0, dtype=torch.int32), requires_grad=False).cuda()
shift = nn.Parameter(torch.tensor(0, dtype=torch.int32), requires_grad=False).cuda()

r1 = torch.rand((3,2,3,3), dtype=torch.float).cuda()
r2 = torch.rand((2,1,2,2), dtype=torch.float).cuda()
r3 = F.conv2d(r1, r2, None, stride=1, groups=2)

s1, z1 = calc_qparams(torch.min(r1), torch.max(r1), 15)
s2, z2 = calc_qparams(torch.min(r2), torch.max(r2), 15)
s3, z3 = calc_qparams(torch.min(r3), torch.max(r3), 15)

q1 = quantize_matrix(r1, s1, z1, 15)
q2 = quantize_matrix(r2, s2, z2, 15)
q3 = quantize_matrix(r3, s3, z3, 15)

M0, shift = quantize_M(s1*s2/s3)
sum_q1q2 = F.conv2d(q1, q2, None, stride=1, groups=2).type(torch.cuda.IntTensor)

input_batch, input_ch, input_col, input_row = q1.shape[0], q1.shape[1], q1.shape[2], q1.shape[3]
filter_batch, filter_ch, filter_col, filter_row = q2.shape[0], q2.shape[1], q2.shape[2], q2.shape[3]
stride = 1

output_col = sum_q1q2.shape[2]
output_row = sum_q1q2.shape[3]
sum_a1 = torch.zeros((input_batch, output_col, output_row), dtype=torch.int32).cuda()
sum_a2 = torch.zeros(filter_batch, dtype=torch.int32).cuda()
sum_a3 = torch.zeros((input_batch, filter_batch, output_col, output_row), dtype=torch.int32).cuda()
sum_a4 = torch.zeros((input_batch, filter_batch), dtype=torch.int32).cuda()

for i in range(filter_batch):
    sum_a2[i] = torch.sum(q2[i,:]).mul(z1)
for i in range(filter_batch):
    sum_q1q2[:,i,:,:] = torch.sub(sum_q1q2[:,i,:,:], sum_a2[i])

for o_col in range(output_col):
    for o_row in range(output_row):
        col_st, col_end = o_col * stride, o_col * stride + filter_col
        row_st, row_end = o_row * stride, o_row * stride + filter_row
        sum_q1q2[:, :, o_col, o_row] = torch.sub(sum_q1q2[:, :, o_col, o_row], torch.sum(q1[:, :, col_st: col_end, row_st: row_end], (2,3)).mul(z2))

nz1z2 = input_ch * filter_col * filter_row * z1 * z2
sum_q1q2 = sum_q1q2.add(nz1z2)

for i_batch in range(input_batch):
    sum_q1q2[i_batch, :] = torch.sub(sum_q1q2[i_batch, :], sum_a1[i_batch])

if shift < 0:
    multiplied = multiply_M((sum_q1q2.type(torch.cuda.LongTensor) << - shift.item()), M0)
    total = shifting(multiplied, 0)
else:
    multiplied = multiply_M(sum_q1q2.type(torch.cuda.LongTensor), M0)
    total = shifting(multiplied, shift.item())
total = total.add(z3)

total = torch.clamp(total, 0, 15)
total = total.type(torch.cuda.FloatTensor)

total_float = quantize_matrix(r3, s3, z3, 15).cuda()
print("\n", total_float,"\n")
print(total)
diff = torch.abs(torch.sub(total_float, total))
avg = torch.sum(diff) / (total.shape[0] * total.shape[1] * total.shape[2] * total.shape[3])
print(torch.min(diff).item(), torch.max(diff).item())
print(avg)
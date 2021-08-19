import torch.nn as nn
import torch
import torch.nn.functional as F

from quantization_utils import *

def hs_totalsum(x, weight, sum_q1q2, s1, s2, s3, z1, z2, z3, M0, shift, s_activation, z_activation):
    input_batch, input_ch, input_col, input_row = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
    filter_batch, filter_ch, filter_col, filter_row = weight.shape[0], weight.shape[1], weight.shape[2], \
                                                      weight.shape[3]
    stride = 2

    output_col = sum_q1q2.shape[2]
    output_row = sum_q1q2.shape[3]
    sum_a1 = torch.zeros((input_batch, output_col, output_row), dtype=torch.int32).cuda()
    sum_a2 = torch.zeros(filter_batch, dtype=torch.int32).cuda()

    for output_ch in range(0, filter_batch):
        sum_a2[output_ch] = torch.sum(weight.data[output_ch, :, :, :]).mul(z1)

    for o_col in range(0, output_col):
        for o_row in range(0, output_row):
            col_st, col_end = o_col * stride, o_col * stride + filter_col
            row_st, row_end = o_row * stride, o_row * stride + filter_row
            sum_a1[:, o_col, o_row] = torch.sum(x[:, :, col_st: col_end, row_st: row_end], (1, 2, 3)).mul(z2)
    nz1z2 = input_ch * filter_col * filter_row * z1 * z2
    sum_q1q2 = sum_q1q2.add(nz1z2)

    for i_batch in range(0, input_batch):
        sum_q1q2[i_batch, :] = torch.sub(sum_q1q2[i_batch, :], sum_a1[i_batch])
    for out_c in range(0, filter_batch):
        sum_q1q2[:, out_c] = torch.sub(sum_q1q2[:, out_c], sum_a2[out_c])

    multiplied = multiply_M(sum_q1q2.type(torch.cuda.LongTensor), M0)
    total = shifting(multiplied, shift.item())
    total = total.add(z3)

    hs_3 = torch.round(3 / s_activation + z_activation)
    hs_6 = torch.round(6 / s_activation + z_activation)
    hs = (total + hs_3)
    hs = torch.clamp(hs, z3.item(), hs_6.item())
    total = total * hs / hs_6.item()
    total = torch.clamp(total, 0, 15)
    return total.type(torch.cuda.FloatTensor)

def hs2_totalsum(x, weight, sum_q1q2, s1, s2, s3, z1, z2, z3, M0, shift, s_activation, z_activation):
    input_batch, input_ch, input_col, input_row = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
    filter_batch, filter_ch, filter_col, filter_row = weight.shape[0], weight.shape[1], weight.shape[2], \
                                                      weight.shape[3]
    stride = 2

    output_col = sum_q1q2.shape[2]
    output_row = sum_q1q2.shape[3]
    sum_a1 = torch.zeros((input_batch, output_col, output_row), dtype=torch.int32).cuda()
    sum_a2 = torch.zeros(filter_batch, dtype=torch.int32).cuda()

    for output_ch in range(0, filter_batch):
        sum_a2[output_ch] = torch.sum(weight.data[output_ch, :, :, :]).mul(z1)

    for o_col in range(0, output_col):
        for o_row in range(0, output_row):
            col_st, col_end = o_col * stride, o_col * stride + filter_col
            row_st, row_end = o_row * stride, o_row * stride + filter_row
            sum_a1[:, o_col, o_row] = torch.sum(x[:, :, col_st: col_end, row_st: row_end], (1, 2, 3)).mul(z2)
    nz1z2 = input_ch * filter_col * filter_row * z1 * z2
    sum_q1q2 = sum_q1q2.add(nz1z2)

    for i_batch in range(0, input_batch):
        sum_q1q2[i_batch, :] = torch.sub(sum_q1q2[i_batch, :], sum_a1[i_batch])
    for out_c in range(0, filter_batch):
        sum_q1q2[:, out_c] = torch.sub(sum_q1q2[:, out_c], sum_a2[out_c])

    multiplied = multiply_M(sum_q1q2.type(torch.cuda.LongTensor), M0)
    total = shifting(multiplied, shift.item())
    total = total.add(z3)

    hs_3 = torch.round(3 / s_activation + z_activation)
    hs_6 = torch.round(6 / s_activation + z_activation)
    hs = (total + hs_3)
    hs = torch.clamp(hs, z_activation.item(), hs_6.item())
    total = total * hs / hs_6.item()
    total = torch.clamp(total, 0, 15)
    return total.type(torch.cuda.FloatTensor)

# case 1
# float conv - hardswish - quantize
# x = torch.rand((2,3,5,5), dtype=torch.float).cuda()
x = torch.randn((128,3,226,226), dtype=torch.float).cuda()
# x = torch.randn(128,3,226,226).cuda()
s1, z1 = calc_qparams(torch.min(x), torch.max(x), 15)
# weight = torch.rand((1,3,3,3), dtype=torch.float).cuda()
weight = torch.randn((16,3,3,3), dtype=torch.float).cuda()
# weight = torch.rand(16,3,3,3).cuda()
s2, z2 = calc_qparams(torch.min(weight), torch.max(weight), 15)
fp_conv = F.conv2d(x, weight, None, 2, (0, 0))
s3, z3 = calc_qparams(torch.min(fp_conv), torch.max(fp_conv), 15)
case1 = nn.Hardswish()(fp_conv)
print(case1)
exit()
s_activation, z_activation = calc_qparams(torch.min(case1), torch.max(case1), 15)
case_1 = quantize_matrix(case1, s_activation, z_activation, 15)

# case 2 # total sum
q_x = quantize_matrix(x, s1, z1, 15)
q_weight = quantize_matrix(weight, s2, z2, 15)
sum_q1q2 = F.conv2d(q_x, q_weight, bias=None, stride=2)
M0, shift = quantize_M(s1 * s2 / s3)
case_2 = hs_totalsum(q_x, q_weight, sum_q1q2.type(torch.cuda.IntTensor), s1, s2, s3, z1, z2, z3, M0, shift, s_activation, z_activation)
case_3 = hs2_totalsum(q_x, q_weight, sum_q1q2.type(torch.cuda.IntTensor), s1, s2, s3, z1, z2, z3, M0, shift, s_activation, z_activation)
# # case 3
# s_conv, z_conv = calc_qparams(torch.min(fp_conv), torch.max(fp_conv), 15)
#
# M0, shift = quantize_M(s1 * s2 / s_conv)
# case_3 = general_totalsum(q_x, q_weight, sum_q1q2.type(torch.cuda.IntTensor), s1, s2, s_conv, z1, z2, z_conv, M0, shift)
#
# q_case1 = quantize_matrix(case1, s3, z3, 15)
# output_min, output_max = torch.min(q_case1).type(torch.cuda.IntTensor).type(torch.cuda.FloatTensor), torch.max(q_case1).type(torch.cuda.IntTensor).type(torch.cuda.FloatTensor)
#
# inverse_output_scale = 1.0 / s3
#
# lookup_table = nn.Parameter(torch.zeros(16).type(torch.cuda.ByteTensor), requires_grad=False)
# for i in range(16):
#     y = s_conv * (i - z_conv)
#     y2 = y + 3.0
#     y2 = y2 if y2 > 0 else 0
#     y2 = y2 if y2 < 6.0 else 6.0
#     y2 = y * y2 / 6.0
#
#     scaled_hardswish = inverse_output_scale * y2 + z3
#     if scaled_hardswish < torch.tensor(0, dtype=torch.float32):
#         scaled_hardswish = torch.tensor(0, dtype=torch.float32)
#     if scaled_hardswish > torch.tensor(15, dtype=torch.float32):
#         scaled_hardswish = torch.tensor(15, dtype=torch.float32)
#     lookup_table[i] = torch.round(scaled_hardswish)
#
# print(lookup_table)
# for i in range(case_3.shape[0]):
#     for j in range(case_3.shape[1]):
#         for k in range(case_3.shape[2]):
#             for m in range(case_3.shape[3]):
#                 case_3[i][j][k][m] = lookup_table[int(case_3[i][j][k][m].item())]

print("\n")
print(s3, s_activation, z3, z_activation)
print("-------------------------------------- CASE 1 --------------------------------------")
print(case_1)
print("-------------------------------------- CASE 2 --------------------------------------")
print(case_2)
print("-------------------------------------- CASE 3 --------------------------------------")
print(case_3)

print("case2 Min Max",torch.min(torch.abs(torch.sub(case_1,case_2))).item(), torch.max(torch.abs(torch.sub(case_1,case_2))).item())
print("case 3 Min Max",torch.min(torch.abs(torch.sub(case_1,case_3))).item(), torch.max(torch.abs(torch.sub(case_1,case_3))).item())
# print("Case1 - Case2 ",torch.sum(torch.abs(torch.sub(case_1,case_2))).item())
# print("Case1 - Case3 ",torch.sum(torch.abs(torch.sub(case_1,case_3))))
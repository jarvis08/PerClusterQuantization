import torch
import sys
import numpy as np
import os
import torch.nn.functional as F
from models.quantization_utils import multiply_M, quantize_M, shifting, calc_qparams, quantize_matrix

def transform(param, dtype):
    return param.flatten().detach().numpy().astype(dtype)


def transfer_qparams(param, dtype):
    return param.cpu().numpy().astype(dtype)


def general_totalsum(x, weight, sum_q1q2, s1, z1, s2, z2, s3, z3, path):
    input_feature, output_feature = sum_q1q2.shape[0], sum_q1q2.shape[1]

    N = x.shape[1]

    M0, shift = quantize_M(s1 * s2 / s3)
    act_qmax = 65535
    # act_qmax = 255

    sum_a1 = torch.zeros(input_feature, dtype=torch.int32)
    sum_a2 = torch.zeros(output_feature, dtype=torch.int32)
    for out_f in range(output_feature):
        sum_a2[out_f] = torch.sum(weight[out_f, :]).mul(z1)
    for in_f in range(input_feature):
        sum_a1[in_f] = torch.sum(x[in_f, :]).mul(z2)

    nz1z2 = N * z1 * z2
    sub_sum = sum_q1q2.add(nz1z2)
    for in_f in range(input_feature):
        sub_sum[in_f, :] = torch.sub(sub_sum[in_f, :], sum_a1[in_f])
    for out_f in range(output_feature):
        sub_sum[:, out_f] = torch.sub(sub_sum[:, out_f], sum_a2[out_f])

    if shift < 0:
        multiplied = multiply_M((sub_sum.type(torch.cuda.LongTensor) << - shift.item()), M0)
        total = shifting(multiplied, 0)
    else:
        multiplied = multiply_M(sub_sum.type(torch.cuda.LongTensor), M0)
        total = shifting(multiplied, shift.item())
    total = total.add(z3)

    if act_qmax == 15:
        total = torch.clamp(total, 0, 15)
    elif act_qmax == 255:
        total = torch.clamp(total, -128, 127)
    elif act_qmax == 65535:  # INT 16
        total = torch.clamp(total, -32768, 32767)
    elif act_qmax == 4294967295:  # INT 32
        total = torch.clamp(total, -2147483648, 2147483647)

    output = total.type(torch.cuda.FloatTensor)

    path_weights = os.path.join(path, 'test_fc.torch.int8.weights')
    path_qparams = os.path.join(path, 'test_fc.torch.int8.qparams')

    tr_w = transform(weight, 'int8')
    with open(path_weights, 'wb') as fw:
        tr_w.tofile(fw)
    with open(path_qparams, 'wb') as f:
        transfer_qparams(s1, 'float32').tofile(f)
        transfer_qparams(s2, 'float32').tofile(f)
        transfer_qparams(s3, 'float32').tofile(f)
        transfer_qparams(z1, 'int8').tofile(f)
        transfer_qparams(z2, 'int8').tofile(f)
        transfer_qparams(z3, 'int8').tofile(f)
        transfer_qparams(M0, 'int32').tofile(f)
        transfer_qparams(shift, 'int32').tofile(f)

    np.set_printoptions(threshold=sys.maxsize)
    print("Func", x.shape)
    print("weight: ", np.array(weight.detach().cpu()))
    print("sum_q1q1: ", np.array(sum_q1q2.cpu()))
    print("output: ", output)
    print("s1: {}, s2: {}, s3: {}, z1: {}, z2: {}, z3: {}, M0:{}, shift: {}".format(s1, s2, s3, z1, z2, z3, M0, shift))
    return output

x = torch.rand((256, 16 * 32 * 32))
x = x * torch.randint(-10, 10, (256, 16 * 32 * 32))
weight = torch.rand((1000, 16 * 32 * 32))
w = weight * torch.randint(-4, 4, (1000, 16 * 32 * 32))

# x = torch.rand((128, 3, 32, 32))
# x = x * torch.randint(-10, 10, (128, 3, 32, 32))

np.set_printoptions(threshold=sys.maxsize)
print("Input : ",np.array(x))

# conv = torch.nn.Conv2d(3, 16, 3, bias=False, stride=1, padding=0)
# w = conv.weight

bit = 8
s1, z1 = calc_qparams(torch.min(x), torch.max(x), bit)
s2, z2 = calc_qparams(torch.min(w), torch.max(w), bit)
quantized_x = quantize_matrix(x, s1, z1, bit)
quantized_w = quantize_matrix(w, s2, z2, bit)

path = './result/darknet'
path_input = os.path.join(path, 'test_fc.torch.int8.inputs')
tr_x = transform(quantized_x.data, 'int8')
with open(path_input, 'wb') as fi:
    tr_x.tofile(fi)

# general_out = conv(x)
# sum_q1q2 = F.conv2d(quantized_x, quantized_w, conv.bias, conv.stride, conv.dilation, conv.groups)
general_out = F.linear(x, w, None)
sum_q1q2 = F.linear(quantized_x, quantized_w, None)

a_bit = 8
s3, z3 = calc_qparams(torch.min(general_out), torch.max(general_out), a_bit)
out = general_totalsum(quantized_x, quantized_w, sum_q1q2.type(torch.cuda.IntTensor), s1, z1, s2, z2, s3, z3, path)






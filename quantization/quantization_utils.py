import torch
import torch.nn as nn
import numpy as np
import os


class SkipBN(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


def calc_qparams(_min, _max, q_max):
    assert q_max == 15 or q_max == 255, print("Not Supported int type!\nPlz use uint4 or int8")
    if q_max == 15:
        s = _max.sub(_min).div(q_max)
        z = - torch.round(_min.div(s))
        return nn.Parameter(s, requires_grad=False), nn.Parameter(torch.clamp(z, 0, q_max), requires_grad=False)
    elif q_max == 255:
        s = _max.sub(_min).div(q_max)
        z = -128 - torch.round(_min.div(s))
        return nn.Parameter(s, requires_grad=False), nn.Parameter(torch.clamp(z, -128, 127), requires_grad=False)


def quantize_M(M):
    assert M > 0 or M < 1
    shift = 0
    while M < 0.5:
        M *= 2
        shift += 1
    q_M = torch.tensor(torch.round(M * (1 << 31)), dtype=torch.int64, device='cuda:0').clone().detach()
    assert (q_M <= (1 << 31))
    if q_M == (1 << 31):
        q_M /= 2
        shift -= 1
    assert shift >= 0
    max_int = 9223372036854775807
    assert q_M <= max_int
    return torch.nn.Parameter(q_M, requires_grad=False), torch.nn.Parameter(torch.tensor(shift, dtype=torch.int32), requires_grad=False)


def multiply_M(sub_sum, q_M):
    max_int = torch.tensor(9223372036854775807, dtype=torch.int64, device='cuda:0')

    overflow_max = torch.where(sub_sum == q_M, True, False)
    overflow_min = torch.where(sub_sum == -max_int - 1, True, False)
    overflow = torch.logical_and(overflow_max, overflow_min)

    subsummultiplier = sub_sum.mul(q_M)

    nudge =  torch.where(subsummultiplier >= 0, (1 << 30), (1 - (1 << 30))).type(torch.cuda.IntTensor)

    subsummultiplier_high = ((subsummultiplier + nudge) / (1 << 31)).type(torch.cuda.LongTensor)

    return torch.where(overflow, max_int, subsummultiplier_high)


def shifting(cur, shift):
    assert shift >= 0 or shift <= 31

    mask = torch.tensor((1 << shift) - 1, dtype=torch.int32, device='cuda:0')
    zero = torch.tensor(0, dtype=torch.int32, device='cuda:0')
    one = torch.tensor(1, dtype=torch.int32, device='cuda:0')

    remainder = (cur & mask).type(torch.cuda.IntTensor) 
    maskiflessthan = torch.where(cur < zero, ~zero, zero)        
    threshold = ((mask >> 1) + (maskiflessthan & 1)).type(torch.cuda.IntTensor)
    maskifgreaterthan = torch.where(remainder > threshold, ~zero, zero)

    total = ((cur >> shift).add(maskifgreaterthan & 1)).type(torch.cuda.IntTensor)    
    return total


def quantize_params(_fp, _int):
    if _int.layer_type == 'QuantizedConv2d':
        _int.weight.data = torch.round(torch.nn.Parameter(_fp.conv.weight.data).div(_int.s2).add(_int.z2))
        _int.bias.data = torch.round(torch.nn.Parameter(_fp.conv.bias.data).div(_int.s1 * _int.s2))
    elif _int.layer_type == 'QuantizedLinear':
        _int.weight.data = torch.round(torch.nn.Parameter(_fp.fc.weight.data).div(_int.s2).add(_int.z2))
        _int.bias.data = torch.round(torch.nn.Parameter(_fp.fc.bias.data).div(_int.s1 * _int.s2))
    return _int


def transfer_qparams(_fp, _int):
    _int.s1 = torch.nn.Parameter(_fp.s1, requires_grad=False)
    _int.s2 = torch.nn.Parameter(_fp.s2, requires_grad=False)
    _int.s3 = torch.nn.Parameter(_fp.s3, requires_grad=False)
    _int.z1 = torch.nn.Parameter(_fp.z1, requires_grad=False)
    _int.z2 = torch.nn.Parameter(_fp.z2, requires_grad=False)
    _int.z3 = torch.nn.Parameter(_fp.z3, requires_grad=False)
    _int.M0 = torch.nn.Parameter(_fp.M0, requires_grad=False)
    _int.shift = torch.nn.Parameter(_fp.shift, requires_grad=False)
    return _int


def quantize(_fp, _int):
    _int = transfer_qparams(_fp, _int)
    _int = quantize_params(_fp, _int)
    return _int


def transform(param):
    return param.flatten().numpy().astype('float32')


def save_qparams(m, f):
    assert m.layer_type in ['FusedConv2d', 'FusedLinear'],\
        "Can't parse Q-params from {}".format(type(m))
    print("S1: {} | Z1: {}".format(m.s1, m.z1))
    print("S2: {} | Z2: {}".format(m.s2, m.z2))
    print("S3: {} | Z3: {}".format(m.s3, m.z3))
    m.s1.numpy().astype('float32').tofile(f)
    m.s2.numpy().astype('float32').tofile(f)
    m.s3.numpy().astype('float32').tofile(f)
    m.z1.numpy().astype('int32').tofile(f)
    m.z2.numpy().astype('int32').tofile(f)
    m.z3.numpy().astype('int32').tofile(f)


def save_block_qparams(block, f):
    # Downsampling after bypass-connection
    if block.downsample:
        save_qparams(block.downsample, f)

    # CONV after bypass-connection
    save_qparams(block.conv1, f)

    # 2nd CONV in a block
    save_qparams(block.conv2, f)

    # SHORTCUT layer in Darknet
    if block.downsample:
        block.downsample.s3.numpy().astype('float32').tofile(f)
    else:
        block.conv1.s1.numpy().astype('float32').tofile(f)
    block.conv2.s3.numpy().astype('float32').tofile(f)
    block.s3.numpy().astype('float32').tofile(f)

    if block.downsample:
        block.downsample.z3.numpy().astype('int32').tofile(f)
    else:
        block.conv1.z1.numpy().astype('int32').tofile(f)
    block.conv2.z3.numpy().astype('int32').tofile(f)
    block.z3.numpy().astype('int32').tofile(f)


def save_fused_alexnet_qparams(model, path):
    with open(path, 'w') as f:
        for name, m in model.named_children():
            if 'features' in name:
                for i in [0, 2, 4, 5, 6]:
                    save_qparams(m[i], f)

            elif 'classifier' in name:
                for i in [0, 1, 2]:
                    save_qparams(m[i], f)


def save_fused_resnet_qparams(model, path):
    with open(path, 'wb') as f:
        for name, m in model.named_children():
            if "layer" in name:
                for i in range(len(m)):
                    save_block_qparams(m[i], f)
            elif name in ["first_conv", "fc"]:
                save_qparams(m, f)


def save_params(model, path):
    with open(path, 'w') as f:
        weight = None
        for name, param in model.named_parameters():
            if 'weight' in name:
                print(">> Layer: {}\tshape={}".format(name, str(param.data.shape).replace('torch.Size', '')))
                weight = transform(param.data)
            elif 'bias' in name:
                print(">> Layer: {}\tshape={}".format(name, str(param.data.shape).replace('torch.Size', '')))
                transform(param.data).tofile(f)
                weight.tofile(f)


def save_fused_network_in_darknet_form(model, arch):
    path = './result/darknet'
    if not os.path.exists(path):
        os.makedirs(path)
    path = os.path.join(path, '{}.fused.torch.'.format(arch))

    model.cpu()
    save_params(model, path + 'weights')
    if arch == "resnet":
        save_fused_resnet_qparams(model, path + 'qparams')
    elif arch == "alexnet":
        save_fused_alexnet_qparams(model, path + 'qparams')

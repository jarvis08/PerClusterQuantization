import os
import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy


class STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, original, modified):
        return modified.detach()

    @staticmethod
    def backward(ctx, grad):
        return grad, None


class QuantizationTool(object):
    def __init__(self):
        self.fuser = None
        self.quantizer = None
        self.pretrained_model_initializer = None
        self.fused_model_initializer = None
        self.quantized_model_initializer = None


def calc_qparams(_min, _max, q_max):
    s = (_max - _min) / q_max
    if q_max == 255:
        z = -128 - torch.round(_min / s)
        return s, torch.clamp(z, -128, 127)
    else:
        z = - torch.round(_min / s)
        return s, torch.clamp(z, 0, q_max)


def ema(x, averaged, smooth):
    _min = torch.min(x).item()
    _max = torch.max(x).item()
    rst_min = averaged[0] * smooth + _min * (1 - smooth)
    rst_max = averaged[1] * smooth + _max * (1 - smooth)
    return rst_min, rst_max


def fake_quantize(x, scale, zero_point, q_max, use_ste=False):
    if q_max == 255:
        _x = (torch.clamp(torch.round(x / scale + zero_point), -128, 127) - zero_point) * scale
    else:
        _x = (torch.clamp(torch.round(x / scale + zero_point), 0, q_max) - zero_point) * scale
    if use_ste:
        return STE.apply(x, _x)
    return _x


def quantize_matrix(x, scale, zero_point, q_max=None):
    if q_max is None:
        return torch.round(x / scale + zero_point)  # sth like bias
    elif q_max == 255:
        return torch.clamp(torch.round(x / scale + zero_point), -128, 127)
    else:
        return torch.clamp(torch.round(x / scale + zero_point), 0, q_max)


def dequantize_matrix(x, scale, zero_point):
    return (x - zero_point) * scale


def quantize_M(M):
    assert M > 0

    shift = 0
    while M < 0.5:
        M *= 2
        shift += 1
    while M > 1:
        M /= 2
        shift -= 1

    q_M = torch.round(M.clone().detach() * (1 << 31)).cuda()
    assert (q_M <= (1 << 31))
    if q_M == (1 << 31):
        q_M /= 2
        shift -= 1

    q_M = q_M.type(torch.cuda.IntTensor)
    shift = torch.tensor(shift, dtype=torch.int32).cuda()
    max_int = 2147483647
    assert q_M <= max_int
    return torch.nn.Parameter(q_M, requires_grad=False), torch.nn.Parameter(shift, requires_grad=False)


def multiply_M(sub_sum, q_M):
    max_int = torch.tensor(9223272036854775807, dtype=torch.int64, device='cuda:0')
    overflow_max = torch.where(sub_sum == q_M, True, False)
    overflow_min = torch.where(sub_sum == -max_int -1, True, False)
    overflow = torch.logical_and(overflow_max, overflow_min)

    subsummultiplier = sub_sum.mul(q_M).type(torch.cuda.LongTensor)
    nudge =  torch.where(subsummultiplier >= 0, (1 << 30), (1 - (1 << 30))).type(torch.cuda.IntTensor)
    subsummultiplier_high = ((subsummultiplier + nudge) / (1 << 31)).type(torch.cuda.LongTensor)
    return torch.where(overflow, max_int, subsummultiplier_high)


def shifting(cur, shift):
    assert shift >= 0
    # assert shift <= 31
    # mask = torch.tensor((1 << shift) - 1, dtype=torch.int32, device='cuda:0')
    mask = torch.tensor((1 << shift) - 1, dtype=torch.int64, device='cuda:0')
    zero = torch.tensor(0, dtype=torch.int32, device='cuda:0')
    one = torch.tensor(1, dtype=torch.int32, device='cuda:0')

    remainder = (cur & mask).type(torch.cuda.IntTensor)
    maskiflessthan = torch.where(cur < zero, ~zero, zero)
    threshold = ((mask >> one) + (maskiflessthan & one)).type(torch.cuda.IntTensor)
    maskifgreaterthan = torch.where(remainder > threshold, ~zero, zero)

    total = ((cur >> shift).add(maskifgreaterthan & one)).type(torch.cuda.IntTensor)
    return total


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


def quantize_and_transfer_params(_fp, _int):
    assert _int.layer_type in ['QuantizedConv2d', 'QuantizedLinear'], "Not supported quantized layer"
    if _int.layer_type == 'QuantizedConv2d':
        fp_layer = _fp.conv
    else:
        fp_layer = _fp.fc

    with torch.no_grad():
        _int.weight.data.copy_(quantize_matrix(fp_layer.weight.clone().detach(), _int.s2, _int.z2, _int.q_max))
        if _int.num_clusters > 1:
            for c in range(_int.num_clusters):
                _int.quantized_bias[c].copy_(quantize_matrix(fp_layer.bias.clone().detach(), _int.s1[c] * _int.s2, 0))
        else:
            if fp_layer.bias is not None:
                _int.quantized_bias[0].copy_(quantize_matrix(fp_layer.bias.clone().detach(), _int.s1 * _int.s2, 0))
    return _int


def quantize(_fp, _int):
    _int = transfer_qparams(_fp, _int)
    _int = quantize_and_transfer_params(_fp, _int)
    return _int


def copy_from_pretrained(_to, _from, norm_layer=None):
    # Copy weights from pretrained FP model
    with torch.no_grad():
        if 'Conv' in _to.layer_type:
            _to.conv.weight.copy_(_from.weight)
            if norm_layer:
                    _to._norm_layer = deepcopy(norm_layer)
            else:
                if _from.bias is not None:
                    _to.conv.bias.copy_(_from.bias)
        elif 'Linear' in _to.layer_type:
            _to.fc.weight.copy_(_from.weight)
            _to.fc.bias.copy_(_from.bias)
        else:
            _to._norm_layer = deepcopy(_from)
    return _to


def transform(param):
    return param.flatten().numpy().astype('float32')


def save_qparams(m, f):
    assert m.layer_type in ['FusedConv2d', 'FusedLinear', 'PCQConv2d', 'PCQLinear'],\
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
            if 'conv' in name or 'fc' in name:
                save_qparams(m, f)


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


def save_fused_network_in_darknet_form(model, args):
    path = './result/darknet'
    if not os.path.exists(path):
        os.makedirs(path)
    path = os.path.join(path, '{}.fused.torch.int{}'.format(args.arch, args.bit))

    model.cpu()
    save_params(model, path + 'weights')
    if 'ResNet' in args.arch:
        save_fused_resnet_qparams(model, path + 'qparams')
    elif 'AlexNet' in args.arch:
        save_fused_alexnet_qparams(model, path + 'qparams')


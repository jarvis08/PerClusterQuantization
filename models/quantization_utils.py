import os
import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy


class STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, original, fake_quantized):
        return fake_quantized.detach()

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
    if q_max == 15:            # UINT 4
        z = - torch.round(_min / s)
        return s, torch.clamp(z, 0, q_max)
    elif q_max == 255:         # INT 8
        z = -128 - torch.round(_min / s)
        return s, torch.clamp(z, -128, 127)
    elif q_max == 65535:       # INT 16
        z = -32768 - torch.round(_min / s)
        return s, torch.clamp(z, -32768, 32767)
    elif q_max == 4294967295:  # INT 32
        return s, torch.nn.Parameter(torch.tensor(0), requires_grad=False)
    else:
        z = - torch.round(_min / s)
        return s, z


def calc_qparams_per_cluster(ranges, q_max):
    s = ranges[:, 1].sub(ranges[:, 0]).div(q_max)

    if q_max == 15:
        z = - torch.round(ranges[:, 0] / s)
        return s, torch.clamp(z, 0, 15).cuda()
    elif q_max == 255:
        z = -128 - torch.round(ranges[:, 0] / s)
        return s, torch.clamp(z, -128, 127).cuda()
    elif q_max == 65535:
        z = -32768 - torch.round(ranges[:, 0] / s)
        return s, torch.clamp(z, -32768, 32767).cuda()

    # If 32bit or larger, use zero-point as 0 which doesn't need to be clamped
    return s, torch.nn.Parameter(torch.zeros(s.shape), requires_grad=False).cuda()


def ema(x, averaged, smooth):
    _min = torch.min(x).item()
    _max = torch.max(x).item()
    rst_min = averaged[0] * smooth + _min * (1 - smooth)
    rst_max = averaged[1] * smooth + _max * (1 - smooth)
    return rst_min, rst_max


def ema_per_cluster(x, averaged_ranges, num_clusters, smooth):
    _x = x.view(num_clusters, -1)
    _min = _x.min(-1, keepdim=True).values
    _max = _x.max(-1, keepdim=True).values
    batch_range = torch.cat([_min, _max], 1)
    averaged_ranges.mul_(smooth).add_(batch_range * (1 - smooth))


def bn_ema(cur, pre, smooth):
    mean = pre[0] * smooth + cur[0].running_mean * (1 - smooth)
    var = pre[1] * smooth + cur[1].running_var * (1 - smooth)
    return mean, var


def fake_quantize(x, scale, zero_point, q_max, use_ste=False):
    _x = x.detach()
    if q_max == 15:
        _qmin, _qmax = 0, 15
    elif q_max == 255:
        _qmin, _qmax = -128, 127
    elif q_max == 65535:
        _qmin, _qmax = -32768, 32767
    else:
        _qmin, _qmax = -2147483648, 2147483647

    _x = (torch.clamp(torch.round(_x / scale + zero_point), _qmin, _qmax) - zero_point) * scale
    if use_ste:
        return STE.apply(x, _x)
    return _x


def fake_quantize_per_cluster_2d(x, scale, zero_point, q_max, cluster_per_data, use_ste=False):
    _x = x.detach()
    s = torch.index_select(scale, 0, cluster_per_data)[:, None]
    z = torch.index_select(zero_point, 0, cluster_per_data)[:, None]
    if q_max == 15:
        _qmin, _qmax = 0, 15
    elif q_max == 255:
        _qmin, _qmax = -128, 127
    elif q_max == 65535:
        _qmin, _qmax = -32768, 32767
    else:
        _qmin, _qmax = -2147483648, 2147483647

    _x = (torch.clamp(torch.round(_x / s + z), _qmin, _qmax) - z) * s
    if use_ste:
        return STE.apply(x, _x)
    return _x


def fake_quantize_per_cluster_4d(x, scale, zero_point, q_max, cluster_per_data, use_ste=False):
    _x = x.detach()
    s = torch.index_select(scale, 0, cluster_per_data)[:, None, None, None]
    z = torch.index_select(zero_point, 0, cluster_per_data)[:, None, None, None]
    if q_max == 15:
        _qmin, _qmax = 0, 15
    elif q_max == 255:
        _qmin, _qmax = -128, 127
    elif q_max == 65535:
        _qmin, _qmax = -32768, 32767
    else:
        _qmin, _qmax = -2147483648, 2147483647

    _x = (torch.clamp(torch.round(_x / s + z), _qmin, _qmax) - z) * s
    if use_ste:
        return STE.apply(x, _x)
    return _x


def apply_qn(x, scale, zero_point, q_max, qn_prob, kernel_size=None, each_channel=False, in_feature=0, out_feature=0):
    _x = x.detach()
    if q_max == 15:
        _qmin, _qmax = 0, 15
    elif q_max == 255:
        _qmin, _qmax = -128, 127
    elif q_max == 65535:
        _qmin, _qmax = -32768, 32767
    else:
        _qmin, _qmax = -2147483648, 2147483647

    fq_x = (torch.clamp(torch.round(_x / scale + zero_point), _qmin, _qmax) - zero_point) * scale

    if kernel_size is None:
        mask = torch.zeros_like(_x)
        mask.bernoulli_(1 - qn_prob)
        noise = (fq_x - _x).masked_fill(mask.bool(), 0)
        qn_x = _x + noise
    else:  # Conv
        if each_channel:
            mask = torch.zeros(in_feature, out_feature).cuda()
            mask.bernoulli_(qn_prob)
            mask = mask.view(-1, in_feature)
            mask = (mask.unsqueeze(2).unsqueeze(3).repeat(1, 1, kernel_size[0], kernel_size[1]))

            noise = (fq_x - _x).masked_fill(mask.bool(), 0)
            qn_x = _x + noise
        else:
            mask = torch.zeros_like(_x)
            mask.bernoulli_(1 - qn_prob)
            noise = (fq_x - _x).masked_fill(mask.bool(), 0)
            qn_x = _x + noise
    return STE.apply(x, qn_x)


def quantize_matrix(x, scale, zero_point, q_max=None):
    x = x.detach()
    quantized = torch.round(x / scale + zero_point)
    if q_max == 15:            # UINT 4
        return torch.clamp(quantized, 0, 15)
    elif q_max == 255:         # INT 8
        return torch.clamp(quantized, -128, 127)
    elif q_max == 65535:       # INT 16
        return torch.clamp(quantized, -32768, 32767)
    elif q_max == 4294967295:  # INT 32
        return torch.clamp(quantized, -2147483648, 2147483647)
    else:
        return quantized


def quantize_matrix_2d(x, scale, zero_point, batch_cluster, q_max=None):
    scale = torch.index_select(scale, 0, batch_cluster)[:, None]
    zero_point = torch.index_select(zero_point, 0, batch_cluster)[:, None]
    quantized = torch.round(x / scale + zero_point)
    if q_max == 15:            # UINT 4
        return torch.clamp(quantized, 0, 15)
    elif q_max == 255:         # INT 8
        return torch.clamp(quantized, -128, 127)
    elif q_max == 65535:       # INT 16
        return torch.clamp(quantized, -32768, 32767)
    elif q_max == 4294967295:  # INT 32
        return torch.clamp(quantized, -2147483648, 2147483647)
    else:
        return quantized


def quantize_matrix_4d(x, scale, zero_point, batch_cluster, q_max=None):
    scale = torch.index_select(scale, 0, batch_cluster)[:, None, None, None]
    zero_point = torch.index_select(zero_point, 0, batch_cluster)[:, None, None, None]
    quantized = torch.round(x / scale + zero_point)
    if q_max == 15:            # UINT 4
        return torch.clamp(quantized, 0, 15)
    elif q_max == 255:         # INT 8
        return torch.clamp(quantized, -128, 127)
    elif q_max == 65535:       # INT 16
        return torch.clamp(quantized, -32768, 32767)
    elif q_max == 4294967295:  # INT 32
        return torch.clamp(quantized, -2147483648, 2147483647)
    else:
        return quantized


def dequantize_matrix(x, scale, zero_point):
    return (x - zero_point) * scale


def dequantize_matrix_4d(x, scale, zero_point, batch_cluster):
    s = torch.index_select(scale, 0, batch_cluster)[:, None, None, None]
    z = torch.index_select(zero_point, 0, batch_cluster)[:, None, None, None]
    return x.sub_(z).mul_(s)


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
    max_int = torch.tensor(9223372036854775807, dtype=torch.int64, device='cuda:0')
    overflow_max = torch.where(sub_sum == q_M, True, False)
    overflow_min = torch.where(sub_sum == -max_int -1, True, False)
    overflow = torch.logical_and(overflow_max, overflow_min)

    subsummultiplier = sub_sum.mul(q_M).type(torch.cuda.LongTensor)
    nudge =  torch.where(subsummultiplier >= 0, (1 << 30), (1 - (1 << 30))).type(torch.cuda.IntTensor)
    subsummultiplier_high = ((subsummultiplier + nudge) / (1 << 31)).type(torch.cuda.LongTensor)
    return torch.where(overflow, max_int, subsummultiplier_high)


def shifting(cur, shift):
    assert shift >= 0
    mask = torch.tensor((1 << shift) - 1, dtype=torch.int64, device='cuda:0')
    zero = torch.tensor(0, dtype=torch.int32, device='cuda:0')
    one = torch.tensor(1, dtype=torch.int32, device='cuda:0')

    remainder = (cur & mask).type(torch.cuda.IntTensor)
    maskiflessthan = torch.where(cur < zero, ~zero, zero)
    threshold = ((mask >> one) + (maskiflessthan & one)).type(torch.cuda.IntTensor)
    maskifgreaterthan = torch.where(remainder > threshold, ~zero, zero)

    total = ((cur >> shift).add(maskifgreaterthan & one)).type(torch.cuda.IntTensor)
    return total


def shifting2d(cur, shift):
    mask = torch.ones((cur.shape[0],1), dtype=torch.int64, device='cuda:0')
    mask = (mask << shift) - 1
    zero = torch.zeros((cur.shape[0],1), dtype=torch.int32, device='cuda:0')
    one = torch.ones((cur.shape[0],1), dtype=torch.int32, device='cuda:0')

    remainder = (cur & mask).type(torch.cuda.IntTensor)
    maskiflessthan = torch.where(cur < zero, ~zero, zero)
    threshold = ((mask >> one) + (maskiflessthan & one)).type(torch.cuda.IntTensor)
    maskifgreaterthan = torch.where(remainder > threshold, ~zero, zero)

    total = ((cur >> shift).add(maskifgreaterthan & one)).type(torch.cuda.IntTensor)
    return total


def shifting4d(cur, shift):
    mask = torch.ones((cur.shape[0],1,1,1), dtype=torch.int64, device='cuda:0')
    mask = (mask << shift) - 1
    zero = torch.zeros((cur.shape[0],1,1,1), dtype=torch.int32, device='cuda:0')
    one = torch.ones((cur.shape[0],1,1,1), dtype=torch.int32, device='cuda:0')

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
    if _int.layer_type in ['QuantizedConv2d', 'QuantizedLinear', 'QuantizedBn2d']:
        _int.act_qmax = nn.Parameter(torch.tensor(_fp.act_qmax), requires_grad=False)
    return _int


def quantize_layer_and_transfer(_fp, _int):
    assert _int.layer_type in ['QuantizedConv2d', 'QuantizedLinear', 'QuantizedBn2d'], "Not supported quantized layer"
    with torch.no_grad():
        if _int.layer_type == 'QuantizedBn2d':
            if _int.num_clusters > 1:
                weight = _fp.weights.clone().detach().div(torch.sqrt(_fp.running_vars.clone().detach() + _fp.eps))
                bias = _fp.biases.clone().detach() - weight * _fp.running_means.clone().detach()
                weight = quantize_matrix(weight, _int.s2, _int.z2, _fp.w_qmax)
                _int.weight.copy_(weight.type(torch.cuda.IntTensor))
                for c in range(_int.num_clusters):
                    b = quantize_matrix(bias[c], _int.s1[c] * _int.s2, 0, 2 ** 32 - 1)
                    _int.bias[c].copy_(b.type(torch.cuda.IntTensor))
            else:
                w = _fp.bn.weight.clone().detach().div(torch.sqrt(_fp.bn.running_var.clone().detach() + _fp.bn.eps))
                b = _fp.bn.bias.clone().detach() - w * _fp.bn.running_mean.clone().detach()
                w = quantize_matrix(w, _int.s2, _int.z2, _fp.w_qmax)
                b = quantize_matrix(b, _int.s1 * _int.s2, 0, 2 ** 32 - 1)

                _int.weight[0].copy_(w.type(torch.cuda.IntTensor))
                _int.bias[0].copy_(b.type(torch.cuda.IntTensor))
        else:
            if _int.layer_type == 'QuantizedConv2d':
                fp_layer = _fp.conv
            else:
                fp_layer = _fp.fc

            _int.weight.data.copy_(quantize_matrix(fp_layer.weight.clone().detach(), _int.s2, _int.z2, _int.q_max))
            if fp_layer.bias is not None:
                _int.is_bias = nn.Parameter(torch.tensor(True, dtype=torch.bool), requires_grad=False)
                if _int.num_clusters > 1:
                    for c in range(_int.num_clusters):
                        _int.quantized_bias[c].copy_(quantize_matrix(fp_layer.bias.clone().detach(),
                                                                     _int.s1[c] * _int.s2, 0, 2 ** 32 - 1))
                else:
                    _int.quantized_bias[0].copy_(quantize_matrix(fp_layer.bias.clone().detach(),
                                                                 _int.s1 * _int.s2, 0, 2 ** 32 - 1))
    return _int


def quantize(_fp, _int):
    _int = transfer_qparams(_fp, _int)
    _int = quantize_layer_and_transfer(_fp, _int)
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


def copy_bn_from_pretrained(_to, _from):
    with torch.no_grad():
        _to.bn = deepcopy(_from)
    return _to


def copy_pcq_bn_from_pretrained(_to, _from, num_clusters):
    with torch.no_grad():
        _to.weights = nn.Parameter(_from.weight.clone().detach().unsqueeze(0)
                                   .repeat(num_clusters, 1), requires_grad=True)
        _to.biases = nn.Parameter(_from.bias.clone().detach().unsqueeze(0)
                                  .repeat(num_clusters, 1), requires_grad=True)
        _to.running_means = nn.Parameter(_from.running_mean.clone().detach()
                                         .unsqueeze(0).repeat(num_clusters, 1), requires_grad=False)
        _to.running_vars = nn.Parameter(_from.running_var.clone().detach()
                                        .unsqueeze(0).repeat(num_clusters, 1), requires_grad=False)
    return _to


def copy_weight_from_pretrained(_to, _from):
    # Copy weights from pretrained FP model
    with torch.no_grad():
        if 'Conv' in _to.layer_type:
            _to.conv.weight.copy_(_from.weight)
        else:
            _to.fc.weight.copy_(_from.weight)
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

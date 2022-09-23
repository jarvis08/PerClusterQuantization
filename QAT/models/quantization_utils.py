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


class SKT_MIX(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, fixed_indices, grad_method, const_portion):
        # max_per_ch = torch.max(input_[:, fixed_indices].view(fixed_indices.size(0), -1).min(dim=1).values.abs(), input_[:, fixed_indices].view(fixed_indices.size(0), -1).max(dim=1).values.abs())
        max_per_ch = input_[:, fixed_indices].transpose(1, 0).reshape(fixed_indices.size(0), -1).max(dim=1).values.abs()[None, :, None, None]
        mask = input_[:, fixed_indices] > (max_per_ch / 2)

        # 이거 넘어가는 부분 가중치 주는거 : max range - 현 을 over로 설정하면 값이 너무 클 수 있을 것 같아서 일단은 max /
        dist_ratio = (max_per_ch - input_[:, fixed_indices].abs()) / max_per_ch
        dist_ratio = torch.where(dist_ratio == 0.0, torch.tensor(1.0, device='cuda'), dist_ratio)
        # dist_ratio = max_per_ch / (max_per_ch - input_[:, fixed_indices])
        ctx.save_for_backward(fixed_indices, mask,  max_per_ch, const_portion * dist_ratio, grad_method)

        # ctx.save_for_backward(fixed_indices, mask,  max_per_ch * const_portion)
        # ctx.save_for_backward(input_, indices, mask)
        return input_

    @staticmethod
    def backward(ctx, grad):
        fixed_indices, mask, max_per_ch, const_portion, grad_method = ctx.saved_tensors

        # import pdb
        # pdb.set_trace()
        if grad_method:
            grad[:, fixed_indices] = torch.where(mask > 0, grad[:, fixed_indices].abs() * const_portion[:, fixed_indices], grad[:, fixed_indices])
        else:
            tmp = torch.masked_select(grad[:, fixed_indices], mask).mean().abs() * const_portion
            grad[:, fixed_indices] = torch.where(mask > 0, grad[:, fixed_indices].abs() - tmp, grad[:, fixed_indices])

        # # consider loss like weight
        # input_tensor, indices, mask = ctx.saved_tensors
        # data = input_tensor.transpose(1, 0).reshape(input_tensor.size(1), -1)
        # _max = data.max(dim=1).values
        # _min = data.min(dim=1).values
        # range_ = _max - _min
        # max_range = range_.max()
        # max_indices = ((max_range / range_[indices]) < 2.0).nonzero(as_tuple=True)[0]
        #
        # # # without considering loss
        # # grad[:, max_indices] = 0
        #
        # ## channel wise version
        # grad_sum = (-1 * input_tensor[:, max_indices] * 0.9 * grad[:, max_indices]).sum(dim=(0, 2, 3))
        # indices_ = torch.where(grad_sum > 0, 1, 0).nonzero(as_tuple=True)[0]
        # grad[:, indices_] = 0
        # import pdb
        # pdb.set_trace()
        return grad, None, None, None


class QuantizationTool(object):
    def __init__(self):
        self.fuser = None
        self.quantizer = None
        self.pretrained_model_initializer = None
        self.fused_model_initializer = None
        self.quantized_model_initializer = None


def get_range(x):
    _x = x.detach()
    return _x.min().item(), _x.max().item()


def get_scale_and_zeropoint(_min, _max, bit):
    if bit == 4:
        s = (_max - _min) / 15
        z = - torch.round(_min / s)
        return s, torch.clamp(z, 0, 15)
    elif bit == 8:
        s = (_max - _min) / 255
        z = - torch.round(_min / s)
        return s, torch.clamp(z, 0, 255)
    elif bit == 16:
        s = (_max - _min) / 65535
        z = -32768 - torch.round(_min / s)
        return s, torch.clamp(z, -32768, 32767)
    elif bit == 24:
        s = _max.sub(_min).div(16777215)
        return s, torch.zeros(s.shape, device='cuda')
    s = (_max - _min) / 4294967295
    return s, torch.tensor(0, device='cuda')


def calc_qparams(range_min, range_max, bit, symmetric=False, zero=None):
    if bit.view(-1).size(0) > 1:
        bit = bit.max()
    if symmetric:
        return calc_symmetric_qparams(range_min, range_max, bit)
    if zero is None:
        zero = torch.tensor(0.0, device='cuda')
    _min = zero if range_min > 0.0 else range_min
    _max = zero if range_max < 0.0 else range_max
    return get_scale_and_zeropoint(_min, _max, bit)


def calc_qparams_per_input_channel_with_range(_min, _max, low_group, high_group, symmetric=False, zero=None):
    if symmetric:
        return calc_symmetric_qparams_per_input_channel(_min, _max, low_group, high_group)
    else:
        if zero is None:
            zero = torch.tensor(0.0, device='cuda')
        _min = torch.where(_min <= zero, _min, zero)
        _max = torch.where(_max >= zero, _max, zero)
    return get_scale_and_zeropoint_per_input_channel(_min, _max, low_group, high_group)

def calc_qparams_last_conv(_min, _max, bit, symmetric=False, zero=None):
    if symmetric:
        return calc_symmetric_qparams(_min, _max, bit)
    else:
        if zero is None:
            zero = torch.tensor(0.0, device='cuda')
        _min = torch.where(_min <= zero, _min, zero)
        _max = torch.where(_max >= zero, _max, zero)
    return get_scale_and_zeropoint(_min, _max, bit)


def calc_symmetric_qparams(_min, _max, bit, per_channel=False):
    with torch.no_grad():
        n = 2 ** (bit - 1) - 1
        s = max(_min.abs(), _max.abs())
        s = torch.clamp(s, min=1e-8) / n
        # if bit == 4:
        #     s = (_max - _min) / 15
        # elif bit == 8:
        #     s = (_max - _min) / 255
        # elif bit == 16:
        #     s = (_max - _min) / 65535
        # elif bit == 24:
        #     s = _max.sub(_min).div(16777215)
        # else:
        #     s = (_max - _min) / 4294967295
    return s, torch.zeros_like(s, device='cuda')    #


def calc_qparams_per_output_channel(mat, bit, symmetric=False, zero=None):
    _mat = mat.view(mat.size(0), -1)
    _min = _mat.min(dim=1).values
    _max = _mat.max(dim=1).values

    if symmetric:
        return calc_symmetric_qparams(_min, _max, bit, True)
    else:
        if zero is None:
            zero = torch.tensor(0.0, device='cuda')
        _min = torch.where(_min <= zero, _min, zero)
        _max = torch.where(_max >= zero, _max, zero)
    return get_scale_and_zeropoint(_min, _max, bit)


def calc_qparams_per_cluster(ranges, bit, zero=None):
    if zero is None:
        zero = torch.tensor(0.0, device='cuda')
    _min = torch.where(ranges[:, 0] <= 0, ranges[:, 0], zero)
    _max = torch.where(ranges[:, 1] >= 0, ranges[:, 1], zero)
    return get_scale_and_zeropoint(_min, _max, bit)


@torch.no_grad()
def ema(x, averaged, smooth):
    _min, _max = torch.min(x).item(), torch.max(x).item()
    updated_min = averaged[0] * smooth + _min * (1 - smooth)
    updated_max = averaged[1] * smooth + _max * (1 - smooth)
    return updated_min, updated_max


def fake_quantize(x, scale, zero_point, bit, symmetric=False, use_ste=False):
    _x = x.detach()
    _x = (clamp_matrix(torch.round(_x / scale + zero_point), bit, symmetric) - zero_point) * scale
    if use_ste:
        return STE.apply(x, _x)
    return _x

## per in channel version
# def calc_symmetric_qparams_per_input_channel(_min, _max, low_group, high_group):
#     scale = torch.zeros(_min.size(0), device='cuda')
#
#     if scale[low_group].size(0):
#         scale[low_group], _ = torch.max(torch.stack([_max[low_group].abs(), _min[low_group].abs()], dim=1), dim=1)
#         scale[low_group] = torch.clamp(scale[low_group], min=1e-8) / 31
#
#     if scale[high_group].size(0):
#         scale[high_group], _ = torch.max(torch.stack([_max[high_group].abs(), _min[high_group].abs()], dim=1), dim=1)
#         scale[high_group] = torch.clamp(scale[high_group], min=1e-8) / 127
#     return scale, torch.zeros_like(scale, device='cuda')

# def calc_qparams_per_input_channel_with_range(_min, _max, low_group, high_group, symmetric=False, zero=None):
#     if symmetric:
#         return calc_symmetric_qparams_per_input_channel(_min, _max, low_group, high_group)
#     else:
#         if zero is None:
#             zero = torch.tensor(0.0, device='cuda')
#         _min = torch.where(_min <= zero, _min, zero)
#         _max = torch.where(_max >= zero, _max, zero)
#     return get_scale_and_zeropoint_per_input_channel(_min, _max, low_group, high_group)

# def calc_qparams_per_input_channel(mat, low_group, high_group, symmetric=False, zero=None):
#     _mat = mat.transpose(1, 0).reshape(mat.size(1), -1)
#     _min = _mat.min(dim=1).values
#     _max = _mat.max(dim=1).values
#     if symmetric:
#         return calc_symmetric_qparams_per_input_channel(_min, _max, low_group, high_group)
#     else:
#         if zero is None:
#             zero = torch.tensor(0.0, device='cuda')
#         _min = torch.where(_min <= zero, _min, zero)
#         _max = torch.where(_max >= zero, _max, zero)
#         return get_scale_and_zeropoint_per_input_channel(_min, _max, low_group, high_group)

# def get_scale_and_zeropoint_per_input_channel(_min, _max, low_group, high_group):
#     _size = _min.size(0)
#     scale = torch.zeros(_size, device='cuda')
#     zero_point = torch.zeros(_size, device='cuda')
#
#     if scale[low_group].size(0):
#         scale[low_group] = (_max[low_group] - _min[low_group]) / 63
#         zero_point[low_group] = torch.clamp(- torch.round(_min[low_group] / scale[low_group]), 0, 63)
#
#     if scale[high_group].size(0):
#         scale[high_group] = (_max[high_group] - _min[high_group]) / 255
#         zero_point[high_group] = torch.clamp(- torch.round(_min[high_group] / scale[high_group]), 0, 255)
#     return scale, zero_point

# def fake_quantize_per_input_channel(x, low_bit, low_group, high_group, zero=None, symmetric=False, use_ste=False, scale=None, zero_point=None):
#     _x = x.detach()
#
#     if scale is None and zero_point is None:
#         scale, zero_point = calc_qparams_per_input_channel(_x, low_group, high_group, symmetric=symmetric, zero=zero)
#     scale = scale[None, :, None, None]
#     zero_point = zero_point[None, :, None, None]
#
#     _x = clamp_matrix_per_input_channel(torch.round(_x / scale + zero_point), low_bit, low_group, high_group, symmetric)
#
#     # drop 2 bits
#     if low_group.view(-1).size(0):
#         _x[:, low_group] = truncate_lower_bits(_x[:, low_group], low_bit, symmetric=symmetric)
#
#     _x = (_x - zero_point) * scale
#
#     if use_ste:
#         return STE.apply(x, _x)
#     return _x


def calc_symmetric_qparams_per_input_channel(_min, _max):
    scale = torch.max(_min.abs(), _max.abs())
    scale = torch.clamp(scale, min=1e-8) / 127
    return scale, torch.zeros_like(scale, device='cuda')

def calc_qparams_per_input_channel_with_range(_min, _max, symmetric=False, zero=None):
    if symmetric:
        return calc_symmetric_qparams_per_input_channel(_min, _max)
    else:
        if zero is None:
            zero = torch.tensor(0.0, device='cuda')
        _min = torch.where(_min <= zero, _min, zero)
        _max = torch.where(_max >= zero, _max, zero)
    return get_scale_and_zeropoint_per_input_channel(_min, _max)

def calc_qparams_per_input_channel(mat, symmetric=False, zero=None):
    _min = mat.min()
    _max = mat.max()

    if symmetric:
        return calc_symmetric_qparams_per_input_channel(_min, _max)

    else:
        if zero is None:
            zero = torch.tensor(0.0, device='cuda')
        _min = torch.where(_min <= zero, _min, zero)
        _max = torch.where(_max >= zero, _max, zero)
    return get_scale_and_zeropoint_per_input_channel(_min, _max)

def get_scale_and_zeropoint_per_input_channel(_min, _max):
    scale = (_max - _min) / 255
    zero_point = torch.clamp(- torch.round(_min / scale), 0, 255)
    return scale, zero_point

def clamp_matrix_per_input_channel(x, low_bit, low_group, high_group, symmetric=False, only_low=False):
    if low_group.view(-1).size(0):
        if symmetric:
            q_range = 2 ** (low_bit - 1) - 1
            low_qmin, low_qmax = -q_range - 1, q_range
        else:
            q_range = 2 ** low_bit - 1
            low_qmin, low_qmax = 0, q_range

        x[:, low_group] = torch.clamp(x[:, low_group], low_qmin, low_qmax)

    if not only_low and high_group.view(-1).size(0):
        if symmetric:
            high_qmin, high_qmax = -128, 127
        else:
            high_qmin, high_qmax = 0, 255

        x[:,high_group] = torch.clamp(x[:,high_group], high_qmin, high_qmax)
    return x

# def truncate_lower_bits(x, bit, symmetric=False):
#     if bit == 8:
#         return x
#     if symmetric:
#         if bit == 6:
#             truncator = -4
#         elif bit == 7:
#             truncator = -16
#     else:
#         if bit == 6:
#             truncator = 60
#         elif bit == 7:
#             truncator = 120
#     bit_truncator = torch.tensor(truncator, dtype=torch.int8, device='cuda')
#     return x.type(torch.cuda.CharTensor).bitwise_and(bit_truncator).type(torch.cuda.FloatTensor)


def truncate_lower_bits(x, bit, low_group, high_group, symmetric=False):
    if bit == 8:
        return x

    # bit_truncator = torch.tensor(4, dtype=torch.int8, device='cuda')
    x[:, low_group] = torch.where(x[:, low_group] % 4 >= 2, x[:, low_group] // 4 + 1, x[:, low_group] // 4) * 4
    return clamp_matrix_per_input_channel(x, bit, low_group, high_group, symmetric=symmetric, only_low=True)

    # return x.type(torch.cuda.CharTensor).bitwise_and(bit_truncator).type(torch.cuda.FloatTensor)


def fake_quantize_per_input_channel(x, low_bit, low_group, high_group, zero=None, symmetric=False, use_ste=False, scale=None, zero_point=None):
    _x = x.detach()

    if scale is None and zero_point is None:
        scale, zero_point = calc_qparams_per_input_channel(_x, symmetric=symmetric)

    _x = clamp_matrix_per_input_channel(torch.round(_x / scale + zero_point), low_bit, low_group, high_group, symmetric=symmetric)

    # drop 2 bits
    if low_group.view(-1).size(0):
        _x = truncate_lower_bits(_x, low_bit, low_group, high_group, symmetric=symmetric)
        # _x[:, low_group] = truncate_lower_bits(_x[:, low_group], low_bit, symmetric=symmetric)

    _x = (_x - zero_point) * scale

    if use_ste:
        return STE.apply(x, _x)
    return _x

def fake_quantize_per_output_channel(x, bit, zero, symmetric=False, use_ste=False):
    _x = x.detach()
    scale, zero_point = calc_qparams_per_output_channel(_x, bit, symmetric, zero)
    scale = scale[:, None, None, None]
    zero_point = zero_point[:, None, None, None]

    _x = (clamp_matrix(torch.round(_x / scale + zero_point), bit, symmetric) - zero_point) * scale
    if use_ste:
        return STE.apply(x, _x)
    return _x


def fake_quantize_per_cluster_2d(x, scale, zero_point, bit, cluster_per_data, use_ste=False):
    _x = x.detach()
    s = torch.index_select(scale, 0, cluster_per_data)[:, None]
    z = torch.index_select(zero_point, 0, cluster_per_data)[:, None]
    _x = (clamp_matrix(torch.round(_x / s + z), bit) - z) * s
    if use_ste:
        return STE.apply(x, _x)
    return _x


def fake_quantize_per_cluster_4d(x, scale, zero_point, bit, cluster_per_data, use_ste=False):
    _x = x.detach()
    s = torch.index_select(scale, 0, cluster_per_data)[:, None, None, None]
    z = torch.index_select(zero_point, 0, cluster_per_data)[:, None, None, None]
    _x = (clamp_matrix(torch.round(_x / s + z), bit) - z) * s
    if use_ste:
        return STE.apply(x, _x)
    return _x


def apply_qn(x, scale, zero_point, bit, qn_prob, kernel_size=None, each_channel=False, in_feature=0, out_feature=0):
    _x = x.detach()
    fq_x = (clamp_matrix(torch.round(_x / scale + zero_point), bit) - zero_point) * scale
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


def quantize_matrix(x, scale, zero_point, bit=None, symmetric=False):
    quantized = torch.round(x / scale + zero_point)
    return clamp_matrix(quantized, bit, symmetric)


def quantize_matrix_per_in_channel(x, scale, zero_point, low_bit, low_group, high_group, symmetric=False):
    x = torch.round(x / scale + zero_point)

    if low_group.view(-1).size(0):
        # x[:, low_group] = truncate_lower_bits(x[:, low_group], low_bit, symmetric=symmetric)
        x = truncate_lower_bits(x, low_bit, low_group, high_group, symmetric=symmetric)
    return clamp_matrix_per_input_channel(x, low_bit, low_group, high_group, symmetric=symmetric)


def quantize_matrix_2d(x, scale, zero_point, batch_cluster, bit=None):
    scale = torch.index_select(scale, 0, batch_cluster)[:, None]
    zero_point = torch.index_select(zero_point, 0, batch_cluster)[:, None]
    quantized = torch.round(x / scale + zero_point)
    return clamp_matrix(quantized, bit)


def quantize_matrix_4d(x, scale, zero_point, batch_cluster, bit=None):
    scale = torch.index_select(scale, 0, batch_cluster)[:, None, None, None]
    zero_point = torch.index_select(zero_point, 0, batch_cluster)[:, None, None, None]
    quantized = torch.round(x / scale + zero_point)
    return clamp_matrix(quantized, bit)


def rescale_matrix(x, z_from, z_to, m0, shift, target_bit, runtime_helper):
    bc = runtime_helper.qat_batch_cluster

    if bc is None:
        z1 = z_from
        z2 = z_to
        _m0 = m0
        _shift = shift
    else:
        if len(x.shape) == 2:
            z1 = torch.index_select(z_from, 0, bc)[:, None]
            z2 = torch.index_select(z_to, 0, bc)[:, None]
            _m0 = torch.index_select(m0, 0, bc)[:, None]
            _shift = torch.index_select(shift, 0, bc)[:, None]
        else:
            z1 = torch.index_select(z_from, 0, bc)[:, None, None, None]
            z2 = torch.index_select(z_to, 0, bc)[:, None, None, None]
            _m0 = torch.index_select(m0, 0, bc)[:, None, None, None]
            _shift = torch.index_select(shift, 0, bc)[:, None, None, None]

    _x = x - z1
    _x = multiply_M(_x, _m0)
    _x = shifting_without_cast(_x, _shift, getattr(runtime_helper, 'mask_{}d'.format(len(x.shape))))
    _x = _x.add(z2)
    return clamp_matrix(_x, target_bit)


def rescale_matrix_2d(x, z_from, z_to, m0, shift, target_bit, runtime_helper):
    bc = runtime_helper.qat_batch_cluster
    batch_size = x.size(0)

    z1 = torch.index_select(z_from, 0, bc)[:, None]
    z2 = torch.index_select(z_to, 0, bc)[:, None]
    _m0 = torch.index_select(m0, 0, bc)[:, None]
    _shift = torch.index_select(shift, 0, bc)[:, None]

    _x = x - z1
    _x = multiply_M(_x, _m0)
    _x = shifting_without_cast(_x, _shift, runtime_helper.mask_2d[:batch_size])
    _x = _x.add(z2)
    return clamp_matrix(_x, target_bit)


def dequantize_matrix(x, scale, zero_point):
    return (x - zero_point) * scale


def dequantize_matrix_2d(x, scale, zero_point, batch_cluster):
    s = torch.index_select(scale, 0, batch_cluster)[:, None]
    z = torch.index_select(zero_point, 0, batch_cluster)[:, None]
    return x.sub(z).mul(s)


def dequantize_matrix_4d(x, scale, zero_point, batch_cluster):
    s = torch.index_select(scale, 0, batch_cluster)[:, None, None, None]
    z = torch.index_select(zero_point, 0, batch_cluster)[:, None, None, None]
    return x.sub(z).mul(s)


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
    return q_M, shift


def multiply_M(x, q_M):
    max_int = 9223372036854775807
    overflow_max = torch.where(x == q_M, True, False)
    overflow_min = torch.where(x == -max_int - 1, True, False)
    overflow = torch.logical_and(overflow_max, overflow_min)

    subsummultiplier = x.mul(q_M)
    nudge = torch.where(subsummultiplier >= 0, (1 << 30), (1 - (1 << 30))).type(torch.cuda.IntTensor)
    subsummultiplier_high = ((subsummultiplier + nudge) / (1 << 31)).type(torch.cuda.LongTensor)
    return torch.where(overflow, max_int, subsummultiplier_high)


def shifting_without_cast(cur, shift, mask=1):
    _mask = (mask << shift) - 1
    zero, one = 0, 1

    remainder = (cur & _mask).type(torch.cuda.IntTensor)
    maskiflessthan = torch.where(cur < zero, ~zero, zero)
    threshold = ((_mask >> one) + (maskiflessthan & one)).type(torch.cuda.IntTensor)
    maskifgreaterthan = torch.where(remainder > threshold, ~zero, zero)
    return (cur >> shift).add(maskifgreaterthan & one)


def shifting(cur, shift, mask=1):
    _mask = (mask << shift) - 1
    zero, one = 0, 1

    remainder = (cur & _mask).type(torch.cuda.IntTensor)
    maskiflessthan = torch.where(cur < zero, ~zero, zero)
    threshold = ((_mask >> one) + (maskiflessthan & one)).type(torch.cuda.IntTensor)
    maskifgreaterthan = torch.where(remainder > threshold, ~zero, zero)
    total = ((cur >> shift).add(maskifgreaterthan & one)).type(torch.cuda.IntTensor)
    return total


def clamp_matrix(x, bit=None, symmetric=False):
    if bit == 4:
        if symmetric:
            qmin, qmax = -8, 7
        else:
            qmin, qmax = 0, 15
    elif bit == 6:
        if symmetric:
            qmin, qmax = -32, 31
        else:
            qmin, qmax = 0, 63
    elif bit == 7:
        if symmetric:
            qmin, qmax = -64, 63
        else:
            qmin, qmax = 0, 127
    elif bit == 8:
        if symmetric:
            qmin, qmax = -128, 127
        else:
            qmin, qmax = 0, 255
    elif bit == 16:
        qmin, qmax = -32768, 32767
    elif bit == 24:
        qmin, qmax = -8388608, 8388607
    else:
        qmin, qmax = -2147483648, 2147483647
    return torch.clamp(x, qmin, qmax)


def mul_and_shift(x, M0, shift, mask=1):
    multiplied = multiply_M(x, M0)
    return shifting_without_cast(multiplied, shift, mask)


def add_pos_and_neg_shift(x, M0, shift, mask, out):
    neg = (shift < 0).nonzero(as_tuple=True)[0]
    pos = (shift >= 0).nonzero(as_tuple=True)[0]
    n_neg, n_pos = len(neg), len(pos)
    if n_neg > 0:
        out[neg] = out[neg] + mul_and_shift(x[neg] << - shift[neg], M0[neg], 0, mask[:n_neg])
    if n_pos > 0:
        out[pos] = out[pos] + mul_and_shift(x[pos], M0[pos], shift[pos], mask[:n_pos])
    return out


def transfer_qparams(_fp, _int):
    _int.s1.data = _fp.s1
    _int.s2.data = _fp.s2
    _int.s3.data = _fp.s3
    _int.z1.data = _fp.z1
    _int.z2.data = _fp.z2
    _int.z3.data = _fp.z3
    _int.M0.data = _fp.M0
    _int.shift.data = _fp.shift
    if _int.layer_type in ['QuantizedConv2d', 'QuantizedLinear', 'QuantizedBn2d']:
        if _fp.w_bit.view(-1).size(0) > 1:
            _fp.w_bit.data = _fp.w_bit.max()
            _int.low_group = _fp.low_group.cuda()
            _int.high_group = _fp.high_group.cuda()
            _int.low_bit = _fp.low_bit
        _int.w_bit.data = _fp.w_bit.data
        _int.a_bit.data = _fp.a_bit.data
        negative_values = (_int.shift < 0).nonzero(as_tuple=True)[0]
        if len(negative_values):
            _int.is_shift_neg.data = torch.tensor(True, dtype=torch.bool)
    return _int


def quantize_folded_conv2d_weight_and_bias(_fp, _int, symmetric):
    # _int.weight.data.copy_(quantize_matrix(_fp.weight, _int.s2, _int.z2, _int.w_bit))
    # _int.sum_a2.data.copy_(torch.sum(_int.weight, dim=(1, 2, 3)).reshape(1, _int.out_channels, 1, 1))
    _int.is_bias.data = torch.tensor(True, dtype=torch.bool)
    # pcq fold
    if _fp.num_clusters > 1:
        if _fp.num_norms > 1:  # multiple batch norm parameters
            for c in range(_fp.num_clusters):
                _int.folded_weight[c].data.copy_(quantize_matrix(_fp.folded_weight[c], _int.s2[c], _int.z2[c], _int.w_bit))
                _int.folded_bias[c].data.copy_(
                    quantize_matrix(_fp.folded_bias[c], _int.s1[c] * _int.s2[c], 0, bit=32, symmetric=True))
                _int.sum_a2[c].data.copy_(
                    torch.sum(_int.folded_weight[c], dim=(1, 2, 3))[:, None, None])
        else:                  # single batch norm parameters
            _int.folded_weight[0].data.copy_(quantize_matrix(_fp.folded_weight[0], _int.s2, _int.z2, _int.w_bit))
            for c in range(_fp.num_clusters):
                _int.folded_bias[c].data.copy_(quantize_matrix(_fp.folded_bias[0], _int.s1[c] * _int.s2, 0, bit=32, symmetric=True))
            _int.sum_a2.data.copy_(torch.sum(_int.folded_weight[0], dim=(1, 2, 3)).reshape(1, _int.out_channels, 1, 1))

    else:   # fused fold
        if _int.per_channel:
            _int.folded_weight.data.copy_(quantize_matrix(_fp.folded_weight, _int.s2[:, None, None, None], _int.z2[:, None, None, None], _int.w_bit, symmetric=symmetric))
            _int.folded_bias.data.copy_(quantize_matrix(_fp.folded_bias[0], _int.s1 * _int.s2, 0, bit=32, symmetric=True))
        else:
            _int.folded_weight.data.copy_(quantize_matrix(_fp.folded_weight, _int.s2, _int.z2, _int.w_bit))
            _int.folded_bias.data.copy_(
                quantize_matrix(_fp.folded_bias, _int.s1 * _int.s2, 0, bit=32, symmetric=True))
        _int.sum_a2.data.copy_(torch.sum(_int.folded_weight, dim=(1, 2, 3)).reshape(1, _int.out_channels, 1, 1))
    return _int


def quantize_conv2d_weight(_fp, _int, symmetric):
    # _int.weight.data.copy_(quantize_matrix(_fp.weight, _int.s2, _int.z2, _int.w_bit))
    # _int.sum_a2.data.copy_(torch.sum(_int.weight, dim=(1, 2, 3)).reshape(1, _int.out_channels, 1, 1))
    if _int.per_channel:
        _int.weight.data.copy_(quantize_matrix(_fp.weight, _int.s2[:, None, None, None],
                                               _int.z2[:, None, None, None], _int.w_bit, symmetric=symmetric))
    else:
        _int.weight.data.copy_(quantize_matrix(_fp.weight, _int.s2, _int.z2, _int.w_bit, symmetric=symmetric))
    _int.sum_a2.data.copy_(torch.sum(_int.weight, dim=(1, 2, 3)).reshape(1, _int.out_channels, 1, 1))
    return _int


def quantize_conv2d_weight_in_channel(_fp, _int, symmetric):
    _int.weight.data.copy_(
        quantize_matrix_per_in_channel(_fp.conv.weight, _int.s2, _int.z2, _int.low_bit, _int.low_group, _int.high_group,
                                       symmetric=symmetric))
    _int.sum_a2.data.copy_(torch.sum(_int.weight, dim=(1, 2, 3)).reshape(1, _int.out_channels, 1, 1))
    return _int


def quantize_fc_weight(_fp, _int, symmetric):
    _int.weight.data.copy_(quantize_matrix(_fp.weight, _int.s2, _int.z2, _int.w_bit, symmetric=symmetric))
    _int.sum_a2.data.copy_(torch.sum(_int.weight, dim=1).reshape(1, _int.out_features))
    return _int


def quantize_bn(_fp, _int):
    if _int.num_clusters > 1:
        _size = (_fp.num_clusters, _fp.num_features)
        _weights = torch.zeros(_size, device='cuda')
        _biases = torch.zeros(_size, device='cuda')
        _means = torch.zeros(_size, device='cuda')
        _vars = torch.zeros(_size, device='cuda')
        for c in range(_fp.num_clusters):
            _weights[c] = _fp.norms[c].weight.clone().detach()
            _biases[c] = _fp.norms[c].bias.clone().detach()
            _means[c] = _fp.norms[c].running_mean.clone().detach()
            _vars[c] = _fp.norms[c].running_var.clone().detach()

        weight = _weights.div(torch.sqrt(_vars + _fp.norms[0].eps))
        bias = _biases - weight * _means
        weight = quantize_matrix(weight, _int.s2, _int.z2, _fp.w_bit)
        _int.weight.copy_(weight.type(torch.cuda.IntTensor))
        for c in range(_int.num_clusters):
            b = quantize_matrix(bias[c], _int.s1[c] * _int.s2, 0, 32)
            _int.bias[c].copy_(b.type(torch.cuda.IntTensor))
    else:
        w = _fp.bn.weight.clone().detach().div(torch.sqrt(_fp.bn.running_var.clone().detach() + _fp.bn.eps))
        b = _fp.bn.bias.clone().detach() - w * _fp.bn.running_mean.clone().detach()
        w = quantize_matrix(w, _int.s2, _int.z2, _fp.w_bit)
        b = quantize_matrix(b, _int.s1 * _int.s2, 0, 32)

        _int.weight[0].copy_(w.type(torch.cuda.IntTensor))
        _int.bias[0].copy_(b.type(torch.cuda.IntTensor))
    return _int


def quantize_layer(_fp, _int):
    with torch.no_grad():
        fold_flag = False
        if _int.layer_type == 'QuantizedBn2d':
            return quantize_bn(_fp, _int)

        if _int.layer_type == 'QuantizedConv2d':
            fp_layer = _fp.conv
            if _fp.fold_convbn:
                fold_flag = True
                _int = quantize_folded_conv2d_weight_and_bias(_fp, _int, _fp.symmetric)
            elif _fp.mixed_precision:
                _int = quantize_conv2d_weight_in_channel(_fp, _int, _fp.symmetric)
            else:
                _int = quantize_conv2d_weight(fp_layer, _int, _fp.symmetric)
        else:
            fp_layer = _fp.fc
            _int = quantize_fc_weight(fp_layer, _int, _fp.symmetric)

        if fp_layer.bias is not None:
            _int.is_bias.data = torch.tensor(True, dtype=torch.bool)

            if not fold_flag:
                if _int.num_clusters > 1:
                    for c in range(_int.num_clusters):
                        _int.quantized_bias[c].copy_(quantize_matrix(fp_layer.bias, _int.s1[c] * _int.s2, 0, bit=32, symmetric=True))
                else:
                    _int.quantized_bias[0].copy_(quantize_matrix(fp_layer.bias, _int.s1 * _int.s2, 0, bit=32, symmetric=True))
    return _int


def quantize(_fp, _int):
    assert _int.layer_type in ['QuantizedConv2d', 'QuantizedLinear', 'QuantizedBn2d'], "Not supported quantized layer"
    _int = transfer_qparams(_fp, _int)
    _int = quantize_layer(_fp, _int)
    return _int


def copy_from_pretrained(_to, _from, norm_layer=None, is_folded=False):
    # Copy weights from pretrained FP model
    with torch.no_grad():
        if 'Conv' in _to.layer_type:
            _to.conv.weight.copy_(_from.weight)
            if norm_layer:
                _to._norm_layer = deepcopy(norm_layer)
            else:
                if is_folded:
                    assert norm_layer is not None, "batch norm layer should be not None while folding"
                if _from.bias is not None:
                    _to.conv.bias.copy_(_from.bias)
        elif 'Linear' in _to.layer_type:
            _to.fc.weight.copy_(_from.weight)
            _to.fc.bias.copy_(_from.bias)
        else:
            _to._norm_layer = deepcopy(_from)
    return _to


def folded_pcq_copy_from_pretrained(_to, _from, norm_layer, momentum, num_norms):
    # Copy weights from pretrained FP model
    with torch.no_grad():
        if 'Conv' in _to.layer_type:
            _to.conv.weight.copy_(_from.weight)
            assert norm_layer is not None, "batch norm layer should be not None while folding"
            if norm_layer:
                for c in range(num_norms):
                    _to._norm_layer[c] = deepcopy(norm_layer)
                    _to._norm_layer[c].momentum = momentum
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


def copy_pcq_bn_from_pretrained(_to, _from, num_clusters, momentum):
    with torch.no_grad():
        for c in range(num_clusters):
            _to.norms[c] = deepcopy(_from)
            _to.norms[c].momentum = momentum
    return _to


def copy_weight_from_pretrained(_to, _from):
    # Copy weights from pretrained FP model
    with torch.no_grad():
        # Origin
        if 'Conv' in _to.layer_type:
            _to.conv.weight.copy_(_from.weight)

        # To HAWQ
        # if 'Conv' in str(type(_to)).split('.')[-1]:
        #     _to.weight.copy_(_from.weight)
        else:
            _to.fc.weight.copy_(_from.weight)       # Origin
            # _to.weight.copy_(_from.weight)        # To HAWQ
    return _to

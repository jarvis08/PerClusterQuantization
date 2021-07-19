import torch.nn as nn
import torch
import torch.nn.functional as F

from ..quant_noise import _quant_noise
from ..quantization_utils import *
from .activation import *
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Sequence


class QuantizedConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, activation_layer=None, dilation=1, groups=1, bias=False, bit=8, num_clusters=1):
        super(QuantizedConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.layer_type = 'QuantizedConv2d'
        self.bit = bit
        self.q_max = 2 ** bit - 1
        self.num_clusters = num_clusters
        self.quantized_bias = nn.Parameter(torch.zeros((num_clusters, out_channels)), requires_grad=False)
        t_init = list(range(num_clusters)) if num_clusters > 1 else 0
        self.s1 = nn.Parameter(torch.tensor(t_init, dtype=torch.float32), requires_grad=False)
        self.s2 = nn.Parameter(torch.tensor(0, dtype=torch.float32), requires_grad=False)
        self.s3 = nn.Parameter(torch.tensor(t_init, dtype=torch.float32), requires_grad=False)
        self.z1 = nn.Parameter(torch.tensor(t_init, dtype=torch.int32), requires_grad=False)
        self.z2 = nn.Parameter(torch.tensor(0, dtype=torch.int32), requires_grad=False)
        self.z3 = nn.Parameter(torch.tensor(t_init, dtype=torch.int32), requires_grad=False)
        self.M0 = nn.Parameter(torch.tensor(t_init, dtype=torch.int32), requires_grad=False)
        self.shift = nn.Parameter(torch.tensor(t_init, dtype=torch.int32), requires_grad=False)
        self.activation_layer = activation_layer
        self.lookup_table = nn.Parameter(torch.zeros(self.q_max), requires_grad=False)

    def forward(self, x, cluster_info):
        if cluster_info is not None:
            return self.pcq(x, cluster_info)
        else:
            if self.activation_layer:
                x = self.general(x)
                return QuantizedActivation(self.activation_layer, bit=self.bit)(x)
            return self.general(x)

    def pcq(self, x, cluster_info):
        if self.padding[0] > 0 or self.padding[1] > 0:
            done = 0
            padded = torch.zeros((x.shape[0], x.shape[1], x.shape[2] + self.padding[0] * 2, x.shape[3] + self.padding[1] * 2)).cuda()
            for i in range(cluster_info.shape[0]):
                c = cluster_info[i][0].item()
                n = cluster_info[i][1].item()
                padded[done:done + n] = F.pad(x[done:done + n], (self.padding[0], self.padding[0], self.padding[1], self.padding[1]), mode='constant', value=self.z1[c].item())
                done += n
            sum_q1q2 = F.conv2d(padded, self.weight, None, self.stride, (0, 0), self.dilation, self.groups)
            return self.pcq_totalsum(padded, sum_q1q2.type(torch.cuda.IntTensor), cluster_info)
        else:
            sum_q1q2 = F.conv2d(x, self.weight, None, self.stride, (0, 0), self.dilation, self.groups)
            return self.pcq_totalsum(x, sum_q1q2.type(torch.cuda.IntTensor), cluster_info)

    def general(self, x):
        if self.padding[0] > 0 or self.padding[1] > 0:
            x = F.pad(x, (self.padding[0], self.padding[0], self.padding[1], self.padding[1]), mode='constant', value=self.z1.item())
        sum_q1q2 = F.conv2d(x, self.weight, None, self.stride, (0, 0), self.dilation, self.groups)
        return self.general_totalsum(x, sum_q1q2.type(torch.cuda.IntTensor))

    def pcq_totalsum(self, x, sum_q1q2, cluster_info):
        input_batch, input_ch, input_col, input_row = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
        filter_batch, filter_ch, filter_col, filter_row = self.weight.shape[0], self.weight.shape[1], self.weight.shape[
            2], self.weight.shape[3]
        stride = self.stride[0]

        done = 0
        for i in range(cluster_info.shape[0]):
            c = cluster_info[i][0].item()
            n = cluster_info[i][1].item()
            for output_ch in range(filter_batch):
                sum_q1q2[done:done + n, output_ch, :, :] = sum_q1q2[done:done + n, output_ch, :, :].add(self.quantized_bias[c][output_ch])
            done += n

        output_col = sum_q1q2.shape[2]
        output_row = sum_q1q2.shape[3]
        sum_a1 = torch.zeros((input_batch, output_col, output_row), dtype=torch.int32).cuda()
        sum_a2 = torch.zeros((cluster_info.shape[0], filter_batch), dtype=torch.int32).cuda()

        for i in range(cluster_info.shape[0]):
            c = cluster_info[i][0].item()
            for output_ch in range(0, filter_batch):
                sum_a2[i, output_ch] = torch.sum(self.weight.data[output_ch, :, :, :]).mul(self.z1[c])

        for o_col in range(0,output_col):
            for o_row in range(0, output_row):
                col_st, col_end = o_col * stride, o_col * stride + filter_col
                row_st, row_end = o_row * stride, o_row * stride + filter_row
                sum_a1[:, o_col, o_row] = torch.sum(x[:, :, col_st: col_end, row_st: row_end], (1, 2, 3)).mul(self.z2)

        done = 0
        for i in range(cluster_info.shape[0]):
            c = cluster_info[i][0].item()
            n = cluster_info[i][1].item()
            nz1z2 = input_ch * filter_col * filter_row * self.z1[c] * self.z2
            sum_q1q2[done:done + n] = sum_q1q2[done:done + n].add(nz1z2)
            done += n

        for i_batch in range(input_batch):
            sum_q1q2[i_batch] = torch.sub(sum_q1q2[i_batch], sum_a1[i_batch])

        done = 0
        for i in range(cluster_info.shape[0]):
            n = cluster_info[i][1].item()
            for out_c in range(filter_batch):
                sum_q1q2[done:done + n, out_c] = torch.sub(sum_q1q2[done:done + n:, out_c], sum_a2[i, out_c])
            done += n

        done = 0
        total = torch.zeros(sum_q1q2.shape, dtype=torch.int32).cuda()
        for i in range(cluster_info.shape[0]):
            c = cluster_info[i][0].item()
            n = cluster_info[i][1].item()
            multiplied = multiply_M(sum_q1q2[done:done + n].type(torch.cuda.LongTensor), self.M0[c])
            total[done:done + n] = shifting(multiplied, self.shift[c].item())
            total[done:done + n] = total[done:done + n].add(self.z3[c])
            done += n

        if self.bit == 4:
            total = torch.clamp(total, 0, 15)
        else: 
            total = torch.clamp(total, -128, 127)
        return total.type(torch.cuda.FloatTensor)

    def general_totalsum(self, x, sum_q1q2):
        input_batch, input_ch, input_col, input_row = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
        filter_batch, filter_ch, filter_col, filter_row = self.weight.shape[0], self.weight.shape[1], self.weight.shape[2], self.weight.shape[3]
        stride = self.stride[0]

        for output_ch in range(filter_batch):
            sum_q1q2[:, output_ch, :, :] = sum_q1q2[:, output_ch, :, :].add(self.quantized_bias[0][output_ch])

        output_col = sum_q1q2.shape[2]
        output_row = sum_q1q2.shape[3]
        sum_a1 = torch.zeros((input_batch, output_col, output_row), dtype=torch.int32).cuda()
        sum_a2 = torch.zeros(filter_batch, dtype=torch.int32).cuda()

        for output_ch in range(0, filter_batch):
            sum_a2[output_ch] = torch.sum(self.weight.data[output_ch, :, :, :]).mul(self.z1)

        for o_col in range(0, output_col):
            for o_row in range(0, output_row):
                col_st, col_end = o_col * stride, o_col * stride + filter_col
                row_st, row_end = o_row * stride, o_row * stride + filter_row
                sum_a1[:, o_col, o_row] = torch.sum(x[:, :, col_st: col_end, row_st: row_end], (1, 2, 3)).mul(self.z2)
        nz1z2 = input_ch * filter_col * filter_row * self.z1 * self.z2
        sum_q1q2 = sum_q1q2.add(nz1z2)

        for i_batch in range(0, input_batch):
            sum_q1q2[i_batch, :] = torch.sub(sum_q1q2[i_batch, :], sum_a1[i_batch])
        for out_c in range(0, filter_batch):
            sum_q1q2[:, out_c] = torch.sub(sum_q1q2[:, out_c], sum_a2[out_c])

        multiplied = multiply_M(sum_q1q2.type(torch.cuda.LongTensor), self.M0)
        total = shifting(multiplied, self.shift.item())
        total = total.add(self.z3)
        if self.bit == 4:
            total = torch.clamp(total, 0, 15)
        else:
            total = torch.clamp(total, -128, 127)
        return total.type(torch.cuda.FloatTensor)


class PCQConv2d(nn.Module):
    """
        Fused Layer to calculate Quantization Parameters(S & Z) with multiple clusters
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, bias=False,
                 norm_layer=None, activation=None, bit=8, smooth=0.999, num_clusters=10):
        super(PCQConv2d, self).__init__()
        self.layer_type = 'PCQConv2d'
        self.out_channels = out_channels
        self.groups = groups
        self.bit = bit
        self.q_max = 2 ** bit - 1
        self.smooth = smooth
        self.flag_ema_init = np.zeros(num_clusters, dtype=bool)
        self.flag_fake_quantization = False
        self.act_range = nn.Parameter(torch.zeros((num_clusters, 2)), requires_grad=False)
        self.num_clusters = num_clusters

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              groups=groups,  bias=bias)
        self._norm_layer = norm_layer(out_channels) if norm_layer else None
        self._activation = activation(inplace=False) if activation else None

    def forward(self, x, cluster_info):
        if self.training:
            s, z = calc_qparams(torch.min(self.conv.weight), torch.max(self.conv.weight), self.q_max)
            with torch.no_grad():
                self.conv.weight.copy_(fake_quantize(self.conv.weight.detach(), s, z, self.q_max))

        x = self.conv(x)
        if self._norm_layer:
            x = self._norm_layer(x)
        if self._activation:
            x = self._activation(x)

        if self.training:
            done = 0
            for i in range(cluster_info.shape[0]):
                c = cluster_info[i][0].item()
                n = cluster_info[i][1].item()
                if self.flag_ema_init[c]:
                    self.act_range[c][0], self.act_range[c][1] = ema(x[done:done + n], self.act_range[c], self.smooth)
                    if self.flag_fake_quantization:
                        s, z = calc_qparams(self.act_range[c][0], self.act_range[c][1], self.q_max)
                        with torch.no_grad():
                            x[done:done + n].copy_(fake_quantize(x[done:done + n].detach(), s, z, self.q_max))
                else:
                    self.act_range[c][0] = torch.min(x[done:done + n]).item()
                    self.act_range[c][1] = torch.max(x[done:done + n]).item()
                    self.flag_ema_init[c] = True
                done += n
        return x

    def set_fake_quantization_flag(self):
        self.flag_fake_quantization = True

    def fuse_conv_and_bn(self):
        # In case of validation, fuse pretrained Conv&BatchNorm params
        assert self.training == False, "Do not fuse layers while training."
        alpha, beta, mean, var, eps = self._norm_layer.weight, self._norm_layer.bias, self._norm_layer.running_mean,\
                                      self._norm_layer.running_var, self._norm_layer.eps
        n_channel = self.conv.weight.shape[0]
        self.conv.bias = nn.Parameter(beta)
        for c in range(n_channel):
            self.conv.weight.data[c] = self.conv.weight.data[c].mul(alpha[c]).div(torch.sqrt(var[c].add(eps)))
            self.conv.bias.data[c] = self.conv.bias.data[c].sub(alpha[c].mul(mean[c]).div(torch.sqrt(var[c])))
        self._norm_layer = nn.Identity()

    def set_qparams(self, s1, z1):
        self.s1, self.z1 = torch.nn.Parameter(s1, requires_grad=False), torch.nn.Parameter(z1, requires_grad=False)

        self.s2, self.z2 = calc_qparams(torch.min(self.conv.weight), torch.max(self.conv.weight), self.q_max)

        self.s3 = nn.Parameter(torch.zeros(self.num_clusters, dtype=torch.float32), requires_grad=False)
        self.z3 = nn.Parameter(torch.zeros(self.num_clusters, dtype=torch.int32), requires_grad=False)
        self.M0 = nn.Parameter(torch.zeros(self.num_clusters, dtype=torch.int32), requires_grad=False)
        self.shift = nn.Parameter(torch.zeros(self.num_clusters, dtype=torch.int32), requires_grad=False)
        for c in range(self.num_clusters):
            self.s3[c], self.z3[c] = calc_qparams(self.act_range[c][0], self.act_range[c][1], self.q_max)
            self.M0[c], self.shift[c] = quantize_M(self.s1[c] * self.s2 / self.s3[c])
        return self.s3, self.z3


class FusedConv2d(nn.Module):
    """
        Fused Layer to calculate Quantization Parameters (S & Z)
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False,
                 norm_layer=None, activation=None, smooth=0.995, bit=32, quant_noise=False, q_prob=0.1):
        super(FusedConv2d, self).__init__()
        self.layer_type = 'FusedConv2d'
        self.bit = bit
        self.q_max = 2 ** bit - 1
        self.smooth = smooth
        self.flag_ema_init = False
        self.flag_fake_quantization = False
        self.act_range = nn.Parameter(torch.zeros(2), requires_grad=False)
        self.groups = groups
        
        self.quant_noise = quant_noise
        if quant_noise:
            self.q_prob = q_prob
            self.conv = _quant_noise(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=self.groups, bias=bias), self.q_prob, 1, q_max=self.q_max)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              groups=self.groups, bias=bias)
        self._norm_layer = norm_layer(out_channels) if norm_layer else None
        self._activation = activation(inplace=False) if activation else None

    def forward(self, x):
        if self.training and not self.quant_noise:
            s, z = calc_qparams(torch.min(self.conv.weight), torch.max(self.conv.weight), self.q_max)
            with torch.no_grad():
                self.conv.weight.copy_(fake_quantize(self.conv.weight.detach(), s, z, self.q_max))

        x = self.conv(x)
        if self._norm_layer:
            x = self._norm_layer(x)
        if self._activation:
            x = self._activation(x)

        if self.training:
            if self.flag_ema_init:
                self.act_range[0], self.act_range[1] = ema(x.detach(), self.act_range, self.smooth)
                if self.flag_fake_quantization:
                    s, z = calc_qparams(self.act_range[0], self.act_range[1], self.q_max)
                    with torch.no_grad():
                        x.copy_(fake_quantize(x, s, z, self.q_max))
            else:
                self.act_range[0] = torch.min(x).item()
                self.act_range[1] = torch.max(x).item()
                self.flag_ema_init = True
        return x

    def set_fake_quantization_flag(self):
        self.flag_fake_quantization = True

    def fuse_conv_and_bn(self):
        # In case of validation, fuse pretrained Conv&BatchNorm params
        assert self.training == False, 'Do not fuse layers while training.'
        alpha, beta, mean, var, eps = self._norm_layer.weight, self._norm_layer.bias, self._norm_layer.running_mean,\
                                      self._norm_layer.running_var, self._norm_layer.eps
        n_channel = self.conv.weight.shape[0]
        self.conv.bias = nn.Parameter(beta)
        for c in range(n_channel):
            self.conv.weight.data[c] = self.conv.weight.data[c].mul(alpha[c]).div(torch.sqrt(var[c].add(eps)))
            self.conv.bias.data[c] = self.conv.bias.data[c].sub(alpha[c].mul(mean[c]).div(torch.sqrt(var[c])))
        self._norm_layer = nn.Identity()

    def set_qparams(self, s1, z1):
        self.s1, self.z1 = nn.Parameter(s1, requires_grad=False), nn.Parameter(z1, requires_grad=False)
        self.s2, self.z2 = calc_qparams(torch.min(self.conv.weight), torch.max(self.conv.weight), self.q_max)
        self.s3, self.z3 = calc_qparams(self.act_range[0], self.act_range[1], self.q_max)
        self.M0, self.shift = quantize_M(self.s1 * self.s2 / self.s3)
        return self.s3, self.z3

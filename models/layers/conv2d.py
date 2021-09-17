from operator import itemgetter

import torch.nn as nn
import torch
import torch.nn.functional as F

from ..quant_noise import _quant_noise
from ..quantization_utils import *
from .activation import *


class QuantizedConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, activation=None,
                 dilation=1, groups=1, bias=False, arg_dict=None):
        super(QuantizedConv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                                              padding, dilation, groups, bias)
        self.layer_type = 'QuantizedConv2d'
        self.bit, self.num_clusters, self.runtime_helper = itemgetter('bit', 'cluster', 'runtime_helper')(arg_dict)
        self.q_max = 2 ** self.bit - 1
        self.act_qmax = nn.Parameter(torch.tensor(0, dtype=torch.int32), requires_grad=False)

        self.is_bias = nn.Parameter(torch.tensor(False, dtype=torch.bool), requires_grad=False)
        self.quantized_bias = nn.Parameter(torch.zeros((self.num_clusters, out_channels), dtype=torch.int32), requires_grad=False)

        t_init = list(range(self.num_clusters)) if self.num_clusters > 1 else 0
        self.s1 = nn.Parameter(torch.tensor(t_init, dtype=torch.float32), requires_grad=False)
        self.s2 = nn.Parameter(torch.tensor(0, dtype=torch.float32), requires_grad=False)
        self.s3 = nn.Parameter(torch.tensor(t_init, dtype=torch.float32), requires_grad=False)
        self.z1 = nn.Parameter(torch.tensor(t_init, dtype=torch.int32), requires_grad=False)
        self.z2 = nn.Parameter(torch.tensor(0, dtype=torch.int32), requires_grad=False)
        self.z3 = nn.Parameter(torch.tensor(t_init, dtype=torch.int32), requires_grad=False)
        self.M0 = nn.Parameter(torch.tensor(t_init, dtype=torch.int32), requires_grad=False)
        self.shift = nn.Parameter(torch.tensor(t_init, dtype=torch.int32), requires_grad=False)
        self.hardswish_6 = nn.Parameter(torch.tensor(t_init, dtype=torch.int32), requires_grad=False)
        self.hardswish_3 = nn.Parameter(torch.tensor(t_init, dtype=torch.int32), requires_grad=False)
        self.s_activation = nn.Parameter(torch.tensor(t_init, dtype=torch.float32), requires_grad=False)
        self.z_activation = nn.Parameter(torch.tensor(t_init, dtype=torch.int32), requires_grad=False)
        self.activation = activation

    def forward(self, x):
        if self.runtime_helper.batch_cluster is not None:
            return self.pcq(x)
        else:
            return self.general(x)

    def pcq(self, x):
        if self.padding[0] > 0 or self.padding[1] > 0:
            bc = self.runtime_helper.batch_cluster
            exists = torch.unique(bc)
            padded = torch.zeros((x.shape[0], x.shape[1], x.shape[2] + self.padding[0] * 2, x.shape[3] + self.padding[1] * 2)).cuda()
            for c in exists:
                indices = (bc == c).nonzero(as_tuple=True)[0]
                padded[indices] = F.pad(x[indices], (self.padding[0], self.padding[0], self.padding[1], self.padding[1]), mode='constant', value=self.z1[c])
            #z1 = torch.index_select(self.z1, 0, bc)
            #for i in range(len(self.runtime_helper.batch_cluster)):
            #    padded[i] = F.pad(x[i], (self.padding[0], self.padding[0], self.padding[1], self.padding[1]), mode='constant', value=z1[i])
            sum_q1q2 = F.conv2d(padded, self.weight, None, self.stride, (0, 0), self.dilation, self.groups)
            return self.pcq_totalsum(padded, sum_q1q2.type(torch.cuda.IntTensor))
        else:
            sum_q1q2 = F.conv2d(x, self.weight, None, self.stride, (0, 0), self.dilation, self.groups)
            return self.pcq_totalsum(x, sum_q1q2.type(torch.cuda.IntTensor))

    def general(self, x):
        if self.padding[0] > 0 or self.padding[1] > 0:
            x = F.pad(x, (self.padding[0], self.padding[0], self.padding[1], self.padding[1]), mode='constant', value=self.z1.item())
        sum_q1q2 = F.conv2d(x, self.weight, None, self.stride, (0, 0), self.dilation, self.groups)

        if self.groups > 1:
            return self.depthwise_totalsum(x, sum_q1q2.type(torch.cuda.IntTensor))
        return self.general_totalsum(x, sum_q1q2.type(torch.cuda.IntTensor))

    def pcq_totalsum(self, x, sum_q1q2):
        bc = self.runtime_helper.batch_cluster
        z1 = torch.index_select(self.z1, 0, bc)
        z3 = torch.index_select(self.z3, 0, bc).reshape(bc.shape[0], 1, 1, 1)
        M0 = torch.index_select(self.M0, 0, bc).reshape(bc.shape[0], 1, 1, 1)
        shift = torch.index_select(self.shift, 0, bc)

        input_batch, input_ch, input_col, input_row = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
        filter_batch, filter_ch, filter_col, filter_row =\
            self.weight.shape[0], self.weight.shape[1], self.weight.shape[2], self.weight.shape[3]
        stride = self.stride[0]

        if self.is_bias:
            bias = torch.index_select(self.quantized_bias, 0, bc)
            for output_ch in range(filter_batch):
                sum_q1q2[:, output_ch, :, :] = sum_q1q2[:, output_ch, :, :].add_(bias[:, output_ch].reshape(bc.shape[0], 1, 1))

        output_col = sum_q1q2.shape[2]
        output_row = sum_q1q2.shape[3]
        sum_a1 = torch.zeros((input_batch, output_col, output_row), dtype=torch.int32).cuda()
        sum_a2 = torch.zeros((bc.shape[0], filter_batch), dtype=torch.int32).cuda()

        for output_ch in range(0, filter_batch):
            sum_a2[:, output_ch] = torch.sum(self.weight.data[output_ch, :, :, :]).mul(z1)

        for o_col in range(0,output_col):
            for o_row in range(0, output_row):
                col_st, col_end = o_col * stride, o_col * stride + filter_col
                row_st, row_end = o_row * stride, o_row * stride + filter_row
                sum_a1[:, o_col, o_row] = torch.sum(x[:, :, col_st: col_end, row_st: row_end], (1, 2, 3)).mul(self.z2)

        nz1z2 = input_ch * filter_col * filter_row * z1 * self.z2
        sum_q1q2 = sum_q1q2.add(nz1z2.type(torch.cuda.IntTensor).reshape(bc.shape[0], 1, 1, 1))

        for i_batch in range(input_batch):
            sum_q1q2[i_batch] = torch.sub(sum_q1q2[i_batch], sum_a1[i_batch])

        for out_c in range(filter_batch):
            sum_q1q2[:, out_c] = torch.sub(sum_q1q2[:, out_c], sum_a2[:, out_c].reshape(bc.shape[0], 1, 1))

        total = torch.zeros(sum_q1q2.shape, dtype=torch.int32).cuda()
        neg = (shift < 0).nonzero(as_tuple=True)[0]
        pos = (shift >= 0).nonzero(as_tuple=True)[0]
        if len(neg) > 0:
            s = - shift[neg].reshape(neg.shape[0], 1, 1, 1)
            subsum = multiply_M((sum_q1q2[neg].type(torch.cuda.LongTensor) << s), M0[neg])
            total[neg] = shifting(subsum, 0)
        if len(pos) > 0:
            s = shift[pos].reshape(pos.shape[0], 1, 1, 1)
            subsum = multiply_M(sum_q1q2[pos].type(torch.cuda.LongTensor), M0[pos])
            total[pos] = shifting4d(subsum, s)
        total = total.add(z3)

        if self.act_qmax == 15:
            total = torch.clamp(total, 0, 15)
        elif self.act_qmax == 255:
            total = torch.clamp(total, -128, 127)
        elif self.act_qmax == 65535:  # INT 16
            total = torch.clamp(total, -32768, 32767)
        elif self.act_qmax == 4294967295:  # INT 32
            total = torch.clamp(total, -2147483648, 2147483647)
        return total.type(torch.cuda.FloatTensor)

    def general_totalsum(self, x, sum_q1q2):
        input_batch, input_ch, input_col, input_row = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
        filter_batch, filter_ch, filter_col, filter_row = self.weight.shape[0], self.weight.shape[1], self.weight.shape[2], self.weight.shape[3]
        stride = self.stride[0]

        if self.is_bias:
            for output_ch in range(filter_batch):
                sum_q1q2[:, output_ch, :, :] = sum_q1q2[:, output_ch, :, :].add(self.quantized_bias[0][output_ch])
        output_col = sum_q1q2.shape[2]
        output_row = sum_q1q2.shape[3]
        sum_a1 = torch.zeros((input_batch, output_col, output_row), dtype=torch.int32).cuda()
        sum_a2 = torch.zeros(filter_batch, dtype=torch.int32).cuda()

        for output_ch in range(filter_batch):
            sum_a2[output_ch] = torch.sum(self.weight.data[output_ch, :, :, :]).mul(self.z1)

        for o_col in range(output_col):
            for o_row in range(output_row):
                col_st, col_end = o_col * stride, o_col * stride + filter_col
                row_st, row_end = o_row * stride, o_row * stride + filter_row
                sum_a1[:, o_col, o_row] = torch.sum(x[:, :, col_st: col_end, row_st: row_end], (1, 2, 3)).mul(self.z2)
        nz1z2 = input_ch * filter_col * filter_row * self.z1 * self.z2
        sum_q1q2 = sum_q1q2.add(nz1z2)

        for i_batch in range(input_batch):
            sum_q1q2[i_batch, :] = torch.sub(sum_q1q2[i_batch, :], sum_a1[i_batch])

        for out_c in range(filter_batch):
            sum_q1q2[:, out_c] = torch.sub(sum_q1q2[:, out_c], sum_a2[out_c])

        if self.shift < 0:
            multiplied = multiply_M((sum_q1q2.type(torch.cuda.LongTensor) << - self.shift.item()), self.M0)
            total = shifting(multiplied, 0)
        else:
            multiplied = multiply_M(sum_q1q2.type(torch.cuda.LongTensor), self.M0)
            total = shifting(multiplied, self.shift.item())
        total = total.add(self.z3)

        if self.activation is not None:
            # hs_total = total + self.hardswish_3
            # hs_total = torch.clamp(hs_total, self.z3.item(), self.hardswish_6.item())
            # if self.activation == 'Hardswish':
            #     total = total * hs_total / self.hardswish_6.item()
            # else:
            #     total = hs_total / self.hardswish_6.item()
            total = dequantize_matrix(total, self.s_activation, self.z_activation)
            total = nn.Hardswish(inplace=False)(total)
            total = quantize_matrix(total, self.s_activation, self.z_activation, self.q_max)

        if self.act_qmax == 15:
            total = torch.clamp(total, 0, 15)
        elif self.act_qmax == 255:
            total = torch.clamp(total, -128, 127)
        elif self.act_qmax == 65535:  # INT 16
            total = torch.clamp(total, -32768, 32767)
        elif self.act_qmax == 4294967295:  # INT 32
            total = torch.clamp(total, -2147483648, 2147483647)
        return total.type(torch.cuda.FloatTensor)

    def depthwise_totalsum(self, x, sum_q1q2):
        input_batch, input_ch, input_col, input_row = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
        filter_batch, filter_ch, filter_col, filter_row = self.weight.shape[0], self.weight.shape[1], self.weight.shape[2], self.weight.shape[3]
        stride = self.stride[0]

        for output_ch in range(filter_batch):
            sum_q1q2[:, output_ch, :, :] = sum_q1q2[:, output_ch, :, :].add(self.quantized_bias[0][output_ch])
        output_col = sum_q1q2.shape[2]
        output_row = sum_q1q2.shape[3]

        for output_ch in range(filter_batch):
            sum_q1q2[:, output_ch, :, :] = torch.sub(sum_q1q2[:, output_ch, :, :], torch.sum(self.weight.data[output_ch, :]).mul(self.z1))

        for o_col in range(output_col):
            for o_row in range(output_row):
                col_st, col_end = o_col * stride, o_col * stride + filter_col
                row_st, row_end = o_row * stride, o_row * stride + filter_row
                sum_q1q2[:, :, o_col, o_row] = torch.sub(sum_q1q2[:, :, o_col, o_row],
                                                         torch.sum(x[:, :, col_st: col_end, row_st: row_end], (2, 3)).mul(self.z2))

        nz1z2 = input_ch * filter_col * filter_row * self.z1 * self.z2
        sum_q1q2 = sum_q1q2.add(nz1z2)

        if self.shift < 0:
            multiplied = multiply_M((sum_q1q2.type(torch.cuda.LongTensor) << - self.shift.item()), self.M0)
            total = shifting(multiplied, 0)
        else:
            multiplied = multiply_M(sum_q1q2.type(torch.cuda.LongTensor), self.M0)
            total = shifting(multiplied, self.shift.item())
        total = total.add(self.z3)

        if self.activation is not None:
            hs_total = total + self.hardswish_3
            hs_total = torch.clamp(hs_total, self.z3.item(), self.hardswish_6.item())
            if self.activation == 'Hardswish':
                total = total * hs_total / self.hardswish_6.item()
            else:
                total = hs_total / self.hardswish_6.item()

        if self.bit == 4:
            total = torch.clamp(total, 0, 15)
        else:
            total = torch.clamp(total, -128, 127)
        return total.type(torch.cuda.FloatTensor)


class PCQConv2d(nn.Module):
    """
        Fused Layer to calculate Quantization Parameters(S & Z) with multiple clusters
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, dilation=1,
                 bias=False, activation=None, act_qmax=None, arg_dict=None):
        super(PCQConv2d, self).__init__()
        self.layer_type = 'PCQConv2d'
        self.out_channels = out_channels
        self.groups = groups

        self.bit, self.smooth, self.num_clusters, self.runtime_helper, self.use_ste, self.quant_noise, self.qn_prob\
            = itemgetter('bit', 'smooth', 'cluster', 'runtime_helper', 'ste', 'quant_noise', 'qn_prob')(arg_dict)
        self.q_max = 2 ** self.bit - 1
        self.act_qmax = act_qmax if act_qmax else self.q_max
        self.act_range = nn.Parameter(torch.zeros((self.num_clusters, 2)), requires_grad=False)
        self.apply_ema = False

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              groups=groups,  bias=bias, dilation=dilation)
        if self.quant_noise:
            self.conv = _quant_noise(self.conv, self.qn_prob, 1, self.q_max)

        self._activation = activation(inplace=False) if activation else None

    def forward(self, x, external_range=None):
        if not self.training:
            if self.runtime_helper.range_update_phase:  # Phase-2
                out = self._fake_quantized_conv(x)
                self._update_activation_ranges(out, external_range)
                if self.runtime_helper.apply_fake_quantization:
                    out = self._fake_quantize_activation(out, external_range)
                return out
            else:
                return self._forward_impl(x)

        # Phase-1
        out = self._fake_quantized_conv(x)
        if self.runtime_helper.apply_fake_quantization:
            out = self._fake_quantize_activation(out, external_range)
        return out

    def _forward_impl(self, x):
        x = self.conv(x)
        if self._activation:
            x = self._activation(x)
        return x

    def _fake_quantized_conv(self, x):
        is_phase1 = not self.runtime_helper.range_update_phase
        w = self.conv.weight
        if not self.quant_noise:
            s, z = calc_qparams(self.conv.weight.min(), self.conv.weight.max(), self.q_max)
            w = fake_quantize(self.conv.weight, s, z, self.q_max, use_ste=is_phase1)
        out = F.conv2d(x, w, self.conv.bias, self.conv.stride, self.conv.padding, self.conv.dilation, self.conv.groups)
        if self._activation:
            out = self._activation(out)
        return out

    def _fake_quantize_activation(self, x, external_range=None):
        if external_range is not None:
            s, z = calc_qparams_per_cluster(external_range, self.act_qmax)
        else:
            s, z = calc_qparams_per_cluster(self.act_range, self.act_qmax)
        return fake_quantize_per_cluster_4d(x, s, z, self.act_qmax, self.runtime_helper.batch_cluster, self.use_ste)

    def _update_activation_ranges(self, x, external_range=None):
        if external_range is not None:
            return None
        # Update of ranges only occures in Phase-2 :: data are sorted by cluster number
        # (number of data per cluster in batch) == (args.data_per_cluster)
        n = self.runtime_helper.data_per_cluster
        if self.apply_ema:
            for c in range(self.num_clusters):
                self.act_range[c][0], self.act_range[c][1] = ema(x[c * n: (c + 1) * n], self.act_range[c], self.smooth)
        else:
            for c in range(self.num_clusters):
                self.act_range[c][0] = x[c * n: (c + 1) * n].min().item()
                self.act_range[c][1] = x[c * n: (c + 1) * n].max().item()
            self.apply_ema = True

    def set_qparams(self, s1, z1, s_external=None, z_external=None):
        self.s1, self.z1 = torch.nn.Parameter(s1, requires_grad=False), torch.nn.Parameter(z1, requires_grad=False)

        self.s2, self.z2 = calc_qparams(torch.min(self.conv.weight), torch.max(self.conv.weight), self.q_max)

        if s_external:
            self.s3, self.z3 = nn.Parameter(s_external, requires_grad=False), \
                               nn.Parameter(z_external, requires_grad=False)
        else:
            self.s3, self.z3 = calc_qparams_per_cluster(self.act_range, self.act_qmax)

        self.M0 = nn.Parameter(torch.zeros(self.num_clusters, dtype=torch.int32), requires_grad=False)
        self.shift = nn.Parameter(torch.zeros(self.num_clusters, dtype=torch.int32), requires_grad=False)
        for c in range(self.num_clusters):
            self.M0[c], self.shift[c] = quantize_M(self.s1[c] * self.s2 / self.s3[c])
        return self.s3, self.z3


class FusedConv2d(nn.Module):
    """
        Fused Layer to calculate Quantization Parameters (S & Z)
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False,
                 norm_layer=None, activation=None, act_qmax=None, arg_dict=None):
        super(FusedConv2d, self).__init__()
        self.layer_type = 'FusedConv2d'
        self.groups = groups

        self.arg_dict = arg_dict
        self.bit, self.smooth, self.folded_fq, self.use_ste, self.runtime_helper, self.quant_noise, self.qn_prob\
            = itemgetter('bit', 'smooth', 'folded_fq', 'ste', 'runtime_helper', 'quant_noise', 'qn_prob')(arg_dict)

        self.q_max = 2 ** self.bit - 1
        self.act_qmax = act_qmax if act_qmax else 2 ** self.bit - 1
        self.act_range = nn.Parameter(torch.zeros(2), requires_grad=False)

        self.apply_ema = False

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              groups=self.groups, bias=bias, dilation=dilation)
        if self.quant_noise:
            self.conv = _quant_noise(self.conv, self.qn_prob, 1, q_max=self.q_max)
        self._norm_layer = norm_layer(out_channels) if norm_layer else None
        self._activation = activation(inplace=False) if activation else None
        self.out_channels = out_channels

    def forward(self, x, external_range=None):
        if not self.training:
            x = self.conv(x)
            if self._norm_layer:
                x = self._norm_layer(x)
            if self._activation:
                x = self._activation(x)
            return x

        if self.folded_fq:
            return self._norm_folded(x, external_range)
        else:
            return self._general(x, external_range)

    def _general(self, x, external_range=None):
        w = self.conv.weight
        if not self.quant_noise:
            s, z = calc_qparams(self.conv.weight.min(), self.conv.weight.max(), self.q_max)
            w = fake_quantize(self.conv.weight, s, z, self.q_max, self.use_ste)

        x = F.conv2d(x, w, self.conv.bias, self.conv.stride, self.conv.padding, self.conv.dilation, self.conv.groups)
        if self._norm_layer:
            x = self._norm_layer(x)
        if self._activation:
            x = self._activation(x)

        out = x
        if external_range is not None:
            if self.runtime_helper.apply_fake_quantization:
                s, z = calc_qparams(external_range[0], external_range[1], self.act_qmax)
                out = fake_quantize(x, s, z, self.act_qmax, self.use_ste)
        else:
            if self.apply_ema:
                self.act_range[0], self.act_range[1] = ema(x, self.act_range, self.smooth)
                if self.runtime_helper.apply_fake_quantization:
                    s, z = calc_qparams(self.act_range[0], self.act_range[1], self.act_qmax)
                    out = fake_quantize(x, s, z, self.act_qmax, self.use_ste)
            else:
                self.act_range[0] = torch.min(x).item()
                self.act_range[1] = torch.max(x).item()
                self.apply_ema = True
        return out

    def _norm_folded(self, x, external_range=None):
        general_out = self.conv(x)
        general_out = self._norm_layer(general_out)
        if self._activation:
            general_out = self._activation(general_out)

        with torch.no_grad():
            alpha, beta, mean, var, eps = self._norm_layer.weight, self._norm_layer.bias, self._norm_layer.running_mean, \
                                          self._norm_layer.running_var, self._norm_layer.eps
            n_channel = self.conv.weight.shape[0]

            folded_weight = self.conv.weight.clone().detach()
            folded_bias = beta.clone().detach()
            for c in range(n_channel):
                folded_weight.data[c] = folded_weight.data[c].mul(alpha[c]).div(torch.sqrt(var[c].add(eps)))
                folded_bias.data[c] = folded_bias.data[c].sub(alpha[c].mul(mean[c]).div(torch.sqrt(var[c])))

            s, z = calc_qparams(torch.min(folded_weight), torch.max(folded_weight), self.q_max)
            fq_folded_weight = fake_quantize(folded_weight, s, z, self.q_max, use_ste=False)

            folded_out = F.conv2d(x, fq_folded_weight, folded_bias, self.conv.stride, self.conv.padding,
                                  self.conv.dilation, self.conv.groups)
            if self._activation:
                folded_out = self._activation(folded_out)

            if external_range is None:
                if self.apply_ema:
                    self.act_range[0], self.act_range[1] = ema(folded_out, self.act_range, self.smooth)
                    if self.runtime_helper.apply_fake_quantization:
                        s, z = calc_qparams(self.act_range[0], self.act_range[1], self.act_qmax)
                        folded_out = fake_quantize(folded_out, s, z, self.act_qmax, use_ste=False)
                else:
                    self.act_range[0] = torch.min(folded_out).item()
                    self.act_range[1] = torch.max(folded_out).item()
                    self.apply_ema = True
            else:
                if self.runtime_helper.apply_fake_quantization:
                    s, z = calc_qparams(external_range[0], external_range[1], self.act_qmax)
                    folded_out = fake_quantize(folded_out, s, z, self.act_qmax, use_ste=False)
        return STE.apply(general_out, folded_out)

    def fold_conv_and_bn(self):
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

    def set_qparams(self, s1, z1, s_external=None, z_external=None):
        self.s1, self.z1 = nn.Parameter(s1, requires_grad=False), nn.Parameter(z1, requires_grad=False)
        self.s2, self.z2 = calc_qparams(self.conv.weight.min(), self.conv.weight.max(), self.q_max)

        if s_external:
            self.s3, self.z3 = nn.Parameter(s_external, requires_grad=False), nn.Parameter(z_external, requires_grad=False)
        else:
            self.s3, self.z3 = calc_qparams(self.act_range[0], self.act_range[1], self.act_qmax)

        self.M0, self.shift = quantize_M(self.s1 * self.s2 / self.s3)
        return self.s3, self.z3


import torch.nn as nn
import torch
import torch.nn.functional as F

from ..quantization_utils import *


class QuantizedConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, bit=8):
        super(QuantizedConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.layer_type = 'QuantizedConv2d'
        self.bit = bit
        self.q_max = 2 ** self.bit - 1
        self.s1 = nn.Parameter(torch.tensor(0, dtype=torch.float32), requires_grad=False)
        self.s2 = nn.Parameter(torch.tensor(0, dtype=torch.float32), requires_grad=False)
        self.s3 = nn.Parameter(torch.tensor(0, dtype=torch.float32), requires_grad=False)
        self.z1 = nn.Parameter(torch.tensor(0, dtype=torch.int32), requires_grad=False)
        self.z2 = nn.Parameter(torch.tensor(0, dtype=torch.int32), requires_grad=False)
        self.z3 = nn.Parameter(torch.tensor(0, dtype=torch.int32), requires_grad=False)
        self.M0 = nn.Parameter(torch.tensor(0, dtype=torch.int32), requires_grad=False)
        self.shift = nn.Parameter(torch.tensor(0, dtype=torch.int32), requires_grad=False)

    def forward(self, x):
        if self.padding[0] > 0 or self.padding[1] > 0:
             x = F.pad(x, (self.padding[0], self.padding[0], self.padding[1], self.padding[1]), mode='constant', value=self.z1.item())
        sum_q1q2 = F.conv2d(x, self.weight, None, self.stride, (0, 0), self.dilation, self.groups)
        return self.totalsum(x, sum_q1q2.type(torch.cuda.IntTensor))

    def totalsum(self, x, sum_q1q2):
        input_batch, input_ch, input_col, input_row = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
        filter_batch, filter_ch, filter_col, filter_row = self.weight.shape[0], self.weight.shape[1], self.weight.shape[2], self.weight.shape[3]
        stride = self.stride[0]

        if self.bias is not None:
            for output_ch in range(0, filter_batch):
                sum_q1q2[:, output_ch, :, :] = sum_q1q2[:, output_ch, :, :].add(self.bias[output_ch])

        output_col = sum_q1q2.shape[2]
        output_row = sum_q1q2.shape[3]
        sum_a1 = torch.zeros((input_batch, output_col, output_row), dtype=torch.int32).cuda()
        sum_a2 = torch.zeros(filter_batch, dtype=torch.int32).cuda()

        for output_ch in range(0, filter_batch):
            sum_a2[output_ch] = torch.sum(self.weight.data[output_ch, :, :, :]).mul(self.z1)
        for o_col in range(0,output_col):
            for o_row in range(0, output_row):
                col_st, col_end = o_col * stride, o_col * stride + filter_col
                row_st, row_end = o_row * stride, o_row * stride + filter_row
                sum_a1[:, o_col, o_row] = self.z2 * torch.sum(x[:, :, col_st: col_end, row_st: row_end], (1, 2, 3))
               
        nz1z2 = input_ch * filter_col * filter_row * self.z1 * self.z2
        sub_sum = sum_q1q2.add(nz1z2)
        for i_batch in range(0, input_batch):
            for out_c in range(0, filter_batch):
                sub_sum[i_batch, out_c] = torch.sub(sub_sum[i_batch, out_c], sum_a1[i_batch])
                sub_sum[i_batch, out_c] = torch.sub(sub_sum[i_batch, out_c], sum_a2[out_c])

        multiplied = multiply_M(sub_sum.type(torch.cuda.LongTensor), self.M0)
        total = shifting(multiplied, self.shift.item())
        total = total.add(self.z3)
        if self.bit == 4:
            total = torch.clamp(total, 0, 15)
        else: 
            total = torch.clamp(total, -128, 127)
        return total.type(torch.cuda.FloatTensor)


class FakeConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False, bit=32):
        super(FakeConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.layer_type = 'FakeConv2d'
        self.bit = bit
        self.q_max = 2 ** self.bit - 1

    def forward(self, x):
        if self.training:
            self.fake_quantize_weight()
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

    def fake_quantize_weight(self):
        s, z = calc_qparams(torch.min(self.weight), torch.max(self.weight), self.q_max)
        self.weight.data = torch.round(self.weight.div(s).add(z)).sub(z).mul(s)


class FusedConv2d(nn.Module):
    """
        Fused Layer to calculate Quantization Parameters (S & Z)
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False, bn=False, relu=True, smooth=0.995, bit=32):
        super(FusedConv2d, self).__init__()
        self.layer_type = 'FusedConv2d'
        self.bit = bit
        self.q_max = 2 ** bit - 1
        self.ema_init = False
        self.smooth = smooth
        self.act_range = nn.Parameter(torch.zeros(2), requires_grad=False)
        self.s1 = nn.Parameter(torch.tensor(0, dtype=torch.float32), requires_grad=False)
        self.s2 = nn.Parameter(torch.tensor(0, dtype=torch.float32), requires_grad=False)
        self.s3 = nn.Parameter(torch.tensor(0, dtype=torch.float32), requires_grad=False)
        self.z1 = nn.Parameter(torch.tensor(0, dtype=torch.int32), requires_grad=False)
        self.z2 = nn.Parameter(torch.tensor(0, dtype=torch.int32), requires_grad=False)
        self.z3 = nn.Parameter(torch.tensor(0, dtype=torch.int32), requires_grad=False)
        self.M0 = nn.Parameter(torch.tensor(0, dtype=torch.int32), requires_grad=False)
        self.shift = nn.Parameter(torch.tensor(0, dtype=torch.int32), requires_grad=False)

        self.out_channels = out_channels
        self.conv = FakeConv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=bias, bit=bit)
        self.bn = nn.BatchNorm2d(out_channels) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        out = self.conv(x)
        if self.bn:
            out = self.bn(out)
        if self.relu:
            out = self.relu(out)

        if self.training:
            if self.ema_init:
                self.ema(out)
                out = self.fake_quantize_activation(out)
            else:
                self.act_range[0] = torch.min(out).item()
                self.act_range[1] = torch.max(out).item()
                self.ema_init = True
        return out

    def ema(self, x):
        _min = torch.min(x).item()
        _max = torch.max(x).item()
        self.act_range[0] = self.act_range[0] * self.smooth + _min * (1 - self.smooth)
        self.act_range[1] = self.act_range[1] * self.smooth + _max * (1 - self.smooth)

    def fake_quantize_activation(self, x):
        s, z = calc_qparams(self.act_range[0], self.act_range[1], self.q_max)
        x = torch.round(x.div(s).add(z)).sub(z).mul(s)
        return x

    def copy_from_pretrained(self, conv, bn):
        # Copy weights from pretrained FP model
        self.conv.weight.data = torch.nn.Parameter(conv.weight.data)
        if bn:
            self.bn = bn
        else:
            self.conv.bias.data = torch.nn.Parameter(conv.bias.data)

    def fuse_conv_and_bn(self):
        # In case of validation, fuse pretrained Conv&BatchNorm params
        assert self.training == False, "Do not fuse layers while training."
        alpha, beta, mean, var, eps = self.bn.weight, self.bn.bias, self.bn.running_mean, self.bn.running_var, self.bn.eps
        n_channel = self.conv.weight.shape[0]
        self.conv.bias = torch.nn.Parameter(beta)
        for c in range(n_channel):
            self.conv.weight.data[c] = self.conv.weight.data[c].mul(alpha[c]).div(torch.sqrt(var[c].add(eps)))
            self.conv.bias.data[c] = self.conv.bias.data[c].sub(alpha[c].mul(mean[c]).div(torch.sqrt(var[c])))
        self.bn = SkipBN()

    def set_conv_qparams(self, s1, z1):
        self.s1, self.z1 = torch.nn.Parameter(s1, requires_grad=False), torch.nn.Parameter(z1, requires_grad=False)
        self.s2, self.z2 = calc_qparams(torch.min(self.conv.weight), torch.max(self.conv.weight), self.q_max)
        self.s3, self.z3 = calc_qparams(self.act_range[0], self.act_range[1], self.q_max)
        self.M0, self.shift = quantize_M(self.s1 * self.s2 / self.s3)
        return self.s3, self.z3

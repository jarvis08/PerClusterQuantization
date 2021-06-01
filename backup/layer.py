# Additive Power-of-Two Quantization: An Efficient Non-uniform Discretization For Neural Networks
# Yuhang Li, Xin Dong, Wei Wang
# International Conference on Learning Representations (ICLR), 2020.


import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np


class SkipBN(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


def _fuse(conv, bn):
    conv.fuse_conv_and_bn(bn.weight, bn.bias, bn.running_mean, bn.running_var, bn.eps)
    bn = SkipBN()
    return conv, bn


def fuse_resnet18_layers(model):
    model.conv1, model.bn1 = _fuse(model.conv1, model.bn1)
    model.layer1[0].conv1, model.layer1[0].bn1 = _fuse(model.layer1[0].conv1, model.layer1[0].bn1)
    model.layer1[0].conv2, model.layer1[0].bn2 = _fuse(model.layer1[0].conv2, model.layer1[0].bn2)
    model.layer1[1].conv1, model.layer1[1].bn1 = _fuse(model.layer1[1].conv1, model.layer1[1].bn1)
    model.layer1[1].conv2, model.layer1[1].bn2 = _fuse(model.layer1[1].conv2, model.layer1[1].bn2)
    model.layer2[0].conv1, model.layer2[0].bn1 = _fuse(model.layer2[0].conv1, model.layer2[0].bn1)
    model.layer2[0].conv2, model.layer2[0].bn2 = _fuse(model.layer2[0].conv2, model.layer2[0].bn2)
    model.layer2[0].downsample[0], model.layer2[0].downsample[1] = _fuse(model.layer2[0].downsample[0], model.layer2[0].downsample[1])
    model.layer2[1].conv1, model.layer2[1].bn1 = _fuse(model.layer2[1].conv1, model.layer2[1].bn1)
    model.layer2[1].conv2, model.layer2[1].bn2 = _fuse(model.layer2[1].conv2, model.layer2[1].bn2)
    model.layer3[0].conv1, model.layer3[0].bn1 = _fuse(model.layer3[0].conv1, model.layer3[0].bn1)
    model.layer3[0].conv2, model.layer3[0].bn2 = _fuse(model.layer3[0].conv2, model.layer3[0].bn2)
    model.layer3[0].downsample[0], model.layer3[0].downsample[1] = _fuse(model.layer3[0].downsample[0], model.layer3[0].downsample[1])
    model.layer3[1].conv1, model.layer3[1].bn1 = _fuse(model.layer3[1].conv1, model.layer3[1].bn1)
    model.layer3[1].conv2, model.layer3[1].bn2 = _fuse(model.layer3[1].conv2, model.layer3[1].bn2)
    model.layer4[0].conv1, model.layer4[0].bn1 = _fuse(model.layer4[0].conv1, model.layer4[0].bn1)
    model.layer4[0].conv2, model.layer4[0].bn2 = _fuse(model.layer4[0].conv2, model.layer4[0].bn2)
    model.layer4[0].downsample[0], model.layer4[0].downsample[1] = _fuse(model.layer4[0].downsample[0], model.layer4[0].downsample[1])
    model.layer4[1].conv1, model.layer4[1].bn1 = _fuse(model.layer4[1].conv1, model.layer4[1].bn1)
    model.layer4[1].conv2, model.layer4[1].bn2 = _fuse(model.layer4[1].conv2, model.layer4[1].bn2)
    return model


class FakeQConv2d(nn.Conv2d):
    cur_cluster = 0

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False, bit=32, smooth=0.995, n_cluster=1):
        super(FakeQConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.layer_type = 'FakeQConv2d'
        self.bit = bit
        self.ema_init = [False for _ in range(n_cluster)]
        self.act_range = np.zeros((n_cluster, 2), dtype=np.float)
        self.smooth = smooth
        self.w_scale = 0
        self.w_zero = 0
        self.a_scale = 0
        self.a_zero = 0

    def forward(self, x):
        if self.bit == 32 or self.training == False:
            return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

        self.ema_matrix(x)
        if self.ema_init[self.cur_cluster]:
            self.set_quantization_params()
            x = self.fake_quantize(x)
        else:
            print("Set CONV layer to apply fake quantization..")
            self.ema_init[self.cur_cluster] = True
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

    def ema_matrix(self, x):
        k = self.cur_cluster
        _min = torch.min(x).item()
        _max = torch.max(x).item()
        if self.ema_init[k]:
            self.act_range[k][0] = self.act_range[k][0] * self.smooth + _min * (1 - self.smooth)
            self.act_range[k][1] = self.act_range[k][1] * self.smooth + _max * (1 - self.smooth)
        else:
            self.act_range[k][0] = _min
            self.act_range[k][1] = _max

    def set_quantization_params(self):
        q_max = 2 ** self.bit - 1

        f_min = self.act_range[self.cur_cluster][0]
        f_max = self.act_range[self.cur_cluster][1]
        self.a_scale = (f_max - f_min) / q_max
        self.a_zero = - f_min / self.a_scale
        self.a_zero = np.clip(self.a_zero, 0, q_max)

        f_min = torch.min(self.weight).item()
        f_max = torch.max(self.weight).item()
        self.w_scale = (f_max - f_min) / q_max
        self.w_zero = - f_min / self.w_scale
        self.w_zero = np.clip(self.w_zero, 0, q_max)

    def fake_quantize(self, x):
        self.weight.data = torch.round(self.weight.div(self.w_scale).add(self.w_zero)).sub(self.w_zero).mul(self.w_scale)
        x = torch.round(x.div(self.a_scale).add(self.a_zero)).sub(self.a_zero).mul(self.a_scale)
        return x

    def get_weight_qparmas(self):
        # Need to recalculate S/Z because of the last back-prop.
        q_max = 2 ** self.bit - 1
        w_min = torch.min(self.weight).item()
        w_max = torch.max(self.weight).item()
        s = (w_max - w_min) / q_max
        z = - w_min / s
        z = np.round(np.clip(z, 0, q_max))
        return s, z

    def fuse_conv_and_bn(self, alpha, beta, mean, var, eps):
        n_channel = self.weight.shape[0]
        self.bias = torch.nn.Parameter(beta)
        for c in range(n_channel):
            self.conv.weight.data[c] = self.conv.weight.data[c].mul(alpha[c]).div(torch.sqrt(var[c].add(eps)))
            self.conv.bias.data[c] = self.conv.bias.data[c].sub(alpha[c].mul(mean[c]).div(torch.sqrt(var[c])))


class FakeQLinear(nn.Linear):
    cur_cluster = 0

    def __init__(self, in_features, out_features, bias=True, bit=32, smooth=0.995, n_cluster=1):
        super(FakeQLinear, self).__init__(in_features, out_features, bias)
        self.layer_type = 'FakeQLinear'
        self.bit = bit
        self.ema_init = [False for _ in range(n_cluster)]
        self.act_range = np.zeros((n_cluster, 2), dtype=np.float)
        self.smooth = smooth
        self.w_scale = 0
        self.w_zero = 0
        self.a_scale = 0
        self.a_zero = 0

    def forward(self, x):
        if self.bit == 32 or self.training == False:
            return F.linear(x, self.weight, self.bias)
        self.ema_matrix(x)
        if self.ema_init[self.cur_cluster]:
            self.set_quantization_params()
            x = self.fake_quantize(x)
        else:
            print("Set Linear layer to apply fake quantization..")
            self.ema_init[self.cur_cluster] = True
        return F.linear(x, self.weight, self.bias)

    def ema_matrix(self, x):
        k = self.cur_cluster
        _min = torch.min(x).item()
        _max = torch.max(x).item()
        if self.ema_init[k]:
            self.act_range[k][0] = self.act_range[k][0] * self.smooth + _min * (1 - self.smooth)
            self.act_range[k][1] = self.act_range[k][1] * self.smooth + _max * (1 - self.smooth)
        else:
            self.act_range[k][0] = _min
            self.act_range[k][1] = _max

    def set_quantization_params(self):
        q_max = 2 ** self.bit - 1

        f_min = self.act_range[self.cur_cluster][0]
        f_max = self.act_range[self.cur_cluster][1]
        self.a_scale = (f_max - f_min) / q_max
        self.a_zero = - f_min / self.a_scale
        self.a_zero = np.clip(self.a_zero, 0, q_max)

        f_min = torch.min(self.weight).item()
        f_max = torch.max(self.weight).item()
        self.w_scale = (f_max - f_min) / q_max
        self.w_zero = - f_min / self.w_scale
        self.w_zero = np.clip(self.w_zero, 0, q_max)

    def fake_quantize(self, x):
        self.weight.data = torch.round(self.weight.div(self.w_scale).add(self.w_zero)).sub(self.w_zero).mul(self.w_scale)
        x = torch.round(x.div(self.a_scale).add(self.a_zero)).sub(self.a_zero).mul(self.a_scale)
        return x

    def get_weight_qparmas(self):
        # Need to recalculate S/Z because of the last back-prop.
        q_max = 2 ** self.bit - 1
        w_min = torch.min(self.weight).item()
        w_max = torch.max(self.weight).item()
        s = (w_max - w_min) / q_max
        z = - w_min / s
        z = np.round(np.clip(z, 0, q_max))
        return s, z

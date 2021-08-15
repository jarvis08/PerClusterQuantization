from operator import itemgetter

import torch.nn as nn
import torch
import torch.nn.functional as F

from ..quant_noise import _quant_noise
from ..quantization_utils import *


class QuantizedBn2d(nn.Module):
    def __init__(self, num_features, activation=None, arg_dict=None):
        super(QuantizedBn2d, self).__init__()
        self.layer_type = 'QuantizedBn2d'
        self.bit, self.num_clusters, self.runtime_helper =\
                itemgetter('bit', 'cluster', 'runtime_helper')(arg_dict)
        self.q_max = 2 ** self.bit - 1

        self.quantized_bias = nn.Parameter(torch.zeros((self.num_clusters, num_features)), requires_grad=False)

        t_init = list(range(self.num_clusters)) if self.num_clusters > 1 else 0
        self.s1 = nn.Parameter(torch.tensor(t_init, dtype=torch.float32), requires_grad=False)
        self.s2 = nn.Parameter(torch.tensor(t_init, dtype=torch.float32), requires_grad=False)
        self.s3 = nn.Parameter(torch.tensor(t_init, dtype=torch.float32), requires_grad=False)
        self.z1 = nn.Parameter(torch.tensor(t_init, dtype=torch.int32), requires_grad=False)
        self.z2 = nn.Parameter(torch.tensor(t_init, dtype=torch.int32), requires_grad=False)
        self.z3 = nn.Parameter(torch.tensor(t_init, dtype=torch.int32), requires_grad=False)

        self.M0 = nn.Parameter(torch.tensor(t_init, dtype=torch.int32), requires_grad=False)
        self.shift = nn.Parameter(torch.tensor(t_init, dtype=torch.int32), requires_grad=False)

        self.weight = nn.Parameter(torch.tensor((self.num_clusters, num_featrues), dtype=torch.int32), requires_grad=False)
        self.bias = nn.Parameter(torch.tensor((self.num_clusters, num_featrues), dtype=torch.int32), requires_grad=False)

        self.activation = activation
        self.hardswish_6 = nn.Parameter(torch.tensor(t_init, dtype=torch.int32), requires_grad=False)
        self.hardswish_3 = nn.Parameter(torch.tensor(t_init, dtype=torch.int32), requires_grad=False)
        self.s_activation = nn.Parameter(torch.tensor(t_init, dtype=torch.float32), requires_grad=False)
        self.z_activation = nn.Parameter(torch.tensor(t_init, dtype=torch.int32), requires_grad=False)


    def forward(self, x):
        if self.runtime_helper.batch_cluster is not None:
            return self.pcq(x)
        else:
            return self.general(x.type(torch.cuda.IntTensor))

    def general(self, x):
        bc = self.runtime_helper.batch_cluster

        done = 0
        total = torch.zeros(x.shape, dtype=torch.int32).cuda()
        for i in range(bc.shape[0]):
            c = bc[i][0]
            n = bc[i][1]
            sum_q1q2 = x[done:done + n].mul(weight[c])
            q1z2 = x[done:done + n].mul(z2[c])
            q2z1 = self.weight[c].mul(z1[c])
            subsum = sum_q1q2 - q1z2 - q2z1 + self.bias[c]

            subsum = multiply_M(subsum.type(torch.cuda.LongTensor), self.M0)
            subsum = shifting(subsum, self.shift.item())
            total[done:done + n] = subsum.add(self.z3)
            done += n

        if self.bit == 4:
            total = torch.clamp(total, 0, 15)
        else:
            total = torch.clamp(total, -128, 127)
        return total.type(torch.cuda.FloatTensor)


class FusedBnReLU(nn.Module):
    def __init__(self, num_features: int, activation=None, arg_dict=None):
        super(FusedBnReLU, self).__init__()
        self.layer_type = 'FusedBnReLU'
        self.bit, self.smooth, self.use_ste, self.runtime_helper =\
                itemgetter('bit', 'smooth', 'ste', 'runtime_helper')(arg_dict)
        self.q_max = 2 ** self.bit - 1

        self.act_range = nn.Parameter(torch.zeros(2), requires_grad=False)
        self.apply_ema = False

        self._norm_layer = nn.BatchNorm2d(num_features)
        self._activation = activation(inplace=False) if activation else None

    def forward(self, x):
        x = self._norm_layer(x)
        if self._activation is not None:
            x = self._activation(x)
        if not self.training:
            return x

        out = x
        if self.apply_ema:
            self.act_range[0], self.act_range[1] = ema(x, self.act_range, self.smooth)
            if self.runtime_helper.apply_fake_quantization:
                s, z = calc_qparams(self.act_range[0], self.act_range[1], self.q_max)
                out = fake_quantize(x, s, z, self.q_max, self.use_ste)
        else:
            self.act_range[0] = torch.min(x).item()
            self.act_range[1] = torch.max(x).item()
            self.apply_ema = True
        return out

    def set_qparams(self, s1, z1):
        self.s1, self.z1 = nn.Parameter(s1, requires_grad=False), nn.Parameter(z1, requires_grad=False)
        self.alpha = self._norm_layer.weight / torch.sqrt(self.running_var + self.eps)
        self.beta = self.bias - (self._norm_layer.running_mean / torch.sqrt(self.running_var + self.eps))
        self.s2, self.z2 = calc_qparams(self.alpha.min(), self.alpha.max(), self.q_max)
        self.s3, self.z3 = calc_qparams(self.act_range[0], self.act_range[1], self.q_max)
        return self.s3, self.z3

    def fold_bn(self):
        # In case of validation, fuse pretrained Conv&BatchNorm params
        assert self.training == False, 'Do not fuse layers while training.'
        alpha, beta, mean, var, eps = self._norm_layer.weight, self._norm_layer.bias, self._norm_layer.running_mean,\
                                      self._norm_layer.running_var, self._norm_layer.eps
        n_channel = self.conv.weight.shape[0]
        self.quantized_weight = nn.Parameter(beta)
        self.quantized_bias = nn.Parameter(beta)
        for c in range(n_channel):
            self.quantized_weight[c] = self.quantized_weight.data[c].mul(alpha[c]).div(torch.sqrt(var[c].add(eps)))
            self.quantized_bias[c] = self.quantized_bias[c].sub(alpha[c].mul(mean[c]).div(torch.sqrt(var[c].add(eps))))
        self._norm_layer = nn.Identity()


from operator import itemgetter

import torch.nn as nn
import torch
import torch.nn.functional as F

from ..quantization_utils import *


class PCQActivation(nn.Module):
    """
        Activation Layer(Hardswish, Hardsigmoid, gelu ..) with clusters
    """
    batch_cluster = None

    def __init__(self, activation=None, arg_dict=None):
        super(PCQActivation, self).__init__()
        self.layer_type = 'PCQActivation'
        self.bit, self.smooth, self.num_clusters, self.runtime_helper, self.use_ste, self.quant_noise, self.qn_prob\
            = itemgetter('bit', 'smooth', 'cluster', 'runtime_helper', 'ste', 'quant_noise', 'qn_prob')(arg_dict)
        self.q_max = 2 ** self.bit - 1
        self.act_range = nn.Parameter(torch.zeros((self.num_clusters, 2)), requires_grad=False)

        self.apply_ema = False

        self._activation = activation(inplace=False)

    def forward(self, x):
        x = self._activation(x)
        if not self.training:
            return x

        if self.runtime_helper.apply_fake_quantization and self.use_ste:
            out = torch.zeros(x.shape).cuda()
        else:
            out = x

        done = 0
        for i in range(self.runtime_helper.batch_cluster.shape[0]):
            c = self.runtime_helper.batch_cluster[i][0].item()
            n = self.runtime_helper.batch_cluster[i][1].item()
            if self.apply_ema[c]:
                self.act_range[c][0], self.act_range[c][1] = ema(x[done:done + n], self.act_range[c], self.smooth)
                if self.runtime_helper.apply_fake_quantization:
                    s, z = calc_qparams(self.act_range[c][0], self.act_range[c][1], self.q_max)
                    out[done:done + n] = fake_quantize(x[done:done + n], s, z, self.q_max, self.use_ste)
            else:
                self.act_range[c][0] = torch.min(x[done:done + n]).item()
                self.act_range[c][1] = torch.max(x[done:done + n]).item()
                self.apply_ema[c] = True
            done += n
        return out

    def set_qparams(self, s1, z1):
        self.s1, self.z1 = nn.Parameter(s1, requires_grad=False), nn.Parameter(z1, requires_grad=False)
        self.s3 = nn.Parameter(torch.zeros(self.num_clusters, dtype=torch.float32), requires_grad=False)
        self.z3 = nn.Parameter(torch.zeros(self.num_clusters, dtype=torch.int32), requires_grad=False)
        for c in range(self.num_clusters):
            self.s3[c], self.z3[c] = calc_qparams(self.act_range[c][0], self.act_range[c][1], self.q_max)
        return self.s3, self.z3


class QActivation(nn.Module):
    """
        Activation Layer(Hardswish, Hardsigmoid, gelu ..)
    """

    def __init__(self, activation=None, arg_dict=None):
        super(QActivation, self).__init__()
        self.layer_type = 'QActivation'
        self.bit, self.smooth, self.runtime_helper, self.use_ste, self.quant_noise, self.qn_prob \
            = itemgetter('bit', 'smooth', 'runtime_helper', 'ste', 'quant_noise', 'qn_prob')(arg_dict)
        self.q_max = 2 ** self.bit - 1
        self.act_range = nn.Parameter(torch.zeros(2), requires_grad=False)

        self.apply_ema = False

        self._activation = activation(inplace=False)

    def forward(self, x):
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
        self.s3, self.z3 = calc_qparams(self.act_range[0], self.act_range[1], self.q_max)
        return self.s3, self.z3

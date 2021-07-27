import torch.nn as nn
import torch
import torch.nn.functional as F

from ..quantization_utils import *


class QActivation(nn.Module):
    """
        Activation Layer(Hardswish, Hardsigmoid, gelu ..)
    """
    def __init__(self, activation=None, smooth=0.999, bit=32):
        super(QActivation, self).__init__()
        self.layer_type = 'QActivation'
        self.bit = bit
        self.q_max = 2 ** bit - 1
        self.flag_ema_init = False
        self.flag_fake_quantization = True
        self.smooth = smooth
        self.act_range = nn.Parameter(torch.zeros(2), requires_grad=False)
        self._activation = activation(inplace=False)

    def forward(self, x):
        x = self._activation(x)

        if self.training:
            if self.flag_ema_init:
                self.act_range[0], self.act_range[1] = ema(x, self.act_range, self.smooth)
                if self.flag_fake_quantization:
                    s, z = calc_qparams(self.act_range[0], self.act_range[1], self.q_max)
                    x = fake_quantize(x, s, z, self.q_max)
            else:
                self.act_range[0] = torch.min(x).item()
                self.act_range[1] = torch.max(x).item()
                self.flag_ema_init = True
        return x

    def set_qparams(self, s1, z1):
        self.s1, self.z1 = nn.Parameter(s1, requires_grad=False), nn.Parameter(z1, requires_grad=False)
        self.s3, self.z3 = calc_qparams(self.act_range[0], self.act_range[1], self.q_max)
        return self.s3, self.z3


class PCQActivation(nn.Module):
    """
        Activation Layer(Hardswish, Hardsigmoid, gelu ..) with clusters
    """
    batch_cluster = None

    def __init__(self, activation=None, smooth=0.999, bit=32, num_clusters=10):
        super(PCQActivation, self).__init__()
        self.layer_type = 'PCQActivation'
        self.bit = bit
        self.q_max = 2 ** bit - 1
        self.num_clusters = num_clusters
        self.flag_ema_init = False
        self.flag_fake_quantization = True
        self.smooth = smooth
        self.act_range = nn.Parameter(torch.zeros((num_clusters, 2)), requires_grad=False)
        self._activation = activation(inplace=False)

    def forward(self, x):
        x = self._activation(x)

        if self.training:
            done = 0
            for i in range(self.batch_cluster.shape[0]):
                c = self.batch_cluster[i][0].item()
                n = self.batch_cluster[i][1].item()
                if self.flag_ema_init[c]:
                    self.act_range[c][0], self.act_range[c][1] = ema(x[done:done + n], self.act_range[c], self.smooth)
                    if self.flag_fake_quantization:
                        s, z = calc_qparams(self.act_range[c][0], self.act_range[c][1], self.q_max)
                        x[done:done + n] = fake_quantize(x[done:done + n], s, z, self.q_max)
                else:
                    self.act_range[c][0] = torch.min(x[done:done + n]).item()
                    self.act_range[c][1] = torch.max(x[done:done + n]).item()
                    self.flag_ema_init[c] = True
                done += n
        return x

    def set_qparams(self, s1, z1):
        self.s1, self.z1 = nn.Parameter(s1, requires_grad=False), nn.Parameter(z1, requires_grad=False)
        self.s3 = nn.Parameter(torch.zeros(self.num_clusters, dtype=torch.float32), requires_grad=False)
        self.z3 = nn.Parameter(torch.zeros(self.num_clusters, dtype=torch.int32), requires_grad=False)
        for c in range(self.num_clusters):
            self.s3[c], self.z3[c] = calc_qparams(self.act_range[c][0], self.act_range[c][1], self.q_max)
        return self.s3, self.z3


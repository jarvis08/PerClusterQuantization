from operator import itemgetter

import torch.nn as nn
import torch

from ..quantization_utils import *


class QuantizedBn2d(nn.Module):
    def __init__(self, num_features, arg_dict=None):
        super(QuantizedBn2d, self).__init__()
        self.layer_type = 'QuantizedBn2d'
        self.bit, self.num_clusters, self.runtime_helper = itemgetter('bit', 'cluster', 'runtime_helper')(arg_dict)
        self.q_max = 2 ** self.bit - 1
        self.num_features = num_features

        t_init = list(range(self.num_clusters)) if self.num_clusters > 1 else 0
        self.s1 = nn.Parameter(torch.tensor(t_init, dtype=torch.float32), requires_grad=False)
        self.s2 = nn.Parameter(torch.tensor(t_init, dtype=torch.float32), requires_grad=False)
        self.s3 = nn.Parameter(torch.tensor(t_init, dtype=torch.float32), requires_grad=False)
        self.z1 = nn.Parameter(torch.tensor(t_init, dtype=torch.int32), requires_grad=False)
        self.z2 = nn.Parameter(torch.tensor(t_init, dtype=torch.int32), requires_grad=False)
        self.z3 = nn.Parameter(torch.tensor(t_init, dtype=torch.int32), requires_grad=False)
        self.M0 = nn.Parameter(torch.tensor(t_init, dtype=torch.int32), requires_grad=False)
        self.shift = nn.Parameter(torch.tensor(t_init, dtype=torch.int32), requires_grad=False)

        #self.weight = nn.Parameter(torch.zeros((self.num_clusters, num_features), dtype=torch.int32), requires_grad=False)
        self.bias = nn.Parameter(torch.zeros((self.num_clusters, num_features), dtype=torch.int32), requires_grad=False)
        self.mean = nn.Parameter(torch.zeros((self.num_clusters, num_features), dtype=torch.int32), requires_grad=False)
        self.weight_divided_by_std = nn.Parameter(torch.zeros((self.num_clusters, num_features), dtype=torch.int32), requires_grad=False)

    def forward(self, x):
        if self.runtime_helper.batch_cluster is not None:
            return self._pcq(x.type(torch.cuda.IntTensor))
        else:
            return self._general(x.type(torch.cuda.IntTensor))

    def _pcq(self, x):
        bc = self.runtime_helper.batch_cluster
        batch, channel, width, height = x.shape

        #weight = torch.index_select(self.weight.repeat_interleave(_size * _size)
        #                            .reshape(self.num_clusters, self.num_features, _size, _size), 0, bc)
        #bias = torch.index_select(self.bias.repeat_interleave(_size * _size)
        #                          .reshape(self.num_clusters, self.num_features, _size, _size), 0, bc)
        #z1 = torch.index_select(self.z1, 0, bc).reshape(bc.shape[0], 1, 1, 1)
        #z2 = torch.index_select(self.z2, 0, bc).reshape(bc.shape[0], 1, 1, 1)
        z3 = torch.index_select(self.z3, 0, bc).reshape(bc.shape[0], 1, 1, 1)
        M0 = torch.index_select(self.M0, 0, bc).reshape(bc.shape[0], 1, 1, 1)
        shift = torch.index_select(self.shift, 0, bc)
        weight_divided_by_std = torch.index_select(self.weight_divided_by_std.repeat_interleave(width * height)
                                                   .reshape(self.num_clusters, channel, width, height), 0, bc)
        bias = torch.index_select(self.bias.repeat_interleave(width * height)
                                  .reshape(self.num_clusters, channel, width, height), 0, bc)
        mean = torch.index_select(self.mean.repeat_interleave(width * height)
                                  .reshape(self.num_clusters, channel, width, height), 0, bc)

        #q1q2 = x.mul(weight)
        #q1z2 = x.mul(z2)
        #q2z1 = weight.mul(z1)
        #subsum = q1q2 - q1z2 - q2z1 + z1 * z2 + bias
        subsum = (x - mean) * weight_divided_by_std + bias

        total = torch.zeros(subsum.shape, dtype=torch.int32).cuda()
        neg = (shift < 0).nonzero(as_tuple=True)[0]
        pos = (shift >= 0).nonzero(as_tuple=True)[0]
        if len(neg) > 0:
            s = - shift[neg].reshape(neg.shape[0], 1, 1, 1)
            multiplied = multiply_M((subsum[neg].type(torch.cuda.LongTensor) << s), M0[neg])
            total[neg] = shifting(multiplied, 0)
        if len(pos) > 0:
            s = shift[pos].reshape(pos.shape[0], 1, 1, 1)
            multiplied = multiply_M(subsum[pos].type(torch.cuda.LongTensor), M0[pos])
            total[pos] = shifting4d(multiplied, s)
        total = total.add(z3)

        if self.bit == 4:
            total = torch.clamp(total, 0, 15)
        else:
            total = torch.clamp(total, -128, 127)
        return total.type(torch.cuda.FloatTensor)

    def _general(self, x):
        _size = x.shape[-1]
        weight = self.weight[0].repeat_interleave(_size * _size)\
                               .reshape(self.num_features, _size, _size)\
                               .repeat(x.shape[0], 1, 1, 1)
        bias = self.bias[0].repeat_interleave(_size * _size)\
                           .reshape(self.num_features, _size, _size)\
                           .repeat(x.shape[0], 1, 1, 1)
        q1q2 = x.mul(weight)
        q1z2 = x.mul(self.z2)
        q2z1 = weight.mul(self.z1)
        subsum = q1q2 - q1z2 - q2z1 + self.z1 * self.z2 + bias

        if self.shift.item() < 0:
            multiplied = multiply_M((subsum.type(torch.cuda.LongTensor) << - self.shift.item()), self.M0)
            total = shifting(multiplied, 0)
        else:
            multiplied = multiply_M(subsum.type(torch.cuda.LongTensor), self.M0)
            total = shifting(multiplied, self.shift.item())
        total = total.add(self.z3)

        if self.bit == 4:
            total = torch.clamp(total, 0, 15)
        else:
            total = torch.clamp(total, -128, 127)
        return total.type(torch.cuda.FloatTensor)


class PCQBnReLU(nn.Module):
    def __init__(self, num_features, activation=None, arg_dict=None):
        super(PCQBnReLU, self).__init__()
        self.layer_type = 'PCQBnReLU'
        self.runtime_helper, self.num_clusters = itemgetter('runtime_helper', 'cluster')(arg_dict)

        self.norms = nn.ModuleList([FusedBnReLU(num_features, activation=activation, arg_dict=arg_dict)\
                                    for _ in range(self.num_clusters)])
        # self.initial_params = torch.zeros((self.num_clusters, 2))
        # self.initial_range = torch.zeros((self.num_clusters, 2))
        # self.apply_ema = False

    def forward(self, x, conv_range=None):
        bc = self.runtime_helper.batch_cluster
        done = 0
        out = []
        for i in range(bc.shape[0]):
            c = bc[i][0].item()
            n = bc[i][1].item()
            out.append(self.norms[c](x[done:done + n], conv_range[c]))
            done += n
        return torch.cat(out)

    def set_qparams(self, s1, z1):
        self.s1, self.z1 = nn.Parameter(s1, requires_grad=False), nn.Parameter(z1, requires_grad=False)

        self.s2 = nn.Parameter(torch.zeros(self.num_clusters, dtype=torch.float32), requires_grad=False)
        self.s3 = nn.Parameter(torch.zeros(self.num_clusters, dtype=torch.float32), requires_grad=False)
        self.z2 = nn.Parameter(torch.zeros(self.num_clusters, dtype=torch.int32), requires_grad=False)
        self.z3 = nn.Parameter(torch.zeros(self.num_clusters, dtype=torch.int32), requires_grad=False)
        self.M0 = nn.Parameter(torch.zeros(self.num_clusters, dtype=torch.int32), requires_grad=False)
        self.shift = nn.Parameter(torch.zeros(self.num_clusters, dtype=torch.int32), requires_grad=False)
        for c in range(self.num_clusters):
            self.s3[c], self.z3[c] = self.norms[c].set_qparams(self.s1[c], self.z1[c])
            self.s2[c], self.z2[c] = self.norms[c].get_weight_qparams()
            self.M0[c], self.shift[c] = self.norms[c].get_multiplier_qparams()
        return self.s3, self.z3

    def fold_bn(self):
        for c in range(self.num_clusters):
            self.norms[c].fold_bn()


class FusedBnReLU(nn.Module):
    def __init__(self, num_features, activation=None, arg_dict=None):
        super(FusedBnReLU, self).__init__()
        self.layer_type = 'FusedBnReLU'
        self.bit, self.smooth, self.use_ste, self.runtime_helper = \
            itemgetter('bit', 'smooth', 'ste', 'runtime_helper')(arg_dict)
        #self.w_qmax = 2 ** 8 - 1
        self.w_qmax = 2 ** 32 - 1
        self.q_max = 2 ** self.bit - 1

        self.act_range = nn.Parameter(torch.zeros(2), requires_grad=False)
        self.apply_ema = False

        self.num_features = num_features
        self.bn = nn.BatchNorm2d(num_features)
        self._activation = activation(inplace=True) if activation else None

    def forward(self, x, conv_range=None):
        if not self.runtime_helper.pcq_initialized:
            # PCQ initialization
            out = self._forward_impl(x)
            self._update_range_without_fq(out)
            return out

        elif not self.training:
            return self._forward_impl(x)

        else:
            if self.runtime_helper.apply_fake_quantization:
                out = self._fake_quantized_bn(x, conv_range)
                return self._fake_quantize_and_update_range(out)
            else:
                out = self._forward_impl(x)
                self._update_range_without_fq(out)
                return out

    def _forward_impl(self, x):
        x = self.bn(x)
        if self._activation:
            x = self._activation(x)
        return x

    def _fake_quantized_bn(self, x, conv_range):
        mean = x.mean(dim=(0, 2, 3))
        var = x.var(dim=(0, 2, 3), unbiased=False)
        self.bn.running_mean = self.bn.running_mean * (1 - self.bn.momentum) + mean * self.bn.momentum
        self.bn.running_var = self.bn.running_var * (1 - self.bn.momentum) + var * self.bn.momentum

        s, z = calc_qparams(conv_range[0], conv_range[1], self.w_qmax)
        m = fake_quantize(self.bn.running_mean, s, z, self.w_qmax, use_ste=False)
        v = fake_quantize(self.bn.running_var, s, z, self.w_qmax, use_ste=False)
        w = fake_quantize(self.bn.weight, s, z, self.w_qmax, self.use_ste)
        b = fake_quantize(self.bn.bias, s, z, self.w_qmax, self.use_ste)

        # batch, channel, width, height = x.shape
        # m = m.repeat_interleave(width * height).reshape(channel, width, height).repeat(batch, 1, 1, 1)
        # v = v.repeat_interleave(width * height).reshape(channel, width, height).repeat(batch, 1, 1, 1)
        # w = w.repeat_interleave(width * height).reshape(channel, width, height).repeat(batch, 1, 1, 1)
        # b = b.repeat_interleave(width * height).reshape(channel, width, height).repeat(batch, 1, 1, 1)

        # out = (x - m) / torch.sqrt(v + self.bn.eps) * w + b
        x = (x - m[None, :, None, None]) / torch.sqrt(v[None, :, None, None] + self.bn.eps) * w[None, :, None, None] + b[None, :, None, None]
        if self._activation:
            x = self._activation(x)
        return x

    def _update_range_without_fq(self, x):
        if self.apply_ema:
            self.act_range[0], self.act_range[1] = ema(x, self.act_range, self.smooth)
        else:
            self.act_range[0] = torch.min(x).item()
            self.act_range[1] = torch.max(x).item()
            self.apply_ema = True

    def _fake_quantize_and_update_range(self, x):
        self.act_range[0], self.act_range[1] = ema(x, self.act_range, self.smooth)
        s, z = calc_qparams(self.act_range[0], self.act_range[1], self.q_max)
        return fake_quantize(x, s, z, self.q_max, self.use_ste)

    # def fold_bn(self):
    #     # In case of validation, fuse pretrained Conv&BatchNorm params
    #     assert self.training == False, 'Do not fuse layers while training.'
    #     alpha, beta, mean, var, eps = self.bn.weight, self.bn.bias, self.bn.running_mean,\
    #                                   self.bn.running_var, self.bn.eps
    #     self.weight = nn.Parameter(alpha / torch.sqrt(var + eps), requires_grad=False)
    #     self.bias = nn.Parameter(beta - alpha * mean / torch.sqrt(var + eps), requires_grad=False)

    # def set_qparams(self, s1, z1):
    #     self.s1, self.z1 = nn.Parameter(s1, requires_grad=False), nn.Parameter(z1, requires_grad=False)
    #     if self.weight.min() > 0:
    #         self.s2, self.z2 = calc_qparams(torch.tensor(0), self.weight.max(), self.w_qmax)
    #     elif self.weight.max() < 0:
    #         self.s2, self.z2 = calc_qparams(self.weight.min(), torch.tensor(0), self.w_qmax)
    #     else:
    #         self.s2, self.z2 = calc_qparams(self.weight.min(), self.weight.max(), self.w_qmax)
    #     self.s3, self.z3 = calc_qparams(self.act_range[0], self.act_range[1], self.q_max)
    #     self.M0, self.shift = quantize_M(self.s1 * self.s2 / self.s3)
    #     return self.s3, self.z3

    def set_qparams(self, s1, z1):
        self.s1, self.z1 = nn.Parameter(s1, requires_grad=False), nn.Parameter(z1, requires_grad=False)
        self.s2, self.z2 = nn.Parameter(s1, requires_grad=False), nn.Parameter(z1, requires_grad=False)
        self.s3, self.z3 = calc_qparams(self.act_range[0], self.act_range[1], self.q_max)
        self.M0, self.shift = quantize_M(self.s1 / self.s3)

    def get_weight_qparams(self):
        return self.s2, self.z2

    def get_multiplier_qparams(self):
        return self.M0, self.shift

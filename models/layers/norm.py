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

        self.weight = nn.Parameter(torch.zeros((self.num_clusters, num_features), dtype=torch.int32), requires_grad=False)
        self.bias = nn.Parameter(torch.zeros((self.num_clusters, num_features), dtype=torch.int32), requires_grad=False)

    def forward(self, x):
        if self.runtime_helper.batch_cluster is not None:
            return self._pcq(x.type(torch.cuda.IntTensor))
        else:
            return self._general(x.type(torch.cuda.IntTensor))

    def _pcq(self, x):
        bc = self.runtime_helper.batch_cluster
        _size = x.shape[-1]

        weight = torch.index_select(self.weight.repeat_interleave(_size * _size)
                                    .reshape(self.num_clusters, self.num_features, _size, _size), 0, bc)
        bias = torch.index_select(self.bias.repeat_interleave(_size * _size)
                                  .reshape(self.num_clusters, self.num_features, _size, _size), 0, bc)
        z1 = torch.index_select(self.z1, 0, bc).reshape(bc.shape[0], 1, 1, 1)
        z2 = torch.index_select(self.z2, 0, bc).reshape(bc.shape[0], 1, 1, 1)
        z3 = torch.index_select(self.z3, 0, bc).reshape(bc.shape[0], 1, 1, 1)
        M0 = torch.index_select(self.M0, 0, bc).reshape(bc.shape[0], 1, 1, 1)
        shift = torch.index_select(self.shift, 0, bc)

        q1q2 = x.mul(weight)
        q1z2 = x.mul(z2)
        q2z1 = weight.mul(z1)
        subsum = q1q2 - q1z2 - q2z1 + z1 * z2 + bias

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
        #self.bn = nn.BatchNorm2d(num_features)

        self.ema_params = torch.zeros((self.num_clusters, 2))
        self.bn_act_range = torch.zeros((self.num_clusters, 2))
        self.bn_ema = False

    def forward(self, x):
        bc = self.runtime_helper.batch_cluster
        # clusters_in_batch = torch.unique(bc)
        # for c in clusters_in_batch:
        #     data_idx_of_c = (bc == c).nonzero(as_tuple=True)[0].cuda()
        #     out.append(self.norms[c](torch.index_select(x, 0, data_idx_of_c)))

        done = 0
        out = []

        if not self.training and self.runtime_helper.bn_init:
            for c in range(self.num_clusters):
                cur_res = self.norms[c](x[done:done + 8])
                if self.bn_ema:
                    self.ema_params[c][0], self.ema_params[c][1] = bn_ema(self.ema_params[c], self.norms[c], 0.9)
                    self.bn_act_range[c][0], self.bn_act_range[c][1] = ema(cur_res, self.bn_act_range[c], self.smooth)
                else:
                    self.bn_act_range[c][0], self.bn_act_range[c][1] = torch.min(cur_res).item(), torch.max(cur_res).item()
                    self.ema_params[c][0], self.ema_params[c][1] = self.norms[c].bn.running_mean, self.norms[c].bn.running_var
                    self.bn_ema = True
                out.append(cur_res)
                done += 8
                return torch.cat(out)

        for i in range(bc.shape[0]):
            c = bc[i][0].item()
            n = bc[i][1].item()
            out.append(self.norms[c](x[done:done + n]))
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

    def fold_norms(self):
        for c in range(self.num_clusters):
            self.norms[c].fold_bn()

    def pass_bn_params(self):
        for c in range(self.num_clusters):
            self.norms[c].bn.running_mean.copy_(self.ema_params[c][0])
            self.norms[c].bn.running_var.copy_(self.ema_params[c][1])
            self.norms[c].act_range.copy_(self.bn_act_range[c])


class FusedBnReLU(nn.Module):
    def __init__(self, num_features, activation=None, arg_dict=None):
        super(FusedBnReLU, self).__init__()
        self.layer_type = 'FusedBnReLU'
        self.bit, self.smooth, self.use_ste, self.runtime_helper = \
            itemgetter('bit', 'smooth', 'ste', 'runtime_helper')(arg_dict)
        self.w_qmax = 2 ** 8 - 1
        self.q_max = 2 ** self.bit - 1

        self.act_range = nn.Parameter(torch.zeros(2), requires_grad=False)
        self.apply_ema = False

        self.num_features = num_features
        self.bn = nn.BatchNorm2d(num_features)
        self._activation = activation(inplace=True) if activation else None

    def forward(self, x):
        general_out = self.bn(x)
        if self._activation:
            general_out = self._activation(general_out)
        if not self.training:
            return general_out

        with torch.no_grad():
            alpha, beta, mean, var, eps = self.bn.weight, self.bn.bias, self.bn.running_mean, \
                                          self.bn.running_var, self.bn.eps
            folded_weight = alpha.div(torch.sqrt(var.add(eps)))
            folded_bias = beta.sub(alpha.mul(mean).div(torch.sqrt(var.add(eps))))

            if folded_weight.min() > 0:
                s, z = calc_qparams(torch.tensor(0), folded_weight.max(), self.w_qmax)
            elif folded_weight.max() < 0:
                s, z = calc_qparams(folded_weight.min(), torch.tensor(0), self.w_qmax)
            else:
                s, z = calc_qparams(folded_weight.min(), folded_weight.max(), self.w_qmax)
            fq_weight = fake_quantize(folded_weight, s, z, self.w_qmax, use_ste=False)

            batch, channel, width, height = x.shape
            w = fq_weight.repeat_interleave(width * height).reshape(channel, width, height).repeat(batch, 1, 1, 1)
            b = folded_bias.repeat_interleave(width * height).reshape(channel, width, height).repeat(batch, 1, 1, 1)

            folded_out = w * x + b
            if self._activation:
                folded_out = self._activation(folded_out)

            if self.apply_ema:
                self.act_range[0], self.act_range[1] = ema(folded_out, self.act_range, self.smooth)
                if self.runtime_helper.apply_fake_quantization:
                    s, z = calc_qparams(self.act_range[0], self.act_range[1], self.q_max)
                    folded_out = fake_quantize(folded_out, s, z, self.q_max, use_ste=False)
            else:
                self.act_range[0] = torch.min(folded_out).item()
                self.act_range[1] = torch.max(folded_out).item()
                self.apply_ema = True
        return STE.apply(general_out, folded_out)

    def fold_bn(self):
        # In case of validation, fuse pretrained Conv&BatchNorm params
        assert self.training == False, 'Do not fuse layers while training.'
        alpha, beta, mean, var, eps = self.bn.weight, self.bn.bias, self.bn.running_mean,\
                                      self.bn.running_var, self.bn.eps
        self.weight = nn.Parameter(alpha / torch.sqrt(var + eps), requires_grad=False)
        self.bias = nn.Parameter(beta - alpha * mean / torch.sqrt(var + eps), requires_grad=False)

    def set_qparams(self, s1, z1):
        self.s1, self.z1 = nn.Parameter(s1, requires_grad=False), nn.Parameter(z1, requires_grad=False)
        if self.weight.min() > 0:
            self.s2, self.z2 = calc_qparams(torch.tensor(0), self.weight.max(), self.w_qmax)
        elif self.weight.max() < 0:
            self.s2, self.z2 = calc_qparams(self.weight.min(), torch.tensor(0), self.w_qmax)
        else:
            self.s2, self.z2 = calc_qparams(self.weight.min(), self.weight.max(), self.w_qmax)
        self.s3, self.z3 = calc_qparams(self.act_range[0], self.act_range[1], self.q_max)
        self.M0, self.shift = quantize_M(self.s1 * self.s2 / self.s3)
        return self.s3, self.z3

    def get_weight_qparams(self):
        return self.s2, self.z2

    def get_multiplier_qparams(self):
        return self.M0, self.shift

from operator import itemgetter

import torch.nn as nn
import torch

from ..quantization_utils import *


class QuantizedBn2d(nn.Module):
    def __init__(self, num_features, arg_dict=None):
        super(QuantizedBn2d, self).__init__()
        self.layer_type = 'QuantizedBn2d'
        self.bit, self.num_clusters, self.runtime_helper = itemgetter('bit', 'cluster', 'runtime_helper')(arg_dict)
        self.num_features = num_features
        self.q_max = 2 ** self.bit - 1
        self.act_qmax = nn.Parameter(torch.tensor(0, dtype=torch.int32), requires_grad=False)

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
            return self._pcq(x.type(torch.cuda.LongTensor))
        else:
            return self._general(x.type(torch.cuda.LongTensor))

    def _pcq(self, x):
        bc = self.runtime_helper.batch_cluster
        batch, channel, width, height = x.shape
        weight = torch.index_select(self.weight.repeat_interleave(width * height)
                                    .reshape(self.num_clusters, self.num_features, width, height), 0, bc)
        bias = torch.index_select(self.bias.repeat_interleave(width * height)
                                  .reshape(self.num_clusters, self.num_features, width, height), 0, bc)
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
            multiplied = multiply_M((subsum[neg] << s), M0[neg])
            total[neg] = shifting(multiplied, 0)
        if len(pos) > 0:
            s = shift[pos].reshape(pos.shape[0], 1, 1, 1)
            multiplied = multiply_M(subsum[pos], M0[pos])
            total[pos] = shifting4d(multiplied, s)
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

    def _general(self, x):
        q1q2 = x.mul(self.weight[0][None, :, None, None])
        q1z2 = x.mul(self.z2)
        q2z1 = self.weight[0].mul(self.z1)
        subsum = q1q2 - q1z2 - q2z1[None, :, None, None] + self.z1 * self.z2 + self.bias[0][None, :, None, None]

        if self.shift.item() < 0:
            multiplied = multiply_M((subsum.type(torch.cuda.LongTensor) << - self.shift.item()), self.M0)
            total = shifting(multiplied, 0)
        else:
            multiplied = multiply_M(subsum.type(torch.cuda.LongTensor), self.M0)
            total = shifting(multiplied, self.shift.item())
        total = total.add(self.z3)

        if self.act_qmax == 15:
            total = torch.clamp(total, 0, 15)
        elif self.act_qmax == 255:
            total = torch.clamp(total, -128, 127)
        elif self.act_qmax == 65535:  # INT 16
            total = torch.clamp(total, -32768, 32767)
        elif self.act_qmax == 4294967295:  # INT 32
            total = torch.clamp(total, -2147483648, 2147483647)
        return total.type(torch.cuda.FloatTensor)


class PCQBnReLU(nn.Module):
    def __init__(self, num_features, momentum=0.1, eps=1e-05, activation=None, act_qmax=None, w_qmax=2 ** 8 - 1, arg_dict=None):
        super(PCQBnReLU, self).__init__()
        self.layer_type = 'PCQBnReLU'
        self.bit, self.smooth, self.use_ste, self.runtime_helper, self.num_clusters = \
            itemgetter('bit', 'smooth', 'ste', 'runtime_helper', 'cluster')(arg_dict)

        self.w_qmax = w_qmax
        self.act_qmax = act_qmax if act_qmax else 2 ** self.bit - 1

        self.act_range = nn.Parameter(torch.zeros((self.num_clusters, 2)), requires_grad=False)
        self.apply_ema = False

        self.weights = nn.Parameter(torch.ones(self.num_clusters, num_features), requires_grad=True)
        self.biases = nn.Parameter(torch.zeros(self.num_clusters, num_features), requires_grad=True)
        self.running_means = nn.Parameter(torch.zeros(self.num_clusters, num_features), requires_grad=False)
        self.running_vars = nn.Parameter(torch.ones(self.num_clusters, num_features), requires_grad=False)
        self.eps = nn.Parameter(torch.tensor(eps), requires_grad=False)
        self.momentum = nn.Parameter(torch.tensor(momentum), requires_grad=False)

        self.activation = activation

    def forward(self, x, external_range=None):

        bc = self.runtime_helper.batch_cluster
        existing_clusters = torch.unique(bc)
        for c in existing_clusters:
            indices = (bc == c).nonzero(as_tuple=True)[0]
            out[indices] = self.norms[c](x[indices])

        w = torch.index_select(self.weights, 0, bc)
        b = torch.index_select(self.biases, 0, bc)
        mean = x.detach().mean(dim=(0, 2, 3))
        var = x.detach().var(dim=(0, 2, 3), unbiased=False)

        if not self.training:
            return out

        if not self.runtime_helper.pcq_initialized:
            self._update_activation_ranges(out, external_range)
            return out

        if self.runtime_helper.range_update_phase:
            self._update_activation_ranges(out, external_range)

        if self.runtime_helper.apply_fake_quantization:
            return self._fake_quantize_activation(out, external_range)
        else:
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
                self.act_range[c][0], self.act_range[c][1] = \
                    ema(x[c * n: (c + 1) * n], self.act_range[c], self.smooth)
        else:
            for c in range(self.num_clusters):
                self.act_range[c][0] = x[c * n: (c + 1) * n].min().item()
                self.act_range[c][1] = x[c * n: (c + 1) * n].max().item()
            self.apply_ema = True

    def set_qparams(self, s1, z1, s_external=None, z_external=None):
        self.s1, self.z1 = nn.Parameter(s1, requires_grad=False), nn.Parameter(z1, requires_grad=False)

        self.s2 = nn.Parameter(torch.zeros(self.num_clusters, dtype=torch.float32), requires_grad=False)
        self.z2 = nn.Parameter(torch.zeros(self.num_clusters, dtype=torch.int32), requires_grad=False)
        for c in range(self.num_clusters):
            self.s2[c], self.z2[c] = self.norms[c].get_weight_qparams()

        if s_external:
            self.s3, self.z3 = nn.Parameter(s_external, requires_grad=False),\
                               nn.Parameter(z_external, requires_grad=False)
        else:
            self.s3, self.z3 = calc_qparams_per_cluster(self.act_range, self.act_qmax)

        self.M0 = nn.Parameter(torch.zeros(self.num_clusters, dtype=torch.int32), requires_grad=False)
        self.shift = nn.Parameter(torch.zeros(self.num_clusters, dtype=torch.int32), requires_grad=False)
        for c in range(self.num_clusters):
            self.M0[c], self.shift[c] = quantize_M(self.s1[c] * self.s2[c] / self.s3[c])
        return self.s3, self.z3


class FusedBnReLU(nn.Module):
    def __init__(self, num_features, activation=None, act_qmax=None, w_qmax=None, is_pcq=False, arg_dict=None):
        super(FusedBnReLU, self).__init__()
        self.layer_type = 'FusedBnReLU'
        self.bit, self.smooth, self.use_ste, self.runtime_helper, self.num_clusters = \
            itemgetter('bit', 'smooth', 'ste', 'runtime_helper', 'cluster')(arg_dict)

        self.is_pcq = is_pcq
        self.w_qmax = w_qmax if w_qmax else 2 ** 8 - 1
        if not is_pcq:
            self.act_qmax = act_qmax if act_qmax else 2 ** self.bit - 1
            self.act_range = nn.Parameter(torch.zeros(2), requires_grad=False)
            self.apply_ema = False

        self.num_features = num_features
        self.bn = nn.BatchNorm2d(num_features)
        self._activation = activation(inplace=True) if activation else None

    def forward(self, x, external_range=None):
        if not self.training:
            return self._forward_impl(x)

        if not self.runtime_helper.pcq_initialized:
            return self._forward_impl(x)

        out = self._fake_quantized_bn(x)
        if self.is_pcq:
            return out

        self._update_activation_range(out, external_range)
        if self.runtime_helper.apply_fake_quantization:
            return self._fake_quantize_activation(out, external_range)
        else:
            return out

    def _forward_impl(self, x):
        x = self.bn(x)
        if self._activation:
            x = self._activation(x)
        return x

    def _fake_quantized_bn(self, x):
        out = self.bn(x)
        if self._activation:
            out = self._activation(out)

        with torch.no_grad():
            mean = x.detach().mean(dim=(0, 2, 3))
            var = x.detach().var(dim=(0, 2, 3), unbiased=False)
            weight = self.bn.weight.div(torch.sqrt(var + self.bn.eps))
            bias = self.bn.bias - weight * mean
            if weight.min() > 0:
                s, z = calc_qparams(torch.tensor(0), weight.max(), self.w_qmax)
            elif weight.max() < 0:
                s, z = calc_qparams(weight.min(), torch.tensor(0), self.w_qmax)
            else:
                s, z = calc_qparams(weight.min(), weight.max(), self.w_qmax)

            weight = fake_quantize(weight, s, z, self.w_qmax, use_ste=False)

            fake_out = x * weight[None, :, None, None] + bias[None, :, None, None]
            if self._activation:
                fake_out = self._activation(fake_out)
        return STE.apply(out, fake_out)

    def _fake_quantize_activation(self, x, external_range=None):
        if external_range is not None:
            s, z = calc_qparams(external_range[0], external_range[1], self.act_qmax)
        else:
            s, z = calc_qparams(self.act_range[0], self.act_range[1], self.act_qmax)
        return fake_quantize(x, s, z, self.act_qmax, self.use_ste)

    def _update_activation_range(self, x, external_range=None):
        if external_range is None:
            self.act_range[0], self.act_range[1] = ema(x, self.act_range, self.smooth)

    def set_qparams(self, s1, z1, s_external=None, z_external=None):
        self.s1, self.z1 = nn.Parameter(s1, requires_grad=False), nn.Parameter(z1, requires_grad=False)

        weight = self.bn.weight.div(torch.sqrt(self.bn.running_var + self.bn.eps))
        if weight.min() > 0:
            self.s2, self.z2 = calc_qparams(torch.tensor(0), weight.max(), self.w_qmax)
        elif weight.max() < 0:
            self.s2, self.z2 = calc_qparams(weight.min(), torch.tensor(0), self.w_qmax)
        else:
            self.s2, self.z2 = calc_qparams(weight.min(), weight.max(), self.w_qmax)

        if s_external:
            self.s3, self.z3 = nn.Parameter(s_external, requires_grad=False), nn.Parameter(z_external, requires_grad=False)
        else:
            self.s3, self.z3 = calc_qparams(self.act_range[0], self.act_range[1], self.act_qmax)

        self.M0, self.shift = quantize_M(self.s1 * self.s2 / self.s3)
        return self.s3, self.z3

    def get_weight_qparams(self):
        weight = self.bn.weight.div(torch.sqrt(self.bn.running_var + self.bn.eps))
        if weight.min() > 0:
            s, z = calc_qparams(torch.tensor(0), weight.max(), self.w_qmax)
        elif weight.max() < 0:
            s, z = calc_qparams(weight.min(), torch.tensor(0), self.w_qmax)
        else:
            s, z = calc_qparams(weight.min(), weight.max(), self.w_qmax)
        return s, z

    def get_multiplier_qparams(self):
        return self.M0, self.shift

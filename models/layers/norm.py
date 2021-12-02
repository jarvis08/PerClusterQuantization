from operator import itemgetter

import torch.nn as nn
import torch
import torch.nn.functional as F
from ..quantization_utils import *


class QuantizedBn2d(nn.Module):
    def __init__(self, num_features, multiplication=True, arg_dict=None):
        super(QuantizedBn2d, self).__init__()
        self.layer_type = 'QuantizedBn2d'
        self.num_clusters, self.runtime_helper = itemgetter('cluster', 'runtime_helper')(arg_dict)
        self.num_features = num_features
        self.w_bit = nn.Parameter(torch.tensor(0, dtype=torch.int8), requires_grad=False)
        self.a_bit = nn.Parameter(torch.tensor(0, dtype=torch.int8), requires_grad=False)

        t_init = list(range(self.num_clusters)) if self.num_clusters > 1 else 0
        self.s1 = nn.Parameter(torch.tensor(t_init, dtype=torch.float32), requires_grad=False)
        self.s2 = nn.Parameter(torch.tensor(0, dtype=torch.float32), requires_grad=False)
        self.s3 = nn.Parameter(torch.tensor(t_init, dtype=torch.float32), requires_grad=False)
        self.z1 = nn.Parameter(torch.tensor(t_init, dtype=torch.int32), requires_grad=False)
        self.z2 = nn.Parameter(torch.tensor(0, dtype=torch.int32), requires_grad=False)
        self.z3 = nn.Parameter(torch.tensor(t_init, dtype=torch.int32), requires_grad=False)
        self.M0 = nn.Parameter(torch.tensor(t_init, dtype=torch.int32), requires_grad=False)
        self.shift = nn.Parameter(torch.tensor(t_init, dtype=torch.int32), requires_grad=False)
        self.is_shift_neg = nn.Parameter(torch.tensor(False, dtype=torch.bool), requires_grad=False)
        self.multiplication = multiplication

        self.weight = nn.Parameter(torch.zeros((self.num_clusters, num_features), dtype=torch.int32), requires_grad=False)
        self.bias = nn.Parameter(torch.zeros((self.num_clusters, num_features), dtype=torch.int32), requires_grad=False)

    def forward(self, x):
        out = self._subsum(x)
        if self.multiplication:
            out = self._totalsum(out)
        return out

    def _subsum(self, x):
        if self.num_clusters > 1:
            return self._pcq_subsum(x)
        else:
            return self._general_subsum(x)

    def _totalsum(self, x):
        if self.num_clusters > 1:
            out = self._pcq_totalsum(x)
        else:
            out = self._general_totalsum(x)
        return clamp_matrix(out, self.a_bit)

    def _pcq_subsum(self, x):
        bc = self.runtime_helper.batch_cluster
        weight = torch.index_select(self.weight, 0, bc)[:, :, None, None]
        bias = torch.index_select(self.bias, 0, bc)[:, :, None, None]
        z1 = torch.index_select(self.z1, 0, bc)[:, None, None, None]

        q1q2 = x.mul(weight)
        q1z2 = x.mul(self.z2)
        q2z1 = weight.mul(z1)
        return q1q2 - q1z2 - q2z1 + z1 * self.z2 + bias

    def _pcq_totalsum(self, subsum):
        bc = self.runtime_helper.batch_cluster
        z3 = torch.index_select(self.z3, 0, bc)[:, None, None, None]
        M0 = torch.index_select(self.M0, 0, bc)[:, None, None, None]
        shift = torch.index_select(self.shift, 0, bc)[:, None, None, None]
        shape = subsum.shape
        mask = self.runtime_helper.mask_4d[:shape[0]]

        if not self.is_shift_neg:
            total = mul_and_shift(subsum, M0, shift, mask)
        else:
            zero = self.runtime_helper.izero
            neg_shift = torch.where(shift < zero, - shift, zero)
            shift = torch.where(shift >= zero, shift, zero)
            subsum = subsum << neg_shift
            total = mul_and_shift(subsum, M0, shift, mask)
        return total.add(z3)

    def _general_subsum(self, x):
        q1q2 = x.mul(self.weight[0][None, :, None, None])
        q1z2 = x.mul(self.z2)
        q2z1 = self.weight[0].mul(self.z1)
        return q1q2 - q1z2 - q2z1[None, :, None, None] + self.z1 * self.z2 + self.bias[0][None, :, None, None]

    def _general_totalsum(self, subsum):
        if self.shift < 0:
            total = mul_and_shift(subsum << - self.shift.item(), self.M0, 0)
        else:
            total = mul_and_shift(subsum, self.M0, self.shift.item())
        return total.add(self.z3)


class PCQBnReLU(nn.Module):
    def __init__(self, num_features, activation=None, a_bit=None, w_bit=None, arg_dict=None):
        super(PCQBnReLU, self).__init__()
        self.layer_type = 'PCQBnReLU'
        self.momentum, arg_w_bit, self.smooth, self.runtime_helper, self.num_clusters, self.use_ste = \
            itemgetter('bn_momentum', 'bit', 'smooth', 'runtime_helper', 'cluster', 'ste')(arg_dict)

        w_bit = w_bit if w_bit is not None else arg_dict['bit_bn_w']
        a_bit = a_bit if a_bit is not None else arg_dict['bit']
        self.w_bit = torch.nn.Parameter(torch.tensor(w_bit, dtype=torch.int8), requires_grad=False)
        self.a_bit = torch.nn.Parameter(torch.tensor(a_bit, dtype=torch.int8), requires_grad=False)

        self.act_range = nn.Parameter(torch.zeros((self.num_clusters, 2)), requires_grad=False)
        self.apply_ema = nn.Parameter(torch.zeros(self.num_clusters, dtype=torch.bool), requires_grad=False)

        self.num_features = num_features
        self.norms = nn.ModuleList([nn.BatchNorm2d(num_features) for _ in range(self.num_clusters)])
        self.activation = activation(inplace=True) if activation else None

    def forward(self, x, external_range=None):
        if not self.training:
            return self._forward_impl(x)

        out = self._pcq(x)
        if external_range is None:
            self._update_activation_ranges(out)
        if self.runtime_helper.apply_fake_quantization:
            out = self._fake_quantize_activation(out, external_range)
        return out

    def _forward_impl(self, x):
        bc = self.runtime_helper.batch_cluster
        exists = torch.unique(bc)
        out = torch.zeros(x.shape, device='cuda')
        for c in exists:
            indices = (bc == c).nonzero(as_tuple=True)[0]
            out[indices] = self.norms[c](x[indices])
        if self.activation:
            out = self.activation(out)
        return out

    def _pcq(self, x):
        cluster = self.runtime_helper.batch_cluster
        bn = self.norms[cluster]
        out = bn(x)
        out = self.activation(out)

        with torch.no_grad():
            _x = x.detach()
            mean = _x.mean(dim=(0, 2, 3))
            var = _x.var(dim=(0, 2, 3), unbiased=False)

            weight = bn.weight.div(torch.sqrt(var + bn.eps))
            bias = bn.bias - weight * mean
            scale, zero_point = calc_qparams(weight.min(), weight.max(), self.w_bit, self.runtime_helper.fzero)
            weight = fake_quantize(weight, scale, zero_point, self.w_bit)

            fake_out = _x * weight[None, :, None, None] + bias[None, :, None, None]
            fake_out = self.activation(fake_out)
        return STE.apply(out, fake_out)
        # out = STE.apply(out, fake_out)
        # if self.activation:
        #     out = self.activation(out)
        # return out

    @torch.no_grad()
    def _update_activation_ranges(self, x):
        cluster = self.runtime_helper.batch_cluster
        data = x.view(x.size(0), -1)
        _max = data.max(dim=1).values.mean()

        if self.activation:
            if self.apply_ema[cluster]:
                self.act_range[cluster][1] = self.act_range[cluster][1] * self.smooth + _max * (1 - self.smooth)
            else:
                self.act_range[cluster][1] = _max
                self.apply_ema[cluster] = True
        else:
            _min = data.min(dim=1).values.mean()
            if self.apply_ema[cluster]:
                self.act_range[cluster][0] = self.act_range[cluster][0] * self.smooth + _min * (1 - self.smooth)
                self.act_range[cluster][1] = self.act_range[cluster][1] * self.smooth + _max * (1 - self.smooth)
            else:
                self.act_range[cluster][0], self.act_range[cluster][1] = _min, _max
                self.apply_ema[cluster] = True

    def _fake_quantize_activation(self, x, external_range=None):
        cluster = self.runtime_helper.batch_cluster
        zero = self.runtime_helper.fzero
        if external_range is not None:
            s, z = calc_qparams(external_range[cluster][0], external_range[cluster][1], self.a_bit, zero)
        else:
            s, z = calc_qparams(self.act_range[cluster][0], self.act_range[cluster][1], self.a_bit, zero)
        return fake_quantize(x, s, z, self.a_bit, use_ste=self.use_ste)

    @torch.no_grad()
    def set_qparams(self, s1, z1, s_external=None, z_external=None):
        zero = self.runtime_helper.fzero
        self.s1, self.z1 = s1, z1

        _weights = torch.zeros((self.num_clusters, self.num_features), device='cuda')
        _vars = torch.ones((self.num_clusters, self.num_features), device='cuda')
        for c in range(self.num_clusters):
            _weights[c] = self.norms[c].weight
            _vars[c] = self.norms[c].running_var
        weight = _weights.div(torch.sqrt(_vars + self.norms[0].eps))
        self.s2, self.z2 = calc_qparams(weight.min(), weight.max(), self.w_bit, zero)

        if s_external is not None:
            self.s3, self.z3 = s_external, z_external
        else:
            self.s3, self.z3 = calc_qparams_per_cluster(self.act_range, self.a_bit, zero)

        self.M0 = torch.zeros(self.num_clusters, dtype=torch.int32)
        self.shift = torch.zeros(self.num_clusters, dtype=torch.int32)
        for c in range(self.num_clusters):
            self.M0[c], self.shift[c] = quantize_M(self.s1[c] * self.s2 / self.s3[c])
        return self.s3, self.z3


class FusedBnReLU(nn.Module):
    def __init__(self, num_features, activation=None, w_bit=None, a_bit=None, arg_dict=None):
        super(FusedBnReLU, self).__init__()
        self.layer_type = 'FusedBnReLU'
        arg_w_bit, self.smooth, self.use_ste, self.runtime_helper, self.num_clusters = \
            itemgetter('bit', 'smooth', 'ste', 'runtime_helper', 'cluster')(arg_dict)

        w_bit = w_bit if w_bit is not None else arg_dict['bit_bn_w']
        a_bit = a_bit if a_bit is not None else arg_dict['bit']
        self.w_bit = torch.nn.Parameter(torch.tensor(w_bit, dtype=torch.int8), requires_grad=False)
        self.a_bit = torch.nn.Parameter(torch.tensor(a_bit, dtype=torch.int8), requires_grad=False)

        self.act_range = nn.Parameter(torch.zeros(2), requires_grad=False)
        self.apply_ema = nn.Parameter(torch.zeros(1, dtype=torch.bool), requires_grad=False)

        self.num_features = num_features
        self.bn = nn.BatchNorm2d(num_features)
        self._activation = activation(inplace=True) if activation else None

    def forward(self, x, external_range=None):
        if not self.training:
            return self._forward_impl(x)

        out = self._fake_quantized_bn(x)
        if external_range is None:
            self._update_activation_range(out)
        if self.runtime_helper.apply_fake_quantization:
            out = self._fake_quantize_activation(out, external_range)
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
            _x = x.detach()
            mean = _x.mean(dim=(0, 2, 3))
            var = _x.var(dim=(0, 2, 3), unbiased=False)

            weight = self.bn.weight.div(torch.sqrt(var + self.bn.eps))
            bias = self.bn.bias - weight * mean
            s, z = calc_qparams(weight.min(), weight.max(), self.w_bit)
            weight = fake_quantize(weight, s, z, self.w_bit)

            fake_out = _x * weight[None, :, None, None] + bias[None, :, None, None]
            if self._activation:
                fake_out = self._activation(fake_out)
        return STE.apply(out, fake_out)

    def _update_activation_range(self, x):
        if self.apply_ema:
            self.act_range[0], self.act_range[1] = ema(x, self.act_range, self.smooth)
        else:
            self.act_range[0], self.act_range[1] = get_range(x)
            self.apply_ema.data = torch.tensor(True, dtype=torch.bool)

    def _fake_quantize_activation(self, x, external_range=None):
        if external_range is not None:
            s, z = calc_qparams(external_range[0], external_range[1], self.a_bit)
        else:
            s, z = calc_qparams(self.act_range[0], self.act_range[1], self.a_bit)
        return fake_quantize(x, s, z, self.a_bit, self.use_ste)

    def set_qparams(self, s1, z1, s_external=None, z_external=None):
        self.s1, self.z1 = s1, z1

        weight = self.bn.weight.div(torch.sqrt(self.bn.running_var + self.bn.eps))
        if weight.min() > 0:
            self.s2, self.z2 = calc_qparams(torch.tensor(0), weight.max(), self.w_bit)
        elif weight.max() < 0:
            self.s2, self.z2 = calc_qparams(weight.min(), torch.tensor(0), self.w_bit)
        else:
            self.s2, self.z2 = calc_qparams(weight.min(), weight.max(), self.w_bit)

        if s_external is not None:
            self.s3, self.z3 = s_external, z_external
        else:
            self.s3, self.z3 = calc_qparams(self.act_range[0], self.act_range[1], self.a_bit)

        self.M0, self.shift = quantize_M(self.s1 * self.s2 / self.s3)
        return self.s3, self.z3

    def get_weight_qparams(self):
        weight = self.bn.weight.div(torch.sqrt(self.bn.running_var + self.bn.eps))
        if weight.min() > 0:
            s, z = calc_qparams(torch.tensor(0), weight.max(), self.w_bit)
        elif weight.max() < 0:
            s, z = calc_qparams(weight.min(), torch.tensor(0), self.w_bit)
        else:
            s, z = calc_qparams(weight.min(), weight.max(), self.w_bit)
        return s, z

    def get_multiplier_qparams(self):
        return self.M0, self.shift

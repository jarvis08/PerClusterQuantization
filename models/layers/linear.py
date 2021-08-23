from operator import itemgetter

import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

from ..quant_noise import _quant_noise
from ..quantization_utils import *
from .activation import *


class QuantizedLinear(nn.Linear):
    batch_cluster = None

    def __init__(self, in_features, out_features, bias=False, activation=None, arg_dict=None):
        super(QuantizedLinear, self).__init__(in_features, out_features, bias)
        self.layer_type = 'QuantizedLinear'
        self.bit, self.num_clusters, self.runtime_helper =\
                itemgetter('bit', 'cluster', 'runtime_helper')(arg_dict)
        self.q_max = 2 ** self.bit - 1
        self.act_qmax = nn.Parameter(torch.tensor(0, dtype=torch.int32), requires_grad=False)
        self.out_features = out_features

        self.quantized_bias = nn.Parameter(torch.zeros((self.num_clusters, out_features)), requires_grad=False)

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
        sum_q1q2 = F.linear(x, self.weight, None)
        if self.runtime_helper.batch_cluster is not None:
            return self.pcq_totalsum(x, sum_q1q2.type(torch.cuda.IntTensor))
        else:
            return self.general_totalsum(x, sum_q1q2.type(torch.cuda.IntTensor))

    def pcq_totalsum(self, x, sum_q1q2):
        bc = self.runtime_helper.batch_cluster
        bias = torch.index_select(self.quantized_bias, 0, bc)
        z1 = torch.index_select(self.z1, 0, bc)
        z3 = torch.index_select(self.z3, 0, bc).reshape(bc.shape[0], 1)
        M0 = torch.index_select(self.M0, 0, bc).reshape(bc.shape[0], 1)
        shift = torch.index_select(self.shift, 0, bc).reshape(bc.shape[0], 1)

        input_feature, output_feature = sum_q1q2.shape[0], sum_q1q2.shape[1]
        N = x.shape[1]
        for out_f in range(output_feature):
            sum_q1q2[:, out_f] = sum_q1q2[:, out_f].add_(bias[:, out_f].type(torch.cuda.IntTensor))

        sum_a1 = torch.zeros(input_feature, dtype=torch.int32).cuda()
        sum_a2 = torch.zeros((bc.shape[0], output_feature), dtype=torch.int32).cuda()

        for out_f in range(output_feature):
            sum_a2[:, out_f] = torch.sum(self.weight[out_f, :]).mul(z1)

        for in_f in range(input_feature):
            sum_a1[in_f] = torch.sum(x[in_f, :]).mul(self.z2)

        z1 = z1.reshape(bc.shape[0], 1)
        nz1z2 = N * z1 * self.z2
        sum_q1q2 = sum_q1q2.add_(nz1z2.type(torch.cuda.IntTensor))

        for in_f in range(input_feature):
            sum_q1q2[in_f, :] = torch.sub(sum_q1q2[in_f, :], sum_a1[in_f])

        for out_f in range(output_feature):
            sum_q1q2[:, out_f] = torch.sub(sum_q1q2[:, out_f], sum_a2[:, out_f])

        multiplied = multiply_M(sum_q1q2.type(torch.cuda.LongTensor), M0)
        total = shifting2d(multiplied, shift)
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
        input_feature, output_feature = sum_q1q2.shape[0], sum_q1q2.shape[1]
        if self.bias is not None:
            for out_f in range(output_feature):
                sum_q1q2[:, out_f] = sum_q1q2[:, out_f].add(self.bias[out_f])
        N = x.shape[1]

        sum_a1 = torch.zeros(input_feature, dtype=torch.int32)
        sum_a2 = torch.zeros(output_feature, dtype=torch.int32)
        for out_f in range(output_feature):
            sum_a2[out_f] = torch.sum(self.weight[out_f, :]).mul(self.z1)
        for in_f in range(input_feature):
            sum_a1[in_f] = torch.sum(x[in_f, :]).mul(self.z2)

        nz1z2 = N * self.z1 * self.z2
        sub_sum = sum_q1q2.add(nz1z2)
        for in_f in range(input_feature):
            sub_sum[in_f, :] = torch.sub(sub_sum[in_f, :], sum_a1[in_f])
        for out_f in range(output_feature):
            sub_sum[:, out_f] = torch.sub(sub_sum[:, out_f], sum_a2[out_f])

        if self.shift < 0:
            multiplied = multiply_M((sub_sum.type(torch.cuda.LongTensor) << - self.shift.item()), self.M0)
            total = shifting(multiplied, 0)
        else:
            multiplied = multiply_M(sub_sum.type(torch.cuda.LongTensor), self.M0)
            total = shifting(multiplied, self.shift.item())
        total = total.add(self.z3)

        if self.activation is not None:
            hs_total = total + self.hardswish_3
            hs_total = torch.clamp(hs_total, self.z3.item(), self.hardswish_6.item())
            hs_total = hs_total / self.hardswish_6
            if self.activation == 'Hardswish':
                total = total * hs_total
            else:
                total = hs_total

        if self.act_qmax == 15:
            total = torch.clamp(total, 0, 15)
        elif self.act_qmax == 255:
            total = torch.clamp(total, -128, 127)
        elif self.act_qmax == 65535:  # INT 16
            total = torch.clamp(total, -32768, 32767)
        elif self.act_qmax == 4294967295:  # INT 32
            total = torch.clamp(total, -2147483648, 2147483647)
        return total.type(torch.cuda.FloatTensor)


class PCQLinear(nn.Module):
    """
        Fused Layer to calculate Quantization Parameters(S & Z) with multiple clusters
    """
    def __init__(self, in_features, out_features, bias=True, activation=None, arg_dict=None):
        super(PCQLinear, self).__init__()
        self.layer_type = 'PCQLinear'
        
        self.bit, self.smooth, self.num_clusters, self.runtime_helper, self.use_ste, self.quant_noise, self.qn_prob\
            = itemgetter('bit', 'smooth', 'cluster', 'runtime_helper', 'ste', 'quant_noise', 'qn_prob')(arg_dict)
        self.q_max = 2 ** self.bit - 1
        self.act_qmax = 2 ** 4 - 1
        self.act_range = nn.Parameter(torch.zeros((self.num_clusters, 2)), requires_grad=False)

        self.apply_ema = np.zeros(self.num_clusters, dtype=bool)

        self.fc = nn.Linear(in_features, out_features, bias=bias)
        if self.quant_noise:
            self.fc = _quant_noise(self.fc, self.qn_prob, 1, self.q_max)

        self._activation = activation(inplace=False) if activation else None

    def forward(self, x):
        general_out = self.fc(x)
        if self._activation:
            general_out = self._activation(general_out)

        if not self.runtime_helper.pcq_initialized:
            done = 0
            for c in range(self.num_clusters):
                if self.apply_ema[c]:
                    self.act_range[c][0], self.act_range[c][1] = ema(general_out[done:done + 8], self.act_range[c],
                                                                     self.smooth)
                else:
                    self.act_range[c][0] = torch.min(general_out[done:done + 8]).item()
                    self.act_range[c][1] = torch.max(general_out[done:done + 8]).item()
                    self.apply_ema[c] = True
                done += 8
            return general_out

        if not self.training:
            return general_out

        s, z = calc_qparams(self.fc.weight.min(), self.fc.weight.max(), self.q_max)
        fake_weight = fake_quantize(self.fc.weight, s, z, self.q_max, use_ste=False)

        fake_out = F.linear(x, fake_weight, self.fc.bias)
        if self._activation:
            fake_out = self._activation(fake_out)

        done = 0
        for i in range(self.runtime_helper.batch_cluster.shape[0]):
            c = self.runtime_helper.batch_cluster[i][0]
            n = self.runtime_helper.batch_cluster[i][1]
            if self.apply_ema[c]:
                self.act_range[c][0], self.act_range[c][1] = ema(fake_out[done:done + n], self.act_range[c], self.smooth)
                if self.runtime_helper.apply_fake_quantization:
                    s, z = calc_qparams(self.act_range[c][0], self.act_range[c][1], self.act_qmax)
                    fake_out[done:done + n] = fake_quantize(fake_out[done:done + n], s, z, self.act_qmax, use_ste=False)
            else:
                self.act_range[c][0] = torch.min(fake_out[done:done + n]).item()
                self.act_range[c][1] = torch.max(fake_out[done:done + n]).item()
                self.apply_ema[c] = True
            done += n
        return STE.apply(general_out, fake_out)

    def set_qparams(self, s1, z1):
        self.s1, self.z1 = torch.nn.Parameter(s1, requires_grad=False), torch.nn.Parameter(z1, requires_grad=False)

        self.s2, self.z2 = calc_qparams(torch.min(self.fc.weight), torch.max(self.fc.weight), self.q_max)

        self.s3 = nn.Parameter(torch.zeros(self.num_clusters, dtype=torch.float32), requires_grad=False)
        self.z3 = nn.Parameter(torch.zeros(self.num_clusters, dtype=torch.int32), requires_grad=False)
        self.M0 = nn.Parameter(torch.zeros(self.num_clusters, dtype=torch.int32), requires_grad=False)
        self.shift = nn.Parameter(torch.zeros(self.num_clusters, dtype=torch.int32), requires_grad=False)
        for c in range(self.num_clusters):
            self.s3[c], self.z3[c] = calc_qparams(self.act_range[c][0], self.act_range[c][1], self.act_qmax)
            self.M0[c], self.shift[c] = quantize_M(self.s1[c] * self.s2 / self.s3[c])
        return self.s3, self.z3


class FusedLinear(nn.Module):
    """
        Fused Layer to calculate Quantization Parameters (S & Z)
    """
    def __init__(self, in_features, out_features, bias=True, activation=None, arg_dict=None):
        super(FusedLinear, self).__init__()
        self.layer_type = 'FusedLinear'

        self.arg_dict = arg_dict
        self.bit, self.smooth, self.use_ste, self.runtime_helper, self.quant_noise, self.qn_prob \
            = itemgetter('bit', 'smooth', 'ste', 'runtime_helper', 'quant_noise', 'qn_prob')(arg_dict)
        self.q_max = 2 ** self.bit - 1
        self.act_qmax = 2 ** self.bit - 1
        self.act_range = nn.Parameter(torch.zeros(2), requires_grad=False)

        self.apply_ema = False

        self.fc = nn.Linear(in_features, out_features, bias=bias)

        if self.quant_noise:
            self.fc = _quant_noise(self.fc, self.qn_prob + self.runtime_helper.qn_prob_increment, 1, q_max=self.q_max)
        self._activation = activation(inplace=False) if activation else None

    def forward(self, x):
        general_out = self.fc(x)
        if self._activation:
            general_out = self._activation(general_out)
        if not self.training:
            return general_out

        with torch.no_grad():
            s, z = calc_qparams(torch.min(self.fc.weight), torch.max(self.fc.weight), self.q_max)
            fake_weight = fake_quantize(self.fc.weight, s, z, self.q_max, use_ste=False)

            fake_out = F.linear(x, fake_weight, self.fc.bias)
            if self._activation:
                fake_out = self._activation(fake_out)

            if self.apply_ema:
                self.act_range[0], self.act_range[1] = ema(fake_out, self.act_range, self.smooth)
                if self.runtime_helper.apply_fake_quantization:
                    s, z = calc_qparams(self.act_range[0], self.act_range[1], self.act_qmax)
                    fake_out = fake_quantize(fake_out, s, z, self.act_qmax, use_ste=False)
            else:
                self.act_range[0] = torch.min(fake_out).item()
                self.act_range[1] = torch.max(fake_out).item()
                self.apply_ema = True
        return STE.apply(general_out, fake_out)

    def set_qparams(self, s1, z1):
        self.s1, self.z1 = torch.nn.Parameter(s1, requires_grad=False), torch.nn.Parameter(z1, requires_grad=False)
        self.s2, self.z2 = calc_qparams(torch.min(self.fc.weight), torch.max(self.fc.weight), self.q_max)
        self.s3, self.z3 = calc_qparams(self.act_range[0], self.act_range[1], self.q_max)
        self.M0, self.shift = quantize_M(self.s1 * self.s2 / self.s3)
        return self.s3, self.z3

# --task_name MRPC --do_train --do_eval --do_lower_case --data_dir $GLUE_DIR/MRPC/ --bert_model /home/hansung/quantization/bert/pytorch-pretrained-BERT/examples/uncased_L-4_H-512_A-8 --max_seq_length 128 --train_batch_size 8 --learning_rate 2e-5 --num_train_epochs 3.0 --output_dir ./mrpc_output

import torch
import torch.nn as nn
from .layers.linear import *
from .quant_noise import _quant_noise
from .quantization_utils import *


class PCQMLP(nn.Module):
    def __init__(self, arg_dict: dict, n_channels: int = 3, num_classes: int = 10) -> None:
        super(PCQMLP, self).__init__()
        target_bit, bit_first, bit_classifier, self.smooth, self.num_clusters, self.runtime_helper \
            = itemgetter('bit', 'bit_first', 'bit_classifier', 'smooth', 'cluster', 'runtime_helper')(arg_dict)
        self.target_bit = torch.nn.Parameter(torch.tensor(target_bit, dtype=torch.int8), requires_grad=False)
        self.in_bit = torch.nn.Parameter(torch.tensor(bit_first, dtype=torch.int8), requires_grad=False)

        self.in_range = nn.Parameter(torch.zeros((self.num_clusters, 2)), requires_grad=False)
        self.apply_ema = nn.Parameter(torch.zeros(self.num_clusters, dtype=torch.bool), requires_grad=False)

        self.fc1 = PCQLinear(1024 * n_channels, 1024, bias=True, activation=nn.ReLU,
                             w_bit=bit_first, a_bit=bit_first, arg_dict=arg_dict)
        self.fc2 = PCQLinear(1024, 1024, bias=True, activation=nn.ReLU, arg_dict=arg_dict)
        self.fc3 = PCQLinear(1024, 1024, bias=True, activation=nn.ReLU, arg_dict=arg_dict)
        self.fc4 = PCQLinear(1024, num_classes, bias=True, is_classifier=True,
                             w_bit=bit_classifier, a_bit=bit_classifier, arg_dict=arg_dict)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.flatten(x, 1)
        if self.training:
            self._update_input_ranges(x)
            if self.runtime_helper.apply_fake_quantization:
                x = self._fake_quantize_input(x)

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x

    @torch.no_grad()
    def _update_input_ranges(self, x):
        cluster = self.runtime_helper.batch_cluster
        _min = x.min(dim=1).values.mean()
        _max = x.max(dim=1).values.mean()
        if self.apply_ema[cluster]:
            self.in_range[cluster][0] = self.in_range[cluster][0] * self.smooth + _min * (1 - self.smooth)
            self.in_range[cluster][1] = self.in_range[cluster][1] * self.smooth + _max * (1 - self.smooth)
        else:
            self.in_range[cluster][0], self.in_range[cluster][1] = _min, _max
            self.apply_ema[cluster] = True

    def _fake_quantize_input(self, x):
        cluster = self.runtime_helper.batch_cluster
        s, z = calc_qparams(self.in_range[cluster][0], self.in_range[cluster][1], self.in_bit)
        return fake_quantize(x, s, z, self.in_bit)

    def set_quantization_params(self):
        self.scale, self.zero_point = calc_qparams_per_cluster(self.in_range, self.in_bit)
        prev_s, prev_z = self.fc1.set_qparams(self.scale, self.zero_point)
        prev_s, prev_z = self.fc2.set_qparams(prev_s, prev_z)
        prev_s, prev_z = self.fc3.set_qparams(prev_s, prev_z)
        self.fc4.set_qparams(prev_s, prev_z)


def pcq_mlp(arg_dict: dict, n_channels=3, num_classes=10) -> PCQMLP:
    return PCQMLP(arg_dict, n_channels=n_channels, num_classes=num_classes)


def modify_fused_mlp_qn_pre_hook(model):
    model.fc1.fc = _quant_noise(model.fc1.fc, model.runtime_helper.qn_prob, 1, q_max=2 ** model.fc1.w_bit - 1)
    model.fc2.fc = _quant_noise(model.fc2.fc, model.runtime_helper.qn_prob, 1, q_max=2 ** model.fc2.w_bit - 1)
    model.fc3.fc = _quant_noise(model.fc3.fc, model.runtime_helper.qn_prob, 1, q_max=2 ** model.fc3.w_bit - 1)
    model.fc4.fc = _quant_noise(model.fc4.fc, model.runtime_helper.qn_prob, 1, q_max=2 ** model.fc4.w_bit - 1)


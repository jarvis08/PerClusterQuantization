import torch
import torch.nn as nn
from .layers.linear import *
from .quant_noise import _quant_noise
from .quantization_utils import *


class PCQMLP(nn.Module):
    def __init__(self, arg_dict: dict, num_classes: int = 10) -> None:
        super(PCQMLP, self).__init__()
        target_bit, first_bit, classifier_bit, self.smooth, self.num_clusters, self.runtime_helper \
            = itemgetter('bit', 'first_bit', 'classifier_bit', 'smooth', 'cluster', 'runtime_helper')(arg_dict)
        self.target_bit = torch.nn.Parameter(torch.tensor(target_bit, dtype=torch.int8), requires_grad=False)
        self.in_bit = torch.nn.Parameter(torch.tensor(first_bit, dtype=torch.int8), requires_grad=False)

        self.in_range = nn.Parameter(torch.zeros((self.num_clusters, 2)), requires_grad=False)
        self.apply_ema = nn.Parameter(torch.zeros(self.num_clusters, dtype=torch.bool), requires_grad=False)

        self.fc1 = PCQLinear(3072, 1024, bias=True, activation=nn.ReLU,
                             w_bit=first_bit, a_bit=first_bit, arg_dict=arg_dict)
        self.fc2 = PCQLinear(1024, 1024, bias=True, activation=nn.ReLU, arg_dict=arg_dict)
        self.fc3 = PCQLinear(1024, 1024, bias=True, activation=nn.ReLU, arg_dict=arg_dict)
        self.fc4 = PCQLinear(1024, num_classes, bias=True, is_classifier=True,
                             w_bit=classifier_bit, a_bit=classifier_bit, arg_dict=arg_dict)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            self._update_input_ranges(x)
            if self.runtime_helper.apply_fake_quantization:
                x = self._fake_quantize_input(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x

    @torch.no_grad()
    def _update_input_ranges(self, x):
        cluster = self.runtime_helper.batch_cluster
        data = x.view(x.size(0), -1)
        _min = data.min(dim=1).values.mean()
        _max = data.max(dim=1).values.mean()
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
        self.scale, self.zero_point = calc_qparams(self.in_range[0], self.in_range[1], self.in_bit)
        prev_s, prev_z = self.fc1.set_qparams(self.scale, self.zero_point)
        prev_s, prev_z = self.fc2.set_qparams(prev_s, prev_z)
        prev_s, prev_z = self.fc3.set_qparams(prev_s, prev_z)
        self.fc4.set_qparams(prev_s, prev_z)


def pcq_mlp(arg_dict: dict, num_classes=10) -> PCQMLP:
    return PCQMLP(arg_dict, num_classes=num_classes)


def set_fused_alexnet(fused, pre):
    fused.fc1 = copy_from_pretrained(fused.fc1, pre.fc1)
    fused.fc2 = copy_from_pretrained(fused.fc2, pre.fc2)
    fused.fc3 = copy_from_pretrained(fused.fc3, pre.fc3)
    fused.fc4 = copy_from_pretrained(fused.fc4, pre.fc4)
    return fused

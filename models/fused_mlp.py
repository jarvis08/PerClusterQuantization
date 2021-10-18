import torch
import torch.nn as nn
from .layers.linear import *
from .quant_noise import _quant_noise
from .quantization_utils import *


class FusedMLP(nn.Module):
    def __init__(self, arg_dict: dict, n_channels: int = 3, num_classes: int = 10) -> None:
        super(FusedMLP, self).__init__()
        target_bit, first_bit, classifier_bit, self.smooth, self.runtime_helper \
            = itemgetter('bit', 'first_bit', 'classifier_bit', 'smooth', 'runtime_helper')(arg_dict)
        self.target_bit = torch.nn.Parameter(torch.tensor(target_bit, dtype=torch.int8), requires_grad=False)
        self.in_bit = torch.nn.Parameter(torch.tensor(first_bit, dtype=torch.int8), requires_grad=False)

        self.in_range = nn.Parameter(torch.zeros(2), requires_grad=False)
        self.apply_ema = False

        self.fc1 = FusedLinear(1024 * n_channels, 1024, bias=True, activation=nn.ReLU,
                               w_bit=first_bit, a_bit=first_bit, arg_dict=arg_dict)
        self.fc2 = FusedLinear(1024, 1024, bias=True, activation=nn.ReLU, arg_dict=arg_dict)
        self.fc3 = FusedLinear(1024, 1024, bias=True, activation=nn.ReLU, arg_dict=arg_dict)
        self.fc4 = FusedLinear(1024, num_classes, bias=True, is_classifier=True,
                               w_bit=classifier_bit, a_bit=classifier_bit, arg_dict=arg_dict)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            if self.apply_ema:
                self.in_range[0], self.in_range[1] = ema(x, self.in_range, self.smooth)
                if self.runtime_helper.apply_fake_quantization:
                    s, z = calc_qparams(self.in_range[0], self.in_range[1], self.in_bit)
                    x = fake_quantize(x, s, z, self.in_bit)
            else:
                self.in_range[0], self.in_range[1] = get_range(x)
                self.apply_ema = True

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x

    def set_quantization_params(self):
        self.scale, self.zero_point = calc_qparams(self.in_range[0], self.in_range[1], self.in_bit)
        prev_s, prev_z = self.fc1.set_qparams(self.scale, self.zero_point)
        prev_s, prev_z = self.fc2.set_qparams(prev_s, prev_z)
        prev_s, prev_z = self.fc3.set_qparams(prev_s, prev_z)
        self.fc4.set_qparams(prev_s, prev_z)


def fused_mlp(arg_dict: dict, n_channels=3, num_classes=10) -> FusedMLP:
    return FusedMLP(arg_dict, n_channels=n_channels, num_classes=num_classes)


def set_fused_mlp(fused, pre):
    fused.fc1 = copy_from_pretrained(fused.fc1, pre.fc1)
    fused.fc2 = copy_from_pretrained(fused.fc2, pre.fc2)
    fused.fc3 = copy_from_pretrained(fused.fc3, pre.fc3)
    fused.fc4 = copy_from_pretrained(fused.fc4, pre.fc4)
    return fused


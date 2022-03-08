from operator import itemgetter

import torch
import torch.nn as nn
from .layers.linear import *
from .quantization_utils import *


class QuantizedMLP(nn.Module):
    def __init__(self, arg_dict, n_channels: int = 3, num_classes: int = 10) -> None:
        super(QuantizedMLP, self).__init__()
        bit, self.num_clusters, self.runtime_helper = itemgetter('bit', 'cluster', 'runtime_helper')(arg_dict)

        self.target_bit = nn.Parameter(torch.tensor(bit, dtype=torch.int8), requires_grad=False)
        self.in_bit = nn.Parameter(torch.tensor(bit, dtype=torch.int8), requires_grad=False)

        t_init = list(range(self.num_clusters)) if self.num_clusters > 1 else 0
        self.scale = nn.Parameter(torch.tensor(t_init, dtype=torch.float32), requires_grad=False)
        self.zero_point = nn.Parameter(torch.tensor(t_init, dtype=torch.int32), requires_grad=False)

        self.fc1 = QuantizedLinear(n_channels * 1024, 1024, arg_dict=arg_dict)
        self.fc2 = QuantizedLinear(1024, 1024, arg_dict=arg_dict)
        self.fc3 = QuantizedLinear(1024, 1024, arg_dict=arg_dict)
        self.fc4 = QuantizedLinear(1024, num_classes, arg_dict=arg_dict)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.flatten(x, 1)
        if self.runtime_helper.qat_batch_cluster is not None:
            x = quantize_matrix_2d(x, self.scale, self.zero_point, self.runtime_helper.qat_batch_cluster, self.in_bit)
        else:
            x = quantize_matrix(x, self.scale, self.zero_point, self.in_bit)

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x


def quantized_mlp(arg_dict: dict, n_channels: int = 3, num_classes: int = 10) -> QuantizedMLP:
    return QuantizedMLP(arg_dict, n_channels=n_channels, num_classes=num_classes)


def quantize_mlp(fp_model, int_model):
    int_model.target_bit.data = fp_model.target_bit
    int_model.in_bit.data = fp_model.in_bit
    int_model.scale.data = fp_model.scale
    int_model.zero_point.data = fp_model.zero_point
    int_model.fc1 = quantize(fp_model.fc1, int_model.fc1)
    int_model.fc2 = quantize(fp_model.fc2, int_model.fc2)
    int_model.fc3 = quantize(fp_model.fc3, int_model.fc3)
    int_model.fc4 = quantize(fp_model.fc4, int_model.fc4)
    return int_model

from operator import itemgetter

import torch
import torch.nn as nn
from .layers.linear import *
from .quantization_utils import *


class QuantizedMLP(nn.Module):
    def __init__(self, arg_dict, num_classes: int = 10) -> None:
        super(QuantizedMLP, self).__init__()
        self.num_clusters, self.runtime_helper = itemgetter('cluster', 'runtime_helper')(arg_dict)

        self.target_bit = nn.Parameter(torch.tensor(0, dtype=torch.int8), requires_grad=False)
        self.in_bit = nn.Parameter(torch.tensor(0, dtype=torch.int8), requires_grad=False)

        t_init = list(range(self.num_clusters)) if self.num_clusters > 1 else 0
        self.scale = nn.Parameter(torch.tensor(t_init, dtype=torch.float32), requires_grad=False)
        self.zero_point = nn.Parameter(torch.tensor(t_init, dtype=torch.int32), requires_grad=False)

        self.fc1 = QuantizedLinear(3072, 1024, arg_dict=arg_dict)
        self.fc2 = QuantizedLinear(1024, 1024, arg_dict=arg_dict)
        self.fc3 = QuantizedLinear(1024, 1024, arg_dict=arg_dict)
        self.fc4 = QuantizedLinear(1024, num_classes, arg_dict=arg_dict)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.runtime_helper.batch_cluster is not None:
            x = quantize_matrix_4d(x, self.scale, self.zero_point, self.runtime_helper.batch_cluster, self.in_bit)
        else:
            x = quantize_matrix(x, self.scale, self.zero_point, self.in_bit)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x


def quantized_mlp(arg_dict: dict, num_classes=10) -> QuantizedMLP:
    return QuantizedMLP(arg_dict, num_classes=num_classes)


def quantize_mlp(fp_model, int_model):
    int_model.scale = torch.nn.Parameter(fp_model.scale, requires_grad=False)
    int_model.zero_point = torch.nn.Parameter(fp_model.zero_point, requires_grad=False)
    int_model.fc1 = quantize(fp_model.fc1, int_model.fc1)
    int_model.fc2 = quantize(fp_model.fc2, int_model.fc2)
    int_model.fc3 = quantize(fp_model.fc3, int_model.fc3)
    int_model.fc4 = quantize(fp_model.fc4, int_model.fc4)
    return int_model

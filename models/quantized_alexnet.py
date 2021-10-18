from operator import itemgetter

import torch
import torch.nn as nn
from typing import Any
from .layers.conv2d import *
from .layers.linear import *
from .layers.maxpool2d import *
from .quantization_utils import *


class QuantizedAlexNet(nn.Module):
    batch_cluster = None

    def __init__(self, arg_dict, num_classes: int = 1000) -> None:
        super(QuantizedAlexNet, self).__init__()
        self.bit, self.num_clusters, self.runtime_helper = itemgetter('bit', 'cluster', 'runtime_helper')(arg_dict)
        self.q_max = 2 ** self.bit - 1

        t_init = list(range(self.num_clusters)) if self.num_clusters > 1 else 0
        self.scale = nn.Parameter(torch.tensor(t_init, dtype=torch.float32), requires_grad=False)
        self.zero_point = nn.Parameter(torch.tensor(t_init, dtype=torch.int32), requires_grad=False)

        self.maxpool = QuantizedMaxPool2d(kernel_size=3, stride=2, padding=0, arg_dict=arg_dict)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.conv1 = QuantizedConv2d(3, 64, kernel_size=11, stride=4, padding=2, bias=True, arg_dict=arg_dict)
        self.conv2 = QuantizedConv2d(64, 192, kernel_size=5, stride=1, padding=2, bias=True, arg_dict=arg_dict)
        self.conv3 = QuantizedConv2d(192, 384, kernel_size=3, stride=1, padding=1, bias=True, arg_dict=arg_dict)
        self.conv4 = QuantizedConv2d(384, 256, kernel_size=3, stride=1, padding=1, bias=True, arg_dict=arg_dict)
        self.conv5 = QuantizedConv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True, arg_dict=arg_dict)
        self.fc1 = QuantizedLinear(256 * 6 * 6, 4096, arg_dict=arg_dict)
        self.fc2 = QuantizedLinear(4096, 4096, arg_dict=arg_dict)
        self.fc3 = QuantizedLinear(4096, num_classes, arg_dict=arg_dict)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.runtime_helper.batch_cluster is not None:
            x = quantize_matrix_4d(x, self.scale, self.zero_point, self.runtime_helper.batch_cluster, self.q_max)
        else:
            x = quantize_matrix(x, self.scale, self.zero_point, self.q_max)

        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.maxpool(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.maxpool(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


class QuantizedAlexNetSmall(nn.Module):
    def __init__(self, arg_dict, num_classes: int = 10) -> None:
        super(QuantizedAlexNetSmall, self).__init__()
        bit, self.num_clusters, self.runtime_helper = itemgetter('bit', 'cluster', 'runtime_helper')(arg_dict)

        self.target_bit = nn.Parameter(torch.tensor(bit, dtype=torch.int8), requires_grad=False)
        self.in_bit = nn.Parameter(torch.tensor(bit, dtype=torch.int8), requires_grad=False)

        t_init = list(range(self.num_clusters)) if self.num_clusters > 1 else 0
        self.scale = nn.Parameter(torch.tensor(t_init, dtype=torch.float32), requires_grad=False)
        self.zero_point = nn.Parameter(torch.tensor(t_init, dtype=torch.int32), requires_grad=False)

        self.maxpool = QuantizedMaxPool2d(kernel_size=3, stride=2, padding=0, arg_dict=arg_dict)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv1 = QuantizedConv2d(3, 96, kernel_size=5, stride=1, padding=2, arg_dict=arg_dict)
        self.conv2 = QuantizedConv2d(96, 256, kernel_size=5, stride=1, padding=2, arg_dict=arg_dict)
        self.conv3 = QuantizedConv2d(256, 384, kernel_size=3, stride=1, padding=1, arg_dict=arg_dict)
        self.conv4 = QuantizedConv2d(384, 384, kernel_size=3, stride=1, padding=1, arg_dict=arg_dict)
        self.conv5 = QuantizedConv2d(384, 256, kernel_size=3, stride=1, padding=1, arg_dict=arg_dict)
        self.fc1 = QuantizedLinear(256, 4096, arg_dict=arg_dict)
        self.fc2 = QuantizedLinear(4096, 4096, arg_dict=arg_dict)
        self.fc3 = QuantizedLinear(4096, num_classes, arg_dict=arg_dict)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.runtime_helper.batch_cluster is not None:
            x = quantize_matrix_4d(x, self.scale, self.zero_point, self.runtime_helper.batch_cluster, self.in_bit)
        else:
            x = quantize_matrix(x, self.scale, self.zero_point, self.in_bit)

        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.maxpool(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.maxpool(x)
        x = self.avgpool(x)
        x = x.floor()
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


def quantized_alexnet(arg_dict: dict, **kwargs: Any) -> QuantizedAlexNet:
    return QuantizedAlexNet(arg_dict, **kwargs)


def quantized_alexnet_small(arg_dict: dict, num_classes=10, **kwargs: Any) -> QuantizedAlexNetSmall:
    return QuantizedAlexNetSmall(arg_dict, num_classes=num_classes, **kwargs)


def quantize_alexnet(fp_model, int_model):
    """
        Copy pre model's params & set fused layers.
        Use fused architecture, but not really fused (use CONV & BN seperately)
    """
    int_model.target_bit.data = fp_model.target_bit.data
    int_model.in_bit.data = fp_model.in_bit.data
    int_model.scale.data = fp_model.scale
    int_model.zero_point.data = fp_model.zero_point
    int_model.conv1 = quantize(fp_model.conv1, int_model.conv1)
    int_model.conv2 = quantize(fp_model.conv2, int_model.conv2)
    int_model.conv3 = quantize(fp_model.conv3, int_model.conv3)
    int_model.conv4 = quantize(fp_model.conv4, int_model.conv4)
    int_model.conv5 = quantize(fp_model.conv5, int_model.conv5)
    int_model.fc1 = quantize(fp_model.fc1, int_model.fc1)
    int_model.fc2 = quantize(fp_model.fc2, int_model.fc2)
    int_model.fc3 = quantize(fp_model.fc3, int_model.fc3)
    return int_model


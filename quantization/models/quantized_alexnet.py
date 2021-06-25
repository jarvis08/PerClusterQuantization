import torch
import torch.nn as nn
from typing import Any
from ..layers.conv2d import *
from ..layers.linear import *
from ..layers.maxpool2d import *
from ..quantization_utils import *


class QuantizedAlexNet(nn.Module):
    def __init__(self, num_classes: int = 1000, bit: int = 8) -> None:
        super(QuantizedAlexNet, self).__init__()
        self.bit = bit
        self.q_max = 2 ** self.bit - 1
        self.scale = nn.Parameter(torch.tensor(0, dtype=torch.float32), requires_grad=False)
        self.zero_point = nn.Parameter(torch.tensor(0, dtype=torch.int32), requires_grad=False)

        self.features = nn.Sequential(
            QuantizedConv2d(3, 64, kernel_size=11, stride=4, padding=2, bias=True, bit=bit),
            QuantizedMaxPool2d(kernel_size=3, stride=2, padding=0),
            QuantizedConv2d(64, 192, kernel_size=5, stride=1, padding=2, bias=True, bit=bit),
            QuantizedMaxPool2d(kernel_size=3, stride=2, padding=0),
            QuantizedConv2d(192, 384, kernel_size=3, stride=1, padding=1, bias=True, bit=bit),
            QuantizedConv2d(384, 256, kernel_size=3, stride=1, padding=1, bias=True, bit=bit),
            QuantizedConv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True, bit=bit),
            QuantizedMaxPool2d(kernel_size=3, stride=2, padding=0),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            QuantizedLinear(256 * 6 * 6, 4096, bit=bit),
            QuantizedLinear(4096, 4096, bit=bit),
            QuantizedLinear(4096, num_classes, bit=bit),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.quantize_input(x)
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def quantize_input(self, x):
        x = torch.round(x.div(self.scale).add(self.zero_point))
        return x


class QuantizedAlexNetSmall(nn.Module):
    def __init__(self, num_classes: int = 10, bit: int = 32) -> None:
        super(QuantizedAlexNetSmall, self).__init__()
        self.bit = bit
        self.q_max = 2 ** self.bit - 1
        self.scale = nn.Parameter(torch.tensor(0, dtype=torch.float32), requires_grad=False)
        self.zero_point = nn.Parameter(torch.tensor(0, dtype=torch.int32), requires_grad=False)

        self.features = nn.Sequential(
            QuantizedConv2d(3, 96, kernel_size=5, stride=1, padding=2, bias=True, bit=bit),
            QuantizedMaxPool2d(kernel_size=3, stride=2, padding=0),
            QuantizedConv2d(96, 256, kernel_size=5, stride=1, padding=2, bias=True, bit=bit),
            QuantizedMaxPool2d(kernel_size=3, stride=2, padding=0),
            QuantizedConv2d(256, 384, kernel_size=3, stride=1, padding=1, bias=True, bit=bit),
            QuantizedConv2d(384, 384, kernel_size=3, stride=1, padding=1, bias=True, bit=bit),
            QuantizedConv2d(384, 256, kernel_size=3, stride=1, padding=1, bias=True, bit=bit),
            QuantizedMaxPool2d(kernel_size=3, stride=2, padding=0),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            QuantizedLinear(256, 4096, bit=bit),
            QuantizedLinear(4096, 4096, bit=bit),
            QuantizedLinear(4096, num_classes, bit=bit),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.quantize_input(x)
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def quantize_input(self, x):
        x = torch.round(x.div(self.scale).add(self.zero_point))
        return x


def quantized_alexnet(bit: int = 32, **kwargs: Any) -> QuantizedAlexNet:
    return QuantizedAlexNet(bit=bit, **kwargs)


def quantized_alexnet_small(bit: int = 32, **kwargs: Any) -> QuantizedAlexNetSmall:
    return QuantizedAlexNetSmall(bit=bit, **kwargs)


def quantize_alexnet(fp_model, int_model):
    """
        Copy pre model's params & set fused layers.
        Use fused architecture, but not really fused (use CONV & BN seperately)
    """
    quantize(fp_model.features[0], int_model.features[0])
    int_model.features[1].zero_point = torch.nn.Parameter(int_model.features[0].z3, requires_grad=False)
    quantize(fp_model.features[2], int_model.features[2])
    int_model.features[3].zero_point = torch.nn.Parameter(int_model.features[2].z3, requires_grad=False)
    quantize(fp_model.features[4], int_model.features[4])
    quantize(fp_model.features[5], int_model.features[5])
    quantize(fp_model.features[6], int_model.features[6])
    int_model.features[7].zero_point = torch.nn.Parameter(int_model.features[6].z3, requires_grad=False)
    quantize(fp_model.classifier[0], int_model.classifier[0])
    quantize(fp_model.classifier[1], int_model.classifier[1])
    quantize(fp_model.classifier[2], int_model.classifier[2])
    return int_model

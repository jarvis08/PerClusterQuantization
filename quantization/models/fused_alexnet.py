import torch
import torch.nn as nn
from typing import Any
from ..layers.fused_conv import *
from ..layers.fused_linear import *
from ..quantization_utils import *


class FusedAlexNet(nn.Module):
    def __init__(self, dataset: str = 'imagenet', num_classes: int = 1000, smooth: float = 0.995, bit: int = 32) -> None:
        super(FusedAlexNet, self).__init__()

        self.bit = bit
        self.q_max = 2 ** self.bit - 1
        self.in_range = nn.Parameter(torch.zeros(2), requires_grad=False)
        self.scale = nn.Parameter(torch.tensor(0, dtype=torch.float32), requires_grad=False)
        self.zero_point = nn.Parameter(torch.tensor(0, dtype=torch.int32), requires_grad=False)
        self.ema_init = False
        self.smooth = smooth

        self.features = nn.Sequential(
            FusedConv2d(3, 96, kernel_size=5, stride=1, padding=2, bias=True, bit=bit, smooth=smooth, bn=False, relu=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
            FusedConv2d(96, 256, kernel_size=5, stride=1, padding=2, bias=True, bit=bit, smooth=smooth, bn=False, relu=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
            FusedConv2d(256, 384, kernel_size=3, stride=1, padding=1, bias=True, bit=bit, smooth=smooth, bn=False, relu=True),
            FusedConv2d(384, 384, kernel_size=3, stride=1, padding=1, bias=True, bit=bit, smooth=smooth, bn=False, relu=True),
            FusedConv2d(384, 256, kernel_size=3, stride=1, padding=1, bias=True, bit=bit, smooth=smooth, bn=False, relu=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            FusedLinear(256, 4096, smooth=smooth, bit=bit, relu=True),
            # FusedLinear(256 * 3 * 3, 4096, smooth=smooth, bit=bit, relu=True),
            FusedLinear(4096, 4096, smooth=smooth, bit=bit, relu=True),
            FusedLinear(4096, num_classes, smooth=smooth, bit=bit, relu=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            if self.ema_init:
                self.ema(x)
                self.set_qparams()
                x = self.fake_quantize_input(x)
            else:
                self.in_range[0] = torch.min(x).item()
                self.in_range[1] = torch.max(x).item()
                self.ema_init = True
        else:
            x = self.fake_quantize_input(x)

        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def ema(self, x):
        _min = torch.min(x).item()
        _max = torch.max(x).item()
        self.in_range[0] = self.in_range[0] * self.smooth + _min * (1 - self.smooth)
        self.in_range[1] = self.in_range[1] * self.smooth + _max * (1 - self.smooth)

    def set_qparams(self):
        self.scale, self.zero_point = calc_qprams(self.in_range[0], self.in_range[1], self.q_max)

    def fake_quantize_input(self, x):
        x = torch.round(x.div(self.scale).add(self.zero_point)).sub(self.zero_point).mul(self.scale)
        return x


def create_fused_alexnet(dataset: str = 'imagenet', num_classes: int = 1000, smooth: float = 0.995, bit: int = 32, **kwargs: Any) -> FusedAlexNet:
    model = FusedAlexNet(dataset=dataset, num_classes=num_classes, smooth=smooth, bit=bit, **kwargs)
    return model


def set_fused_alexnet_params(fused, pre):
    """
        Copy pre model's params & set fused layers.
        Use fused architecture, but not really fused (use CONV & BN seperately)
    """
    print(pre.features[0])
    print(pre.features[3])
    print(pre.features[6])
    print(pre.features[8])
    print(pre.features[10])
    fused.features[0].copy_from_pretrained(pre.features[0], False)
    fused.features[2].copy_from_pretrained(pre.features[3], False)
    fused.features[4].copy_from_pretrained(pre.features[6], False)
    fused.features[5].copy_from_pretrained(pre.features[8], False)
    fused.features[6].copy_from_pretrained(pre.features[10], False)

    print(pre.classifier[0])
    print(pre.classifier[2])
    print(pre.classifier[4])
    fused.classifier[0].fc.weight = torch.nn.Parameter(pre.classifier[0].weight)
    fused.classifier[0].fc.bias = torch.nn.Parameter(pre.classifier[0].bias)
    fused.classifier[1].fc.weight = torch.nn.Parameter(pre.classifier[2].weight)
    fused.classifier[1].fc.bias = torch.nn.Parameter(pre.classifier[2].bias)
    fused.classifier[2].fc.weight = torch.nn.Parameter(pre.classifier[4].weight)
    fused.classifier[2].fc.bias = torch.nn.Parameter(pre.classifier[4].bias)
    return fused

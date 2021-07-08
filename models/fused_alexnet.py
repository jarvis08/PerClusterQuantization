import torch
import torch.nn as nn
from typing import Any
from .layers.conv2d import *
from .layers.linear import *
from .quantization_utils import *


class FusedAlexNet(nn.Module):
    def __init__(self, num_classes: int = 1000, smooth: float = 0.995, bit: int = 32) -> None:
        super(FusedAlexNet, self).__init__()
        self.bit = bit
        self.q_max = 2 ** self.bit - 1
        self.in_range = nn.Parameter(torch.zeros(2), requires_grad=False)
        self.ema_init = False
        self.smooth = smooth

        self.features = nn.Sequential(
            FusedConv2d(3, 64, kernel_size=11, stride=4, padding=2, bias=True, bit=bit, smooth=smooth, bn=False, relu=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
            FusedConv2d(64, 192, kernel_size=5, stride=1, padding=2, bias=True, bit=bit, smooth=smooth, bn=False, relu=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
            FusedConv2d(192, 384, kernel_size=3, stride=1, padding=1, bias=True, bit=bit, smooth=smooth, bn=False, relu=True),
            FusedConv2d(384, 256, kernel_size=3, stride=1, padding=1, bias=True, bit=bit, smooth=smooth, bn=False, relu=True),
            FusedConv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True, bit=bit, smooth=smooth, bn=False, relu=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            FusedLinear(256 * 6 * 6, 4096, smooth=smooth, bit=bit, relu=True),
            FusedLinear(4096, 4096, smooth=smooth, bit=bit, relu=True),
            FusedLinear(4096, num_classes, smooth=smooth, bit=bit, relu=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            if self.ema_init:
                self.in_range[0], self.in_range[1] = ema(x, self.in_range, self.smooth)
                s, z = calc_qparams(self.in_range[0], self.in_range[1], self.q_max)
                x = fake_quantize(x, s, z)
            else:
                self.in_range[0] = torch.min(x).item()
                self.in_range[1] = torch.max(x).item()
                self.ema_init = True

        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def set_quantization_params(self):
        self.scale, self.zero_point = calc_qparams(self.in_range[0], self.in_range[1], self.q_max)
        prev_s, prev_z = self.features[0].set_qparams(self.scale, self.zero_point)
        prev_s, prev_z = self.features[2].set_qparams(prev_s, prev_z)
        prev_s, prev_z = self.features[4].set_qparams(prev_s, prev_z)
        prev_s, prev_z = self.features[5].set_qparams(prev_s, prev_z)
        prev_s, prev_z = self.features[6].set_qparams(prev_s, prev_z)
        prev_s, prev_z = self.classifier[0].set_qparams(prev_s, prev_z)
        prev_s, prev_z = self.classifier[1].set_qparams(prev_s, prev_z)
        _, _ = self.classifier[2].set_qparams(prev_s, prev_z)


class FusedAlexNetSmall(nn.Module):
    def __init__(self, num_classes: int = 10, smooth: float = 0.995, bit: int = 32) -> None:
        super(FusedAlexNetSmall, self).__init__()
        self.bit = bit
        self.q_max = 2 ** self.bit - 1
        self.in_range = nn.Parameter(torch.zeros(2), requires_grad=False)
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
            FusedLinear(4096, 4096, smooth=smooth, bit=bit, relu=True),
            FusedLinear(4096, num_classes, smooth=smooth, bit=bit, relu=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            if self.ema_init:
                self.in_range[0], self.in_range[1] = ema(x, self.in_range, self.smooth)
                s, z = calc_qparams(self.in_range[0], self.in_range[1], self.q_max)
                x = fake_quantize(x, s, z)
            else:
                self.in_range[0] = torch.min(x).item()
                self.in_range[1] = torch.max(x).item()
                self.ema_init = True

        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def set_quantization_params(self):
        self.scale, self.zero_point = calc_qparams(self.in_range[0], self.in_range[1], self.q_max)
        prev_s, prev_z = self.features[0].set_qparams(self.scale, self.zero_point)
        prev_s, prev_z = self.features[2].set_qparams(prev_s, prev_z)
        prev_s, prev_z = self.features[4].set_qparams(prev_s, prev_z)
        prev_s, prev_z = self.features[5].set_qparams(prev_s, prev_z)
        prev_s, prev_z = self.features[6].set_qparams(prev_s, prev_z)
        prev_s, prev_z = self.classifier[0].set_qparams(prev_s, prev_z)
        prev_s, prev_z = self.classifier[1].set_qparams(prev_s, prev_z)
        _, _ = self.classifier[2].set_qparams(prev_s, prev_z)


def fused_alexnet(smooth: float = 0.999, bit: int = 32, **kwargs: Any) -> FusedAlexNet:
    return FusedAlexNet(smooth=smooth, bit=bit, **kwargs)


def fused_alexnet_small(smooth: float = 0.999, bit: int = 32, **kwargs: Any) -> FusedAlexNetSmall:
    return FusedAlexNetSmall(smooth=smooth, bit=bit, **kwargs)


def set_fused_alexnet(fused, pre):
    """
        Copy pre model's params & set fused layers.
        Use fused architecture, but not really fused (use CONV & BN seperately)
    """
    fused.features[0].copy_from_pretrained(pre.features[0], False)
    fused.features[2].copy_from_pretrained(pre.features[3], False)
    fused.features[4].copy_from_pretrained(pre.features[6], False)
    fused.features[5].copy_from_pretrained(pre.features[8], False)
    fused.features[6].copy_from_pretrained(pre.features[10], False)

    fused.classifier[0].fc.weight = torch.nn.Parameter(pre.classifier[1].weight)
    fused.classifier[0].fc.bias = torch.nn.Parameter(pre.classifier[1].bias)
    fused.classifier[1].fc.weight = torch.nn.Parameter(pre.classifier[4].weight)
    fused.classifier[1].fc.bias = torch.nn.Parameter(pre.classifier[4].bias)
    fused.classifier[2].fc.weight = torch.nn.Parameter(pre.classifier[6].weight)
    fused.classifier[2].fc.bias = torch.nn.Parameter(pre.classifier[6].bias)

    return fused

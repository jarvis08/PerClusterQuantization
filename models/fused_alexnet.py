import torch
import torch.nn as nn
from typing import Any
from .layers.conv2d import *
from .layers.linear import *
from .quantization_utils import *


class FusedAlexNet(nn.Module):
    def __init__(self, arg_dict: dict, num_classes: int = 1000) -> None:
        super(FusedAlexNet, self).__init__()
        self.bit, self.smooth, self.runtime_helper = itemgetter('bit', 'smooth', 'runtime_helper')(arg_dict)
        self.q_max = 2 ** self.bit - 1
        self.in_range = nn.Parameter(torch.zeros(2), requires_grad=False)

        self.apply_ema = False

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.conv1 = FusedConv2d(3, 64, kernel_size=11, stride=4, padding=2, bias=True,
                                 activation=nn.ReLU, arg_dict=arg_dict)
        self.conv2 = FusedConv2d(64, 192, kernel_size=5, stride=1, padding=2, bias=True,
                                 activation=nn.ReLU, arg_dict=arg_dict)
        self.conv3 = FusedConv2d(192, 384, kernel_size=3, stride=1, padding=1, bias=True,
                                 activation=nn.ReLU, arg_dict=arg_dict)
        self.conv4 = FusedConv2d(384, 256, kernel_size=3, stride=1, padding=1, bias=True,
                                 activation=nn.ReLU, arg_dict=arg_dict)
        self.conv5 = FusedConv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True,
                                 activation=nn.ReLU, arg_dict=arg_dict)
        self.fc1 = FusedLinear(256 * 6 * 6, 4096, activation=nn.ReLU, arg_dict=arg_dict)
        self.fc2 = FusedLinear(4096, 4096, bias=True, activation=nn.ReLU, arg_dict=arg_dict)
        self.fc3 = FusedLinear(4096, num_classes, arg_dict=arg_dict)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            if self.apply_ema:
                self.in_range[0], self.in_range[1] = ema(x, self.in_range, self.smooth)
                if self.runtime_helper.apply_fake_quantization:
                    s, z = calc_qparams(self.in_range[0], self.in_range[1], self.q_max)
                    x = fake_quantize(x, s, z, self.q_max)
            else:
                self.in_range[0] = torch.min(x).item()
                self.in_range[1] = torch.max(x).item()
                self.apply_ema = True

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

    def set_quantization_params(self):
        self.scale, self.zero_point = calc_qparams(self.in_range[0], self.in_range[1], self.q_max)
        prev_s, prev_z = self.conv1.set_qparams(self.scale, self.zero_point)
        prev_s, prev_z = self.conv2.set_qparams(prev_s, prev_z)
        prev_s, prev_z = self.conv3.set_qparams(prev_s, prev_z)
        prev_s, prev_z = self.conv4.set_qparams(prev_s, prev_z)
        prev_s, prev_z = self.conv5.set_qparams(prev_s, prev_z)
        prev_s, prev_z = self.fc1.set_qparams(prev_s, prev_z)
        prev_s, prev_z = self.fc2.set_qparams(prev_s, prev_z)
        _, _ = self.fc3.set_qparams(prev_s, prev_z)


class FusedAlexNetSmall(nn.Module):
    def __init__(self, arg_dict: dict, num_classes: int = 10) -> None:
        super(FusedAlexNetSmall, self).__init__()
        self.bit, self.smooth, self.runtime_helper = itemgetter('bit', 'smooth', 'runtime_helper')(arg_dict)
        self.q_max = 2 ** self.bit - 1
        self.in_range = nn.Parameter(torch.zeros(2), requires_grad=False)

        self.apply_ema = False
        self.apply_fake_quantization = False

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv1 = FusedConv2d(3, 96, kernel_size=5, stride=1, padding=2, bias=True,
                                 activation=nn.ReLU, arg_dict=arg_dict)
        self.conv2 = FusedConv2d(96, 256, kernel_size=5, stride=1, padding=2, bias=True,
                                 activation=nn.ReLU, arg_dict=arg_dict)
        self.conv3 = FusedConv2d(256, 384, kernel_size=3, stride=1, padding=1, bias=True,
                                 activation=nn.ReLU, arg_dict=arg_dict)
        self.conv4 = FusedConv2d(384, 384, kernel_size=3, stride=1, padding=1, bias=True,
                                 activation=nn.ReLU, arg_dict=arg_dict)
        self.conv5 = FusedConv2d(384, 256, kernel_size=3, stride=1, padding=1, bias=True,
                                 activation=nn.ReLU, arg_dict=arg_dict)
        self.fc1 = FusedLinear(256, 4096, bias=True, activation=nn.ReLU, arg_dict=arg_dict)
        self.fc2 = FusedLinear(4096, 4096, bias=True, activation=nn.ReLU, arg_dict=arg_dict)
        self.fc3 = FusedLinear(4096, num_classes, bias=True, arg_dict=arg_dict)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fake_input = x.clone().detach()
        if self.training:
            with torch.no_grad():
                if self.apply_ema:
                    self.in_range[0], self.in_range[1] = ema(x, self.in_range, self.smooth)
                    if self.runtime_helper.apply_fake_quantization:
                        s, z = calc_qparams(self.in_range[0], self.in_range[1], self.q_max)
                        fake_input = fake_quantize(x, s, z, self.q_max, use_ste=False)
                else:
                    self.in_range[0] = torch.min(x).item()
                    self.in_range[1] = torch.max(x).item()
                    self.apply_ema = True

        #x = self.conv1(x)
        x = self.conv1(STE.apply(x, fake_input))
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

    def set_quantization_params(self):
        self.scale, self.zero_point = calc_qparams(self.in_range[0], self.in_range[1], self.q_max)
        prev_s, prev_z = self.conv1.set_qparams(self.scale, self.zero_point)
        prev_s, prev_z = self.conv2.set_qparams(prev_s, prev_z)
        prev_s, prev_z = self.conv3.set_qparams(prev_s, prev_z)
        prev_s, prev_z = self.conv4.set_qparams(prev_s, prev_z)
        prev_s, prev_z = self.conv5.set_qparams(prev_s, prev_z)
        prev_s, prev_z = self.fc1.set_qparams(prev_s, prev_z)
        prev_s, prev_z = self.fc2.set_qparams(prev_s, prev_z)
        _, _ = self.fc3.set_qparams(prev_s, prev_z)


def fused_alexnet(arg_dict: dict, **kwargs: Any) -> FusedAlexNet:
    return FusedAlexNet(arg_dict, **kwargs)


def fused_alexnet_small(arg_dict: dict, **kwargs: Any) -> FusedAlexNetSmall:
    return FusedAlexNetSmall(arg_dict, **kwargs)


def set_fused_alexnet(fused, pre):
    """
        Copy pre model's params & set fused layers.
        Use fused architecture, but not really fused (use CONV & BN seperately)
    """
    fused.conv1 = copy_from_pretrained(fused.conv1, pre.features[0])
    fused.conv2 = copy_from_pretrained(fused.conv2, pre.features[3])
    fused.conv3 = copy_from_pretrained(fused.conv3, pre.features[6])
    fused.conv4 = copy_from_pretrained(fused.conv4, pre.features[8])
    fused.conv5 = copy_from_pretrained(fused.conv5, pre.features[10])
    fused.fc1 = copy_from_pretrained(fused.fc1, pre.classifier[1])
    fused.fc2 = copy_from_pretrained(fused.fc2, pre.classifier[4])
    fused.fc3 = copy_from_pretrained(fused.fc3, pre.classifier[6])
    return fused

import torch
import torch.nn as nn
from typing import Any
from .layers.conv2d import *
from .layers.linear import *
from .quant_noise import _quant_noise
from .quantization_utils import *


class PCQAlexNet(nn.Module):
    batch_cluster = None

    def __init__(self, arg_dict: dict, num_classes: int = 1000) -> None:
        super(PCQAlexNet, self).__init__()
        arg_bit, self.smooth, self.num_clusters, self.runtime_helper, self.quant_noise, self.qn_prob\
            = itemgetter('bit', 'smooth', 'cluster', 'runtime_helper', 'quant_noise', 'qn_prob')(arg_dict)

        self.bit = torch.nn.Parameter(torch.tensor(0, dtype=torch.int8), requires_grad=False)
        self.bit.data = torch.tensor(arg_bit, dtype=torch.int8)
        self.in_range = nn.Parameter(torch.zeros(self.num_clusters, 2), requires_grad=False)

        self.apply_ema = nn.Parameter(torch.zeros(self.num_clusters, dtype=torch.bool), requires_grad=False)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.conv1 = PCQConv2d(3, 64, kernel_size=11, stride=4, padding=2, bias=True,
                               activation=nn.ReLU, arg_dict=arg_dict)
        self.conv2 = PCQConv2d(64, 192, kernel_size=5, stride=1, padding=2, bias=True,
                               activation=nn.ReLU, arg_dict=arg_dict)
        self.conv3 = PCQConv2d(192, 384, kernel_size=3, stride=1, padding=1, bias=True,
                               activation=nn.ReLU, arg_dict=arg_dict)
        self.conv4 = PCQConv2d(384, 256, kernel_size=3, stride=1, padding=1, bias=True,
                               activation=nn.ReLU, arg_dict=arg_dict)
        self.conv5 = PCQConv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True,
                               activation=nn.ReLU, arg_dict=arg_dict)
        self.fc1 = PCQLinear(256 * 6 * 6, 4096, activation=nn.ReLU, arg_dict=arg_dict)
        self.fc2 = PCQLinear(4096, 4096, activation=nn.ReLU, arg_dict=arg_dict)
        self.fc3 = PCQLinear(4096, num_classes, arg_dict=arg_dict)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            self._update_input_ranges(x)
            if self.runtime_helper.apply_fake_quantization:
                x = self._fake_quantize_input(x)

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

    def _fake_quantize_input(self, x):
        s, z = calc_qparams_per_cluster(self.in_range, self.bit)
        return fake_quantize_per_cluster_4d(x, s, z, self.bit, self.runtime_helper.qat_batch_cluster)

    @torch.no_grad()
    def _update_input_ranges(self, x):
        cluster = self.runtime_helper.qat_batch_cluster
        data = x.view(x.size(0), -1).clone().detach()
        _min = data.min(dim=1).values.mean()
        _max = data.max(dim=1).values.mean()
        if self.apply_ema[cluster]:
            self.in_range[cluster][0] = self.in_range[cluster][0] * self.smooth + _min * (1 - self.smooth)
            self.in_range[cluster][1] = self.in_range[cluster][1] * self.smooth + _max * (1 - self.smooth)
        else:
            self.in_range[cluster][0], self.in_range[cluster][1] = _min, _max
            self.apply_ema[cluster] = True

    def set_quantization_params(self):
        self.scale, self.zero_point = calc_qparams(self.in_range[0], self.in_range[1], self.bit)
        prev_s, prev_z = self.conv1.set_qparams(self.scale, self.zero_point)
        prev_s, prev_z = self.conv2.set_qparams(prev_s, prev_z)
        prev_s, prev_z = self.conv3.set_qparams(prev_s, prev_z)
        prev_s, prev_z = self.conv4.set_qparams(prev_s, prev_z)
        prev_s, prev_z = self.conv5.set_qparams(prev_s, prev_z)
        prev_s, prev_z = self.fc1.set_qparams(prev_s, prev_z)
        prev_s, prev_z = self.fc2.set_qparams(prev_s, prev_z)
        _, _ = self.fc3.set_qparams(prev_s, prev_z)


class PCQAlexNetSmall(nn.Module):
    batch_cluster = None

    def __init__(self, arg_dict: dict, num_classes: int = 10) -> None:
        super(PCQAlexNetSmall, self).__init__()
        target_bit, bit_first, bit_classifier, self.smooth, self.num_clusters, self.runtime_helper \
            = itemgetter('bit', 'bit_first', 'bit_classifier', 'smooth', 'cluster', 'runtime_helper')(arg_dict)
        self.target_bit = torch.nn.Parameter(torch.tensor(target_bit, dtype=torch.int8), requires_grad=False)
        self.in_bit = torch.nn.Parameter(torch.tensor(bit_first, dtype=torch.int8), requires_grad=False)
        self.in_range = nn.Parameter(torch.zeros((self.num_clusters, 2)), requires_grad=False)

        self.apply_ema = nn.Parameter(torch.zeros(self.num_clusters, dtype=torch.bool), requires_grad=False)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv1 = PCQConv2d(3, 96, kernel_size=5, stride=1, padding=2, bias=True, activation=nn.ReLU,
                               w_bit=bit_first, arg_dict=arg_dict)
        self.conv2 = PCQConv2d(96, 256, kernel_size=5, stride=1, padding=2, bias=True,
                               activation=nn.ReLU, arg_dict=arg_dict)
        self.conv3 = PCQConv2d(256, 384, kernel_size=3, stride=1, padding=1, bias=True,
                               activation=nn.ReLU, arg_dict=arg_dict)
        self.conv4 = PCQConv2d(384, 384, kernel_size=3, stride=1, padding=1, bias=True,
                               activation=nn.ReLU, arg_dict=arg_dict)
        self.conv5 = PCQConv2d(384, 256, kernel_size=3, stride=1, padding=1, bias=True,
                               activation=nn.ReLU, arg_dict=arg_dict)
        self.fc1 = PCQLinear(256, 4096, bias=True, activation=nn.ReLU, arg_dict=arg_dict)
        self.fc2 = PCQLinear(4096, 4096, bias=True, activation=nn.ReLU, a_bit=bit_classifier, arg_dict=arg_dict)
        self.fc3 = PCQLinear(4096, num_classes, bias=True, w_bit=bit_classifier, a_bit=bit_classifier, arg_dict=arg_dict)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            self._update_input_ranges(x)
            if self.runtime_helper.apply_fake_quantization:
                x = self._fake_quantize_input(x)

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


    @torch.no_grad()
    def _update_input_ranges(self, x):
        cluster = self.runtime_helper.qat_batch_cluster
        if self.runtime_helper.undo_gema:
            _min = x.min().item()
            _max = x.max().item()
        else:
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
        cluster = self.runtime_helper.qat_batch_cluster
        s, z = calc_qparams(self.in_range[cluster][0], self.in_range[cluster][1], self.in_bit)
        return fake_quantize(x, s, z, self.in_bit)


    def set_quantization_params(self):
        zero = self.runtime_helper.fzero
        self.scale, self.zero_point = calc_qparams_per_cluster(self.in_range, self.in_bit, zero)
        prev_s, prev_z = self.conv1.set_qparams(self.scale, self.zero_point)
        prev_s, prev_z = self.conv2.set_qparams(prev_s, prev_z)
        prev_s, prev_z = self.conv3.set_qparams(prev_s, prev_z)
        prev_s, prev_z = self.conv4.set_qparams(prev_s, prev_z)
        prev_s, prev_z = self.conv5.set_qparams(prev_s, prev_z)
        prev_s, prev_z = self.fc1.set_qparams(prev_s, prev_z)
        prev_s, prev_z = self.fc2.set_qparams(prev_s, prev_z)
        _, _ = self.fc3.set_qparams(prev_s, prev_z)


def pcq_alexnet(arg_dict: dict, **kwargs: Any) -> PCQAlexNet:
    return PCQAlexNet(arg_dict, **kwargs)


def pcq_alexnet_small(arg_dict: dict, num_classes=10, **kwargs: Any) -> PCQAlexNetSmall:
    return PCQAlexNetSmall(arg_dict, num_classes=num_classes, **kwargs)


def set_pcq_alexnet(pcq, pre):
    pcq.conv1 = copy_from_pretrained(pcq.conv1, pre.features[0])
    pcq.conv2 = copy_from_pretrained(pcq.conv2, pre.features[3])
    pcq.conv3 = copy_from_pretrained(pcq.conv3, pre.features[6])
    pcq.conv4 = copy_from_pretrained(pcq.conv4, pre.features[8])
    pcq.conv5 = copy_from_pretrained(pcq.conv5, pre.features[10])
    pcq.fc1 = copy_from_pretrained(pcq.fc1, pre.classifier[1])
    pcq.fc2 = copy_from_pretrained(pcq.fc2, pre.classifier[4])
    pcq.fc3 = copy_from_pretrained(pcq.fc3, pre.classifier[6])
    return pcq

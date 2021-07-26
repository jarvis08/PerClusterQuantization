import torch
import torch.nn as nn
from typing import Any
from .layers.conv2d import *
from .layers.linear import *
from .layers.maxpool2d import *
from .quantization_utils import *


class QuantizedAlexNet(nn.Module):
    batch_cluster = None

    def __init__(self, num_classes: int = 1000, bit: int = 8, num_clusters: int = 1) -> None:
        super(QuantizedAlexNet, self).__init__()
        self.bit = bit
        self.q_max = 2 ** bit - 1
        self.num_clusters = num_clusters
        t_init = list(range(num_clusters)) if num_clusters > 1 else 0
        self.scale = nn.Parameter(torch.tensor(t_init, dtype=torch.float32), requires_grad=False)
        self.zero_point = nn.Parameter(torch.tensor(t_init, dtype=torch.int32), requires_grad=False)

        self.maxpool = QuantizedMaxPool2d(kernel_size=3, stride=2, padding=0)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.conv1 = QuantizedConv2d(3, 64, kernel_size=11, stride=4, padding=2, bias=True, bit=bit)
        self.conv2 = QuantizedConv2d(64, 192, kernel_size=5, stride=1, padding=2, bias=True, bit=bit)
        self.conv3 = QuantizedConv2d(192, 384, kernel_size=3, stride=1, padding=1, bias=True, bit=bit)
        self.conv4 = QuantizedConv2d(384, 256, kernel_size=3, stride=1, padding=1, bias=True, bit=bit)
        self.conv5 = QuantizedConv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True, bit=bit)
        self.fc1 = QuantizedLinear(256 * 6 * 6, 4096, bit=bit)
        self.fc2 = QuantizedLinear(4096, 4096, bit=bit)
        self.fc3 = QuantizedLinear(4096, num_classes, bit=bit)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.batch_cluster is not None:
            done = 0
            for i in range(self.batch_cluster.shape[0]):
                c = self.batch_cluster[i][0].item()
                n = self.batch_cluster[i][1].item()
                x[done:done + n].copy_(quantize_matrix(x[done:done + n].detach(), self.scale[c], self.zero_point[c], self.q_max))
                done += n
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

    @classmethod
    def set_cluster_information_of_batch(cls, info):
        cls.batch_cluster = info
        QuantizedConv2d.batch_cluster = info
        QuantizedLinear.batch_cluster = info
        QuantizedMaxPool2d.batch_cluster = info


class QuantizedAlexNetSmall(nn.Module):
    batch_cluster = None

    def __init__(self, num_classes: int = 10, bit: int = 32, num_clusters: int = 1) -> None:
        super(QuantizedAlexNetSmall, self).__init__()
        self.bit = bit
        self.q_max = 2 ** bit - 1
        self.num_clusters = num_clusters
        t_init = list(range(num_clusters)) if num_clusters > 1 else 0
        self.scale = nn.Parameter(torch.tensor(t_init, dtype=torch.float32), requires_grad=False)
        self.zero_point = nn.Parameter(torch.tensor(t_init, dtype=torch.int32), requires_grad=False)

        self.maxpool = QuantizedMaxPool2d(kernel_size=3, stride=2, padding=0)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv1 = QuantizedConv2d(3, 96, kernel_size=5, stride=1, padding=2, bias=False, bit=bit, num_clusters=num_clusters)
        self.conv2 = QuantizedConv2d(96, 256, kernel_size=5, stride=1, padding=2, bias=False, bit=bit, num_clusters=num_clusters)
        self.conv3 = QuantizedConv2d(256, 384, kernel_size=3, stride=1, padding=1, bias=False, bit=bit, num_clusters=num_clusters)
        self.conv4 = QuantizedConv2d(384, 384, kernel_size=3, stride=1, padding=1, bias=False, bit=bit, num_clusters=num_clusters)
        self.conv5 = QuantizedConv2d(384, 256, kernel_size=3, stride=1, padding=1, bias=False, bit=bit, num_clusters=num_clusters)
        self.fc1 = QuantizedLinear(256, 4096, bit=bit, num_clusters=num_clusters)
        self.fc2 = QuantizedLinear(4096, 4096, bit=bit, num_clusters=num_clusters)
        self.fc3 = QuantizedLinear(4096, num_classes, bit=bit, num_clusters=num_clusters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.batch_cluster is not None:
            done = 0
            for i in range(self.batch_cluster.shape[0]):
                c = self.batch_cluster[i][0].item()
                n = self.batch_cluster[i][1].item()
                x[done:done + n] = quantize_matrix(x[done:done + n], self.scale[c], self.zero_point[c], self.q_max)
                done += n
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

    @classmethod
    def set_cluster_information_of_batch(cls, info):
        cls.batch_cluster = info
        QuantizedConv2d.batch_cluster = info
        QuantizedLinear.batch_cluster = info
        QuantizedMaxPool2d.batch_cluster = info


def quantized_alexnet(bit: int = 32, num_clusters: int = 1, **kwargs: Any) -> QuantizedAlexNet:
    return QuantizedAlexNet(bit=bit, num_clusters=num_clusters, **kwargs)


def quantized_alexnet_small(bit: int = 32, num_clusters: int = 1, **kwargs: Any) -> QuantizedAlexNetSmall:
    return QuantizedAlexNetSmall(bit=bit, num_clusters=num_clusters, **kwargs)


def quantize_alexnet(fp_model, int_model):
    """
        Copy pre model's params & set fused layers.
        Use fused architecture, but not really fused (use CONV & BN seperately)
    """
    int_model.scale = torch.nn.Parameter(fp_model.scale, requires_grad=False)
    int_model.zero_point = torch.nn.Parameter(fp_model.zero_point, requires_grad=False)
    int_model.conv1 = quantize(fp_model.conv1, int_model.conv1)
    int_model.conv2 = quantize(fp_model.conv2, int_model.conv2)
    int_model.conv3 = quantize(fp_model.conv3, int_model.conv3)
    int_model.conv4 = quantize(fp_model.conv4, int_model.conv4)
    int_model.conv5 = quantize(fp_model.conv5, int_model.conv5)
    int_model.fc1 = quantize(fp_model.fc1, int_model.fc1)
    int_model.fc2 = quantize(fp_model.fc2, int_model.fc2)
    int_model.fc3 = quantize(fp_model.fc3, int_model.fc3)
    return int_model

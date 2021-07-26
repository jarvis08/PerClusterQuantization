import torch
import torch.nn as nn
from typing import Any
from .layers.conv2d import *
from .layers.linear import *
from .quantization_utils import *


class PCQAlexNet(nn.Module):
    batch_cluster = None

    def __init__(self, num_classes: int = 1000, smooth: float = 0.995, bit: int = 32, num_clusters: int = 10, quant_noise=False, q_prob=0.1) -> None:
        super(PCQAlexNet, self).__init__()
        self.bit = bit
        self.q_max = 2 ** self.bit - 1
        self.in_range = nn.Parameter(torch.zeros(num_clusters, 2), requires_grad=False)
        self.flag_ema_init = np.zeros(num_clusters, dtype=bool)
        self.flag_fake_quantization = False
        self.smooth = smooth

        self.num_clusters = num_clusters
        self.quant_noise = quant_noise
        self.q_prob = q_prob

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.conv1 = PCQConv2d(3, 64, kernel_size=11, stride=4, padding=2, bias=True, bit=bit, smooth=smooth, num_clusters=num_clusters, activation=nn.ReLU6, quant_noise=self.quant_noise, q_prob=self.q_prob)
        self.conv2 = PCQConv2d(64, 192, kernel_size=5, stride=1, padding=2, bias=True, bit=bit, smooth=smooth, num_clusters=num_clusters, activation=nn.ReLU6, quant_noise=self.quant_noise, q_prob=self.q_prob)
        self.conv3 = PCQConv2d(192, 384, kernel_size=3, stride=1, padding=1, bias=True, bit=bit, smooth=smooth, num_clusters=num_clusters, activation=nn.ReLU6, quant_noise=self.quant_noise, q_prob=self.q_prob)
        self.conv4 = PCQConv2d(384, 256, kernel_size=3, stride=1, padding=1, bias=True, bit=bit, smooth=smooth, num_clusters=num_clusters, activation=nn.ReLU6, quant_noise=self.quant_noise, q_prob=self.q_prob)
        self.conv5 = PCQConv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True, bit=bit, smooth=smooth, num_clusters=num_clusters, activation=nn.ReLU6, quant_noise=self.quant_noise, q_prob=self.q_prob)
        self.fc1 = PCQLinear(256 * 6 * 6, 4096, smooth=smooth, bit=bit, num_clusters=num_clusters, activation=nn.ReLU6, quant_noise=self.quant_noise, q_prob=self.q_prob)
        self.fc2 = PCQLinear(4096, 4096, smooth=smooth, bit=bit, num_clusters=num_clusters, activation=nn.ReLU6, quant_noise=self.quant_noise, q_prob=self.q_prob)
        self.fc3 = PCQLinear(4096, num_classes, smooth=smooth, bit=bit, num_clusters=num_clusters, quant_noise=self.quant_noise, q_prob=self.q_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            done = 0
            for i in range(self.batch_cluster.shape[0]):
                c = self.batch_cluster[i][0]
                n = self.batch_cluster[i][1]
                if c == -1:
                    break
                if self.flag_ema_init[c]:
                    self.in_range[c][0], self.in_range[c][1] = ema(x[done:done + n], self.in_range[c], self.smooth)
                    if self.flag_fake_quantization:
                        s, z = calc_qparams(self.in_range[c][0], self.in_range[c][1], self.q_max)
                        x[done:done + n] = fake_quantize(x[done:done + n], s, z)
                else:
                    self.in_range[c][0] = torch.min(x).item()
                    self.in_range[c][1] = torch.max(x).item()
                    self.flag_ema_init[c] = True
                done += n

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
        PCQConv2d.batch_cluster = info
        PCQLinear.batch_cluster = info

    def start_fake_quantization(self):
        self.flag_fake_quantization = True
        self.conv1.flag_fake_quantization = True
        self.conv2.flag_fake_quantization = True
        self.conv3.flag_fake_quantization = True
        self.conv4.flag_fake_quantization = True
        self.conv5.flag_fake_quantization = True
        self.fc1.flag_fake_quantization = True
        self.fc2.flag_fake_quantization = True
        self.fc3.flag_fake_quantization = True

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


class PCQAlexNetSmall(nn.Module):
    batch_cluster = None

    def __init__(self, num_classes: int = 10, smooth: float = 0.995, bit: int = 32, num_clusters: int = 10, quant_noise=False, q_prob=0.1) -> None:
        super(PCQAlexNetSmall, self).__init__()
        self.bit = bit
        self.q_max = 2 ** self.bit - 1
        self.in_range = nn.Parameter(torch.zeros((num_clusters, 2)), requires_grad=False)
        self.flag_ema_init = np.zeros(num_clusters, dtype=bool)
        self.flag_fake_quantization = False
        self.smooth = smooth

        self.num_clusters = num_clusters
        self.quant_noise = quant_noise
        self.q_prob = q_prob

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv1 = PCQConv2d(3, 96, kernel_size=5, stride=1, padding=2, bias=True, bit=bit, num_clusters=num_clusters, smooth=smooth, activation=nn.ReLU6, quant_noise=quant_noise, q_prob=q_prob)
        self.conv2 = PCQConv2d(96, 256, kernel_size=5, stride=1, padding=2, bias=True, bit=bit, num_clusters=num_clusters, smooth=smooth, activation=nn.ReLU6, quant_noise=quant_noise, q_prob=q_prob)
        self.conv3 = PCQConv2d(256, 384, kernel_size=3, stride=1, padding=1, bias=True, bit=bit, num_clusters=num_clusters, smooth=smooth, activation=nn.ReLU6, quant_noise=quant_noise, q_prob=q_prob)
        self.conv4 = PCQConv2d(384, 384, kernel_size=3, stride=1, padding=1, bias=True, bit=bit, num_clusters=num_clusters, smooth=smooth, activation=nn.ReLU6, quant_noise=quant_noise, q_prob=q_prob)
        self.conv5 = PCQConv2d(384, 256, kernel_size=3, stride=1, padding=1, bias=True, bit=bit, num_clusters=num_clusters, smooth=smooth, activation=nn.ReLU6, quant_noise=quant_noise, q_prob=q_prob)
        self.fc1 = PCQLinear(256, 4096, smooth=smooth, bit=bit, num_clusters=num_clusters, activation=nn.ReLU6, quant_noise=quant_noise, q_prob=q_prob)
        self.fc2 = PCQLinear(4096, 4096, smooth=smooth, bit=bit, num_clusters=num_clusters, activation=nn.ReLU6, quant_noise=quant_noise, q_prob=q_prob)
        self.fc3 = PCQLinear(4096, num_classes, smooth=smooth, num_clusters=num_clusters, bit=bit, quant_noise=quant_noise, q_prob=q_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            done = 0
            for i in range(self.batch_cluster.shape[0]):
                c = self.batch_cluster[i][0].item()
                n = self.batch_cluster[i][1].item()
                if c == -1:
                    break
                if self.flag_ema_init[c]:
                    self.in_range[c][0], self.in_range[c][1] = ema(x[done:done + n], self.in_range[c], self.smooth)
                    if self.flag_fake_quantization:
                        s, z = calc_qparams(self.in_range[c][0], self.in_range[c][1], self.q_max)
                        x[done:done + n] = fake_quantize(x[done:done + n], s, z, self.q_max)
                else:
                    self.in_range[c][0] = torch.min(x[done:done + n]).item()
                    self.in_range[c][1] = torch.max(x[done:done + n]).item()
                    self.flag_ema_init[c] = True
                done += n

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
        PCQConv2d.batch_cluster = info
        PCQLinear.batch_cluster = info

    def start_fake_quantization(self):
        self.flag_fake_quantization = True
        self.conv1.flag_fake_quantization = True
        self.conv2.flag_fake_quantization = True
        self.conv3.flag_fake_quantization = True
        self.conv4.flag_fake_quantization = True
        self.conv5.flag_fake_quantization = True
        self.fc1.flag_fake_quantization = True
        self.fc2.flag_fake_quantization = True
        self.fc3.flag_fake_quantization = True

    def set_quantization_params(self):
        self.scale = nn.Parameter(torch.zeros(self.num_clusters, dtype=torch.float32), requires_grad=False)
        self.zero_point = nn.Parameter(torch.zeros(self.num_clusters, dtype=torch.int32), requires_grad=False)
        for c in range(self.num_clusters):
            self.scale[c], self.zero_point[c] = calc_qparams(self.in_range[c][0], self.in_range[c][1], self.q_max)
        prev_s, prev_z = self.conv1.set_qparams(self.scale, self.zero_point)
        prev_s, prev_z = self.conv2.set_qparams(prev_s, prev_z)
        prev_s, prev_z = self.conv3.set_qparams(prev_s, prev_z)
        prev_s, prev_z = self.conv4.set_qparams(prev_s, prev_z)
        prev_s, prev_z = self.conv5.set_qparams(prev_s, prev_z)
        prev_s, prev_z = self.fc1.set_qparams(prev_s, prev_z)
        prev_s, prev_z = self.fc2.set_qparams(prev_s, prev_z)
        _, _ = self.fc3.set_qparams(prev_s, prev_z)


def pcq_alexnet(smooth: float = 0.999, bit: int = 32, num_clusters: int = 10, quant_noise=False, q_prob=0.1, **kwargs: Any) -> PCQAlexNet:
    return PCQAlexNet(smooth=smooth, bit=bit, num_clusters=num_clusters, **kwargs, quant_noise=quant_noise, q_prob=q_prob)


def pcq_alexnet_small(smooth: float = 0.999, bit: int = 32, num_clusters: int = 10, quant_noise=False, q_prob=0.1, **kwargs: Any) -> PCQAlexNetSmall:
    return PCQAlexNetSmall(smooth=smooth, bit=bit, num_clusters=num_clusters, quant_noise=quant_noise, q_prob=q_prob, **kwargs)

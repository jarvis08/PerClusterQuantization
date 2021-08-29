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
        self.bit, self.smooth, self.num_clusters, self.runtime_helper, self.quant_noise, self.qn_prob\
            = itemgetter('bit', 'smooth', 'cluster', 'runtime_helper', 'quant_noise', 'qn_prob')(arg_dict)
        self.q_max = 2 ** self.bit - 1
        self.in_range = nn.Parameter(torch.zeros(self.num_clusters, 2), requires_grad=False)

        self.apply_ema = np.zeros(self.num_clusters, dtype=bool)

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
            done = 0
            for i in range(self.runtime_helper.batch_cluster.shape[0]):
                c = self.runtime_helper.batch_cluster[i][0]
                n = self.runtime_helper.batch_cluster[i][1]
                if self.apply_ema[c]:
                    self.in_range[c][0], self.in_range[c][1] = ema(x[done:done + n], self.in_range[c], self.smooth)
                    if self.runtime_helper.apply_fake_quantization:
                        s, z = calc_qparams(self.in_range[c][0], self.in_range[c][1], self.q_max)
                        x[done:done + n] = fake_quantize(x[done:done + n], s, z)
                else:
                    self.in_range[c][0] = torch.min(x[done:done + n]).item()
                    self.in_range[c][1] = torch.max(x[done:done + n]).item()
                    self.apply_ema[c] = True
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

    def __init__(self, arg_dict: dict, num_classes: int = 10) -> None:
        super(PCQAlexNetSmall, self).__init__()
        self.bit, self.smooth, self.num_clusters, self.runtime_helper, self.quant_noise, self.qn_prob\
            = itemgetter('bit', 'smooth', 'cluster', 'runtime_helper', 'quant_noise', 'qn_prob')(arg_dict)
        self.q_max = 2 ** self.bit - 1
        act_qmax = 2 ** self.bit - 1
        self.in_range = nn.Parameter(torch.zeros((self.num_clusters, 2)), requires_grad=False)

        self.apply_ema = np.zeros(self.num_clusters, dtype=bool)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv1 = PCQConv2d(3, 96, kernel_size=5, stride=1, padding=2, bias=True,
                               activation=nn.ReLU, arg_dict=arg_dict, act_qmax=act_qmax)
        self.conv2 = PCQConv2d(96, 256, kernel_size=5, stride=1, padding=2, bias=True,
                               activation=nn.ReLU, arg_dict=arg_dict, act_qmax=act_qmax)
        self.conv3 = PCQConv2d(256, 384, kernel_size=3, stride=1, padding=1, bias=True,
                               activation=nn.ReLU, arg_dict=arg_dict, act_qmax=act_qmax)
        self.conv4 = PCQConv2d(384, 384, kernel_size=3, stride=1, padding=1, bias=True,
                               activation=nn.ReLU, arg_dict=arg_dict, act_qmax=act_qmax)
        self.conv5 = PCQConv2d(384, 256, kernel_size=3, stride=1, padding=1, bias=True,
                               activation=nn.ReLU, arg_dict=arg_dict, act_qmax=act_qmax)
        self.fc1 = PCQLinear(256, 4096, activation=nn.ReLU, arg_dict=arg_dict, act_qmax=act_qmax)
        self.fc2 = PCQLinear(4096, 4096, activation=nn.ReLU, arg_dict=arg_dict, act_qmax=act_qmax)
        self.fc3 = PCQLinear(4096, num_classes, arg_dict=arg_dict, act_qmax=act_qmax)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            done = 0
            for i in range(self.runtime_helper.batch_cluster.shape[0]):
                c = self.runtime_helper.batch_cluster[i][0]
                n = self.runtime_helper.batch_cluster[i][1]
                if self.apply_ema[c]:
                    self.in_range[c][0], self.in_range[c][1] = ema(x[done:done + n], self.in_range[c], self.smooth)
                    if self.runtime_helper.apply_fake_quantization:
                        s, z = calc_qparams(self.in_range[c][0], self.in_range[c][1], self.q_max)
                        x[done:done + n] = fake_quantize(x[done:done + n], s, z, self.q_max)
                else:
                    self.in_range[c][0] = torch.min(x[done:done + n]).item()
                    self.in_range[c][1] = torch.max(x[done:done + n]).item()
                    self.apply_ema[c] = True
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
        self.fc3.set_qparams(prev_s, prev_z)


def pcq_alexnet(arg_dict: dict, **kwargs: Any) -> PCQAlexNet:
    return PCQAlexNet(arg_dict, **kwargs)


def pcq_alexnet_small(arg_dict: dict, **kwargs: Any) -> PCQAlexNetSmall:
    return PCQAlexNetSmall(arg_dict, **kwargs)

def modify_pcq_alexnet_qn_pre_hook(model):
    model.conv1.conv = _quant_noise(model.conv1.conv, model.runtime_helper.qn_prob, 1, q_max=model.q_max)
    model.conv2.conv = _quant_noise(model.conv2.conv, model.runtime_helper.qn_prob, 1, q_max=model.q_max)
    model.conv3.conv = _quant_noise(model.conv3.conv, model.runtime_helper.qn_prob, 1, q_max=model.q_max)
    model.conv4.conv = _quant_noise(model.conv4.conv, model.runtime_helper.qn_prob, 1, q_max=model.q_max)
    model.conv5.conv = _quant_noise(model.conv5.conv, model.runtime_helper.qn_prob, 1, q_max=model.q_max)
    model.fc1.fc = _quant_noise(model.fc1.fc, model.runtime_helper.qn_prob, 1, q_max=model.q_max)
    model.fc2.fc = _quant_noise(model.fc2.fc, model.runtime_helper.qn_prob, 1, q_max=model.q_max)
    model.fc3.fc = _quant_noise(model.fc3.fc, model.runtime_helper.qn_prob, 1, q_max=model.q_max)
    model.qn_prob = model.runtime_helper.qn_prob
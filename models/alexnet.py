import torch
import torch.nn as nn
from typing import Any
from .quantization_utils import *


class ClampMax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, original, clamped):
        return clamped.detach()

    @staticmethod
    def backward(ctx, grad):
        return grad, None


class AlexNet(nn.Module):
    def __init__(self, num_classes :int = 1000) -> None:
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class AlexNetSmall(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super(AlexNetSmall, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class AlexNetNoSeq(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super(AlexNetNoSeq, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        self.dropout = nn.Dropout()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.conv1 = nn.Conv2d(3, 96, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(256, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            x = self.conv1(x)
            x = self.relu(x)
            x = self.saturate_values(x, 0)
            x = self.maxpool(x)
            x = self.conv2(x)
            x = self.relu(x)
            x = self.saturate_values(x, 0)
            x = self.maxpool(x)
            x = self.conv3(x)
            x = self.relu(x)
            x = self.saturate_values(x, 0)
            x = self.conv4(x)
            x = self.relu(x)
            x = self.conv5(x)
            x = self.relu(x)
            x = self.saturate_values(x, 0)
            x = self.maxpool(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            x = self.relu(x)
            x = self.fc3(x)
            return x

        # f = open('fp_conv_output_max.txt', 'a')
        x = self.conv1(x)
        x = self.relu(x)
        # self.save_max_value(x, f)
        x = self.saturate_values(x, 0)
        x = self.maxpool(x)

        x = self.conv2(x)
        x = self.relu(x)
        # self.save_max_value(x, f)
        x = self.saturate_values(x, 1)
        x = self.maxpool(x)

        x = self.conv3(x)
        x = self.relu(x)
        # self.save_max_value(x, f)
        x = self.saturate_values(x, 2)

        x = self.conv4(x)
        x = self.relu(x)
        # self.save_max_value(x, f)
        x = self.saturate_values(x, 3)

        x = self.conv5(x)
        x = self.relu(x)
        # self.save_max_value(x, f, is_last=True)
        x = self.saturate_values(x, 4)
        x = self.maxpool(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu(x)

        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)

        x = self.fc3(x)
        # f.close()
        return x

    def set_no_seq_alexnet(self, pre):
        from copy import deepcopy
        with torch.no_grad():
            self.conv1.weight = deepcopy(pre.features[0].weight)
            self.conv1.bias = deepcopy(pre.features[0].bias)
            self.conv2.weight = deepcopy(pre.features[3].weight)
            self.conv2.bias = deepcopy(pre.features[3].bias)
            self.conv3.weight = deepcopy(pre.features[6].weight)
            self.conv3.bias = deepcopy(pre.features[6].bias)
            self.conv4.weight = deepcopy(pre.features[8].weight)
            self.conv4.bias = deepcopy(pre.features[8].bias)
            self.conv5.weight = deepcopy(pre.features[10].weight)
            self.conv5.bias = deepcopy(pre.features[10].bias)
            self.fc1.weight = deepcopy(pre.classifier[1].weight)
            self.fc1.bias = deepcopy(pre.classifier[1].bias)
            self.fc2.weight = deepcopy(pre.classifier[4].weight)
            self.fc2.bias = deepcopy(pre.classifier[4].bias)
            self.fc3.weight = deepcopy(pre.classifier[6].weight)
            self.fc3.bias = deepcopy(pre.classifier[6].bias)

    @staticmethod
    def saturate_values(x, layer_idx):
        # with torch.no_grad():
        #     median_of_max_values = [5.823872, 7.904878, 4.921054, 2.202915, 1.642391]
        #     _x = torch.clamp_max(x, median_of_max_values[layer_idx])
        with torch.no_grad():
            # to_quantile = torch.tensor([0.99], device='cuda')
            # value = torch.quantile(x, to_quantile)
            # _x = torch.clamp_max(x, value.item())

            _min, _max = x.min(), x.max()
            s, z = calc_qparams(_min, _max, 4)
            quantized_x = quantize_matrix(x, s, z, 4)
            num_zero = (quantized_x == z).nonzero(as_tuple=True)[0]
            to_quantile = torch.tensor([num_zero.size(0) / x.numel()], device='cuda')
            value = torch.quantile(x, to_quantile)
            y = torch.zeros_like(x, device='cuda')
            _x = torch.where(x > value, x, y)
        return ClampMax.apply(x, _x)

    @staticmethod
    def save_max_value(x, file, is_last=False):
        with torch.no_grad():
            max_values = x.view(x.size(0), -1).max(dim=1).values
            if is_last:
                for i in range(x.size(0) - 1):
                    file.write('{},'.format(max_values[i]))
                file.write('{}\n'.format(max_values[-1]))
            else:
                for i in range(x.size(0)):
                    file.write('{},'.format(max_values[i]))


def alexnet(**kwargs: Any) -> AlexNet:
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return AlexNet(**kwargs)


def alexnet_small(num_classes=10, **kwargs: Any) -> AlexNetSmall:
    return AlexNetSmall(num_classes=num_classes, **kwargs)


def alexnet_no_seq(num_classes=10, **kwargs: Any) -> AlexNetSmall:
    return AlexNetNoSeq(num_classes=num_classes, **kwargs)

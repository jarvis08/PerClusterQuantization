import torch
import torch.nn as nn
from typing import Any


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
    def __init__(self, num_classes: int = 10, smooth=0.999) -> None:
        super(AlexNetSmall, self).__init__()
        self.smooth = smooth
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

        for module in self.features:
            if isinstance(module, nn.Conv2d):
                module.input_range = nn.Parameter(torch.zeros((2, module.in_channels)), requires_grad=False)
                module.apply_ema = nn.Parameter(torch.tensor(0, dtype=torch.bool), requires_grad=False)

    def set_mixed_bits(self, x: torch.Tensor) -> torch.Tensor:
        for module in self.features:
            if isinstance(module, nn.Conv2d):
                data = x.transpose(1,0).reshape(x.size(1), -1)
                _max = data.max(dim=1).values
                _min = data.min(dim=1).values

                if module.apply_ema:
                    updated_min = module.input_range[0] * self.smooth + _min * (1 - self.smooth)
                    updated_max = module.input_range[1] * self.smooth + _max * (1 - self.smooth)

                    module.input_range[0], module.input_range[1] = updated_min, updated_max
                else:
                    module.input_range[0], module.input_range[1] = _min, _max
                    module.apply_ema.data = torch.tensor(True, dtype=torch.bool)
            x = module(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def alexnet(**kwargs: Any) -> AlexNet:
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return AlexNet(**kwargs)


def alexnet_small(num_classes=10, smooth=0.999, **kwargs: Any) -> AlexNetSmall:
    return AlexNetSmall(num_classes=num_classes, smooth=smooth, **kwargs)
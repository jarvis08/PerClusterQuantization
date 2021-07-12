import time
from typing import Any
from .layers.conv2d import *
from .layers.linear import *
from .quant_noise import quant_noise
from .quantization_utils import *

# def set_qn_alexnet(qn, pre):
#     qn.features[0].copy_from_pretrained(pre.features[0])
#     qn.features[3].copy_from_pretrained(pre.features[3])
#     qn.features[6].copy_from_pretrained(pre.features[6])
#     qn.features[8].copy_from_pretrained(pre.features[8])
#     qn.features[10].copy_from_pretrained(pre.features[10])
#
#     qn.classifier[1].weight = torch.nn.Parameter(pre.classifier[1].weight)
#     qn.classifier[1].bias = torch.nn.Parameter(pre.classifier[1].bias)
#     qn.classifier[4].weight = torch.nn.Parameter(pre.classifier[4].weight)
#     qn.classifier[4].bias = torch.nn.Parameter(pre.classifier[4].bias)
#     qn.classifier[6].weight = torch.nn.Parameter(pre.classifier[6].weight)
#     qn.classifier[6].bias = torch.nn.Parameter(pre.classifier[6].bias)
#
#     return qn

def set_qn_fused_alexnet(qn, pre):
    """
        Copy pre model's params & set fused layers.
        Use fused architecture, but not really fused (use CONV & BN seperately)
    """

    qn.features[0].copy_from_pretrained(pre.features[0])
    qn.features[3].copy_from_pretrained(pre.features[3])
    qn.features[6].copy_from_pretrained(pre.features[6])
    qn.features[8].copy_from_pretrained(pre.features[8])
    qn.features[10].copy_from_pretrained(pre.features[10])

    qn.classifier[1].weight = torch.nn.Parameter(pre.classifier[1].weight)
    qn.classifier[1].bias = torch.nn.Parameter(pre.classifier[1].bias)
    qn.classifier[4].weight = torch.nn.Parameter(pre.classifier[4].weight)
    qn.classifier[4].bias = torch.nn.Parameter(pre.classifier[4].bias)
    qn.classifier[6].weight = torch.nn.Parameter(pre.classifier[6].weight)
    qn.classifier[6].bias = torch.nn.Parameter(pre.classifier[6].bias)

    return qn

class AlexNet(nn.Module):
    def __init__(self, num_classes: int = 1000) -> None:
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

class Qn_AlexNet(nn.Module):
    def __init__(self, noise_prob : float=0.1, bit: int =8, num_classes: int=1000) -> None:
        super(Qn_AlexNet, self).__init__()
        self.noise_prob = noise_prob

        self.bit = bit
        self.q_max = 2 ** self.bit - 1

        self.features = nn.Sequential(
            Qn_Conv2d(3, 64, kernel_size=11, stride=4, padding=2, q_prob=noise_prob),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            Qn_Conv2d(64, 192, kernel_size=5, padding=2, q_prob=noise_prob),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            Qn_Conv2d(192, 384, kernel_size=3, padding=1, q_prob=noise_prob),
            nn.ReLU(inplace=True),

            Qn_Conv2d(384, 256, kernel_size=3, padding=1, q_prob=noise_prob),
            nn.ReLU(inplace=True),

            Qn_Conv2d(256, 256, kernel_size=3, padding=1, q_prob=noise_prob),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            quant_noise(nn.Linear(256 * 6 * 6, 4096), noise_prob, 1),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            quant_noise(nn.Linear(4096, 4096), noise_prob, 1),
            nn.ReLU(inplace=True),
            quant_noise(nn.Linear(4096, num_classes), noise_prob, 1),
        )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.features(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.classifier(x)
            return x

        def copy_from_pretrained(self, conv, bn):
            # Copy weights from pretrained FP model
            self.conv.weight.data = torch.nn.Parameter(conv.weight.data)
            if bn:
                self.bn = bn
            else:
                self.conv.bias.data = torch.nn.Parameter(conv.bias.data)

class AlexNetSmall(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super(AlexNetSmall, self).__init__()
        noise_prob = 0.1
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



class Qn_AlexNetSmall(nn.Module):
    def __init__(self, smooth :float=0.999 ,noise_prob : float=0.1, bit: int =8,  num_classes: int = 10) -> None:
        super(Qn_AlexNetSmall, self).__init__()
        # self.noise_prob = noise_prob
        self.noise_prob = noise_prob
        self.bit = bit
        self.q_max = 2 ** self.bit - 1
        self.features = nn.Sequential(
            Qn_Conv2d(3, 96, kernel_size=5, stride=1, padding=2, q_prob=self.noise_prob, q_max=self.q_max),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),

            Qn_Conv2d(96, 256, kernel_size=5, stride=1, padding=2, q_prob=self.noise_prob, q_max=self.q_max),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),

            Qn_Conv2d(256, 384, kernel_size=3, stride=1, padding=1, q_prob=self.noise_prob, q_max=self.q_max),
            nn.ReLU(inplace=True),

            Qn_Conv2d(384, 384, kernel_size=3, stride=1, padding=1, q_prob=self.noise_prob, q_max=self.q_max),
            nn.ReLU(inplace=True),

            Qn_Conv2d(384, 256, kernel_size=3, stride=1, padding=1, q_prob=self.noise_prob, q_max=self.q_max),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            quant_noise(nn.Linear(256, 4096), self.noise_prob, 1, self.q_max),
            nn.ReLU(inplace=True),

            nn.Dropout(),
            quant_noise(nn.Linear(4096, 4096), self.noise_prob, 1, self.q_max),
            nn.ReLU(inplace=True),

            quant_noise(nn.Linear(4096, num_classes), self.noise_prob, 1, self.q_max),
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def copy_from_pretrained(self, conv):
        # Copy weights from pretrained FP model
        self.conv.weight.data = torch.nn.Parameter(conv.weight.data)
        self.conv.bias.data = torch.nn.Parameter(conv.bias.data)


class Qn_Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 q_prob=0.2, q_max=255):
        super(Qn_Conv2d, self).__init__()
        self.layer_type = 'QNConv2d'
        self.q_prob = q_prob
        self.conv = quant_noise(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias), q_prob, 1, q_max)

    def forward(self, x):
        x = self.conv(x)
        return x

    def copy_from_pretrained(self, conv):
        self.conv.weight.data = torch.nn.Parameter(conv.weight.data)
        self.conv.bias.data = torch.nn.Parameter(conv.bias.data)


def qn_alexnet_small(smooth: float = 0.999, bit: int = 8, noise_prob=0.1, **kwargs:Any) -> Qn_AlexNetSmall:
    return Qn_AlexNetSmall(smooth=smooth, bit=bit, noise_prob=noise_prob, **kwargs)

def qn_alexnet(smooth: float = 0.999, bit: int =8, noise_prob=0.1, **kwargs:Any) -> Qn_AlexNet:
    return Qn_AlexNet(smooth=smooth, bit=bit, noise_prob=noise_prob, **kwargs)


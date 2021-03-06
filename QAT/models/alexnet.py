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

    def initialize_counter(self, x, n_clusters):
        self.zero_counter = []
        indices = [2, 5, 8, 10, 12]
        _to = None
        for i in range(len(indices)):
            _from = 0 if i == 0 else indices[i - 1]
            _to = indices[i]
            x = self.features[_from:_to](x)
            n_features = x.view(-1).size(0)
            self.zero_counter.append(torch.zeros((n_clusters, n_features), device='cuda'))

        # Classifier make 0 ratio
        # x = self.features[_to:](x)  # left feature extractions
        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)

        # indices = [3, 6, 7]
        # for i in range(len(indices)):
        #     _from = 0 if i == 0 else indices[i - 1]
        #     _to = indices[i]
        #     x = self.classifier[_from:_to](x)
        #     n_features = x.view(-1).size(0)
        #     self.zero_counter.append(torch.zeros((n_clusters, n_features), device='cuda'))
        # self.zero_counter.append(torch.zeros((n_clusters, x.shape[1]), device='cuda'))

    def count_zeros_per_index(self, x, cluster, n_clusters):
        if not hasattr(self, 'zero_counter'):
            self.initialize_counter(x[0].unsqueeze(0), n_clusters)

        # indices = [2, 5, 8, 10, 12]
        # for l in range(len(indices)):
        #     _from = 0 if l == 0 else indices[l - 1]
        #     _to = indices[l]
        #     x = self.features[_from:_to](x)

        conv_cnt = 0
        for layer_idx, layer in enumerate(self.features):
            if isinstance(layer, nn.Conv2d):
                x = self.features[layer_idx](x)
                x = self.features[layer_idx + 1](x)
                layer_idx += 2
                n_features = self.zero_counter[conv_cnt].size(1)
                for idx in range(x.size(0)):
                    flattened = x[idx].view(-1)
                    zeros_idx = (flattened == 0.0).nonzero(as_tuple=True)[0]
                    zeros_idx %= n_features
                    self.zero_counter[conv_cnt][cluster, zeros_idx] += 1
                conv_cnt += 1
            else:
                x = self.features[layer_idx](x)

    # Classifier make 0 ratio
        # x = self.features[_to:](x)  # left feature extractions
        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)

        # indices = [3, 6, 7]
        # for i in range(len(indices)):
        #     _from = 0 if i == 0 else indices[i - 1]
        #     _to = indices[i]
        #     x = self.classifier[_from:_to](x)

        #     n_features = self.zero_counter[l].size(1)
        #     for idx in range(x.size(0)):
        #         flattened = x[idx].view(-1)
        #         zeros_idx = (flattened == 0.0).nonzero(as_tuple=True)[0]
        #         zeros_idx %= n_features
        #         self.zero_counter[l][cluster, zeros_idx] += 1


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


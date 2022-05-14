import torch
from torch import Tensor
import torch.nn as nn
from torch.utils.model_zoo import load_url
from typing import Type, Any, Callable, Union, List, Optional


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

    def count_zeros_per_index(self, x, cluster, n_clusters, zero_counter, l_idx, initialized):
        if not initialized:
            _x = x[0].unsqueeze(0)
            identity = _x

            out = self.conv1(_x)
            out = self.bn1(out)
            out = self.relu(out)
            n_features = out.view(-1).size(0)
            zero_counter.append(torch.zeros((n_clusters, n_features), device='cuda'))

            out = self.conv2(out)
            out = self.bn2(out)

            if self.downsample is not None:
                identity = self.downsample(_x)

            out += identity
            out = self.relu(out)
            n_features = out.view(-1).size(0)
            zero_counter.append(torch.zeros((n_clusters, n_features), device='cuda'))

        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        l_idx += 1
        n_features = zero_counter[l_idx].size(1)
        for i in range(out.size(0)):
            flattened = out[i].view(-1)
            zeros_idx = (flattened == 0.0).nonzero(as_tuple=True)[0]
            zeros_idx %= n_features
            zero_counter[l_idx][cluster, zeros_idx] += 1

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        l_idx += 1
        n_features = zero_counter[l_idx].size(1)
        for i in range(out.size(0)):
            flattened = out[i].view(-1)
            zeros_idx = (flattened == 0.0).nonzero(as_tuple=True)[0]
            zeros_idx %= n_features
            zero_counter[l_idx][cluster, zeros_idx] += 1
        return out, l_idx


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

    def count_zeros_per_index(self, x, cluster, n_clusters, zero_counter, l_idx, initialized):
        if not initialized:
            _x = x[0].unsqueeze(0)
            identity = _x

            out = self.conv1(_x)
            out = self.bn1(out)
            out = self.relu(out)
            n_features = out.view(-1).size(0)
            zero_counter.append(torch.zeros((n_clusters, n_features), device='cuda'))

            out = self.conv2(out)
            out = self.bn2(out)
            out = self.relu(out)
            n_features = out.view(-1).size(0)
            zero_counter.append(torch.zeros((n_clusters, n_features), device='cuda'))

            out = self.conv3(out)
            out = self.bn3(out)

            if self.downsample is not None:
                identity = self.downsample(_x)

            out += identity
            out = self.relu(out)
            n_features = out.view(-1).size(0)
            zero_counter.append(torch.zeros((n_clusters, n_features), device='cuda'))

        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        l_idx += 1
        n_features = zero_counter[l_idx].size(1)
        for i in range(out.size(0)):
            flattened = out[i].view(-1)
            zeros_idx = (flattened == 0.0).nonzero(as_tuple=True)[0]
            zeros_idx %= n_features
            zero_counter[l_idx][cluster, zeros_idx] += 1

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        l_idx += 1
        n_features = zero_counter[l_idx].size(1)
        for i in range(out.size(0)):
            flattened = out[i].view(-1)
            zeros_idx = (flattened == 0.0).nonzero(as_tuple=True)[0]
            zeros_idx %= n_features
            zero_counter[l_idx][cluster, zeros_idx] += 1

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        l_idx += 1
        n_features = zero_counter[l_idx].size(1)
        for i in range(out.size(0)):
            flattened = out[i].view(-1)
            zeros_idx = (flattened == 0.0).nonzero(as_tuple=True)[0]
            zeros_idx %= n_features
            zero_counter[l_idx][cluster, zeros_idx] += 1
        return out, l_idx


class ResNet(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.num_blocks = 4

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

    def count_zeros_per_index(self, x, cluster, n_clusters):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        initialized = True
        if not hasattr(self, 'zero_counter'):
            initialized = False
            n_features = x.view(-1).size(0)
            self.zero_counter = []
            self.zero_counter.append(torch.zeros((n_clusters, n_features), device='cuda'))

        l_idx = 0
        n_features = self.zero_counter[l_idx].size(1)
        for i in range(x.size(0)):
            flattened = x[i].view(-1)
            zeros_idx = (flattened == 0.0).nonzero(as_tuple=True)[0]
            zeros_idx %= n_features
            self.zero_counter[l_idx][cluster, zeros_idx] += 1

        blocks = [self.layer1, self.layer2, self.layer3, self.layer4]
        for block in blocks:
            for b in range(len(block)):
                x, l_idx = block[b].count_zeros_per_index(x, cluster, n_clusters, self.zero_counter, l_idx, initialized)


class ResNet20(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNet20, self).__init__()
        self._norm_layer = nn.BatchNorm2d
        self.inplanes = 16
        self.num_blocks = 3

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = self._norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                self._norm_layer(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def show_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.show_params()

    def count_zeros_per_index(self, x, cluster, n_clusters):
        # cluster -> ready cluster idx, n_cluster -> args.sub_cluster
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        initialized = True
        if not hasattr(self, 'zero_counter'):
            initialized = False
            n_features = x.view(-1).size(0)
            self.zero_counter = []
            # 리스트에 박은거 (8, 524288)
            self.zero_counter.append(torch.zeros((n_clusters, n_features), device='cuda'))

        # 처음 0 block 지날때마다 l_idx 증
        l_idx = 0
        # 데이터 핀거 shape 524288
        n_features = self.zero_counter[l_idx].size(1)
        for i in range(x.size(0)):
            # 16384
            flattened = x[i].view(-1)
            # 5686
            zeros_idx = (flattened == 0.0).nonzero(as_tuple=True)[0]
            # n_features가 더 큰데 왜 굳이 넣었을까?
            zeros_idx %= n_features
            # 클러스터별로 zero_idx를
            self.zero_counter[l_idx][cluster, zeros_idx] += 1

        blocks = [self.layer1, self.layer2, self.layer3]
        for block in blocks:
            for b in range(len(block)):
                x, l_idx = block[b].count_zeros_per_index(x, cluster, n_clusters, self.zero_counter, l_idx, initialized)


def _resnet(
    arch: str,
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    pretrained: bool,
    progress: bool,
    **kwargs: Any
) -> ResNet:
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model


def resnet18(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress, **kwargs)


def resnet20(num_classes=10) -> ResNet20:
    return ResNet20(BasicBlock, [3, 3, 3], num_classes=num_classes)


def resnet50(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)

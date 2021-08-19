import torch
import torch.nn as nn
import torch.utils.checkpoint as cp

from collections import OrderedDict
from typing import List, Tuple
from torch import Tensor

from .layers import *
from .quantization_utils import *


class QuantizedDenseLayer(nn.Module):
    def __init__(
        self,
        arg_dict,
        num_input_features: int,
        growth_rate: int,
        bn_size: int,
        memory_efficient: bool = False,
    ) -> None:
        super(QuantizedDenseLayer, self).__init__()
        self.arg_dict = arg_dict
        self.bit, self.num_clusters = itemgetter('bit', 'cluster')(arg_dict)
        self.q_max = 2 ** self.bit - 1

        self.bn = QuantizedBnReLU(num_input_features, arg_dict)
        self.conv1 = QuantizedConv2d(num_input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=False,
                                 arg_dict=arg_dict)
        self.conv2 = QuantizedConv2d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False, arg_dict=arg_dict)
        self.memory_efficient = memory_efficient

    # torchscript does not yet support *args, so we overload method
    # allowing it to take either a List[Tensor] or single Tensor
    def forward(self, input):  # noqa: F811
        if isinstance(input, Tensor):
            prev_features = [input]
        else:
            prev_features = input

        x = torch.cat(prev_features, 1)
        out = self.bn(x)
        out = self.conv1(out)
        out = self.conv2(out)
        return out


class QuantizedTransition(nn.Sequential):
    def __init__(self, arg_dict, num_input_features: int, num_output_features: int) -> None:
        super(QuantizedTransition, self).__init__()
        self.arg_dict = arg_dict
        self.bit, self.num_clusters = itemgetter('bit', 'cluster')(arg_dict)
        self.q_max = 2 ** self.bit - 1

        self.norm = QuantizedBnReLU(num_input_features, arg_dict)
        self.conv = QuantizedConv2d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False, arg_dict=arg_dict)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        out = self.norm(x)
        out = self.conv(out)
        out = self.pool(out)
        return out


class QuantizedDenseBlock(nn.ModuleDict):
    _version = 2

    def __init__(
        self,
        arg_dict,
        num_layers: int,
        num_input_features: int,
        bn_size: int,
        growth_rate: int,
        memory_efficient: bool = False,
    ) -> None:
        super(QuantizedDenseBlock, self).__init__()
        self.arg_dict = arg_dict
        self.num_layers = num_layers
        self.bit, self.num_clusters = itemgetter('bit', 'cluster')(arg_dict)
        self.q_max = 2 ** self.bit - 1

        for i in range(num_layers):
            layer = FusedDenseLayer(
                arg_dict,
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                memory_efficient=memory_efficient,
            )
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features: Tensor) -> Tensor:
        features = [init_features]
        for name, layer in self.items():
            new_features = layer(features)
            features.append(new_features)
        out = torch.cat(features, 1)
        return out


class QuantizedDenseNet(nn.Module):
    def __init__(
        self,
        growth_rate: int = 32,
        block_config: Tuple[int, int, int, int] = (6, 12, 24, 16),
        num_init_features: int = 64,
        arg_dict: dict = None,
        bn_size: int = 4,
        num_classes: int = 1000,
        memory_efficient: bool = False
    ) -> None:
        super(QuantizedDenseNet, self).__init__()
        self.arg_dict = arg_dict
        self.bit, self.num_clusters, self.runtime_helper = itemgetter('bit', 'cluster', 'runtime_helper')(arg_dict)
        self.q_max = 2 ** self.bit - 1

        t_init = list(range(self.num_clusters)) if self.num_clusters > 1 else 0
        self.scale = nn.Parameter(torch.tensor(t_init, dtype=torch.float32), requires_grad=False)
        self.zero_point = nn.Parameter(torch.tensor(t_init, dtype=torch.int32), requires_grad=False)

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('first_conv', QuantizedConv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False, arg_dict=arg_dict)),
            ('maxpool', QuantizedMaxPool2d(kernel_size=3, stride=2, padding=1, arg_dict=arg_dict))
        ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = QuantizedDenseBlock(
                arg_dict=arg_dict,
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                memory_efficient=memory_efficient,
            )
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = QuantizedTransition(arg_dict=arg_dict, num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2
        # Last Norm
        self.features.add_module('last_norm', QuantizedBnReLU(num_features, arg_dict))
        # Linear layer
        self.classifier = QuantizedLinear(num_features, num_classes, arg_dict=arg_dict)

    def forward(self, x: Tensor) -> Tensor:
        if self.runtime_helper.batch_cluster is not None:
            done = 0
            for i in range(self.runtime_helper.batch_cluster.shape[0]):
                c = self.runtime_helper.batch_cluster[i][0].item()
                n = self.runtime_helper.batch_cluster[i][1].item()
                x[done:done + n] = quantize_matrix(x[done:done + n], self.scale[c], self.zero_point[c], self.q_max)
                done += n
        else:
            x = quantize_matrix(x, self.scale, self.zero_point, self.q_max)

        out = self.features(x)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out


def quantized_densenet(arg_dict: dict, **kwargs):
    return QuantizedDenseNet(32, (6, 12, 24, 16), 64, arg_dict, **kwargs)


def quantize_block(_fp, _int):
    for layer_idx in range(1, _fp.num_layers+1):
        fp_layer = getattr(_fp, 'denselayer%d' % layer_idx)
        int_layer = getattr(_int, 'denselayer%d' % layer_idx)

        int_layer.bn = quantize(fp_layer.bn, int_layer.bn)
        int_layer.conv1 = quantize(fp_layer.conv1, int_layer.conv1)
        int_layer.conv2 = quantize(fp_layer.conv2, int_layer.conv2)
    return _int


def quantize_PCQblock(_fp, _int):
    for layer_idx in range(1, _fp.num_layers+1):
        fp_layer = getattr(_fp, 'denselayer%d' % layer_idx)
        int_layer = getattr(_int, 'denselayer%d' % layer_idx)

        int_layer.bn1 = quantize(fp_layer.bn1, int_layer.bn1)
        int_layer.conv1 = quantize(fp_layer.conv1, int_layer.conv1)
        int_layer.bn2 = quantize(fp_layer.bn2, int_layer.bn2)
        int_layer.conv2 = quantize(fp_layer.conv2, int_layer.conv2)
    return _int


def quantize_trans(_fp, _int):
    _int.norm = quantize(_fp.norm, _int.norm)
    _int.conv = quantize(_fp.conv, _int.conv)
    return _int


def quantize_densenet(fp_model, int_model):
    int_model.scale = torch.nn.Parameter(fp_model.scale, requires_grad=False)
    int_model.zero_point = torch.nn.Parameter(fp_model.zero_point, requires_grad=False)
    int_model.first_conv = quantize(fp_model.first_conv, int_model.first_conv)
    int_model.features.denseblock1 = quantize_block(fp_model.features.denseblock1, int_model.features.denseblock1)
    int_model.features.transition1 = quantize_trans(fp_model.features.transition1, int_model.features.transition1)
    int_model.features.denseblock2 = quantize_block(fp_model.features.denseblock2, int_model.features.denseblock2)
    int_model.features.transition2 = quantize_trans(fp_model.features.transition2, int_model.features.transition2)
    int_model.features.denseblock3 = quantize_block(fp_model.features.denseblock3, int_model.features.denseblock3)
    int_model.features.transition3 = quantize_trans(fp_model.features.transition3, int_model.features.transition3)
    int_model.features.denseblock4 = quantize_block(fp_model.features.denseblock4, int_model.features.denseblock4)
    int_model.features.last_norm = quantize(fp_model.features.last_norm, int_model.features.last_norm)
    int_model.classifier = quantize(fp_model.classifier, int_model.classifier)
    return int_model


def quantize_pcq_densenet(fp_model, int_model):
    int_model.scale = torch.nn.Parameter(fp_model.scale, requires_grad=False)
    int_model.zero_point = torch.nn.Parameter(fp_model.zero_point, requires_grad=False)
    int_model.first_conv = quantize(fp_model.first_conv, int_model.first_conv)
    int_model.first_norm = quantize(fp_model.first_norm, int_model.first_norm)
    int_model.features.denseblock1 = quantize_block(fp_model.features.denseblock1, int_model.features.denseblock1)
    int_model.features.transition1 = quantize_trans(fp_model.features.transition1, int_model.features.transition1)
    int_model.features.denseblock2 = quantize_block(fp_model.features.denseblock2, int_model.features.denseblock2)
    int_model.features.transition2 = quantize_trans(fp_model.features.transition2, int_model.features.transition2)
    int_model.features.denseblock3 = quantize_block(fp_model.features.denseblock3, int_model.features.denseblock3)
    int_model.features.transition3 = quantize_trans(fp_model.features.transition3, int_model.features.transition3)
    int_model.features.denseblock4 = quantize_block(fp_model.features.denseblock4, int_model.features.denseblock4)
    int_model.features.last_norm = quantize(fp_model.features.last_norm, int_model.features.last_norm)
    int_model.classifier = quantize(fp_model.classifier, int_model.classifier)
    return int_model
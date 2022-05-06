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
        self.num_clusters, self.runtime_helper = itemgetter('cluster', 'runtime_helper')(arg_dict)

        self.bn1 = QuantizedBn2d(num_input_features, arg_dict=arg_dict)
        self.conv1 = QuantizedConv2d(num_input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=False, arg_dict=arg_dict)
        self.bn2 = QuantizedBn2d(bn_size * growth_rate, arg_dict=arg_dict)
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
        out = self.bn1(x)
        out = self.conv1(out.type(torch.cuda.FloatTensor))
        out = self.bn2(out)
        out = self.conv2(out.type(torch.cuda.FloatTensor))
        return out


class QuantizedTransition(nn.Sequential):
    def __init__(self, arg_dict, num_input_features: int, num_output_features: int) -> None:
        super(QuantizedTransition, self).__init__()
        self.arg_dict = arg_dict
        self.num_clusters, self.runtime_helper = itemgetter('cluster', 'runtime_helper')(arg_dict)

        self.bn = QuantizedBn2d(num_input_features, arg_dict=arg_dict)
        self.conv = QuantizedConv2d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False, arg_dict=arg_dict)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        out = self.bn(x)
        out = self.conv(out.type(torch.cuda.FloatTensor))
        out = self.pool(out.type(torch.cuda.FloatTensor))
        out = out.floor()
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
        self.num_clusters, self.runtime_helper = itemgetter('cluster', 'runtime_helper')(arg_dict)

        for i in range(num_layers):
            layer = QuantizedDenseLayer(
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
        self.num_clusters, self.bit_first, self.runtime_helper = itemgetter('cluster', 'bit_first', 'runtime_helper')(arg_dict)

        self.target_bit = nn.Parameter(torch.tensor(0, dtype=torch.int8), requires_grad=False)
        self.a_bit = nn.Parameter(torch.tensor(0, dtype=torch.int8), requires_grad=False)
        self.in_bit = nn.Parameter(torch.tensor(self.bit_first, dtype=torch.int8), requires_grad=False)
        self.arg_dict = arg_dict

        t_init = list(range(self.num_clusters)) if self.num_clusters > 1 else 0
        self.scale = nn.Parameter(torch.tensor(t_init, dtype=torch.float32), requires_grad=False)
        self.zero_point = nn.Parameter(torch.tensor(t_init, dtype=torch.int32), requires_grad=False)

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('first_conv', QuantizedConv2d(3, num_init_features, kernel_size=7, stride=2, padding=3,
                                           is_first=True, arg_dict=arg_dict)),
            ('first_norm', QuantizedBn2d(num_init_features, arg_dict=arg_dict)),
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
        self.features.add_module('last_norm', QuantizedBn2d(num_features, arg_dict=arg_dict))
        # Linear layer
        self.classifier = QuantizedLinear(num_features, num_classes, arg_dict=arg_dict)

    def forward(self, x: Tensor) -> Tensor:
        if self.runtime_helper.qat_batch_cluster is not None:
            x = quantize_matrix_4d(x, self.scale, self.zero_point, self.runtime_helper.qat_batch_cluster, self.in_bit)
        else:
            x = quantize_matrix(x, self.scale, self.zero_point, self.in_bit)

        out = self.features.first_conv(x.type(torch.cuda.FloatTensor))
        out = self.features.first_norm(out)

        out = self.features.maxpool(out.type(torch.cuda.FloatTensor))
        out = self.features.denseblock1(out)
        out = self.features.transition1(out)
        out = self.features.denseblock2(out)
        out = self.features.transition2(out)
        out = self.features.denseblock3(out)
        out = self.features.transition3(out)
        out = self.features.denseblock4(out)
        out = self.features.last_norm(out)

        out = F.adaptive_avg_pool2d(out.type(torch.cuda.FloatTensor), (1, 1))
        out = out.floor()

        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out.type(torch.cuda.FloatTensor)


def quantized_densenet(arg_dict: dict, **kwargs):
    return QuantizedDenseNet(32, (6, 12, 24, 16), 64, arg_dict, **kwargs)


def quantize_block(_fp, _int):
    for layer_idx in range(1, _fp.num_layers+1):
        fp_layer = getattr(_fp, 'denselayer%d' % layer_idx)
        int_layer = getattr(_int, 'denselayer%d' % layer_idx)

        int_layer.bn1 = quantize(fp_layer.bn1, int_layer.bn1)
        int_layer.conv1 = quantize(fp_layer.conv1, int_layer.conv1)
        int_layer.bn2 = quantize(fp_layer.bn2, int_layer.bn2)
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
    _int.bn = quantize(_fp.bn, _int.bn)
    _int.conv = quantize(_fp.conv, _int.conv)
    return _int


def quantize_densenet(fp_model, int_model):
    int_model.target_bit.data = fp_model.target_bit
    int_model.in_bit.data = fp_model.in_bit
    int_model.scale.data = fp_model.scale
    int_model.zero_point.data = fp_model.zero_point
    int_model.features.first_conv = quantize(fp_model.features.first_conv, int_model.features.first_conv)
    int_model.features.first_norm = quantize(fp_model.features.first_norm, int_model.features.first_norm)
    int_model.features.denseblock1 = quantize_block(fp_model.features.denseblock1, int_model.features.denseblock1)
    int_model.features.transition1 = quantize_trans(fp_model.features.transition1, int_model.features.transition1)
    int_model.features.denseblock2 = quantize_block(fp_model.features.denseblock2, int_model.features.denseblock2)
    int_model.features.transition2 = quantize_trans(fp_model.features.transition2, int_model.features.transition2)
    int_model.features.denseblock3 = quantize_block(fp_model.features.denseblock3, int_model.features.denseblock3)
    int_model.features.transition3 = quantize_trans(fp_model.features.transition3, int_model.features.transition3)
    int_model.features.denseblock4 = quantize_block(fp_model.features.denseblock4, int_model.features.denseblock4)
    int_model.features.last_norm = quantize(fp_model.features.last_norm, int_model.features.last_norm)
    int_model.classifier = quantize(fp_model.classifier, int_model.classifier)

    int_model.a_bit.data = fp_model.a_bit
    #int_model.s1.data = fp_model.s1  # S, Z of 8/16/32 bit
    #int_model.z1.data = fp_model.z1
    #int_model.s_target.data = fp_model.s_target  # S, Z of 4/8 bit
    #int_model.z_target.data = fp_model.z_target
    #int_model.M0.data = fp_model.M0
    #int_model.shift.data = fp_model.shift
    return int_model


def quantize_pcq_densenet(fp_model, int_model):
    int_model.scale = torch.nn.Parameter(fp_model.scale, requires_grad=False)
    int_model.zero_point = torch.nn.Parameter(fp_model.zero_point, requires_grad=False)
    int_model.features.first_conv = quantize(fp_model.features.first_conv, int_model.features.first_conv)
    int_model.features.first_norm = quantize(fp_model.features.first_norm, int_model.features.first_norm)
    int_model.features.maxpool.bit.data = int_model.features.first_norm.a_bit.data
    int_model.features.maxpool.zero_point.data = int_model.features.first_norm.z3.data
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
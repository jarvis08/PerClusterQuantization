
import torch
import torch.nn as nn
import torch.utils.checkpoint as cp

from collections import OrderedDict
from typing import List, Tuple
from torch import Tensor

from .layers import *
from .quantization_utils import *


class PCQDenseLayer(nn.Module):
    def __init__(
        self,
        arg_dict,
        num_input_features: int,
        growth_rate: int,
        bn_size: int,
        memory_efficient: bool = False,
        a_bit = None
    ) -> None:
        super(PCQDenseLayer, self).__init__()
        self.arg_dict = arg_dict

        target_bit, bit_conv_act, bit_addcat, self.smooth, self.use_ste, self.num_clusters, self.runtime_helper \
            = itemgetter('bit', 'bit_conv_act', 'bit_addcat', 'smooth', 'ste', 'cluster', 'runtime_helper')(arg_dict)
        self.a_bit = torch.nn.Parameter(torch.tensor(bit_addcat, dtype=torch.int8), requires_grad=False)
        self.target_bit = torch.nn.Parameter(torch.tensor(target_bit, dtype=torch.int8), requires_grad=False)

        self.bn1 = PCQBnReLU(num_input_features, activation=nn.ReLU, a_bit=target_bit, arg_dict=arg_dict)
        self.conv1 = PCQConv2d(num_input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=False,
                               arg_dict=arg_dict, a_bit=bit_conv_act)
        self.bn2 = PCQBnReLU(bn_size * growth_rate, activation=nn.ReLU, a_bit=target_bit, arg_dict=arg_dict)
        self.conv2 = PCQConv2d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False,
                               arg_dict=arg_dict, a_bit=bit_conv_act)
        self.memory_efficient = memory_efficient

    # torchscript does not yet support *args, so we overload method
    # allowing it to take either a List[Tensor] or single Tensor
    def forward(self, input, external_range):  # noqa: F811
        if isinstance(input, Tensor):
            prev_features = [input]
        else:
            prev_features = input

        x = torch.cat(prev_features, 1)
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.conv2(out, external_range)
        return out

    def set_layer_qparams(self, s1, z1):
        prev_s, prev_z = self.bn1.set_qparams(s1, z1)
        prev_s, prev_z = self.conv1.set_qparams(prev_s, prev_z)
        prev_s, prev_z = self.bn2.set_qparams(prev_s, prev_z)
        self.conv2.set_qparams(prev_s, prev_z, s1, z1)


class PCQTransition(nn.Sequential):
    def __init__(self, arg_dict, num_input_features: int, num_output_features: int, a_bit=None) -> None:
        super(PCQTransition, self).__init__()
        self.arg_dict = arg_dict
        target_bit, bit_conv_act, bit_addcat, self.smooth, self.use_ste, self.num_clusters, self.runtime_helper \
            = itemgetter('bit', 'bit_conv_act', 'bit_addcat', 'smooth', 'ste', 'cluster', 'runtime_helper')(arg_dict)
        self.a_bit = torch.nn.Parameter(torch.tensor(bit_addcat, dtype=torch.int8), requires_grad=False)

        self.bn = PCQBnReLU(num_input_features, activation=nn.ReLU, a_bit=target_bit, arg_dict=arg_dict)
        self.conv = PCQConv2d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False,
                              arg_dict=arg_dict, a_bit=bit_addcat)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x, next_block_range):
        out = self.bn(x)
        out = self.conv(out, next_block_range)
        out = self.pool(out)
        return out

    def set_transition_qparams(self, s1, z1, next_block_s, next_block_z):
        prev_s, prev_z = self.bn.set_qparams(s1, z1)
        self.conv.set_qparams(prev_s, prev_z, next_block_s, next_block_z)
        return next_block_s, next_block_z


class PCQDenseBlock(nn.ModuleDict):
    _version = 2

    def __init__(
        self,
        arg_dict,
        num_layers: int,
        num_input_features: int,
        bn_size: int,
        growth_rate: int,
        memory_efficient: bool = False,
        a_bit=None
    ) -> None:
        super(PCQDenseBlock, self).__init__()
        self.arg_dict = arg_dict
        self.num_layers = num_layers
        target_bit, bit_conv_act, bit_addcat, self.smooth, self.use_ste, self.num_clusters, self.runtime_helper \
            = itemgetter('bit', 'bit_conv_act', 'bit_addcat', 'smooth', 'ste', 'cluster', 'runtime_helper')(arg_dict)
        self.a_bit = torch.nn.Parameter(torch.tensor(bit_addcat, dtype=torch.int8), requires_grad=False)

        self.act_range = nn.Parameter(torch.zeros(self.num_clusters, 2), requires_grad=False)
        self.apply_ema = nn.Parameter(torch.zeros(self.num_clusters, dtype=torch.bool), requires_grad=False)

        for i in range(num_layers):
            layer = PCQDenseLayer(
                arg_dict,
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                memory_efficient=memory_efficient,
                a_bit=a_bit
            )
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features: Tensor) -> Tensor:
        features = [init_features]
        for name, layer in self.items():
            new_features = layer(features, self.act_range)
            features.append(new_features)
        out = torch.cat(features, 1)

        if self.training:
            self._update_activation_ranges(out)
        return out

    @torch.no_grad()
    def _update_activation_ranges(self, x):
        cluster = self.runtime_helper.qat_batch_cluster
        data = x.view(x.size(0), -1)
        # _min = data.min(dim=1).values.mean()
        _max = data.max(dim=1).values.mean()
        if self.apply_ema[cluster]:
            # self.act_range[cluster][0] = self.act_range[cluster][0] * self.smooth + _min * (1 - self.smooth)
            self.act_range[cluster][1] = self.act_range[cluster][1] * self.smooth + _max * (1 - self.smooth)
        else:
            # self.act_range[cluster][0], self.act_range[cluster][1] = _min, _max
            self.act_range[cluster][1] = _max
            self.apply_ema[cluster] = True

    def set_block_qparams(self):
        zero = self.runtime_helper.fzero
        self.s3, self.z3 = calc_qparams_per_cluster(self.act_range, self.a_bit, zero)
        for name, layer in self.items():
            layer.set_layer_qparams(self.s3, self.z3)
        return self.s3, self.z3


class PCQDenseNet(nn.Module):
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
        super(PCQDenseNet, self).__init__()
        self.arg_dict = arg_dict
        target_bit, bit_conv_act, bit_addcat, bit_first, bit_classifier, self.smooth, self.num_clusters, self.runtime_helper \
            = itemgetter('bit', 'bit_conv_act', 'bit_addcat', 'bit_first', 'bit_classifier', 'smooth', 'cluster',
                         'runtime_helper')(arg_dict)
        self.target_bit = torch.nn.Parameter(torch.tensor(target_bit, dtype=torch.int8), requires_grad=False)
        self.a_bit = torch.nn.Parameter(torch.tensor(bit_addcat, dtype=torch.int8), requires_grad=False)
        self.in_bit = torch.nn.Parameter(torch.tensor(bit_first, dtype=torch.int8), requires_grad=False)

        self.in_range = nn.Parameter(torch.zeros(self.num_clusters, 2), requires_grad=False)
        self.apply_ema = nn.Parameter(torch.zeros(self.num_clusters, dtype=torch.bool), requires_grad=False)

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('first_conv', PCQConv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False,
                                     w_bit=bit_first, a_bit=bit_conv_act, arg_dict=arg_dict)),
            ('first_norm', PCQBnReLU(num_init_features, activation=nn.ReLU, a_bit=bit_addcat, arg_dict=arg_dict)),
            ('maxpool', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = PCQDenseBlock(
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
                trans = PCQTransition(arg_dict=arg_dict, num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2
        # Last Norm
        self.features.add_module('last_norm', PCQBnReLU(num_features, activation=nn.ReLU, a_bit=target_bit, arg_dict=arg_dict))
        # Linear layer
        self.classifier = PCQLinear(num_features, num_classes, is_classifier=True,
                                    w_bit=bit_classifier, a_bit=bit_classifier, arg_dict=arg_dict)

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            self._update_input_ranges(x)
            if self.runtime_helper.apply_fake_quantization:
                x = self._fake_quantize_input(x)

        out = self.features.first_conv(x)
        out = self.features.first_norm(out, self.features.denseblock1.act_range)
        out = self.features.maxpool(out)
        out = self.features.denseblock1(out)
        out = self.features.transition1(out, self.features.denseblock2.act_range)
        out = self.features.denseblock2(out)
        out = self.features.transition2(out, self.features.denseblock3.act_range)
        out = self.features.denseblock3(out)
        out = self.features.transition3(out, self.features.denseblock4.act_range)
        out = self.features.denseblock4(out)
        out = self.features.last_norm(out)

        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out

    @torch.no_grad()
    def _update_input_ranges(self, x):
        cluster = self.runtime_helper.qat_batch_cluster
        data = x.view(x.size(0), -1)
        _min = data.min(dim=1).values.mean()
        _max = data.max(dim=1).values.mean()
        if self.apply_ema[cluster]:
            self.in_range[cluster][0] = self.in_range[cluster][0] * self.smooth + _min * (1 - self.smooth)
            self.in_range[cluster][1] = self.in_range[cluster][1] * self.smooth + _max * (1 - self.smooth)
        else:
            self.in_range[cluster][0], self.in_range[cluster][1] = _min, _max
            self.apply_ema[cluster] = True

    def _fake_quantize_input(self, x):
        cluster = self.runtime_helper.qat_batch_cluster
        s, z = calc_qparams(self.in_range[cluster][0], self.in_range[cluster][1], self.in_bit, self.runtime_helper.fzero)
        return fake_quantize(x, s, z, self.in_bit)

    def set_quantization_params(self):
        self.scale, self.zero_point = calc_qparams_per_cluster(self.in_range, self.in_bit)
        conv_s, conv_z = self.features.first_conv.set_qparams(self.scale, self.zero_point)
        block1_s, block1_z = self.features.denseblock1.set_block_qparams()
        block2_s, block2_z = self.features.denseblock2.set_block_qparams()
        block3_s, block3_z = self.features.denseblock3.set_block_qparams()
        block4_s, block4_z = self.features.denseblock4.set_block_qparams()
        self.features.first_norm.set_qparams(conv_s, conv_z, block1_s, block1_z)
        self.features.transition1.set_transition_qparams(block1_s, block1_z, block2_s, block2_z)
        self.features.transition2.set_transition_qparams(block2_s, block2_z, block3_s, block3_z)
        self.features.transition3.set_transition_qparams(block3_s, block3_z, block4_s, block4_z)
        prev_s, prev_z = self.features.last_norm.set_qparams(block4_s, block4_z)
        self.classifier.set_qparams(prev_s, prev_z)


def pcq_densenet(arg_dict: dict, **kwargs):
    return PCQDenseNet(32, (6, 12, 24, 16), 64, arg_dict, **kwargs)


def set_pcq_densenet(fused, pre):
    n = fused.arg_dict['cluster']
    momentum = fused.arg_dict['bn_momentum']
    # first conv & norm
    fused.features.first_conv = copy_weight_from_pretrained(fused.features.first_conv, pre.features.conv0)
    fused.features.first_norm = copy_pcq_bn_from_pretrained(fused.features.first_norm, pre.features.norm0, n, momentum)

    # dense block & Transition
    for block_idx in range(1,5):
        fused_block = getattr(fused.features, 'denseblock%d' % block_idx)
        pre_block = getattr(pre.features, 'denseblock%d' % block_idx)
        if block_idx < 4:
            fused_trans = getattr(fused.features, 'transition%d' % block_idx)
            pre_trans = getattr(pre.features, 'transition%d' % block_idx)

        # dense layer
        for layer_idx in range(1, fused_block.num_layers+1):
            fused_layer = getattr(fused_block,'denselayer%d' % layer_idx)
            pre_layer = getattr(pre_block,'denselayer%d' % layer_idx)
            fused_layer.conv1 = copy_weight_from_pretrained(fused_layer.conv1, pre_layer.conv1)
            fused_layer.conv2 = copy_weight_from_pretrained(fused_layer.conv2, pre_layer.conv2)
            fused_layer.bn1 = copy_pcq_bn_from_pretrained(fused_layer.bn1, pre_layer.norm1, n, momentum)
            fused_layer.bn2 = copy_pcq_bn_from_pretrained(fused_layer.bn2, pre_layer.norm2, n, momentum)

        # transition
        if block_idx < 4:
            fused_trans.conv = copy_weight_from_pretrained(fused_trans.conv, pre_trans.conv)
            fused_trans.bn = copy_pcq_bn_from_pretrained(fused_trans.bn, pre_trans.norm, n, momentum)
    # Last BatchNorm
    fused.features.last_norm = copy_pcq_bn_from_pretrained(fused.features.last_norm, pre.features.norm5, n, momentum)

    # Classifier
    fused.classifier = copy_from_pretrained(fused.classifier, pre.classifier)
    return fused


# def fold_pcq_densenet(model):
#     # first norm
#     model.features.first_norm.fold_norms()
#     # dense block & Transition
#     for block_idx in range(1, 5):
#         block = getattr(model.features, 'denseblock%d' % block_idx)
#         if block_idx < 4:
#             trans = getattr(model.features, 'transition%d' % block_idx)
#         # dense layer
#         for layer_idx in range(1, block.num_layers + 1):
#             fused_layer = getattr(block, 'denselayer%d' % layer_idx)
#             fused_layer.bn1.fold_norms()
#             fused_layer.bn2.fold_norms()
#         # transition
#         if block_idx < 4:
#             trans.norm.fold_norms()
#     # Last BatchNorm
#     model.features.last_norm.fold_norms()
#     return model
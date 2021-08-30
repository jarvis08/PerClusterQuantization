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
    ) -> None:
        super(PCQDenseLayer, self).__init__()
        self.arg_dict = arg_dict

        self.bit, self.smooth, self.num_clusters, self.runtime_helper, self.use_ste, self.quant_noise, self.qn_prob \
            = itemgetter('bit', 'smooth', 'cluster', 'runtime_helper', 'ste', 'quant_noise', 'qn_prob')(arg_dict)
        self.q_max = 2 ** self.bit - 1
        self.activation_qmax = 2 ** 16 - 1

        self.act_range = nn.Parameter(torch.zeros(self.num_clusters, 2), requires_grad=False)
        self.apply_ema = np.zeros(self.num_clusters, dtype=bool)

        self.bn1 = PCQBnReLU(num_input_features, nn.ReLU, arg_dict=arg_dict)
        self.conv1 = PCQConv2d(num_input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=False,
                               arg_dict=arg_dict, act_qmax=self.activation_qmax)
        self.bn2 = PCQBnReLU(bn_size * growth_rate, nn.ReLU, arg_dict=arg_dict)
        self.conv2 = PCQConv2d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False,
                               arg_dict=arg_dict, act_qmax=self.activation_qmax)
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
        _, _ = self.conv2.set_qparams(prev_s, prev_z, s1, z1)


class PCQTransition(nn.Sequential):
    def __init__(self, arg_dict, num_input_features: int, num_output_features: int) -> None:
        super(PCQTransition, self).__init__()
        self.arg_dict = arg_dict
        self.bit, self.smooth, self.num_clusters, self.runtime_helper, self.use_ste, self.quant_noise, self.qn_prob \
            = itemgetter('bit', 'smooth', 'cluster', 'runtime_helper', 'ste', 'quant_noise', 'qn_prob')(arg_dict)
        self.q_max = 2 ** self.bit - 1
        self.activation_qmax = 2 ** 16 - 1

        self.bn = PCQBnReLU(num_input_features, nn.ReLU, arg_dict=arg_dict)
        self.conv = PCQConv2d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False,
                              arg_dict=arg_dict, act_qmax=self.activation_qmax)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x, next_block_range):
        out = self.bn(x)
        out = self.conv(out, next_block_range)
        out = self.pool(out)
        return out

    def set_transition_qparams(self, s1, z1, next_block_s, next_block_z):
        prev_s, prev_z = self.bn.set_qparams(s1, z1)
        _, _ = self.conv.set_qparams(prev_s, prev_z, next_block_s, next_block_z)
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
    ) -> None:
        super(PCQDenseBlock, self).__init__()
        self.arg_dict = arg_dict
        self.num_layers = num_layers
        self.bit, self.smooth, self.num_clusters, self.runtime_helper, self.use_ste, self.quant_noise, self.qn_prob \
            = itemgetter('bit', 'smooth', 'cluster', 'runtime_helper', 'ste', 'quant_noise', 'qn_prob')(arg_dict)
        self.q_max = 2 ** self.bit - 1
        self.activation_qmax = 2 ** 16 - 1

        self.act_range = nn.Parameter(torch.zeros(self.num_clusters, 2), requires_grad=False)
        self.apply_ema = np.zeros(self.num_clusters, dtype=bool)

        for i in range(num_layers):
            layer = PCQDenseLayer(
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
            new_features = layer(features, self.act_range)
            features.append(new_features)
        out = torch.cat(features, 1)

        if not self.runtime_helper.pcq_initialized:
            done = 0
            for c in range(self.num_clusters):
                if self.apply_ema[c]:
                    self.act_range[c][0], self.act_range[c][1] = ema(out[done:done + 8], self.act_range[c], self.smooth)
                else:
                    self.act_range[c][0] = torch.min(out[done:done + 8]).item()
                    self.act_range[c][1] = torch.max(out[done:done + 8]).item()
                    self.apply_ema[c] = True
                done += 8
            return out

        if not self.training:
            return out

        # if self.runtime_helper.apply_fake_quantization and self.use_ste:
        #     _out = torch.zeros(out.shape).cuda()
        # else:
        #     _out = out
        _out = out

        done = 0
        for i in range(self.runtime_helper.batch_cluster.shape[0]):
            c = self.runtime_helper.batch_cluster[i][0].item()
            n = self.runtime_helper.batch_cluster[i][1].item()
            if self.apply_ema[c]:
                self.act_range[c][0], self.act_range[c][1] = ema(out[done:done + n], self.act_range[c], self.smooth)
            else:
                self.act_range[c][0] = torch.min(out).item()
                self.act_range[c][1] = torch.max(out).item()
                self.apply_ema[c] = True
            done += n
        return _out

    def set_block_qparams(self, prev_s, prev_z):
        self.s3 = nn.Parameter(torch.zeros(self.num_clusters, dtype=torch.float32), requires_grad=False)
        self.z3 = nn.Parameter(torch.zeros(self.num_clusters, dtype=torch.int32), requires_grad=False)
        for c in range(self.num_clusters):
            self.s3[c], self.z3[c] = calc_qparams(self.act_range[c][0], self.act_range[c][1], self.activation_qmax)

        for name, layer in self.items():
            prev_s, prev_z = layer.set_layer_qparams(prev_s, prev_z)
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
        self.bit, self.smooth, self.num_clusters, self.runtime_helper, self.use_ste, self.quant_noise, self.qn_prob \
            = itemgetter('bit', 'smooth', 'cluster', 'runtime_helper', 'ste', 'quant_noise', 'qn_prob')(arg_dict)
        self.q_max = 2 ** self.bit - 1
        self.activation_qmax = 2 ** 16 - 1
        self.in_range = nn.Parameter(torch.zeros(self.num_clusters, 2), requires_grad=False)

        self.apply_ema = np.zeros(self.num_clusters, dtype=bool)

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('first_conv', PCQConv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False,
                                     arg_dict=arg_dict, act_qmax=self.activation_qmax)),
            ('first_norm', PCQBnReLU(num_init_features, nn.ReLU, act_qmax=self.activation_qmax, arg_dict=arg_dict)),
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
        self.features.add_module('last_norm', PCQBnReLU(num_features, nn.ReLU, arg_dict=arg_dict))
        # Linear layer
        self.classifier = PCQLinear(num_features, num_classes, arg_dict=arg_dict)

    def forward(self, x: Tensor) -> Tensor:
        if not self.runtime_helper.pcq_initialized:
            done = 0
            for c in range(self.num_clusters):
                if self.apply_ema[c]:
                    self.in_range[c][0], self.in_range[c][1] = ema(x[done:done + 8], self.in_range[c], self.smooth)
                else:
                    self.in_range[c][0] = torch.min(x[done:done + 8]).item()
                    self.in_range[c][1] = torch.max(x[done:done + 8]).item()
                    self.apply_ema[c] = True
                done += 8

        elif self.training:
            done = 0
            for i in range(self.runtime_helper.batch_cluster.shape[0]):
                c = self.runtime_helper.batch_cluster[i][0].item()
                n = self.runtime_helper.batch_cluster[i][1].item()
                if self.apply_ema[c]:
                    self.in_range[c][0], self.in_range[c][1] = ema(x[done:done + n], self.in_range[c], self.smooth)
                    if self.runtime_helper.apply_fake_quantization:
                        s, z = calc_qparams(self.in_range[c][0], self.in_range[c][1], self.q_max)
                        x[done:done + n] = fake_quantize(x[done:done + n], s, z, self.q_max)
                else:
                    self.in_range[c][0] = torch.min(x).item()
                    self.in_range[c][1] = torch.max(x).item()
                    self.apply_ema[c] = True
                done += n

        # out = self.features(x)
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

    def set_quantization_params(self):
        self.scale = nn.Parameter(torch.zeros(self.num_clusters, dtype=torch.float32), requires_grad=False)
        self.zero_point = nn.Parameter(torch.zeros(self.num_clusters, dtype=torch.int32), requires_grad=False)
        for c in range(self.num_clusters):
            self.scale[c], self.zero_point[c] = calc_qparams(self.in_range[c][0], self.in_range[c][1], self.q_max)
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
        _, _ = self.classifier.set_qparams(prev_s, prev_z)


def pcq_densenet(arg_dict: dict, **kwargs):
    return PCQDenseNet(32, (6, 12, 24, 16), 64, arg_dict, **kwargs)


def set_pcq_densenet(fused, pre):
    # first conv & norm
    fused.features.first_conv = copy_from_pretrained(fused.features.first_conv, pre.features.conv0)
    fused.features.first_norm = copy_bn_from_pretrained(fused.features.first_norm, pre.features.norm0)
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
            fused_layer.bn1 = copy_bn_from_pretrained(fused_layer.bn1, pre_layer.norm1)
            fused_layer.conv1 = copy_from_pretrained(fused_layer.conv1, pre_layer.conv1)
            fused_layer.bn2 = copy_bn_from_pretrained(fused_layer.bn2, pre_layer.norm2)
            fused_layer.conv2 = copy_from_pretrained(fused_layer.conv2, pre_layer.conv2)

        # transition
        if block_idx < 4:
            fused_trans.bn = copy_bn_from_pretrained(fused_trans.bn, pre_trans.norm)
            fused_trans.conv = copy_from_pretrained(fused_trans.conv, pre_trans.conv)
    # Last BatchNorm
    fused.features.last_norm = copy_bn_from_pretrained(fused.features.last_norm, pre.features.norm5)
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

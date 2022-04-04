"""
    Quantized ResNet for ImageNet-1K, implemented in PyTorch.
    Original paper: 'Deep Residual Learning for Image Recognition,' https://arxiv.org/abs/1512.03385.
"""

import torch
import torch.nn as nn
import copy
from ..quantization_utils.quant_modules import *
from pytorchcv.models.common import ConvBlock
from pytorchcv.models.shufflenetv2 import ShuffleUnit, ShuffleInitBlock
import time
import logging


class Q_ResNet18(nn.Module):
    """
        Quantized ResNet50 model from 'Deep Residual Learning for Image Recognition,' https://arxiv.org/abs/1512.03385.
    """
    def __init__(self, model):
        super().__init__()
        features = getattr(model, 'features')
        init_block = getattr(features, 'init_block')

        self.quant_input = QuantAct()

        self.quant_init_block_convbn = QuantBnConv2d()
        self.quant_init_block_convbn.set_param(init_block.conv.conv, init_block.conv.bn)

        self.quant_act_int32 = QuantAct()

        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.act = nn.ReLU()

        self.channel = [2, 2, 2, 2]

        for stage_num in range(0, 4):
            stage = getattr(features, "stage{}".format(stage_num + 1))
            for unit_num in range(0, self.channel[stage_num]):
                unit = getattr(stage, "unit{}".format(unit_num + 1))
                quant_unit = Q_ResBlockBn()
                quant_unit.set_param(unit)
                setattr(self, f"stage{stage_num + 1}.unit{unit_num + 1}", quant_unit)

        self.final_pool = QuantAveragePool2d(kernel_size=7, stride=1)

        self.quant_act_output = QuantAct()

        output = getattr(model, 'output')
        self.quant_output = QuantLinear()
        self.quant_output.set_param(output)

    def forward(self, x):
        x, act_scaling_factor = self.quant_input(x)

        x, weight_scaling_factor = self.quant_init_block_convbn(x, act_scaling_factor)

        x = self.pool(x)
        x, act_scaling_factor = self.quant_act_int32(x, act_scaling_factor, weight_scaling_factor)

        x = self.act(x)

        for stage_num in range(0, 4):
            for unit_num in range(0, self.channel[stage_num]):
                tmp_func = getattr(self, f"stage{stage_num+1}.unit{unit_num+1}")
                x, act_scaling_factor = tmp_func(x, act_scaling_factor)

        x = self.final_pool(x, act_scaling_factor)

        x, act_scaling_factor = self.quant_act_output(x, act_scaling_factor)
        x = x.view(x.size(0), -1)
        x = self.quant_output(x, act_scaling_factor)

        return x


# class Q_ResNet20_unfold_Daq(nn.Module):
#     """
#         Quantized ResNet20 model for dataset CIFAR100, CIFAR10
#     """
#     def __init__(self, model, runtime_helper):
#         super().__init__()
#         features = getattr(model, 'features')
#         init_block = getattr(features, 'init_block')

#         self.quant_input = QuantAct_Daq(runtime_helper=runtime_helper)

#         self.quant_init_block_conv = QuantConv2d()
#         self.quant_init_block_conv.set_param(init_block.conv)
       
#         self.quant_init_block_conv_act = QuantAct_Daq(runtime_helper=runtime_helper)

#         self.quant_init_block_bn = QuantBn()
#         self.quant_init_block_bn.set_param(init_block.bn)

#         self.quant_act_int32 = QuantAct_Daq(runtime_helper=runtime_helper)

#         self.act = nn.ReLU()

#         self.channel = [3, 3, 3]

#         for stage_num in range(0, 3):
#             stage = getattr(features, "stage{}".format(stage_num + 1))
#             for unit_num in range(0, self.channel[stage_num]):
#                 unit = getattr(stage, 'unit{}'.format(unit_num + 1))
#                 quant_unit = Q_ResBlockBn_unfold_Daq()
#                 quant_unit.set_param(unit, runtime_helper)
#                 setattr(self, f'stage{stage_num + 1}.unit{unit_num + 1}', quant_unit)

#         self.final_pool = QuantAveragePool2d(kernel_size=8 , stride=1)

#         self.quant_act_output = QuantAct_Daq(runtime_helper=runtime_helper)

#         output = getattr(model, 'output')
#         self.quant_output = QuantLinear()
#         self.quant_output.is_classifier = True
#         self.quant_output.set_param(output)

#     def forward(self, x):
#         x, act_scaling_factor = self.quant_input(x)

#         x, conv_scaling_factor = self.quant_init_block_conv(x, act_scaling_factor)
#         x, act_scaling_factor = self.quant_init_block_conv_act(x, act_scaling_factor, conv_scaling_factor)

#         x, bn_scaling_factor = self.quant_init_block_bn(x, act_scaling_factor)
#         x, act_scaling_factor = self.quant_act_int32(x, act_scaling_factor, bn_scaling_factor)

#         x = self.act(x)

#         for stage_num in range(0,3):
#             for unit_num in range(0, self.channel[stage_num]):
#                 tmp_func = getattr(self, f'stage{stage_num + 1}.unit{unit_num + 1}')
#                 x, act_scaling_factor = tmp_func(x, act_scaling_factor)

#         x = self.final_pool(x, act_scaling_factor)

#         x, act_scaling_factor = self.quant_act_output(x, act_scaling_factor)
#         x = x.view(x.size(0), -1)
#         x = self.quant_output(x, act_scaling_factor)

#         return x


# class Q_ResNet20_unfold(nn.Module):
#     """
#         Quantized ResNet20 model for dataset CIFAR100, CIFAR10
#     """
#     def __init__(self, model):
#         super().__init__()
#         features = getattr(model, 'features')
#         init_block = getattr(features, 'init_block')

#         self.quant_input = QuantAct()

#         self.quant_init_block_conv = QuantConv2d()
#         self.quant_init_block_conv.set_param(init_block.conv)
       
#         self.quant_init_block_conv_act = QuantAct()

#         self.quant_init_block_bn = QuantBn()
#         self.quant_init_block_bn.set_param(init_block.bn)

#         self.quant_act_int32 = QuantAct()

#         self.act = nn.ReLU()

#         self.channel = [3, 3, 3]

#         for stage_num in range(0, 3):
#             stage = getattr(features, "stage{}".format(stage_num + 1))
#             for unit_num in range(0, self.channel[stage_num]):
#                 unit = getattr(stage, 'unit{}'.format(unit_num + 1))
#                 quant_unit = Q_ResBlockBn_unfold()
#                 quant_unit.set_param(unit)
#                 setattr(self, f'stage{stage_num + 1}.unit{unit_num + 1}', quant_unit)

#         self.final_pool = QuantAveragePool2d(kernel_size=8 , stride=1)

#         self.quant_act_output = QuantAct()

#         output = getattr(model, 'output')
#         self.quant_output = QuantLinear()
#         self.quant_output.is_classifier = True
#         self.quant_output.set_param(output)

#     def forward(self, x):
#         x, act_scaling_factor = self.quant_input(x)

#         x, conv_scaling_factor = self.quant_init_block_conv(x, act_scaling_factor)
#         x, act_scaling_factor = self.quant_init_block_conv_act(x, act_scaling_factor, conv_scaling_factor)

#         x, bn_scaling_factor = self.quant_init_block_bn(x, act_scaling_factor)
#         x, act_scaling_factor = self.quant_act_int32(x, act_scaling_factor, bn_scaling_factor)

#         x = self.act(x)

#         for stage_num in range(0,3):
#             for unit_num in range(0, self.channel[stage_num]):
#                 tmp_func = getattr(self, f'stage{stage_num + 1}.unit{unit_num + 1}')
#                 x, act_scaling_factor = tmp_func(x, act_scaling_factor)

#         x = self.final_pool(x, act_scaling_factor)

#         x, act_scaling_factor = self.quant_act_output(x, act_scaling_factor)
#         x = x.view(x.size(0), -1)
#         x = self.quant_output(x, act_scaling_factor)

#         return x


class Q_ResNet20_Daq(nn.Module):
    """
        Quantized ResNet20 model for dataset CIFAR100, CIFAR10
    """
    def __init__(self, model, runtime_helper):
        super().__init__()
        features = getattr(model, 'features')
        init_block = getattr(features, 'init_block')

        self.runtime_helper = runtime_helper

        self.quant_input = QuantAct_Daq(runtime_helper=runtime_helper)

        self.quant_init_block_convbn = QuantBnConv2d()
        self.quant_init_block_convbn.set_param(init_block.conv, init_block.bn)

        self.quant_act_int32 = QuantAct_Daq(runtime_helper=runtime_helper)

        self.act = nn.ReLU()

        self.channel = [3, 3, 3]

        for stage_num in range(0, 3):
            stage = getattr(features, "stage{}".format(stage_num + 1))
            for unit_num in range(0, self.channel[stage_num]):
                unit = getattr(stage, 'unit{}'.format(unit_num + 1))
                quant_unit = Q_ResBlockBn_Daq()
                quant_unit.set_param(unit, runtime_helper)
                setattr(self, f'stage{stage_num + 1}.unit{unit_num + 1}', quant_unit)

        self.final_pool = QuantAveragePool2d(kernel_size=8 , stride=1)

        self.quant_act_output = QuantAct_Daq(runtime_helper=runtime_helper)

        output = getattr(model, 'output')
        self.quant_output = QuantLinear()
        self.quant_output.is_classifier = True
        self.quant_output.set_param(output)

    def forward(self, x):
        x, act_scaling_factor = self.quant_input(x)

        x, weight_scaling_factor = self.quant_init_block_convbn(x, act_scaling_factor)
        x, act_scaling_factor = self.quant_act_int32(x, act_scaling_factor, weight_scaling_factor)

        x = self.act(x)

        for stage_num in range(0,3):
            for unit_num in range(0, self.channel[stage_num]):
                tmp_func = getattr(self, f'stage{stage_num + 1}.unit{unit_num + 1}')
                x, act_scaling_factor = tmp_func(x, act_scaling_factor)

        x, act_scaling_factor = self.final_pool(x, act_scaling_factor)

        x, act_scaling_factor = self.quant_act_output(x, act_scaling_factor)
        x = x.view(x.size(0), -1)
        x = self.quant_output(x, act_scaling_factor)

        return x

    def initialize_counter(self, x, n_clusters):
        self.zero_counter = []

        self.features = nn.Sequential(self.quant_init_block_convbn, self.act)
        
        x = self.features(x)
        n_features = x.view(-1).size(0)
        self.zero_counter.append(torch.zeros((n_clusters, n_features), device='cuda'))

        for stage_num in range(0,3):
            for unit_num in range(0, self.channel[stage_num]):
                tmp_func = getattr(self, f'stage{stage_num + 1}.unit{unit_num + 1}')
                x = tmp_func.initialize_counter(x, n_clusters, self.zero_counter)


    # def count_zeros_per_index(self, x, cluster, n_clusters):
    #     x = self.quant_input(x)
    #     x = self.quant_init_block_convbn(x)
    #     x = self.quant_act_int32(x)

    #     initialized = True
    #     if not hasattr(self, 'zero_counter'):
    #         initialized = False
    #         n_features = x.view(-1).size(0)
    #         self.zero_counter = []
    #         self.zero_counter.append(torch.zeros((n_clusters, n_features), device='cuda'))

    #     l_idx = 0
    #     n_features = self.zero_counter[l_idx].size(1)
    #     for i in range(x.size(0)):
    #         flattened = x[i].view(-1)
    #         zeros_idx = (flattened == 0.0).nonzero(as_tuple=True)[0]
    #         zeros_idx %= n_features
    #         self.zero_counter[l_idx][cluster, zeros_idx] += 1

    #     for stage_num in range(0,3):
    #         for unit_num in range(0, self.channel[stage_num]):
    #             tmp_func = getattr(self, f'stage{stage_num + 1}.unit{unit_num + 1}')
    #             x, l_idx = tmp_func.count_zeros_per_index(x, cluster, n_clusters, self.zero_counter, l_idx, initialized)

    def toggle_full_precision(self):
        print('Model Toggle full precision FUNC')
        for module in self.modules():
            if isinstance(module, (QuantAct_Daq, QuantLinear, QuantBnConv2d)):
                precision = getattr(module, 'full_precision_flag')
                if precision:
                    precision = False
                else:
                    precision = True
                setattr(module, 'full_precision_flag', precision)


class Q_ResNet20(nn.Module):
    """
        Quantized ResNet20 model for dataset CIFAR100, CIFAR10
    """
    def __init__(self, model):
        super().__init__()
        features = getattr(model, 'features')
        init_block = getattr(features, 'init_block')

        self.quant_input = QuantAct()

        self.quant_init_block_convbn = QuantBnConv2d()
        self.quant_init_block_convbn.set_param(init_block.conv, init_block.bn)

        self.quant_act_int32 = QuantAct()

        self.act = nn.ReLU()

        self.channel = [3, 3, 3]

        for stage_num in range(0, 3):
            stage = getattr(features, "stage{}".format(stage_num + 1))
            for unit_num in range(0, self.channel[stage_num]):
                unit = getattr(stage, 'unit{}'.format(unit_num + 1))
                quant_unit = Q_ResBlockBn()
                quant_unit.set_param(unit)
                setattr(self, f'stage{stage_num + 1}.unit{unit_num + 1}', quant_unit)

        self.final_pool = QuantAveragePool2d(kernel_size=8, stride=1)

        self.quant_act_output = QuantAct()

        output = getattr(model, 'output')
        self.quant_output = QuantLinear()
        self.quant_output.is_classifier = True
        self.quant_output.set_param(output)

    def forward(self, x):
        x, act_scaling_factor = self.quant_input(x)

        x, weight_scaling_factor = self.quant_init_block_convbn(x, act_scaling_factor)
        x, act_scaling_factor = self.quant_act_int32(x, act_scaling_factor, weight_scaling_factor)

        x = self.act(x)

        for stage_num in range(0,3):
            for unit_num in range(0, self.channel[stage_num]):
                tmp_func = getattr(self, f'stage{stage_num + 1}.unit{unit_num + 1}')
                x, act_scaling_factor = tmp_func(x, act_scaling_factor)

        x, act_scaling_factor = self.final_pool(x, act_scaling_factor)

        x, act_scaling_factor = self.quant_act_output(x, act_scaling_factor)
        x = x.view(x.size(0), -1)
        x = self.quant_output(x, act_scaling_factor)

        return x


class Q_ResNet50(nn.Module):
    """
        Quantized ResNet50 model from 'Deep Residual Learning for Image Recognition,' https://arxiv.org/abs/1512.03385.
    """
    def __init__(self, model):
        super().__init__()
        features = getattr(model, 'features')
        init_block = getattr(features, 'init_block')

        self.quant_input = QuantAct()

        self.quant_init_convbn = QuantBnConv2d()
        self.quant_init_convbn.set_param(init_block.conv.conv, init_block.conv.bn)

        self.quant_act_int32 = QuantAct()

        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.act = nn.ReLU()

        self.channel = [3, 4, 6, 3]

        for stage_num in range(0, 4):
            stage = getattr(features, "stage{}".format(stage_num + 1))
            for unit_num in range(0, self.channel[stage_num]):
                unit = getattr(stage, "unit{}".format(unit_num + 1))
                quant_unit = Q_ResUnitBn()
                quant_unit.set_param(unit)
                setattr(self, f"stage{stage_num + 1}.unit{unit_num + 1}", quant_unit)

        self.final_pool = QuantAveragePool2d(output=(1, 1))

        self.quant_act_output = QuantAct()

        output = getattr(model, 'output')
        self.quant_output = QuantLinear()
        self.quant_output.is_classifier = True
        self.quant_output.set_param(output)

    def forward(self, x):
        x, act_scaling_factor = self.quant_input(x)

        x, weight_scaling_factor = self.quant_init_convbn(x, act_scaling_factor)
        
        x = self.pool(x)
        x, act_scaling_factor = self.quant_act_int32(x, act_scaling_factor, weight_scaling_factor)
        
        x = self.act(x)

        for stage_num in range(0, 4):
            for unit_num in range(0, self.channel[stage_num]):
                tmp_func = getattr(self, f"stage{stage_num+1}.unit{unit_num+1}")
                x, act_scaling_factor = tmp_func(x, act_scaling_factor)

        x, act_scaling_factor = self.final_pool(x, act_scaling_factor)

        x, act_scaling_factor = self.quant_act_output(x, act_scaling_factor)
        x = x.view(x.size(0), -1)
        x = self.quant_output(x, act_scaling_factor)

        return x

class Q_ResNet50_Daq(nn.Module):
    """
        Quantized ResNet50 model from 'Deep Residual Learning for Image Recognition,' https://arxiv.org/abs/1512.03385.
    """
    def __init__(self, model, runtime_helper=None):
        super().__init__()

        features = getattr(model, 'features')
        init_block = getattr(features, 'init_block')

        self.runtime_helper = runtime_helper

        self.quant_input = QuantAct_Daq(runtime_helper=runtime_helper)

        self.quant_init_convbn = QuantBnConv2d()
        self.quant_init_convbn.set_param(init_block.conv.conv, init_block.conv.bn)

        self.quant_act_int32 = QuantAct_Daq(runtime_helper=runtime_helper)

        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.act = nn.ReLU()

        self.channel = [3, 4, 6, 3]

        for stage_num in range(0, 4):
            stage = getattr(features, "stage{}".format(stage_num + 1))
            for unit_num in range(0, self.channel[stage_num]):
                unit = getattr(stage, "unit{}".format(unit_num + 1))
                quant_unit = Q_ResUnitBn_Daq()
                quant_unit.set_param(unit, runtime_helper)
                setattr(self, f"stage{stage_num + 1}.unit{unit_num + 1}", quant_unit)

        self.final_pool = QuantAveragePool2d(output=(1, 1))

        self.quant_act_output = QuantAct_Daq(runtime_helper=runtime_helper)

        output = getattr(model, 'output')
        self.quant_output = QuantLinear()
        self.quant_output.is_classifier = True
        self.quant_output.set_param(output)

    def forward(self, x):
        x, act_scaling_factor = self.quant_input(x)

        x, weight_scaling_factor = self.quant_init_convbn(x, act_scaling_factor)

        x = self.pool(x)
        x, act_scaling_factor = self.quant_act_int32(x, act_scaling_factor, weight_scaling_factor)

        x = self.act(x)

        for stage_num in range(0, 4):
            for unit_num in range(0, self.channel[stage_num]):
                tmp_func = getattr(self, f"stage{stage_num+1}.unit{unit_num+1}")
                x, act_scaling_factor = tmp_func(x, act_scaling_factor)

        x, act_scaling_factor = self.final_pool(x, act_scaling_factor)

        x, act_scaling_factor = self.quant_act_output(x, act_scaling_factor)
        x = x.view(x.size(0), -1)
        x = self.quant_output(x, act_scaling_factor)

        return x


# class Q_ResNet101(nn.Module):
#     """
#        Quantized ResNet101 model from 'Deep Residual Learning for Image Recognition,' https://arxiv.org/abs/1512.03385.
#     """
#     def __init__(self, model):
#         super().__init__()

#         features = getattr(model, 'features')

#         init_block = getattr(features, 'init_block')
#         self.quant_input = QuantAct()
#         self.quant_init_convbn = QuantBnConv2d()
#         self.quant_init_convbn.set_param(init_block.conv.conv, init_block.conv.bn)

#         self.quant_act_int32 = QuantAct()

#         self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         self.act = nn.ReLU()

#         self.channel = [3, 4, 23, 3]

#         for stage_num in range(0, 4):
#             stage = getattr(features, "stage{}".format(stage_num + 1))
#             for unit_num in range(0, self.channel[stage_num]):
#                 unit = getattr(stage, "unit{}".format(unit_num + 1))
#                 quant_unit = Q_ResUnitBn()
#                 quant_unit.set_param(unit)
#                 setattr(self, f"stage{stage_num + 1}.unit{unit_num + 1}", quant_unit)

#         self.final_pool = QuantAveragePool2d(kernel_size=7, stride=1)

#         self.quant_act_output = QuantAct()

#         output = getattr(model, 'output')
#         self.quant_output = QuantLinear()
#         self.quant_output.set_param(output)

#     def forward(self, x):
#         x, act_scaling_factor = self.quant_input(x)

#         x, weight_scaling_factor = self.quant_init_convbn(x, act_scaling_factor)

#         x = self.pool(x)
#         x, act_scaling_factor = self.quant_act_int32(x, act_scaling_factor, weight_scaling_factor, None, None)

#         x = self.act(x)

#         for stage_num in range(0, 4):
#             for unit_num in range(0, self.channel[stage_num]):
#                 tmp_func = getattr(self, f"stage{stage_num+1}.unit{unit_num+1}")
#                 x, act_scaling_factor = tmp_func(x, act_scaling_factor)

#         x = self.final_pool(x, act_scaling_factor)

#         x, act_scaling_factor = self.quant_act_output(x, act_scaling_factor)
#         x = x.view(x.size(0), -1)
#         x = self.quant_output(x, act_scaling_factor)

#         return x


class Q_ResUnitBn_Daq(nn.Module):
    """
       Quantized ResNet unit with residual path.
       Applying DAQ.
    """
    def __init__(self):
        super(Q_ResUnitBn_Daq, self).__init__()

    def set_param(self, unit, runtime_helper):
        self.resize_identity = unit.resize_identity

        self.quant_act = QuantAct_Daq(runtime_helper=runtime_helper)

        convbn1 = unit.body.conv1
        self.quant_convbn1 = QuantBnConv2d()
        self.quant_convbn1.set_param(convbn1.conv, convbn1.bn)
        self.quant_act1 = QuantAct_Daq(runtime_helper=runtime_helper)

        convbn2 = unit.body.conv2
        self.quant_convbn2 = QuantBnConv2d()
        self.quant_convbn2.set_param(convbn2.conv, convbn2.bn)
        self.quant_act2 = QuantAct_Daq(runtime_helper=runtime_helper)

        convbn3 = unit.body.conv3
        self.quant_convbn3 = QuantBnConv2d()
        self.quant_convbn3.set_param(convbn3.conv, convbn3.bn)

        if self.resize_identity:
            self.quant_identity_convbn = QuantBnConv2d()
            self.quant_identity_convbn.set_param(unit.identity_conv.conv, unit.identity_conv.bn)

        self.quant_act_int32 = QuantAct_Daq(runtime_helper=runtime_helper)

    def forward(self, x, scaling_factor_int32=None):
        # forward using the quantized modules
        if self.resize_identity:
            x, act_scaling_factor = self.quant_act(x, scaling_factor_int32)
            identity_act_scaling_factor = act_scaling_factor.clone()
            identity, identity_weight_scaling_factor = self.quant_identity_convbn(x, act_scaling_factor)
        else:
            identity = x
            x, act_scaling_factor = self.quant_act(x, scaling_factor_int32)

        x, weight_scaling_factor = self.quant_convbn1(x, act_scaling_factor)
        x = nn.ReLU()(x)
        x, act_scaling_factor = self.quant_act1(x, act_scaling_factor, weight_scaling_factor)

        x, weight_scaling_factor = self.quant_convbn2(x, act_scaling_factor)
        x = nn.ReLU()(x)
        x, act_scaling_factor = self.quant_act2(x, act_scaling_factor, weight_scaling_factor)

        x, weight_scaling_factor = self.quant_convbn3(x, act_scaling_factor)

        x = x + identity

        if self.resize_identity:
            x, act_scaling_factor = self.quant_act_int32(x, act_scaling_factor, weight_scaling_factor, identity, identity_act_scaling_factor, identity_weight_scaling_factor)
        else:
            x, act_scaling_factor = self.quant_act_int32(x, act_scaling_factor, weight_scaling_factor, identity, scaling_factor_int32, None)

        x = nn.ReLU()(x)

        return x, act_scaling_factor

class Q_ResUnitBn(nn.Module):
    """
       Quantized ResNet unit with residual path.
    """
    def __init__(self):
        super(Q_ResUnitBn, self).__init__()

    def set_param(self, unit):
        self.resize_identity = unit.resize_identity

        self.quant_act = QuantAct()

        convbn1 = unit.body.conv1
        self.quant_convbn1 = QuantBnConv2d()
        self.quant_convbn1.set_param(convbn1.conv, convbn1.bn)
        self.quant_act1 = QuantAct()

        convbn2 = unit.body.conv2
        self.quant_convbn2 = QuantBnConv2d()
        self.quant_convbn2.set_param(convbn2.conv, convbn2.bn)
        self.quant_act2 = QuantAct()

        convbn3 = unit.body.conv3
        self.quant_convbn3 = QuantBnConv2d()
        self.quant_convbn3.set_param(convbn3.conv, convbn3.bn)

        if self.resize_identity:
            self.quant_identity_convbn = QuantBnConv2d()
            self.quant_identity_convbn.set_param(unit.identity_conv.conv, unit.identity_conv.bn)

        self.quant_act_int32 = QuantAct()

    def forward(self, x, scaling_factor_int32=None):
        # forward using the quantized modules
        if self.resize_identity:
            x, act_scaling_factor = self.quant_act(x, scaling_factor_int32)
            identity_act_scaling_factor = act_scaling_factor.clone()
            identity, identity_weight_scaling_factor = self.quant_identity_convbn(x, act_scaling_factor)
        else:
            identity = x
            x, act_scaling_factor = self.quant_act(x, scaling_factor_int32)

        x, weight_scaling_factor = self.quant_convbn1(x, act_scaling_factor)
        x = nn.ReLU()(x)
        x, act_scaling_factor = self.quant_act1(x, act_scaling_factor, weight_scaling_factor)

        x, weight_scaling_factor = self.quant_convbn2(x, act_scaling_factor)
        x = nn.ReLU()(x)
        x, act_scaling_factor = self.quant_act2(x, act_scaling_factor, weight_scaling_factor)

        x, weight_scaling_factor = self.quant_convbn3(x, act_scaling_factor)

        x = x + identity

        if self.resize_identity:
            x, act_scaling_factor = self.quant_act_int32(x, act_scaling_factor, weight_scaling_factor, identity, identity_act_scaling_factor, identity_weight_scaling_factor)
        else:
            x, act_scaling_factor = self.quant_act_int32(x, act_scaling_factor, weight_scaling_factor, identity, scaling_factor_int32, None)

        x = nn.ReLU()(x)

        return x, act_scaling_factor

class Q_ResBlockBn_unfold_Daq(nn.Module):
    """
        Quantized ResNet block with residual path.
    """
    def __init__(self):
        super(Q_ResBlockBn_unfold_Daq, self).__init__()

    def set_param(self, unit, runtime_helper):
        self.resize_identity = unit.resize_identity

        self.quant_act = QuantAct_Daq(runtime_helper=runtime_helper)

        convbn1 = unit.body.conv1
        self.quant_conv1 = QuantConv2d()
        self.quant_conv1.set_param(convbn1.conv)
        self.quant_conv1_act = QuantAct_Daq(runtime_helper=runtime_helper)

        self.quant_bn1 = QuantBn()
        self.quant_bn1.set_param(convbn1.bn)
        self.quant_act1 = QuantAct_Daq(runtime_helper=runtime_helper)

        convbn2 = unit.body.conv2
        self.quant_conv2 = QuantConv2d()
        self.quant_conv2.set_param(convbn2.conv)
        self.quant_conv2_act = QuantAct_Daq(runtime_helper=runtime_helper)

        self.quant_bn2 = QuantBn()
        self.quant_bn2.set_param(convbn2.bn)

        if self.resize_identity:
            self.quant_identity_conv = QuantConv2d()
            self.quant_identity_conv.set_param(unit.identity_conv.conv)
            self.quant_identity_act = QuantAct_Daq(runtime_helper=runtime_helper)
            self.quant_identity_bn = QuantBn()
            self.quant_identity_bn.set_param(unit.identity_conv.bn)

        self.quant_act_int32 = QuantAct_Daq(runtime_helper=runtime_helper)

    def forward(self, x, scaling_factor_int32=None):
        # forward using the quantized modules
        if self.resize_identity:
            x, act_scaling_factor = self.quant_act(x, scaling_factor_int32)
            identity_act_scaling_factor = act_scaling_factor.clone()
            identity, identity_conv_scaling_factor = self.quant_identity_conv(x, act_scaling_factor)
            identity, identity_act_scaling_factor = self.quant_identity_act(identity, identity_act_scaling_factor, identity_conv_scaling_factor) #
            identity, identity_weight_scaling_factor = self.quant_identity_bn(identity, act_scaling_factor)
        else:
            identity = x
            x, act_scaling_factor = self.quant_act(x, scaling_factor_int32)

        x, conv_scaling_factor = self.quant_conv1(x, act_scaling_factor)
        x, act_scaling_factor = self.quant_conv1_act(x, act_scaling_factor, conv_scaling_factor)
        x, bn_scaling_factor = self.quant_bn1(x, act_scaling_factor)
        
        x = nn.ReLU()(x)
        x, act_scaling_factor = self.quant_act1(x, act_scaling_factor, bn_scaling_factor)

        x, conv_scaling_factor = self.quant_conv2(x, act_scaling_factor)
        x, act_scaling_factor = self.quant_conv2_act(x, act_scaling_factor, conv_scaling_factor)
        x, bn_scaling_factor = self.quant_bn2(x, act_scaling_factor)

        x = x + identity

        if self.resize_identity:
            x, act_scaling_factor = self.quant_act_int32(x, act_scaling_factor, bn_scaling_factor, identity, identity_act_scaling_factor, identity_weight_scaling_factor)
        else:
            x, act_scaling_factor = self.quant_act_int32(x, act_scaling_factor, bn_scaling_factor, identity, scaling_factor_int32, None)

        x = nn.ReLU()(x)

        return x, act_scaling_factor

class Q_ResBlockBn_unfold(nn.Module):
    """
        Quantized ResNet block with residual path.
    """
    def __init__(self):
        super(Q_ResBlockBn_unfold, self).__init__()

    def set_param(self, unit):
        self.resize_identity = unit.resize_identity

        self.quant_act = QuantAct()

        convbn1 = unit.body.conv1
        self.quant_conv1 = QuantConv2d()
        self.quant_conv1.set_param(convbn1.conv)
        self.quant_conv1_act = QuantAct()

        self.quant_bn1 = QuantBn()
        self.quant_bn1.set_param(convbn1.bn)
        self.quant_act1 = QuantAct()

        convbn2 = unit.body.conv2
        self.quant_conv2 = QuantConv2d()
        self.quant_conv2.set_param(convbn2.conv)
        self.quant_conv2_act = QuantAct()

        self.quant_bn2 = QuantBn()
        self.quant_bn2.set_param(convbn2.bn)

        if self.resize_identity:
            self.quant_identity_conv = QuantConv2d()
            self.quant_identity_conv.set_param(unit.identity_conv.conv)
            self.quant_identity_act = QuantAct()
            self.quant_identity_bn = QuantBn()
            self.quant_identity_bn.set_param(unit.identity_conv.bn)

        self.quant_act_int32 = QuantAct()

    def forward(self, x, scaling_factor_int32=None):
        # forward using the quantized modules
        if self.resize_identity:
            x, act_scaling_factor = self.quant_act(x, scaling_factor_int32)
            identity_act_scaling_factor = act_scaling_factor.clone()
            identity, identity_conv_scaling_factor = self.quant_identity_conv(x, act_scaling_factor)
            identity, identity_act_scaling_factor = self.quant_identity_act(identity, identity_act_scaling_factor, identity_conv_scaling_factor) #
            identity, identity_weight_scaling_factor = self.quant_identity_bn(identity, act_scaling_factor)
        else:
            identity = x
            x, act_scaling_factor = self.quant_act(x, scaling_factor_int32)

        x, conv_scaling_factor = self.quant_conv1(x, act_scaling_factor)
        x, act_scaling_factor = self.quant_conv1_act(x, act_scaling_factor, conv_scaling_factor)
        x, bn_scaling_factor = self.quant_bn1(x, act_scaling_factor)
        
        x = nn.ReLU()(x)
        x, act_scaling_factor = self.quant_act1(x, act_scaling_factor, bn_scaling_factor)

        x, conv_scaling_factor = self.quant_conv2(x, act_scaling_factor)
        x, act_scaling_factor = self.quant_conv2_act(x, act_scaling_factor, conv_scaling_factor)
        x, bn_scaling_factor = self.quant_bn2(x, act_scaling_factor)

        x = x + identity

        if self.resize_identity:
            x, act_scaling_factor = self.quant_act_int32(x, act_scaling_factor, bn_scaling_factor, identity, identity_act_scaling_factor, identity_weight_scaling_factor)
        else:
            x, act_scaling_factor = self.quant_act_int32(x, act_scaling_factor, bn_scaling_factor, identity, scaling_factor_int32, None)

        x = nn.ReLU()(x)

        return x, act_scaling_factor

class Q_ResBlockBn_Daq(nn.Module):
    """
        Quantized ResNet block with residual path.
        Applying DAQ.
    """

    def __init__(self):
        super(Q_ResBlockBn_Daq, self).__init__()

    def set_param(self, unit, runtime_helper):
        self.resize_identity = unit.resize_identity

        self.quant_act = QuantAct_Daq(runtime_helper=runtime_helper)

        convbn1 = unit.body.conv1
        self.quant_convbn1 = QuantBnConv2d()
        self.quant_convbn1.set_param(convbn1.conv, convbn1.bn)

        self.act1 = nn.ReLU()

        self.quant_act1 = QuantAct_Daq(runtime_helper=runtime_helper)

        convbn2 = unit.body.conv2
        self.quant_convbn2 = QuantBnConv2d()
        self.quant_convbn2.set_param(convbn2.conv, convbn2.bn)

        if self.resize_identity:
            self.quant_identity_convbn = QuantBnConv2d()
            self.quant_identity_convbn.set_param(unit.identity_conv.conv, unit.identity_conv.bn)

        self.act2 = nn.ReLU()

        self.quant_act_int32 = QuantAct_Daq(runtime_helper=runtime_helper)

    def forward(self, x, scaling_factor_int32=None):
        # forward using the quantized modules
        if self.resize_identity:
            x, act_scaling_factor = self.quant_act(x, scaling_factor_int32)
            identity_act_scaling_factor = act_scaling_factor.clone()
            identity, identity_weight_scaling_factor = self.quant_identity_convbn(x, act_scaling_factor)
        else:
            identity = x
            x, act_scaling_factor = self.quant_act(x, scaling_factor_int32)

        x, weight_scaling_factor = self.quant_convbn1(x, act_scaling_factor)
        x = self.act1(x)
        x, act_scaling_factor = self.quant_act1(x, act_scaling_factor, weight_scaling_factor)

        x, weight_scaling_factor = self.quant_convbn2(x, act_scaling_factor)

        x = x + identity

        if self.resize_identity:
            x, act_scaling_factor = self.quant_act_int32(x, act_scaling_factor, weight_scaling_factor, identity,
                                                         identity_act_scaling_factor, identity_weight_scaling_factor)
        else:
            x, act_scaling_factor = self.quant_act_int32(x, act_scaling_factor, weight_scaling_factor, identity,
                                                         scaling_factor_int32, None)

        x = self.act2(x)

        return x, act_scaling_factor

    def initialize_counter(self, x, n_clusters, zero_counter):
        self.zero_counter = zero_counter

        if self.resize_identity:
            self.features = nn.Sequential(self.quant_identity_convbn,
                                          self.quant_convbn1, self.act1,
                                          self.quant_convbn2, self.act2)

            identity = self.features[0]
            x = self.features[1]
            x = self.features[2]

            n_features = x.view(-1).size(0)
            self.zero_counter.append(torch.zeros((n_clusters, n_features), device='cuda'))

            x = 



        else:
            self.features = nn.Sequential(self.quant_convbn1, self.act1, 
                                          self.quant_convbn2, self.act2)



        for i in range(len(se)):
            _from = 0 if i == 0 else indices[i - 1]
            _to = indices[i]
            x = self.features[_from:_to](x)
            n_features = x.view(-1).size(0)
            self.zero_counter.append(torch.zeros((n_clusters, n_features), device='cuda'))
        return x

    def count_zeros_per_index(self, x, cluster, n_clusters):
        if not hasattr(self, 'zero_counter'):  
            self.initialize_counter(x[0].unsqueeze(0), n_clusters)

        if self.resize_identity:
            indices = [1, 3]
            for layer_idx, layer in enumerate(self.features):
                if layer_idx is in indices:
        else:
            indices = [0, 2]
            identity = x

            x = self.features[0](x)
            x = self.features[1](x)

            n_features = self.zero_counter[conv_cnt].size(1)
            for idx in range(x.size(0)):
                flattened = x[idx].view(-1)
                zeros_idx = (flattened == 0.0).nonzero(as_tuple=True)[0]
                zeros_idx %= n_features
                self.zero_counter[conv_cnt][cluster, zeros_idx] += 1
        

    # def count_zeros_per_index(self, x, cluster, n_clusters, zero_counter, l_idx, initialized):
    #     # make empty list space
    #     if not initialized:
    #         _x = x[0].unsqueeze(0)
    #         if self.resize_identity:
    #             _x = self.quant_act(_x)
    #             identity = self.quant_identity_convbn(_x)
    #         else:
    #             identity = _x
    #             _x = self.quant_act(_x)

    #         _x = self.quant_convbn1(_x)
    #         _x = nn.ReLU()(_x)
    #         _x = self.quant_act1(_x)
    #         ###
    #         n_features = _x.view(-1).size(0)
    #         zero_counter.append(torch.zeros((n_clusters, n_features), device='cuda'))
    #         ###
    #         _x = self.quant_convbn2(_x)

    #         _x = _x + identity

    #         # if self.resize_identity:
    #         #     _x = self.quant_act_int32(_x)
    #         # else:
    #         #     _x = self.quant_act_int32(_x)
    #         _x = nn.ReLU()(_x)
    #         n_features = _x.view(-1).size(0)
    #         zero_counter.append(torch.zeros((n_clusters, n_features), device='cuda'))

    #     if self.resize_identity:
    #         x = self.quant_act(x)
    #         identity = self.quant_identity_convbn(x)
    #     else:
    #         identity = x
    #         x = self.quant_act(x)

    #     x = self.quant_convbn1(x)
    #     x = nn.ReLU()(x)
    #     x = self.quant_act1(x)
    #     ###
    #     l_idx += 1
    #     n_features = zero_counter[l_idx].size(1)
    #     for i in range(x.size(0)):
    #         flatten = x[i].view(-1)
    #         zeros_idx = (flatten == 0.0).nonzero(as_tuple=True)[0]
    #         zeros_idx %= n_features
    #         zero_counter[l_idx][cluster, zeros_idx] += 1
    #     ###

    #     x = self.quant_convbn2(x)

    #     x = x + identity

    #     # if self.resize_identity:
    #     #     x = self.quant_act_int32(x)
    #     # else:
    #     #     x = self.quant_act_int32(x)

    #     x = nn.ReLU()(x)
    #     ###
    #     l_idx += 1
    #     n_features = zero_counter[l_idx].size(1)
    #     for i in range(x.size(0)):
    #         flatten = x[i].view(-1)
    #         zeros_idx = (flatten == 0.0).nonzero(as_tuple=True)[0]
    #         zeros_idx %= n_features
    #         zero_counter[l_idx][cluster, zeros_idx] += 1
    #     ###
    #     return x, l_idx


class Q_ResBlockBn(nn.Module):
    """
        Quantized ResNet block with residual path.
    """
    def __init__(self):
        super(Q_ResBlockBn, self).__init__()

    def set_param(self, unit):
        self.resize_identity = unit.resize_identity

        self.quant_act = QuantAct()

        convbn1 = unit.body.conv1
        self.quant_convbn1 = QuantBnConv2d()
        self.quant_convbn1.set_param(convbn1.conv, convbn1.bn)

        self.quant_act1 = QuantAct()

        convbn2 = unit.body.conv2
        self.quant_convbn2 = QuantBnConv2d()
        self.quant_convbn2.set_param(convbn2.conv, convbn2.bn)

        if self.resize_identity:
            self.quant_identity_convbn = QuantBnConv2d()
            self.quant_identity_convbn.set_param(unit.identity_conv.conv, unit.identity_conv.bn)

        self.quant_act_int32 = QuantAct()

    def forward(self, x, scaling_factor_int32=None):
        # forward using the quantized modules
        if self.resize_identity:
            x, act_scaling_factor = self.quant_act(x, scaling_factor_int32)
            identity_act_scaling_factor = act_scaling_factor.clone()
            identity, identity_weight_scaling_factor = self.quant_identity_convbn(x, act_scaling_factor)
        else:
            identity = x
            x, act_scaling_factor = self.quant_act(x, scaling_factor_int32)

        x, weight_scaling_factor = self.quant_convbn1(x, act_scaling_factor)
        x = nn.ReLU()(x)
        x, act_scaling_factor = self.quant_act1(x, act_scaling_factor, weight_scaling_factor)

        x, weight_scaling_factor = self.quant_convbn2(x, act_scaling_factor)

        x = x + identity

        if self.resize_identity:
            x, act_scaling_factor = self.quant_act_int32(x, act_scaling_factor, weight_scaling_factor, identity, identity_act_scaling_factor, identity_weight_scaling_factor)
        else:
            x, act_scaling_factor = self.quant_act_int32(x, act_scaling_factor, weight_scaling_factor, identity, scaling_factor_int32, None)

        x = nn.ReLU()(x)

        return x, act_scaling_factor


def q_resnet18(model):
    net = Q_ResNet18(model)
    return net

def q_resnet20(model, runtime_helper=None):
    if runtime_helper is None:
        net = Q_ResNet20(model)
    else:
        net = Q_ResNet20_Daq(model, runtime_helper)
    return net

def q_resnet20_unfold(model, runtime_helper=None):
    if runtime_helper is None:
        net = Q_ResNet20_unfold(model)
    else:
        net = Q_ResNet20_unfold_Daq(model,runtime_helper)
    return net

def q_resnet50(model, runtime_helper=None):
    if runtime_helper is None:
        net = Q_ResNet50(model)
    else:
        net = Q_ResNet50_Daq(model, runtime_helper)
    return net

def q_resnet101(model):
    net = Q_ResNet101(model)
    return net

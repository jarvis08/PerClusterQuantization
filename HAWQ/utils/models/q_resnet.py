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

    def set_daq_helper(self, runtime_helper):
        self.runtime_helper = runtime_helper
        self.quant_input.runtime_helper = runtime_helper
        self.quant_init_block_convbn.runtime_helper = runtime_helper
        self.quant_act_int32.runtime_helper = runtime_helper

        for stage_num in range(0, 4):
            for unit_num in range(0, self.channel[stage_num]):
                tmp_func = getattr(self, f"stage{stage_num+1}.unit{unit_num+1}")
                tmp_func.runtime_helper = runtime_helper

        self.final_pool.runtime_helper = runtime_helper
        self.quant_act_output.runtime_helper = runtime_helper
        self.quant_output.runtime_helper = runtime_helper


class Q_ResNet20_unfold(nn.Module):
    """
        Quantized ResNet20 model for dataset CIFAR100, CIFAR10
    """
    def __init__(self, model):
        super().__init__()
        features = getattr(model, 'features')
        init_block = getattr(features, 'init_block')

        self.quant_input = QuantAct()

        self.quant_init_block_conv = QuantConv2d()
        self.quant_init_block_conv.set_param(init_block.conv)
       
        self.quant_init_block_conv_act = QuantAct()

        self.quant_init_block_bn = QuantBn()
        self.quant_init_block_bn.set_param(init_block.bn)

        self.quant_act_int32 = QuantAct()

        self.act = nn.ReLU()

        self.channel = [3, 3, 3]

        for stage_num in range(0, 3):
            stage = getattr(features, "stage{}".format(stage_num + 1))
            for unit_num in range(0, self.channel[stage_num]):
                unit = getattr(stage, 'unit{}'.format(unit_num + 1))
                quant_unit = Q_ResBlockBn_unfold()
                quant_unit.set_param(unit)
                setattr(self, f'stage{stage_num + 1}.unit{unit_num + 1}', quant_unit)

        self.final_pool = QuantAveragePool2d(kernel_size=8 , stride=1)

        self.quant_act_output = QuantAct()

        output = getattr(model, 'output')
        self.quant_output = QuantLinear()
        self.quant_output.set_param(output)

    def forward(self, x):
        x, act_scaling_factor = self.quant_input(x)

        x, conv_scaling_factor = self.quant_init_block_conv(x, act_scaling_factor)
        x, act_scaling_factor = self.quant_init_block_conv_act(x, act_scaling_factor, conv_scaling_factor)

        x, bn_scaling_factor = self.quant_init_block_bn(x, act_scaling_factor)
        x, act_scaling_factor = self.quant_act_int32(x, act_scaling_factor, bn_scaling_factor)

        x = self.act(x)

        for stage_num in range(0,3):
            for unit_num in range(0, self.channel[stage_num]):
                tmp_func = getattr(self, f'stage{stage_num + 1}.unit{unit_num + 1}')
                x, act_scaling_factor = tmp_func(x, act_scaling_factor)

        x = self.final_pool(x, act_scaling_factor)

        x, act_scaling_factor = self.quant_act_output(x, act_scaling_factor)
        x = x.view(x.size(0), -1)
        x, _ = self.quant_output(x, act_scaling_factor)

        return x

    def set_daq_helper(self, runtime_helper):
        self.runtime_helper = runtime_helper
        self.quant_input.runtime_helper = runtime_helper
        self.quant_init_block_convbn.runtime_helper = runtime_helper
        self.quant_act_int32.runtime_helper = runtime_helper

        for stage_num in range(0, 3):
            for unit_num in range(0, self.channel[stage_num]):
                tmp_func = getattr(self, f"stage{stage_num+1}.unit{unit_num+1}")
                tmp_func.runtime_helper = runtime_helper

        self.final_pool.runtime_helper = runtime_helper
        self.quant_act_output.runtime_helper = runtime_helper
        self.quant_output.runtime_helper = runtime_helper


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
        self.quant_act_output.isClassifier = True

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

        x = self.final_pool(x, act_scaling_factor)

        x, act_scaling_factor = self.quant_act_output(x, act_scaling_factor)
        x = x.view(x.size(0), -1)
        x = self.quant_output(x, act_scaling_factor)

        return x

    def set_daq_helper(self, runtime_helper):
        self.runtime_helper = runtime_helper
        self.quant_input.runtime_helper = runtime_helper
        self.quant_init_block_convbn.runtime_helper = runtime_helper
        self.quant_act_int32.runtime_helper = runtime_helper

        for stage_num in range(0, 3):
            for unit_num in range(0, self.channel[stage_num]):
                tmp_func = getattr(self, f"stage{stage_num+1}.unit{unit_num+1}")
                tmp_func.runtime_helper = runtime_helper
                if isinstance(tmp_func, QuantAct):
                    tmp_func.set_daq_ema_params(runtime_helper)

        self.final_pool.runtime_helper = runtime_helper
        self.quant_act_output.runtime_helper = runtime_helper
        self.quant_output.runtime_helper = runtime_helper


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

        x = self.final_pool(x, act_scaling_factor)

        x, act_scaling_factor = self.quant_act_output(x, act_scaling_factor)
        x = x.view(x.size(0), -1)
        x = self.quant_output(x, act_scaling_factor)

        return x

    def set_daq_helper(self, runtime_helper):
        self.runtime_helper = runtime_helper
        self.quant_input.runtime_helper = runtime_helper
        self.quant_init_block_convbn.runtime_helper = runtime_helper
        self.quant_act_int32.runtime_helper = runtime_helper

        for stage_num in range(0, 3):
            for unit_num in range(0, self.channel[stage_num]):
                tmp_func = getattr(self, f"stage{stage_num+1}.unit{unit_num+1}")
                tmp_func.runtime_helper = runtime_helper
                if isinstance(tmp_func, QuantAct):
                    tmp_func.set_daq_ema_params(runtime_helper)

        self.final_pool.runtime_helper = runtime_helper
        self.quant_act_output.runtime_helper = runtime_helper
        self.quant_output.runtime_helper = runtime_helper

        self.quant_input.set_daq_ema_params(runtime_helper)
        self.quant_act_int32.set_daq_ema_params(runtime_helper)
        self.quant_act_output.set_daq_ema_params(runtime_helper)


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

        self.final_pool = QuantAveragePool2d(kernel_size=7, stride=1)

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

        x = self.final_pool(x, act_scaling_factor)

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

        self.quant_act_int32 = QuantAct(runtime_helper=runtime_helper)

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

        self.final_pool = QuantAveragePool2d(kernel_size=7, stride=1)

        self.quant_act_output = QuantAct_Daq(runtime_helper=runtime_helper)
        self.quant_act_output.isClassifier = True

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

        x = self.final_pool(x, act_scaling_factor)

        x, act_scaling_factor = self.quant_act_output(x, act_scaling_factor)
        x = x.view(x.size(0), -1)
        x = self.quant_output(x, act_scaling_factor)

        return x


class Q_ResNet101(nn.Module):
    """
       Quantized ResNet101 model from 'Deep Residual Learning for Image Recognition,' https://arxiv.org/abs/1512.03385.
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

        self.channel = [3, 4, 23, 3]

        for stage_num in range(0, 4):
            stage = getattr(features, "stage{}".format(stage_num + 1))
            for unit_num in range(0, self.channel[stage_num]):
                unit = getattr(stage, "unit{}".format(unit_num + 1))
                quant_unit = Q_ResUnitBn()
                quant_unit.set_param(unit)
                setattr(self, f"stage{stage_num + 1}.unit{unit_num + 1}", quant_unit)

        self.final_pool = QuantAveragePool2d(kernel_size=7, stride=1)

        self.quant_act_output = QuantAct()

        output = getattr(model, 'output')
        self.quant_output = QuantLinear()
        self.quant_output.set_param(output)

    def forward(self, x):
        x, act_scaling_factor = self.quant_input(x)

        x, weight_scaling_factor = self.quant_init_convbn(x, act_scaling_factor)

        x = self.pool(x)
        x, act_scaling_factor = self.quant_act_int32(x, act_scaling_factor, weight_scaling_factor, None, None)

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

        weight_scaling_factor = conv_scaling_factor * bn_scaling_factor
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

        self.quant_act1 = QuantAct_Daq(runtime_helper=runtime_helper)

        convbn2 = unit.body.conv2
        self.quant_convbn2 = QuantBnConv2d()
        self.quant_convbn2.set_param(convbn2.conv, convbn2.bn)

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

        x = x + identity

        if self.resize_identity:
            x, act_scaling_factor = self.quant_act_int32(x, act_scaling_factor, weight_scaling_factor, identity,
                                                         identity_act_scaling_factor, identity_weight_scaling_factor)
        else:
            x, act_scaling_factor = self.quant_act_int32(x, act_scaling_factor, weight_scaling_factor, identity,
                                                         scaling_factor_int32, None)

        x = nn.ReLU()(x)

        return x, act_scaling_factor


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

def q_resnet20_unfold(model):
    net = Q_ResNet20_unfold(model)
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

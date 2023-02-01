"""
    Quantized ResNet for ImageNet-1K, implemented in PyTorch.
    Original paper: 'Deep Residual Learning for Image Recognition,' https://arxiv.org/abs/1512.03385.
"""

import torch
import torch.nn as nn
from ..quantization_utils.quant_modules import *
from pytorchcv.models.common import ConvBlock
from pytorchcv.models.shufflenetv2 import ShuffleUnit, ShuffleInitBlock

ACCUMULATE = {'sum': torch.add, 'max': torch.maximum}


class Q_ResNet18(nn.Module):
    """
        Quantized ResNet50 model from 'Deep Residual Learning for Image Recognition,' https://arxiv.org/abs/1512.03385.
    """

    def __init__(self, model, num_clusters=1):
        super().__init__()
        self.full_precision = False
        features = getattr(model, 'features')
        init_block = getattr(features, 'init_block')

        self.quant_input = QuantAct(num_clusters=num_clusters)

        self.quant_init_block_convbn = QuantBnConv2d()
        self.quant_init_block_convbn.set_param(
            init_block.conv.conv, init_block.conv.bn)

        self.quant_act_int32 = QuantAct(num_clusters=num_clusters)

        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.act = nn.ReLU()

        self.channel = [2, 2, 2, 2]

        for stage_num in range(0, 4):
            stage = getattr(features, "stage{}".format(stage_num + 1))
            for unit_num in range(0, self.channel[stage_num]):
                unit = getattr(stage, "unit{}".format(unit_num + 1))
                quant_unit = Q_ResBlockBn()
                quant_unit.set_param(unit, num_clusters=num_clusters)
                setattr(
                    self, f"stage{stage_num + 1}.unit{unit_num + 1}", quant_unit)

        self.final_pool = QuantAveragePool2d(kernel_size=7, stride=1)

        self.quant_act_output = QuantAct(num_clusters=num_clusters)

        output = getattr(model, 'output')
        self.quant_output = QuantLinear()
        self.quant_output.is_classifier = True
        self.quant_output.set_param(output)

    def forward(self, x, cluster=None):
        x, act_scaling_factor = self.quant_input((x, cluster))

        x, weight_scaling_factor = self.quant_init_block_convbn(
            x, act_scaling_factor)

        x = self.pool(x)
        x, act_scaling_factor = self.quant_act_int32((x, cluster), act_scaling_factor, weight_scaling_factor)

        x = self.act(x)

        for stage_num in range(0, 4):
            for unit_num in range(0, self.channel[stage_num]):
                tmp_func = getattr(
                    self, f"stage{stage_num+1}.unit{unit_num+1}")
                x, act_scaling_factor = tmp_func(
                    x, act_scaling_factor, cluster=cluster)

        x = self.final_pool(x, act_scaling_factor)

        x, act_scaling_factor = self.quant_act_output((x, cluster), act_scaling_factor)
        x = x.view(x.size(0), -1)
        x = self.quant_output(x, act_scaling_factor)

        return x


class Q_ResNet20(nn.Module):
    """
        Quantized ResNet20 model for dataset CIFAR100, CIFAR10
    """

    def __init__(self, model, num_clusters=1):
        super().__init__()
        self.full_precision = False
        features = getattr(model, 'features')
        init_block = getattr(features, 'init_block')

        self.quant_input = QuantAct(num_clusters=num_clusters)

        self.quant_init_block_convbn = QuantBnConv2d()
        self.quant_init_block_convbn.set_param(init_block.conv, init_block.bn)

        self.quant_act_int32 = QuantAct(num_clusters=num_clusters)

        self.act = nn.ReLU()

        self.channel = [3, 3, 3]

        for stage_num in range(0, 3):
            stage = getattr(features, "stage{}".format(stage_num + 1))
            for unit_num in range(0, self.channel[stage_num]):
                unit = getattr(stage, 'unit{}'.format(unit_num + 1))
                quant_unit = Q_ResBlockBn()
                quant_unit.set_param(unit, num_clusters=num_clusters)
                setattr(
                    self, f'stage{stage_num + 1}.unit{unit_num + 1}', quant_unit)

        self.final_pool = QuantAveragePool2d(kernel_size=8, stride=1)

        self.quant_act_output = QuantAct(num_clusters=num_clusters)

        output = getattr(model, 'output')
        self.quant_output = QuantLinear()
        self.quant_output.is_classifier = True
        self.quant_output.set_param(output)

    def forward(self, x, cluster=None):
        x, act_scaling_factor = self.quant_input((x, cluster))

        x, weight_scaling_factor = self.quant_init_block_convbn(
            x, act_scaling_factor)
        
        x, act_scaling_factor = self.quant_act_int32((x, cluster), act_scaling_factor, weight_scaling_factor)

        x = self.act(x)

        for stage_num in range(0, 3):
            for unit_num in range(0, self.channel[stage_num]):
                tmp_func = getattr(
                    self, f'stage{stage_num + 1}.unit{unit_num + 1}')
                x, act_scaling_factor = tmp_func(
                    x, act_scaling_factor, cluster=cluster)

        x, act_scaling_factor = self.final_pool(x, act_scaling_factor)

        x, act_scaling_factor = self.quant_act_output((x, cluster), act_scaling_factor)
        x = x.view(x.size(0), -1)
        x = self.quant_output(x, act_scaling_factor)

        return x


class Q_ResNet50(nn.Module):
    """
        Quantized ResNet50 model from 'Deep Residual Learning for Image Recognition,' https://arxiv.org/abs/1512.03385.
    """

    def __init__(self, model, num_clusters=1):
        super().__init__()
        self.full_precision = False
        features = getattr(model, 'features')
        init_block = getattr(features, 'init_block')

        self.quant_input = QuantAct(num_clusters=num_clusters)

        self.quant_init_convbn = QuantBnConv2d()
        self.quant_init_convbn.set_param(
            init_block.conv.conv, init_block.conv.bn)

        self.quant_act_int32 = QuantAct(num_clusters=num_clusters)

        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.act = nn.ReLU()

        self.channel = [3, 4, 6, 3]

        for stage_num in range(0, 4):
            stage = getattr(features, "stage{}".format(stage_num + 1))
            for unit_num in range(0, self.channel[stage_num]):
                unit = getattr(stage, "unit{}".format(unit_num + 1))
                quant_unit = Q_ResUnitBn()
                quant_unit.set_param(unit, num_clusters=num_clusters)
                setattr(
                    self, f"stage{stage_num + 1}.unit{unit_num + 1}", quant_unit)

        self.final_pool = QuantAveragePool2d(
            kernel_size=7, stride=1, padding=0)

        self.quant_act_output = QuantAct(num_clusters=num_clusters)

        output = getattr(model, 'output')
        self.quant_output = QuantLinear()
        self.quant_output.is_classifier = True
        self.quant_output.set_param(output)

    def forward(self, x, cluster=None):
        x, act_scaling_factor = self.quant_input((x, cluster))

        x, weight_scaling_factor = self.quant_init_convbn(
            x, act_scaling_factor)

        x = self.pool(x)
        x, act_scaling_factor = self.quant_act_int32((x, cluster), act_scaling_factor, weight_scaling_factor)

        x = self.act(x)

        for stage_num in range(0, 4):
            for unit_num in range(0, self.channel[stage_num]):
                tmp_func = getattr(
                    self, f"stage{stage_num+1}.unit{unit_num+1}")
                x, act_scaling_factor = tmp_func(
                    x, act_scaling_factor, cluster=cluster)

        x, act_scaling_factor = self.final_pool(x, act_scaling_factor)

        x, act_scaling_factor = self.quant_act_output((x, cluster), act_scaling_factor)
        x = x.view(x.size(0), -1)
        x = self.quant_output(x, act_scaling_factor)

        return x


class Q_ResUnitBn(nn.Module):
    """
       Quantized ResNet unit with residual path.
    """

    def __init__(self):
        super(Q_ResUnitBn, self).__init__()

    def set_param(self, unit, num_clusters=1):
        self.resize_identity = unit.resize_identity

        self.quant_act = QuantAct(num_clusters=num_clusters)

        convbn1 = unit.body.conv1
        self.quant_convbn1 = QuantBnConv2d()
        self.quant_convbn1.set_param(convbn1.conv, convbn1.bn)
        self.act1 = nn.ReLU()
        self.quant_act1 = QuantAct(num_clusters=num_clusters)

        convbn2 = unit.body.conv2
        self.quant_convbn2 = QuantBnConv2d()
        self.quant_convbn2.set_param(convbn2.conv, convbn2.bn)
        self.act2 = nn.ReLU()
        self.quant_act2 = QuantAct(num_clusters=num_clusters)

        convbn3 = unit.body.conv3
        self.quant_convbn3 = QuantBnConv2d()
        self.quant_convbn3.set_param(convbn3.conv, convbn3.bn)

        if self.resize_identity:
            self.quant_identity_convbn = QuantBnConv2d()
            self.quant_identity_convbn.set_param(
                unit.identity_conv.conv, unit.identity_conv.bn)

        self.quant_act_int32 = QuantAct(num_clusters=num_clusters)
        self.act3 = nn.ReLU()

    def forward(self, x, scaling_factor_int32=None, cluster=None):
        # forward using the quantized modules
        if self.resize_identity:
            x, act_scaling_factor = self.quant_act((x, cluster), scaling_factor_int32)
            identity_act_scaling_factor = act_scaling_factor.clone(
            ) if act_scaling_factor is not None else None
            identity, identity_weight_scaling_factor = self.quant_identity_convbn(
                x, act_scaling_factor)
        else:
            identity = x
            x, act_scaling_factor = self.quant_act((x, cluster), scaling_factor_int32)

        x, weight_scaling_factor = self.quant_convbn1(x, act_scaling_factor)
        x = self.act1(x)
        x, act_scaling_factor = self.quant_act1((x, cluster), act_scaling_factor, weight_scaling_factor)

        x, weight_scaling_factor = self.quant_convbn2(x, act_scaling_factor)
        x = self.act2(x)
        x, act_scaling_factor = self.quant_act2((x, cluster), act_scaling_factor, weight_scaling_factor)

        x, weight_scaling_factor = self.quant_convbn3(x, act_scaling_factor)

        x = x + identity

        if self.resize_identity:
            x, act_scaling_factor = self.quant_act_int32((x, cluster), act_scaling_factor, weight_scaling_factor,
                                                         identity, identity_act_scaling_factor, identity_weight_scaling_factor)
        else:
            x, act_scaling_factor = self.quant_act_int32((x, cluster),  act_scaling_factor, weight_scaling_factor,
                                                         identity, scaling_factor_int32, None)

        x = self.act3(x)

        return x, act_scaling_factor


class Q_ResBlockBn(nn.Module):
    """
        Quantized ResNet block with residual path.
    """

    def __init__(self):
        super(Q_ResBlockBn, self).__init__()

    def set_param(self, unit, num_clusters=1):
        self.resize_identity = unit.resize_identity

        self.quant_act = QuantAct(num_clusters=num_clusters)

        convbn1 = unit.body.conv1
        self.quant_convbn1 = QuantBnConv2d()
        self.quant_convbn1.set_param(convbn1.conv, convbn1.bn)
        self.act1 = nn.ReLU()

        self.quant_act1 = QuantAct(num_clusters=num_clusters)

        convbn2 = unit.body.conv2
        self.quant_convbn2 = QuantBnConv2d()
        self.quant_convbn2.set_param(convbn2.conv, convbn2.bn)

        if self.resize_identity:
            self.quant_identity_convbn = QuantBnConv2d()
            self.quant_identity_convbn.set_param(
                unit.identity_conv.conv, unit.identity_conv.bn)

        self.quant_act_int32 = QuantAct(num_clusters=num_clusters)
        self.act2 = nn.ReLU()

    def forward(self, x, scaling_factor_int32=None, cluster=None):
        # forward using the quantized modules
        if self.resize_identity:
            x, act_scaling_factor = self.quant_act((x, cluster), scaling_factor_int32)
            identity_act_scaling_factor = act_scaling_factor.clone(
            ) if act_scaling_factor is not None else None
            identity, identity_weight_scaling_factor = self.quant_identity_convbn(
                x, act_scaling_factor)
        else:
            identity = x
            x, act_scaling_factor = self.quant_act((x, cluster), scaling_factor_int32)

        x, weight_scaling_factor = self.quant_convbn1(x, act_scaling_factor)
        x = self.act1(x)
        x, act_scaling_factor = self.quant_act1((x, cluster), act_scaling_factor, weight_scaling_factor)

        x, weight_scaling_factor = self.quant_convbn2(x, act_scaling_factor)

        x = x + identity

        if self.resize_identity:
            x, act_scaling_factor = self.quant_act_int32((x, cluster), act_scaling_factor, weight_scaling_factor,
                                                         identity, identity_act_scaling_factor, identity_weight_scaling_factor)
        else:
            x, act_scaling_factor = self.quant_act_int32((x, cluster), act_scaling_factor, weight_scaling_factor,
                                                         identity, scaling_factor_int32, None)

        x = self.act2(x)

        return x, act_scaling_factor


def q_resnet18(model, num_clusters=None):
    return Q_ResNet18(model, num_clusters)


def q_resnet20(model, num_clusters=None):
    return Q_ResNet20(model, num_clusters)


def q_resnet50(model, num_clusters=None):
    return Q_ResNet50(model, num_clusters)

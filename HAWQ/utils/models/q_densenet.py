import torch
import torch.nn as nn
import copy
from ..quantization_utils.quant_modules import *
from pytorchcv.models.common import ConvBlock
from pytorchcv.models.shufflenetv2 import ShuffleUnit, ShuffleInitBlock
import time
import logging


class Q_DenseNet(nn.Module):
    def __init__(self, model, num_clusters=1):
        super().__init__()
        features = getattr(model, 'features')
        init_block = getattr(features, 'init_block')

        self.quant_input = QuantAct(num_clusters=num_clusters)

        self.quant_init_convbn = QuantBnConv2d()
        self.quant_init_convbn.set_param(init_block.conv, init_block.bn)

        self.act1 = nn.ReLU(inplace=True)

        self.pool = QuantMaxPool2d(kernel_size=3, stride=2, padding=1)
        self.quant_act1 = QuantAct(num_clusters=num_clusters)

        units = [6, 12, 24, 16]

        for stage_num in range(4):
            stage = getattr(features, "stage{}".format(stage_num + 1))

            if stage_num != 0:
                trans = getattr(stage, "trans{}".format(stage_num + 1))
                quant_trans = Q_Transition()
                quant_trans.set_param(trans, num_clusters=num_clusters)
                setattr(self, f'trans{stage_num +1}', quant_trans)

            dense_block = Q_DenseBlock()
            dense_block.set_param(
                stage, units[stage_num], num_clusters=num_clusters)
            setattr(self, f'stage{stage_num + 1}', dense_block)

        post_activ = getattr(features, 'post_activ')
        self.batch_norm = QuantBn()
        self.batch_norm.set_param(post_activ.bn)

        self.act2 = nn.ReLU(inplace=True)
        self.quant_act2 = QuantAct(num_clusters=num_clusters)

        self.final_pool = QuantAveragePool2d(kernel_size=7, stride=1)

        self.quant_act_output = QuantAct(num_clusters=num_clusters)

        output = getattr(model, 'output')
        self.quant_output = QuantLinear()
        self.quant_output.is_classifier = True
        self.quant_output.set_param(output)

    def forward(self, x, cluster=None):
        x, act_scaling_factor = self.quant_input(x, cluster=cluster)

        x, weight_scaling_factor = self.quant_init_convbn(
            x, act_scaling_factor)
        x = self.act1(x)
        x, act_scaling_factor = self.pool(x, act_scaling_factor)
        x, act_scaling_factor = self.quant_act1(
            x, act_scaling_factor, weight_scaling_factor, cluster=cluster)

        for stage_num in range(4):
            if stage_num != 0:
                transition = getattr(self, f'trans{stage_num + 1}')
                x, act_scaling_factor = transition(
                    x, act_scaling_factor, cluster=cluster)
            function = getattr(self, f'stage{stage_num + 1}')
            x, act_scaling_factor = function(
                x, act_scaling_factor, cluster=cluster)

        x, bn_scaling_factor = self.batch_norm(x, act_scaling_factor)
        x = self.act2(x)
        x, act_scaling_factor = self.quant_act2(
            x, act_scaling_factor, bn_scaling_factor, cluster=cluster)

        x, act_scaling_factor = self.final_pool(x, act_scaling_factor)
        x, act_scaling_factor = self.quant_act_output(
            x, act_scaling_factor, cluster=cluster)

        x = x.view(x.size(0), -1)
        x = self.quant_output(x, act_scaling_factor)
        return x


class Q_Transition(nn.Module):
    def __init__(self):
        super(Q_Transition, self).__init__()

    def set_param(self, trans, num_clusters=1):
        conv_block = getattr(trans, 'conv')

        self.batch_norm = QuantBn()
        self.batch_norm.set_param(conv_block.bn)
        self.act = nn.ReLU(inplace=True)
        self.quant_act1 = QuantAct(num_clusters=num_clusters)

        self.conv = QuantConv2d()
        self.conv.set_param(conv_block.conv)

        self.quant_act2 = QuantAct(num_clusters=num_clusters)

        self.pool = QuantAveragePool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, x, act_scaling_factor=None, cluster=None):
        x, bn_scaling_factor = self.batch_norm(x, act_scaling_factor)

        x = self.act(x)
        x, act_scaling_factor = self.quant_act1(
            x, act_scaling_factor, bn_scaling_factor, cluster=cluster)

        x, conv_scaling_factor = self.conv(x, act_scaling_factor)
        x, act_scaling_factor = self.quant_act2(
            x, act_scaling_factor, conv_scaling_factor, cluster=cluster)

        x, act_scaling_factor = self.pool(x, act_scaling_factor)
        return x, act_scaling_factor


class Q_DenseUnit(nn.Module):
    def __init__(self):
        super(Q_DenseUnit, self).__init__()

    def set_param(self, unit, num_clusters=1):
        layer1 = getattr(unit, "conv1")
        self.quant_bn1 = QuantBn()
        self.quant_bn1.set_param(layer1.bn)
        self.act1 = nn.ReLU(inplace=True)
        self.quant_act1 = QuantAct(num_clusters=num_clusters)

        layer2 = getattr(unit, "conv2")
        self.quant_convbn = QuantBnConv2d()
        self.quant_convbn.set_param(layer1.conv, layer2.bn)

        self.act2 = nn.ReLU(inplace=True)
        self.quant_act2 = QuantAct(num_clusters=num_clusters)

        self.quant_conv2 = QuantConv2d()
        self.quant_conv2.set_param(layer2.conv)
        self.quant_act_output = QuantAct(num_clusters=num_clusters)

    def forward(self, batch, input_scaling_factor=None, cluster=None):
        x, bn_scaling_factor = self.quant_bn1(batch, input_scaling_factor)
        x = self.act1(x)
        x, act_scaling_factor = self.quant_act1(
            x, input_scaling_factor, bn_scaling_factor, cluster=cluster)

        x, weight_scaling_factor = self.quant_convbn(x, act_scaling_factor)
        x = self.act2(x)
        x, act_scaling_factor = self.quant_act2(
            x, act_scaling_factor, weight_scaling_factor, cluster=cluster)

        x, conv_scaling_factor = self.quant_conv2(x, act_scaling_factor)

        concat_tensor = torch.cat((batch, x), 1)
        x, act_scaling_factor = self.quant_act_output(concat_tensor, act_scaling_factor, conv_scaling_factor,
                                                      concat=True, concat_scaling_factor=input_scaling_factor, cluster=cluster)
        return x, act_scaling_factor


class Q_DenseBlock(nn.Module):
    def __init__(self):
        super(Q_DenseBlock, self).__init__()

    def set_param(self, stage, layers, num_clusters=1):
        self.layers = layers
        for unit_num in range(self.layers):
            unit = getattr(stage, 'unit{}'.format(unit_num + 1))
            quant_unit = Q_DenseUnit()
            quant_unit.set_param(unit, num_clusters=num_clusters)
            setattr(self, f'unit{unit_num + 1}', quant_unit)

    def forward(self, x, act_scaling_factor=None, cluster=None):
        for unit_num in range(self.layers):
            function = getattr(self, f'unit{unit_num + 1}')
            x, act_scaling_factor = function(
                x, act_scaling_factor, cluster=cluster)
        return x, act_scaling_factor


def q_densenet(model, num_clusters=None):
    return Q_DenseNet(model, num_clusters)

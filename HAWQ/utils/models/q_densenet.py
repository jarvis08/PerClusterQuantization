import torch
import torch.nn as nn
import copy
from ..quantization_utils.quant_modules import *
from pytorchcv.models.common import ConvBlock
from pytorchcv.models.shufflenetv2 import ShuffleUnit, ShuffleInitBlock
import time
import logging


class Q_DenseNet(nn.Module):
    def __init__(self, model):
        super().__init__()
        features = getattr(model, 'features')
        init_block = getattr(features, 'init_block')

        self.quant_input = QuantAct()

        self.quant_init_block_conv = QuantConv2d()
        self.quant_init_block_conv.set_param(init_block.conv)
        self.quant_init_block_bn = QuantBn()
        self.quant_init_block_bn.set_param(init_block.bn)

        self.quant_act1 = QuantAct()

        self.act = nn.ReLU()

        self.pool = QuantMaxPool2d()
        self.quant_act2 = QuantAct()

        units = [6, 12, 24, 16] 

        for stage_num in range(4):
            stage = getattr(features, "stage{}".format(stage_num + 1)) 

            if stage_num is not 0:
                trans = getattr(stage, "trans{}".format(stage_num + 1)) 
                quant_trans = Q_Transition()
                quant_trans.set_param(trans)

            dense_block = Q_DenseBlock() 
            dense_block.set_param(stage, units[stage_num])
            setattr(self, f'stage{stage_num + 1}', dense_block)

        post_activ = getattr(features, 'post_activ')
        self.batch_norm = QuantBn()
        self.batch_norm.set_param(post_activ.bn)
        self.quant_act3 = QuantAct() # not confident

        self.act = nn.ReLU()

        self.final_pool = QuantAveragePool2d(kernel_size=7, stride=1)
     
        self.quant_act_output = QuantAct()

        output = getattr(model, 'output')
        self.quant_output = QuantLinear()
        self.quant_output.set_param(output)


    def forward(self, input):

        return out 


class Q_Transition(nn.Module):
    def __init__(self):
        super(Q_Transition, self).__init__()

    def set_param(self, trans):
        self.quant_act = QuantAct()

        conv_block = getattr(trans, 'conv')
        self.batch_norm = QuantBn()
        self.batch_norm.set_param(conv_block.bn)
        self.quant_act1 = QuantAct()

        self.act = nn.ReLU()

        self.conv = QuantConv2d()
        self.conv.set_param(conv_block.conv)
        self.quant_act2 = QuantAct()

        self.pool = QuantAveragePool2d(kernel_size=2, stride=2)
        self.quant_act3 = QuantAct()

    def forward(self, x, scaling_factor_int32=None):
        return out


class Q_DenseLayer(nn.Module):
    def __init__(self):
        super(Q_DenseLayer, self).__init__()

    def set_param(self, layer):
        self.quant_bn = QuantBn()
        self.quant_bn.set_param(layer.bn)
        self.quant_act = QuantAct()

        self.quant_conv = QuantConv2d()
        self.quant_conv.set_param(layer.conv)

        self.quant_output = QuantAct()

    def forward(self, x, act_scaling_factor=None):
        x, bn_scaling_factor = self.quant_bn(x, conv_scaling_factor)
        x = nn.ReLU()(x)
        x, act_scaling_factor = self.quant_act(x, act_scaling_factor, bn_scaling_factor)

        x, conv_scaling_factor = self.quant_conv(x, act_scaling_factor)
        x, act_scaling_factor = self.quant_output(x, act_scaling_factor, conv_scaling_factor)

        return x, act_scaling_factor


class Q_DenseUnit(nn.Module):
    def __init__(self):
        super(Q_DenseUnit, self).__init__()

    def set_param(self, unit):
        self.quant_act_input = QuantAct()

        for block_num in range(2):
            block = getattr(unit, "conv{}".format(block_num + 1))
            conv_block = Q_DenseLayer()
            conv_block.set_param(block)
            setattr(self, f'conv{block_num + 1}', conv_block)

    def forward(self, x, act_scaling_factor=None):
        x = torch.cat(x, 1)
        x, act_scaling_factor = quant_act_input(x, act_scaling_factor)

        for block_num in range(2):
            function = getattr(self, f'conv{block_num + 1}')
            x, act_scaling_factor = function(x, act_scaling_factor)

        return x, act_scaling_factor


class Q_DenseBlock(nn.Module):
    def __init__(self):
        super(Q_DenseBlock, self).__init__()

    def set_param(self, stage, layers):
        self.layers = layers
        for unit_num in range(self.layers):
            unit = getattr(stage, 'unit{}'.format(unit_num + 1))
            quant_unit = Q_DenseUnit()
            quant_unit.set_param(unit)
            setattr(self, f'unit{unit_num + 1}', quant_unit)

    def forward(self, x, act_scaling_factor=None):
        features = [x]
        for unit_num in range(self.layers):
            function = getattr(self, f'unit{unit_num + 1}')
            output, act_scaling_factor = function(features, act_scaling_factor)
            features.append(output)

        return x


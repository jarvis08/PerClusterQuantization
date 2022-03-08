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
        self.quant_init_block_act = QuantAct()
        self.quant_init_block_bn = QuantBn()
        self.quant_init_block_bn.set_param(init_block.bn)

        self.quant_act1 = QuantAct()

        self.act1 = nn.ReLU(inplace=True)

        self.pool = QuantMaxPool2d(kernel_size=2, stride=2, padding=1)
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

        self.act2 = nn.ReLU(inplace=True))

        self.final_pool = QuantAveragePool2d(kernel_size=7, stride=1)
     
        self.quant_act_output = QuantAct()

        output = getattr(model, 'output')
        self.quant_output = QuantLinear()
        self.quant_output.set_param(output)

    def forward(self, input):
        x, act_scaling_factor = self.quant_input(x)

        x, conv_scaling_factor = self.quant_init_block_conv(x, act_scaling_factor)
        x, act_scaling_factor = self.quant_init_block_act(x, act_scaling_factor, conv_scaling_factor)

        x, bn_scaling_factor = self.quant_init_block_bn(x, act_scaling_factor)
        x, act_scaling_factor = self.quant_act1(x, act_scaling_factor, bn_scaling_factor)

        x = self.act1(x) 

        x = self.pool(x, act_scaling_factor)
        x, act_scaling_factor = self.quant_act2(x, act_scaling_factor) 

        for stage_num in range(4):
            function = getattr(self, f'stage{stage_num + 1}')
            x, act_scaling_factor = function(x, act_scaling_factor)

        x, bn_scaling_factor = self.batch_norm(x, act_scaling_factor)
        x, act_scaling_factor = self.quant_act3(x, act_scaling_factor, bn_scaling_factor)

        x = self.act2(x)

        x = self.final_pool(x, act_scaling_factor)

        x, act_scaling_factor = self.quant_act_output(x, act_scaling_factor)
        x = x.view(x.size(0), -1)
        x = self.quant_output(x, act_Scaling_factor)

        return x


class Q_Transition(nn.Module):
    def __init__(self):
        super(Q_Transition, self).__init__()

    def set_param(self, trans):
        conv_block = getattr(trans, 'conv')

        self.batch_norm = QuantBn()
        self.batch_norm.set_param(conv_block.bn)
        self.quant_act1 = QuantAct()

        self.act = nn.ReLU(inplace=True)

        self.conv = QuantConv2d()
        self.conv.set_param(conv_block.conv)
        self.quant_act2 = QuantAct()

        self.pool = QuantAveragePool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True)
        self.quant_output = QuantAct()

    def forward(self, x, act_scaling_factor=None):
        x, bn_scaling_factor = self.batch_norm(x, act_scaling_factor)
        x, act_scaling_factor = self.quant_act1(act_scaling_factor, bn_scaling_factor)

        x = self.act(x)

        x, conv_scaling_factor = self.conv(x, act_scaling_factor)
        x, act_scaling_factor = self.quant_act2(act_scaling_factor, conv_scaling_factor)

        x = self.pool(x, act_scaling_factor)
        x, act_scaling_factor = self.quant_output(x, act_scaling_factor)

        return x, act_scaling_factor



class Q_DenseUnit(nn.Module):
    def __init__(self):
        super(Q_DenseUnit, self).__init__()

    def set_param(self, unit, is_last_layer):
        self.is_last_layer = is_last_layer
        self.quant_act_input = QuantAct()
        
        layer1 = getattr(unit, "conv1")

        self.quant_bn1 = QuantBn()
        self.quant_bn1.set_param(layer1.bn)
        self.quant_act1 = QuantAct()

        self.act1 = nn.ReLU(inplace=True)

        self.quant_conv1 = QuantConv2d()
        self.quant_conv1.set_param(layer1.conv)
        self.quant_act2 = QuantAct()

        layer2 = getattr(unit, "conv2")
        self.quant_bn2 = QuantBn()
        self.quant_bn2.set_param(layer2.bn)
        self.quant_act3 = QuantAct()

        self.act2 = nn.ReLU(inplace=True)
        self.quant_conv2 = QuantConv2d()
        self.quant_conv2.set_param(layer2.conv)

        self.quant_act_output = QuantAct()


    def forward(self, x, act_scaling_factor=None):
        if not self.is_last_layer:
            concat_tensor = [x]
            concat_act_scaling_factor = act_scaling_factor.clone()

        x, bn_scaling_factor = self.quant_bn1(x, act_scaling_factor)
        x, act_scaling_factor = self.quant_act1(x, act_scaling_factor, bn_scaling_factor)

        x = self.act1(x)

        x, conv_scaling_factor = self.quant_conv1(x, act_scaling_factor)
        x, act_scaling_factor = self.quant_act2(x, act_scaling_factor, conv_scaling_factor)

        x, bn_scaling_factor = self.quant_bn2(x, act_scaling_factor)
        x, act_scaling_factor = self.quant_act3(x, act_scaling_factor, bn_scaling_factor)

        x = self.act2(x)

        x, conv_scaling_factor = self.quant_conv2(x, act_scaling_factor)

        if not self.is_last_layer:
            concat_tensor.append(x)
            concat_tensor = torch.cat(concat_tensor, 1)

            x, act_scaling_factor = quant_act_output(concat_tensor, act_scaling_factor, conv_scaling_factor, 
                                                    concat=x, concat_scaling_factor=concat_act_scaling_factor)
        else:
            x, act_scaling_factor = quant_act_output(x, act_scaling_factor, conv_scaling_factor)
        return x, act_scaling_factor


class Q_DenseBlock(nn.Module):
    def __init__(self):
        super(Q_DenseBlock, self).__init__()

    def set_param(self, stage, layers):
        self.layers = layers
        for unit_num in range(self.layers):
            unit = getattr(stage, 'unit{}'.format(unit_num + 1))
            quant_unit = Q_DenseUnit()
            quant_unit.set_param(unit, unit_num is (self.layers - 1))
            setattr(self, f'unit{unit_num + 1}', quant_unit)

    def forward(self, x, act_scaling_factor=None):
        for unit_num in range(self.layers):
            function = getattr(self, f'unit{unit_num + 1}')
            output, act_scaling_factor = function(features, act_scaling_factor)
        return output, act_scaling_factor


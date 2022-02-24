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


class Q_AlexNet(nn.Module):
    def __init__(self, model):
        super().__init__()
        features = getattr(model, 'features')

        self.quant_input = QuantAct()

        conv_block = getattr(features, 'conv')

        self.conv1 = QuantConv2d()
        self.conv1.set_param(conv_block.conv1)
        self.quant_act1 = QuantAct()
        self.act1 = nn.ReLU()

        self.maxpool1 = QuantMaxPool2d()
        self.quant_act1_1 = QuantAct()

        self.conv2 = QuantConv2d()
        self.conv2.set_param(conv_block.conv2)
        self.quant_act2 = QuantAct()
        self.act2 = nn.ReLU()

        self.maxpool2 = QuantMaxPool2d()
        self.quant_act2_1 = QuantAct()

        self.conv3 = QuantConv2d()
        self.conv3.set_param(conv_block.conv3)
        self.quant_act3 = QuantAct()
        self.act3 = nn.ReLU()

        self.conv4 = QuantConv2d()
        self.conv4.set_param(conv_block.conv4)
        self.quant_act4 = QuantAct()
        self.act4 = nn.ReLU()

        self.conv5 = QuantConv2d()
        self.conv5.set_param(conv_block.conv5)
        self.quant_act5 = QuantAct()
        self.act5 = nn.ReLU()

        self.maxpool3 = QuantMaxPool2d()
        self.quant_act3_1 = QuantAct()
        #self.avgpool = QuantAveragePool2d()
        
        fc_block = getattr(features, 'fc')

        self.fc1 = QuantLinear()
        self.fc1.set_param(fc_block.fc1)
        self.quant_act6 = QuantAct()
        self.act6 = nn.ReLU()

        self.fc2 = QuantLinear()
        self.fc2.set_param(fc_block.fc2)
        self.quant_act7 = QuantAct()
        self.act7 = nn.ReLU()

        self.fc3 = QuantLinear()
        self.fc3.set_param(fc_block.fc3)
        
    def forward(self, x):
        x, act_scaling_factor = self.quant_input(x)

        x, conv_scaling_factor = self.conv1(x, act_scaling_factor)
        x, act_scaling_factor = self.quant_act1(x, act_scaling_factor, conv_scaling_factor)

        x = self.act1(x)

        x = self.maxpool1(x, act_scaling_factor)
        x, act_scaling_factor = self.quant_act1_1(x, act_scaling_factor) 

        x, conv_scaling_factor = self.conv2(x, act_scaling_factor)
        x, act_scaling_factor = self.quant_act2(x, act_scaling_factor, conv_scaling_factor)

        x = self.act2(x)

        x = self.maxpool2(x, act_scaling_factor)
        x, act_scaling_factor = self.quant_act2_1(x, act_scaling_factor)

        x, conv_scaling_factor = self.conv3(x, act_scaling_factor)
        x, act_scaling_factor = self.quant_act3(x, act_scaling_factor, conv_scaling_factor)

        x = self.act3(x)

        x, conv_scaling_factor = self.conv4(x, act_scaling_factor)
        x, act_scaling_factor = self.quant_act4(x, act_scaling_factor, conv_scaling_factor)

        x = self.act4(x)
        
        x, conv_scaling_factor = self.conv5(x, act_scaling_factor)
        x, act_scaling_factor = self.quant_act5(x, act_scaling_factor, conv_scaling_factor)

        x = self.act5(x)

        x = self.maxpool3(x, act_scaling_factor)
        x, act_scaling_factor = self.quant_act3_1(x, act_scaling_factor)



        x = torch.flatten(x, 1)

        x, fc_scaling_factor = self.fc1(x, act_scaling_factor)
        x, act_scaling_factor = self.quant_act6(x, act_scaling_factor, fc_scaling_factor)

        x = self.act6(x)

        x, fc_scaling_factor = self.fc2(x, act_scaling_factor)
        x, act_scaling_factor = self.quant_act7(x, act_scaling_factor, fc_scaling_factor)

        x = self.act7(x)

        x, _ = self.fc3(x, act_scaling_factor)

        return x



"""
    Quantized ResNet for ImageNet-1K, implemented in PyTorch.
    Original paper: 'Deep Residual Learning for Image Recognition,' https://arxiv.org/abs/1512.03385.
"""
import torch
import torch.nn as nn
import copy
from ..quantization_utils.quant_modules import *

class Q_AlexNet(nn.Module):
    def __init__(self, model, model_dict=None):
        super().__init__()
        self.full_precision = False
        features = getattr(model, 'features')

        self.quant_input = QuantAct()

        # stage1
        stage = getattr(features, 'stage1')
        conv_block = getattr(stage, 'unit1')
        conv_block.conv.in_channels = 3
        conv_block.conv.out_channels = 96
        conv_block.conv.stride = (1, 1)
        conv_block.conv.kernel_size = (5, 5)
        conv_block.conv.padding = (2, 2)

        self.conv1 = QuantConv2d()
        self.conv1.set_param(conv_block.conv, model_dict, 'features.stage1.unit1.conv')
        self.act1 = nn.ReLU()
        self.quant_act1 = QuantAct()

        self.maxpool1 = QuantMaxPool2d(ceil_mode=True)

        # stage2
        stage = getattr(features, 'stage2')
        conv_block = getattr(stage, 'unit1')
        conv_block.conv.in_channels = 96
        conv_block.conv.out_channel = 256
        conv_block.conv.stride = (1, 1)
        conv_block.conv.kernel_size = (5, 5)
        conv_block.conv.padding = (2, 2)
        self.conv2 = QuantConv2d()
        self.conv2.set_param(conv_block.conv, model_dict, 'features.stage2.unit1.conv')
        self.act2 = nn.ReLU()
        self.quant_act2 = QuantAct()

        self.maxpool2 = QuantMaxPool2d(ceil_mode=True)

        # stage3
        stage = getattr(features, 'stage3')
        conv_block = getattr(stage, 'unit1')
        conv_block.conv.in_channels = 256
        conv_block.conv.out_channels = 384
        conv_block.conv.kernel_size = (3, 3)
        conv_block.conv.stride = (1, 1)
        conv_block.conv.padding = (1, 1)
        self.conv3 = QuantConv2d()
        self.conv3.set_param(conv_block.conv, model_dict, 'features.stage3.unit1.conv')
        self.act3 = nn.ReLU()
        self.quant_act3 = QuantAct()

        conv_block = getattr(stage, 'unit2')
        conv_block.conv.in_channels  = 384
        conv_block.conv.out_channels = 384
        conv_block.conv.kernel_size = (3, 3)
        conv_block.conv.stride = (1, 1)
        conv_block.conv.padding = (1, 1)
        self.conv4 = QuantConv2d()
        self.conv4.set_param(conv_block.conv, model_dict, 'features.stage3.unit2.conv')
        self.act4 = nn.ReLU()
        self.quant_act4 = QuantAct()

        conv_block = getattr(stage, 'unit3')
        conv_block.conv.in_channels = 384
        conv_block.conv.out_channels = 256
        conv_block.conv.kernel_size = (3, 3)
        conv_block.conv.stride = (1, 1)
        conv_block.conv.padding = (1, 1)
        self.conv5 = QuantConv2d()
        self.conv5.set_param(conv_block.conv, model_dict, 'features.stage3.unit3.conv')
        self.act5 = nn.ReLU()
        self.quant_act5 = QuantAct()

        self.maxpool3 = QuantMaxPool2d(ceil_mode=True)
        self.avgpool = QuantAveragePool2d(output=(1, 1))

        output = getattr(model, 'output')

        # fc1
        fc_block = getattr(output, 'fc1')
        fc_block.fc.in_features = 256
        fc_block.fc.out_features = 4096
        self.fc1 = QuantLinear()
        self.fc1.set_param(fc_block.fc, model_dict, 'output.fc1.fc')
        self.act6 = nn.ReLU()
        self.quant_act6 = QuantAct()

        # fc2
        fc_block = getattr(output, 'fc2')
        fc_block.fc.in_features = 4096
        fc_block.fc.out_features = 4096
        self.fc2 = QuantLinear()
        self.fc2.set_param(fc_block.fc, model_dict, 'output.fc2.fc')
        self.act7 = nn.ReLU()
        self.quant_act7 = QuantAct()

        # fc3
        fc = getattr(output, 'fc3')
        fc.in_features = 4096
        fc.out_features = 10
        self.fc3 = QuantLinear()
        self.fc3.is_classifier = True
        self.fc3.set_param(fc, model_dict, 'output.fc3')

    def forward(self, x):
        x, act_scaling_factor = self.quant_input(x)

        x, conv_scaling_factor = self.conv1(x, act_scaling_factor)
        x = self.act1(x)
        x, act_scaling_factor = self.quant_act1(x, act_scaling_factor, conv_scaling_factor)

        x, act_scaling_factor = self.maxpool1(x, act_scaling_factor)

        x, conv_scaling_factor = self.conv2(x, act_scaling_factor)
        x = self.act2(x)
        x, act_scaling_factor = self.quant_act2(x, act_scaling_factor, conv_scaling_factor)

        x, act_scaling_factor = self.maxpool2(x, act_scaling_factor)

        x, conv_scaling_factor = self.conv3(x, act_scaling_factor)
        x = self.act3(x)
        x, act_scaling_factor = self.quant_act3(x, act_scaling_factor, conv_scaling_factor)

        x, conv_scaling_factor = self.conv4(x, act_scaling_factor)
        x = self.act4(x)
        x, act_scaling_factor = self.quant_act4(x, act_scaling_factor, conv_scaling_factor)
        
        x, conv_scaling_factor = self.conv5(x, act_scaling_factor)
        x = self.act5(x)
        x, act_scaling_factor = self.quant_act5(x, act_scaling_factor, conv_scaling_factor)

        x, act_scaling_factor = self.maxpool3(x, act_scaling_factor)
        x, act_scaling_factor = self.avgpool(x, act_scaling_factor)

        x = x.view(x.size(0), -1)

        x, fc_scaling_factor = self.fc1(x, act_scaling_factor)
        x = self.act6(x)
        x, act_scaling_factor = self.quant_act6(x, act_scaling_factor, fc_scaling_factor)

        x, fc_scaling_factor = self.fc2(x, act_scaling_factor)
        x = self.act7(x)
        x, act_scaling_factor = self.quant_act7(x, act_scaling_factor, fc_scaling_factor)

        x = self.fc3(x, act_scaling_factor)

        return x
                

def q_alexnet(model, model_dict=None):
    return Q_AlexNet(model, model_dict)

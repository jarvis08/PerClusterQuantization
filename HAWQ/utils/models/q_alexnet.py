import torch
import torch.nn as nn
import copy
from ..quantization_utils.quant_modules import *
from pytorchcv.models.common import ConvBlock
from pytorchcv.models.shufflenetv2 import ShuffleUnit, ShuffleInitBlock
import time
import logging


class Q_AlexNet(nn.Module):
    """
        Quantized Alexnet model for dataset CIFAR100, CIFAR10
    """
    def __init__(self, model):
        super().__init__()
        features = getattr(model, 'features')

        self.quant_input = QuantAct()

        self.channel = [1, 1, 3]

        for stage_num in range(0, 3):
            stage = getattr(features, "stage{}".format(stage_num + 1))
            for unit_num in range(0, self.channel[stage_num]):
                unit = getattr(stage, 'unit{}'.format(unit_num + 1))
                quant_conv = Q_AlexConv()
                quant_conv.set_param(unit)
                pool = getattr(stage, 'pool{}'.format(stage_num + 1))
                setattr(self, f'stage{stage_num + 1}.unit{unit_num + 1}', quant_conv)
                setattr(self, f'stage{stage_num + 1}.unit{stage_num + 1}', pool)


        self.quant_act_output = QuantAct()
        output = getattr(model, 'output')
        self.quant_output = Q_AlexOutputBlock()
        self.quant_output.set_param(output)

    def forward(self, x):
        x, act_scaling_factor = self.quant_input(x)

        for stage_num in range(0, 3):
            for unit_num in range(0, self.channel[stage_num]):
                tmp_alex_conv = getattr(self, f'stage{stage_num + 1}.unit{unit_num + 1}.quant_conv')
                print(tmp_alex_conv)
                x, act_scaling_factor = tmp_alex_conv(x, act_scaling_factor)


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
                print(tmp_func)
                tmp_func.runtime_helper = runtime_helper
                if isinstance(tmp_func, QuantAct):
                    tmp_func.set_daq_ema_params(runtime_helper)

        self.final_pool.runtime_helper = runtime_helper
        self.quant_act_output.runtime_helper = runtime_helper
        self.quant_output.runtime_helper = runtime_helper

        self.quant_input.set_daq_ema_params(runtime_helper)
        self.quant_act_int32.set_daq_ema_params(runtime_helper)
        self.quant_act_output.set_daq_ema_params(runtime_helper)

class Q_AlexConv(nn.Module):
    """
        Quantized AlexNet unit
    """
    def __init__(self):
        super(Q_AlexConv, self).__init__()

    def set_param(self, unit):
        self.quant_act = QuantAct()

        conv = unit.conv
        self.quant_conv = QuantConv2d()
        self.quant_conv.set_param(conv)

    def forward(self, x, scaling_factor_int32=None):
        x, act_scaling_factor = self.quant_act(x, scaling_factor_int32)

        x, weight_scaling_factor = self.quant_conv(x, act_scaling_factor)
        x = nn.ReLU()(x)

        return x,


class Q_AlexOutputBlock(nn.Module):
    """
        Quantized AlexNet output Block
    """
    def __init__(self):
        super(Q_AlexOutputBlock, self).__init__()

    def set_param(self, block):
        self.quant_act = QuantAct()

        fc1 = block.fc1
        self.quant_fc1 = QuantLinear()
        self.quant_fc1.set_param(fc1.fc)
        self.quant_fc1.is_classifier = False
        self.quant_act1 = QuantAct()

        fc2 = block.fc2
        self.quant_fc2 = QuantLinear()
        self.quant_fc2.set_param(fc2.fc)
        self.quant_fc2.is_classifier = False
        self.quant_act2 = QuantAct()

        fc3 = block.fc3
        self.quant_fc3 = QuantLinear()
        self.quant_fc3.set_param(fc3)
        self.quant_fc3.is_classifier = True
        self.quant_act3 = QuantAct()


    def forward(self,x , scaling_factor_int32=None):
        x, act_scaling_factor = self.quant_act(x, scaling_factor_int32)

        x, weight_scaling_factor = self.quant_fc1(x, act_scaling_factor)
        x = nn.ReLU()(x)
        x = nn.Dropout()(x)
        x, act_scaling_factor = self.quant_act1(x, act_scaling_factor, weight_scaling_factor)

        x, weight_scaling_factor = self.quant_fc21(x, act_scaling_factor)
        x = nn.ReLU()(x)
        x = nn.Dropout()(x)
        x, act_scaling_factor = self.quant_act2(x, act_scaling_factor, weight_scaling_factor)

        x = self.quant_fc3(x, act_scaling_factor)

        return x

def q_alexnet(model):
    net = Q_AlexNet(model)
    return net
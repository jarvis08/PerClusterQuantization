
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
    def __init__(self, model, model_dict=None):
        super().__init__()
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
        # self.avgpool = QuantAveragePool2d(output=(6, 6))
        self.avgpool = QuantAveragePool2d(output=(1, 1))

        output = getattr(model, 'output')
        # fc_block = getattr(features, 'fc')

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

    def toggle_full_precision(self):
        print('Model Toggle full precision FUNC')
        for module in self.modules():
            if isinstance(module, (QuantAct, QuantLinear, QuantBnConv2d, QuantConv2d)):
                precision = getattr(module, 'full_precision_flag')
                if precision:
                    precision = False
                else:
                    precision = True
                setattr(module, 'full_precision_flag', precision)

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


class Q_AlexNet_Daq(nn.Module):
    def __init__(self, model, model_dict=None, runtime_helper=None):
        super().__init__()
        features = getattr(model, 'features')

        self.quant_input = QuantAct_Daq(runtime_helper=runtime_helper)

        # stage1
        stage = getattr(features, 'stage1')
        conv_block = getattr(stage, 'unit1')
        conv_block.conv.in_channels = 3
        conv_block.conv.out_channels = 96
        conv_block.conv.stride = (1, 1)
        conv_block.conv.kernel_size = (5, 5)
        conv_block.conv.padding = (2, 2)
        # conv_block = getattr(features, 'conv')

        self.conv1 = QuantConv2d()
        self.conv1.set_param(conv_block.conv, model_dict, 'features.stage1.unit1.conv')
        self.act1 = nn.ReLU()
        self.quant_act1 = QuantAct_Daq(runtime_helper=runtime_helper)

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
        self.quant_act2 = QuantAct_Daq(runtime_helper=runtime_helper)

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
        self.quant_act3 = QuantAct_Daq(runtime_helper=runtime_helper)

        conv_block = getattr(stage, 'unit2')
        conv_block.conv.in_channels = 384
        conv_block.conv.out_channels = 384
        conv_block.conv.kernel_size = (3, 3)
        conv_block.conv.stride = (1, 1)
        conv_block.conv.padding = (1, 1)
        self.conv4 = QuantConv2d()
        self.conv4.set_param(conv_block.conv, model_dict, 'features.stage3.unit2.conv')
        self.act4 = nn.ReLU()
        self.quant_act4 = QuantAct_Daq(runtime_helper=runtime_helper)

        conv_block = getattr(stage, 'unit3')
        conv_block.conv.in_channels = 384
        conv_block.conv.out_channels = 256
        conv_block.conv.kernel_size = (3, 3)
        conv_block.conv.stride = (1, 1)
        conv_block.conv.padding = (1, 1)
        self.conv5 = QuantConv2d()
        self.conv5.set_param(conv_block.conv, model_dict, 'features.stage3.unit3.conv')
        self.act5 = nn.ReLU()
        self.quant_act5 = QuantAct_Daq(runtime_helper=runtime_helper)

        self.maxpool3 = QuantMaxPool2d(ceil_mode=True)
        # self.avgpool = QuantAveragePool2d(output=(6, 6))
        self.avgpool = QuantAveragePool2d(output=(1, 1))

        output = getattr(model, 'output')
        # fc_block = getattr(features, 'fc')

        # fc1
        fc_block = getattr(output, 'fc1')
        fc_block.fc.in_features = 256
        fc_block.fc.out_features = 4096
        self.fc1 = QuantLinear()
        self.fc1.set_param(fc_block.fc, model_dict, 'output.fc1.fc')
        self.act6 = nn.ReLU()
        self.quant_act6 = QuantAct_Daq(runtime_helper=runtime_helper)
        self.quant_act6.is_classifier = True

        # fc2
        fc_block = getattr(output, 'fc2')
        fc_block.fc.in_features = 4096
        fc_block.fc.out_features = 4096
        self.fc2 = QuantLinear()
        self.fc2.set_param(fc_block.fc, model_dict, 'output.fc2.fc')
        self.act7 = nn.ReLU()
        self.quant_act7 = QuantAct_Daq(runtime_helper=runtime_helper)
        self.quant_act7.is_classifier = True

        # fc3
        fc = getattr(output, 'fc3')
        fc.in_features = 4096
        fc.out_features = 10
        self.fc3 = QuantLinear()
        self.fc3.is_classifier = True
        self.fc3.set_param(fc, model_dict, 'output.fc3')

    def toggle_full_precision(self):
        print('Model Toggle full precision FUNC')
        for module in self.modules():
            if isinstance(module, (QuantAct_Daq, QuantLinear, QuantBnConv2d, QuantConv2d)):
                precision = getattr(module, 'full_precision_flag')
                if precision:
                    precision = False
                else:
                    precision = True
                setattr(module, 'full_precision_flag', precision)


    def initialize_counter(self, x, n_clusters):
        self.zero_counter = []

        self.features = nn.Sequential(self.conv1, self.act1, self.maxpool1,
                                      self.conv2, self.act2, self.maxpool2,
                                      self.conv3, self.act3,
                                      self.conv4, self.act4,
                                      self.conv5, self.act5)

        x, _ = self.features[0](x)
        x = self.features[1](x)

        n_features = x.view(-1).size(0)
        self.zero_counter.append(torch.zeros((n_clusters, n_features), device='cuda'))
        
        x, _ = self.features[2](x)
        x, _ = self.features[3](x)
        x = self.features[4](x)
        
        n_features = x.view(-1).size(0)
        self.zero_counter.append(torch.zeros((n_clusters, n_features), device='cuda'))

        x, _ = self.features[5](x)
        x, _ = self.features[6](x)
        x = self.features[7](x)
        
        n_features = x.view(-1).size(0)
        self.zero_counter.append(torch.zeros((n_clusters, n_features), device='cuda'))

        x, _ = self.features[8](x)
        x = self.features[9](x)
        
        n_features = x.view(-1).size(0)
        self.zero_counter.append(torch.zeros((n_clusters, n_features), device='cuda'))

        x, _ = self.features[10](x)
        x = self.features[11](x)
        
        n_features = x.view(-1).size(0)
        self.zero_counter.append(torch.zeros((n_clusters, n_features), device='cuda'))

        
    def count_zeros_per_index(self, x, cluster, n_clusters):
        if not hasattr(self, 'zero_counter'):
            self.initialize_counter(x[0].unsqueeze(0), n_clusters)

        x, _ = self.features[0](x)
        x = self.features[1](x)

        layer_idx = 0
        n_features = self.zero_counter[layer_idx].size(1)
        for idx in range(x.size(0)):
            flattened = x[idx].view(-1)
            zeros_idx = (flattened == 0.0).nonzero(as_tuple=True)[0]
            zeros_idx %= n_features
            self.zero_counter[layer_idx][cluster, zeros_idx] += 1
        
        x, _ = self.features[2](x)
        x, _ = self.features[3](x)
        x = self.features[4](x)

        layer_idx += 1
        n_features = self.zero_counter[layer_idx].size(1)
        for idx in range(x.size(0)):
            flattened = x[idx].view(-1)
            zeros_idx = (flattened == 0.0).nonzero(as_tuple=True)[0]
            zeros_idx %= n_features
            self.zero_counter[layer_idx][cluster, zeros_idx] += 1

        x, _ = self.features[5](x)
        x, _ = self.features[6](x)
        x = self.features[7](x)

        layer_idx += 1
        n_features = self.zero_counter[layer_idx].size(1)
        for idx in range(x.size(0)):
            flattened = x[idx].view(-1)
            zeros_idx = (flattened == 0.0).nonzero(as_tuple=True)[0]
            zeros_idx %= n_features
            self.zero_counter[layer_idx][cluster, zeros_idx] += 1

        x, _ = self.features[8](x)
        x = self.features[9](x)

        layer_idx += 1
        n_features = self.zero_counter[layer_idx].size(1)
        for idx in range(x.size(0)):
            flattened = x[idx].view(-1)
            zeros_idx = (flattened == 0.0).nonzero(as_tuple=True)[0]
            zeros_idx %= n_features
            self.zero_counter[layer_idx][cluster, zeros_idx] += 1

        x, _ = self.features[10](x)
        x = self.features[11](x)

        layer_idx += 1
        n_features = self.zero_counter[layer_idx].size(1)
        for idx in range(x.size(0)):
            flattened = x[idx].view(-1)
            zeros_idx = (flattened == 0.0).nonzero(as_tuple=True)[0]
            zeros_idx %= n_features
            self.zero_counter[layer_idx][cluster, zeros_idx] += 1


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

        # x = torch.flatten(x, 1)
        x = x.view(x.size(0), -1)

        x, fc_scaling_factor = self.fc1(x, act_scaling_factor)
        x = self.act6(x)
        x, act_scaling_factor = self.quant_act6(x, act_scaling_factor, fc_scaling_factor)

        x, fc_scaling_factor = self.fc2(x, act_scaling_factor)
        x = self.act7(x)
        x, act_scaling_factor = self.quant_act7(x, act_scaling_factor, fc_scaling_factor)

        x = self.fc3(x, act_scaling_factor)
        return x


def q_alexnet(model, model_dict=None, runtime_helper=None):
    if runtime_helper is None:
        return Q_AlexNet(model, model_dict)
    else:
        return Q_AlexNet_Daq(model, model_dict, runtime_helper)


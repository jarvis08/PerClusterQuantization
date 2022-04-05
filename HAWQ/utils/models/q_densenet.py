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

        self.quant_init_convbn = QuantBnConv2d()
        self.quant_init_convbn.set_param(init_block.conv, init_block.bn)

        self.act1 = nn.ReLU(inplace=True)

        self.pool = QuantMaxPool2d(kernel_size=3, stride=2, padding=1)
        self.quant_act1 = QuantAct()

        units = [6, 12, 24, 16] 

        for stage_num in range(4):
            stage = getattr(features, "stage{}".format(stage_num + 1)) 

            if stage_num != 0:
                trans = getattr(stage, "trans{}".format(stage_num + 1)) 
                quant_trans = Q_Transition()
                quant_trans.set_param(trans)
                setattr(self, f'trans{stage_num +1}', quant_trans)

            dense_block = Q_DenseBlock() 
            dense_block.set_param(stage, units[stage_num])
            setattr(self, f'stage{stage_num + 1}', dense_block)

        post_activ = getattr(features, 'post_activ')
        self.batch_norm = QuantBn()
        self.batch_norm.set_param(post_activ.bn)

        self.act2 = nn.ReLU(inplace=True)
        self.quant_act2 = QuantAct()

        self.final_pool = QuantAveragePool2d(kernel_size=7, stride=1)
     
        self.quant_act_output = QuantAct()

        output = getattr(model, 'output')
        self.quant_output = QuantLinear()
        self.quant_output.is_classifier = True
        self.quant_output.set_param(output)

    def forward(self, input):
        x, act_scaling_factor = self.quant_input(input)

        x, weight_scaling_factor = self.quant_init_convbn(x, act_scaling_factor)

        x, act_scaling_factor = self.pool(x, act_scaling_factor)
        x = self.act1(x) 
        x, act_scaling_factor = self.quant_act1(x, act_scaling_factor, weight_scaling_factor)

        for stage_num in range(4):
            if stage_num != 0:
                transition = getattr(self, f'trans{stage_num + 1}')
                x, act_scaling_factor = transition(x, act_scaling_factor)
            function = getattr(self, f'stage{stage_num + 1}')
            x, act_scaling_factor = function(x, act_scaling_factor)

        x, bn_scaling_factor = self.batch_norm(x, act_scaling_factor)
        x = self.act2(x)
        x, act_scaling_factor = self.quant_act2(x, act_scaling_factor, bn_scaling_factor)

        x, act_scaling_factor = self.final_pool(x, act_scaling_factor)
        x, act_scaling_factor = self.quant_act_output(x, act_scaling_factor)

        x = x.view(x.size(0), -1)
        x = self.quant_output(x, act_scaling_factor)
        return x


class Q_Transition(nn.Module):
    def __init__(self):
        super(Q_Transition, self).__init__()

    def set_param(self, trans):
        conv_block = getattr(trans, 'conv')

        self.batch_norm = QuantBn()
        self.batch_norm.set_param(conv_block.bn)
        self.act = nn.ReLU(inplace=True)
        self.quant_act1 = QuantAct()

        self.conv = QuantConv2d()
        self.conv.set_param(conv_block.conv)

        self.quant_act2 = QuantAct()

        self.pool = QuantAveragePool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, x, act_scaling_factor=None):
        x, bn_scaling_factor = self.batch_norm(x, act_scaling_factor)

        x = self.act(x)
        x, act_scaling_factor = self.quant_act1(x, act_scaling_factor, bn_scaling_factor)

        x, conv_scaling_factor = self.conv(x, act_scaling_factor)
        x, act_scaling_factor = self.quant_act2(x, act_scaling_factor, conv_scaling_factor)

        x, act_scaling_factor = self.pool(x, act_scaling_factor)
        return x, act_scaling_factor



class Q_DenseUnit(nn.Module):
    def __init__(self):
        super(Q_DenseUnit, self).__init__()

    def set_param(self, unit):
        layer1 = getattr(unit, "conv1")
        self.quant_bn1 = QuantBn()
        self.quant_bn1.set_param(layer1.bn)
        self.act1 = nn.ReLU(inplace=True)
        self.quant_act1 = QuantAct()

        layer2 = getattr(unit, "conv2")
        self.quant_convbn = QuantBnConv2d()
        self.quant_convbn.set_param(layer1.conv, layer2.bn)

        self.act2 = nn.ReLU(inplace=True)
        self.quant_act2 = QuantAct()

        self.quant_conv2 = QuantConv2d()
        self.quant_conv2.set_param(layer2.conv)
        self.quant_act_output = QuantAct()


    def forward(self, batch, input_scaling_factor=None):
        x, bn_scaling_factor = self.quant_bn1(batch, input_scaling_factor)
        x = self.act1(x)
        x, act_scaling_factor = self.quant_act1(x, input_scaling_factor, bn_scaling_factor)

        x, weight_scaling_factor = self.quant_convbn(x, act_scaling_factor)
        x = self.act2(x)
        x, act_scaling_factor = self.quant_act2(x, act_scaling_factor, weight_scaling_factor)

        x, conv_scaling_factor = self.quant_conv2(x, act_scaling_factor)

        concat_tensor = torch.cat((batch, x), 1)
        x, act_scaling_factor = self.quant_act_output(concat_tensor, act_scaling_factor, conv_scaling_factor, 
                                                concat=True, concat_scaling_factor=input_scaling_factor)
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
        for unit_num in range(self.layers):
            function = getattr(self, f'unit{unit_num + 1}')
            x, act_scaling_factor = function(x, act_scaling_factor)
        return x, act_scaling_factor



class Q_DenseNet_Daq(nn.Module):
    def __init__(self, model, runtime_helper=None):
        super().__init__()
        features = getattr(model, 'features')
        init_block = getattr(features, 'init_block')

        self.quant_input = QuantAct_Daq(runtime_helper=runtime_helper)

        self.quant_init_convbn = QuantBnConv2d()
        self.quant_init_convbn.set_param(init_block.conv, init_block.bn)
        self.quant_act1 = QuantAct_Daq(runtime_helper=runtime_helper)
        self.act1 = nn.ReLU(inplace=True)

        self.pool = QuantMaxPool2d(kernel_size=3, stride=2, padding=1)

        units = [6, 12, 24, 16] 

        for stage_num in range(4):
            stage = getattr(features, "stage{}".format(stage_num + 1)) 

            if stage_num is not 0:
                trans = getattr(stage, "trans{}".format(stage_num + 1)) 
                quant_trans = Q_Transition_Daq()
                quant_trans.set_param(trans, runtime_helper)
                setattr(self, f'trans{stage_num +1}', quant_trans)

            dense_block = Q_DenseBlock_Daq() 
            dense_block.set_param(stage, units[stage_num], runtime_helper)
            setattr(self, f'stage{stage_num + 1}', dense_block)

        post_activ = getattr(features, 'post_activ')
        self.batch_norm = QuantBn()
        self.batch_norm.set_param(post_activ.bn)

        self.act2 = nn.ReLU(inplace=True)
        self.quant_act2 = QuantAct_Daq(runtime_helper=runtime_helper)

        self.final_pool = QuantAveragePool2d(kernel_size=7, stride=1)
     
        self.quant_act_output = QuantAct_Daq(runtime_helper=runtime_helper)

        output = getattr(model, 'output')
        self.quant_output = QuantLinear()
        self.quant_output.is_classifier = True
        self.quant_output.set_param(output)

    def forward(self, input):
        x, act_scaling_factor = self.quant_input(input)

        x, weight_scaling_factor = self.quant_init_convbn(x, act_scaling_factor)

        x, act_scaling_factor = self.pool(x, act_scaling_factor)
        x = self.act1(x) 
        x, act_scaling_factor = self.quant_act1(x, act_scaling_factor, weight_scaling_factor)

        for stage_num in range(4):
            if stage_num != 0:
                transition = getattr(self, f'trans{stage_num + 1}')
                x, act_scaling_factor = transition(x, act_scaling_factor)
            function = getattr(self, f'stage{stage_num + 1}')
            x, act_scaling_factor = function(x, act_scaling_factor)

        x, bn_scaling_factor = self.batch_norm(x, act_scaling_factor)
        x = self.act2(x)
        x, act_scaling_factor = self.quant_act2(x, act_scaling_factor, bn_scaling_factor)

        x, act_scaling_factor = self.final_pool(x, act_scaling_factor)
        x, act_scaling_factor = self.quant_act_output(x, act_scaling_factor)

        x = x.view(x.size(0), -1)
        x = self.quant_output(x, act_scaling_factor)
        return x

    #### NNAC Helper ####
    def toggle_full_precision(self):
        print('Model Toggle full precision FUNC')
        for module in self.modules():
            if isinstance(module, (QuantAct_Daq, QuantLinear, QuantBnConv2d, QuantBn, QuantConv2d)):
                precision = getattr(module, 'full_precision_flag')
                if precision:
                    precision = False
                else:
                    precision = True
                setattr(module, 'full_precision_flag', precision)
    
    def initialize_counter(self, x, n_clusters):
        self.zero_counter = []

        self.features = nn.Sequential(self.quant_init_convbn, self.act1, self.pool)
        
        x, _ = self.features[0](x)
        x = self.features[1](x)
        x, _ = self.features[2](x)

        n_features = x.view(-1).size(0)
        self.zero_counter.append(torch.zeros((n_clusters, n_features), device='cuda'))

        for stage_num in range(0,4):
            if stage_num != 0:
                transition = getattr(self, f'trans{stage_num + 1}')
                x, self.zero_counter = transition.initialize_counter(x, n_clusters, self.zero_counter)
                setattr(self, f'trans{stage_num +1}', transition)
            tmp_func = getattr(self, f'stage{stage_num + 1}.unit{unit_num + 1}')
            x = tmp_func.initialize_counter(x, n_clusters, self.zero_counter)
            setattr(self, f'stage{stage_num + 1}', tmp_func)
        
        self.classifiers = nn.Sequential(self.batch_norm, self.act2)

        x, _ = self.classifiers[0](x)
        x = self.classifiers[1](x)

        n_features = x.view(-1).size(0)
        self.zero_counter.append(torch.zeros((n_clusters, n_features), device='cuda'))

    def count_zeros_per_index(self, x, cluster, n_clusters):
        if not hasattr(self, 'zero_counter'):
            self.initialize_counter(x[0].unsqueeze(0), n_clusters)

        x, _ = self.features[0](x)
        x = self.features[1](x)
        x, _ = self.features[2](x)

        layer_idx = 0
        n_features = self.zero_counter[layer_idx].size(1)
        for idx in range(x.size(0)):
            flattened = x[idx].view(-1)
            zeros_idx = (flattened == 0.0).nonzero(as_tuple=True)[0]
            zeros_idx %= n_features
            self.zero_counter[layer_idx][cluster, zeros_idx] += 1
    
        for stage_num in range(0,4):
            if stage_num != 0:
                transition = getattr(self, f'trans{stage_num + 1}')
                x, layer_idx = transition.count_zeros_per_index(x, layer_idx, cluster, n_clusters)
            tmp_func = getattr(self, f'stage{stage_num + 1}.unit{unit_num + 1}')
            x, layer_idx = tmp_func.count_zeros_per_index(x, layer_idx, cluster, n_clusters)
        
        x, _ = self.classifiers[0](x)
        x = self.classifiers[1](x)

        layer_idx += 1
        n_features = self.zero_counter[layer_idx].size(1)
        for idx in range(x.size(0)):
            flattened = x[idx].view(-1)
            zeros_idx = (flattened == 0.0).nonzero(as_tuple=True)[0]
            zeros_idx %= n_features
            self.zero_counter[layer_idx][cluster, zeros_idx] += 1



class Q_Transition_Daq(nn.Module):
    def __init__(self):
        super(Q_Transition, self).__init__()

    def set_param(self, trans, runtime_helper=None):
        conv_block = getattr(trans, 'conv')

        self.batch_norm = QuantBn()
        self.batch_norm.set_param(conv_block.bn)
        self.act = nn.ReLU(inplace=True)
        self.quant_act1 = QuantAct_Daq(runtime_helper=runtime_helper)

        self.conv = QuantConv2d()
        self.conv.set_param(conv_block.conv)

        self.quant_act2 = QuantAct_Daq(runtime_helper=runtime_helper)

        self.pool = QuantAveragePool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, x, act_scaling_factor=None):
        x, bn_scaling_factor = self.batch_norm(x, act_scaling_factor)

        x = self.act(x)
        x, act_scaling_factor = self.quant_act1(x, act_scaling_factor, bn_scaling_factor)

        x, conv_scaling_factor = self.conv(x, act_scaling_factor)
        x, act_scaling_factor = self.quant_act2(x, act_scaling_factor, conv_scaling_factor)

        x, act_scaling_factor = self.pool(x, act_scaling_factor)
        return x, act_scaling_factor

    def initialize_counter(self, x, n_clusters, zero_counter):
        self.zero_counter = zero_counter
        
        self.features = nn.Sequential(self.batch_norm, self.act, 
                                      self.conv, self.pool)

        x = self.features[0](x)
        x = self.features[1](x)

        n_features = x.view(-1).size(0)
        self.zero_counter.append(torch.zeros((n_clusters, n_features), device='cuda'))

        x = self.features[2](x)
        x = self.features[3](x)
        return x

    def count_zeros_per_index(self, x, layer_idx, cluster, n_clusters):
        x = self.features[0](x)
        x = self.features[1](x)

        layer_idx += 1
        n_features = self.zero_counter[layer_idx].size(1)
        for idx in range(x.size(0)):
            flattened = x[idx].view(-1)
            zeros_idx = (flattened == 0.0).nonzero(as_tuple=True)[0]
            zeros_idx %= n_features
            self.zero_counter[layer_idx][cluster, zeros_idx] += 1

        x = self.features[2](x)
        x = self.features[3](x)
        return x, layer_idx


class Q_DenseUnit_Daq(nn.Module):
    def __init__(self):
        super(Q_DenseUnit, self).__init__()

    def set_param(self, unit, runtime_helper=None):
        layer1 = getattr(unit, "conv1")
        self.quant_bn1 = QuantBn()
        self.quant_bn1.set_param(layer1.bn)
        self.act1 = nn.ReLU(inplace=True)
        self.quant_act1 = QuantAct_Daq(runtime_helper=runtime_helper)

        layer2 = getattr(unit, "conv2")
        self.quant_convbn = QuantBnConv2d()
        self.quant_convbn.set_param(layer1.conv, layer2.bn)

        self.act2 = nn.ReLU(inplace=True)
        self.quant_act3 = QuantAct_Daq(runtime_helper=runtime_helper)

        self.quant_conv2 = QuantConv2d()
        self.quant_conv2.set_param(layer2.conv)
        self.quant_act_output = QuantAct_Daq(runtime_helper=runtime_helper)


    def forward(self, batch, input_scaling_factor=None):
        x, bn_scaling_factor = self.quant_bn1(batch, input_scaling_factor)
        x = self.act1(x)
        x, act_scaling_factor = self.quant_act1(x, input_scaling_factor, bn_scaling_factor)

        x, weight_scaling_factor = self.quant_convbn(x, act_scaling_factor)
        x = self.act2(x)
        x, act_scaling_factor = self.quant_act2(x, act_scaling_factor, weight_scaling_factor)

        x, conv_scaling_factor = self.quant_conv2(x, act_scaling_factor)

        concat_tensor = torch.cat((batch, x), 1)
        x, act_scaling_factor = self.quant_act_output(concat_tensor, act_scaling_factor, conv_scaling_factor, 
                                                concat=True, concat_scaling_factor=input_scaling_factor)
        return x, act_scaling_factor

    def initialize_counter(self, x, n_clusters, zero_counter):
        self.zero_counter = zero_counter
        
        self.features = nn.Sequential(self.quant_bn1, self.act1,
                                      self.quant_convbn, self.act2,
                                      self.quant_conv2)

        batch = x

        x = self.features[0](x)
        x = self.features[1](x)

        n_features = x.view(-1).size(0)
        self.zero_counter.append(torch.zeros((n_clusters, n_features), device='cuda'))

        x = self.features[2](x)
        x = self.features[3](x)
        
        n_features = x.view(-1).size(0)
        self.zero_counter.append(torch.zeros((n_clusters, n_features), device='cuda'))

        x = self.features[4](x)
        x = torch.cat((batch, x), 1)
        
        n_features = x.view(-1).size(0)
        self.zero_counter.append(torch.zeros((n_clusters, n_features), device='cuda'))

        return x

    def count_zeros_per_index(self, x, layer_idx, cluster, n_clusters):
        x = self.features[0](x)
        x = self.features[1](x)

        layer_idx += 1
        n_features = self.zero_counter[layer_idx].size(1)
        for idx in range(x.size(0)):
            flattened = x[idx].view(-1)
            zeros_idx = (flattened == 0.0).nonzero(as_tuple=True)[0]
            zeros_idx %= n_features
            self.zero_counter[layer_idx][cluster, zeros_idx] += 1

        x = self.features[2](x)
        x = self.features[3](x)
        
        layer_idx += 1
        n_features = self.zero_counter[layer_idx].size(1)
        for idx in range(x.size(0)):
            flattened = x[idx].view(-1)
            zeros_idx = (flattened == 0.0).nonzero(as_tuple=True)[0]
            zeros_idx %= n_features
            self.zero_counter[layer_idx][cluster, zeros_idx] += 1

        x = self.features[4](x)
        x = torch.cat((batch, x), 1)

        layer_idx += 1
        n_features = self.zero_counter[layer_idx].size(1)
        for idx in range(x.size(0)):
            flattened = x[idx].view(-1)
            zeros_idx = (flattened == 0.0).nonzero(as_tuple=True)[0]
            zeros_idx %= n_features
            self.zero_counter[layer_idx][cluster, zeros_idx] += 1

        return x, layer_idx


class Q_DenseBlock_Daq(nn.Module):
    def __init__(self):
        super(Q_DenseBlock, self).__init__()

    def set_param(self, stage, layers, runtime_helper=None):
        self.layers = layers
        for unit_num in range(self.layers):
            unit = getattr(stage, 'unit{}'.format(unit_num + 1))
            quant_unit = Q_DenseUnit_Daq()
            quant_unit.set_param(unit, runtime_helper)
            setattr(self, f'unit{unit_num + 1}', quant_unit)

    def forward(self, x, act_scaling_factor=None):
        for unit_num in range(self.layers):
            function = getattr(self, f'unit{unit_num + 1}')
            x, act_scaling_factor = function(x, act_scaling_factor)
        return x, act_scaling_factor

    def initialize_counter(self, x, n_clusters, zero_counter):
        self.zero_counter = zero_counter
        for unit_num in range(self.layers):
            function = getattr(self, f'unit{unit_num + 1}')
            x = function.initialize_counter(x, n_clusters, self.zero_counter)
        return x, self.zero_counter
        
    def count_zeros_per_index(self, x, layer_idx, cluster, n_clusters):
        for unit_num in range(self.layers):
            function = getattr(self, f'unit{unit_num + 1}')
            x, layer_idx = function.count_zeros_per_index(x, layer_idx, cluster, n_clusters)
        return x, layer_idx



def q_densenet(model, runtime_helper=None):
    if runtime_helper is None:
        net = Q_DenseNet(model)
    else:
        net = Q_DenseNet_Daq(model, runtime_helper)
    return net



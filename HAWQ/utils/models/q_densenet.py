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
            dense_block.set_param(stage, units[stage_num], num_clusters=num_clusters)
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

        x, weight_scaling_factor = self.quant_init_convbn(x, act_scaling_factor)
        x = self.act1(x) 
        x, act_scaling_factor = self.pool(x, act_scaling_factor)
        x, act_scaling_factor = self.quant_act1(x, act_scaling_factor, weight_scaling_factor, cluster=cluster)

        for stage_num in range(4):
            if stage_num != 0:
                transition = getattr(self, f'trans{stage_num + 1}')
                x, act_scaling_factor = transition(x, act_scaling_factor, cluster=cluster)
            function = getattr(self, f'stage{stage_num + 1}')
            x, act_scaling_factor = function(x, act_scaling_factor, cluster=cluster)

        x, bn_scaling_factor = self.batch_norm(x, act_scaling_factor)
        x = self.act2(x)
        x, act_scaling_factor = self.quant_act2(x, act_scaling_factor, bn_scaling_factor, cluster=cluster)

        x, act_scaling_factor = self.final_pool(x, act_scaling_factor)
        x, act_scaling_factor = self.quant_act_output(x, act_scaling_factor, cluster=cluster)

        x = x.view(x.size(0), -1)
        x = self.quant_output(x, act_scaling_factor)
        return x

    #### NNAC Helper ####
    def toggle_full_precision(self):
        print('Model Toggle full precision FUNC')
        for module in self.modules():
            if isinstance(module, (QuantAct, QuantLinear, QuantBnConv2d, QuantBn, QuantConv2d)):
                precision = getattr(module, 'full_precision_flag')
                if precision:
                    precision = False
                else:
                    precision = True
                setattr(module, 'full_precision_flag', precision)
    
    def delete_counters(self):
        # del self.zero_counter
        # del self.max_counter
        del self.max_accumulator
        
    def update_max_accumulator(self, x, cluster, l_idx):
        if type(x) is tuple:
            x = x[0]
        _max = torch.scatter_reduce(self.zero_buffer, 0, cluster, src=x.view(x.size(0), -1).max(dim=1).values, reduce=self.reduce)
        self.max_accumulator[l_idx] = self.max_accumulator[l_idx].max(_max)
        return l_idx + 1    
    
    def accumulate_output_max_distribution(self, x, cluster, n_clusters, l_idx=0, reduce='amax'):
        if not hasattr(self, 'max_accumulator'):
            self.reduce = reduce
            self.max_accumulator = torch.zeros([181, n_clusters]).cuda()
            self.zero_buffer = torch.zeros(n_clusters).cuda()
    
        x, act_scaling_factor = self.quant_input(x, cluster=cluster)
        x, weight_scaling_factor = self.quant_init_convbn(x, act_scaling_factor)
        x = self.act1(x) 
        x, act_scaling_factor = self.pool(x, act_scaling_factor)
        
        ##################################
        l_idx = self.update_max_accumulator(x, cluster, l_idx)
        ##################################
        
        x, act_scaling_factor = self.quant_act1(x, act_scaling_factor, weight_scaling_factor, cluster=cluster)
        for stage_num in range(0,4):
            if stage_num != 0:
                transition = getattr(self, f'trans{stage_num + 1}')
                x, l_idx, act_scaling_factor = transition.accumulate_output_max_distribution(x, cluster, n_clusters, self,
                                                                                             l_idx, act_scaling_factor)
            tmp_func = getattr(self, f'stage{stage_num + 1}')
            x, l_idx, act_scaling_factor = tmp_func.accumulate_output_max_distribution(x, cluster, n_clusters, self, 
                                                                                       l_idx, act_scaling_factor)

    
    # def initialize_counter(self, x, n_clusters):
    #     self.zero_counter = []

    #     self.features = nn.Sequential(self.quant_init_convbn, self.act1, self.pool)
        
    #     x, _ = self.features[0](x)
    #     x = self.features[1](x)
    #     x, _ = self.features[2](x)

    #     n_features = x.view(-1).size(0)
    #     self.zero_counter.append(torch.zeros((n_clusters, n_features), device='cuda'))

    #     for stage_num in range(0,4):
    #         if stage_num != 0:
    #             transition = getattr(self, f'trans{stage_num + 1}')
    #             x = transition.initialize_counter(x, n_clusters, self.zero_counter)
    #             setattr(self, f'trans{stage_num +1}', transition)
    #         tmp_func = getattr(self, f'stage{stage_num + 1}')
    #         x = tmp_func.initialize_counter(x, n_clusters, self.zero_counter)
    #         setattr(self, f'stage{stage_num + 1}', tmp_func)
        
    #     self.classifiers = nn.Sequential(self.batch_norm, self.act2)

    #     x, _ = self.classifiers[0](x)
    #     x = self.classifiers[1](x)

    #     n_features = x.view(-1).size(0)
    #     self.zero_counter.append(torch.zeros((n_clusters, n_features), device='cuda'))

    # def count_zeros_per_index(self, x, cluster, n_clusters):
    #     if not hasattr(self, 'zero_counter'):
    #         self.initialize_counter(x[0].unsqueeze(0), n_clusters)

    #     x, _ = self.features[0](x)
    #     x = self.features[1](x)
    #     x, _ = self.features[2](x)

    #     layer_idx = 0
    #     n_features = self.zero_counter[layer_idx].size(1)
    #     for idx in range(x.size(0)):
    #         flattened = x[idx].view(-1)
    #         zeros_idx = (flattened == 0.0).nonzero(as_tuple=True)[0]
    #         zeros_idx %= n_features
    #         self.zero_counter[layer_idx][cluster, zeros_idx] += 1
    
    #     for stage_num in range(0,4):
    #         if stage_num != 0:
    #             transition = getattr(self, f'trans{stage_num + 1}')
    #             x, layer_idx = transition.count_zeros_per_index(x, layer_idx, cluster)
    #         tmp_func = getattr(self, f'stage{stage_num + 1}')
    #         x, layer_idx = tmp_func.count_zeros_per_index(x, layer_idx, cluster)
        
    #     x, _ = self.classifiers[0](x)
    #     x = self.classifiers[1](x)

    #     layer_idx += 1
    #     n_features = self.zero_counter[layer_idx].size(1)
    #     for idx in range(x.size(0)):
    #         flattened = x[idx].view(-1)
    #         zeros_idx = (flattened == 0.0).nonzero(as_tuple=True)[0]
    #         zeros_idx %= n_features
    #         self.zero_counter[layer_idx][cluster, zeros_idx] += 1

    def get_output_max_distribution(self, x, cluster, n_clusters):
        initialized = True
        if not hasattr(self, 'max_counter'):
            initialized = False
            self.max_counter = []
            self.max_counter.append([[] for _ in range(n_clusters)])

        x, act_scaling_factor = self.quant_input(x, cluster=cluster)
        x, weight_scaling_factor = self.quant_init_convbn(x, act_scaling_factor)
        x = self.act1(x) 
        x, act_scaling_factor = self.pool(x, act_scaling_factor)
        
        l_idx = 0
        _max = x.view(x.size(0), -1).max(dim=1).values
        if self.max_counter[l_idx][cluster] == []:
            self.max_counter[l_idx][cluster] = _max
        else:
            self.max_counter[l_idx][cluster] = torch.cat([self.max_counter[l_idx][cluster], _max])
        
        x, act_scaling_factor = self.quant_act1(x, act_scaling_factor, weight_scaling_factor, cluster=cluster)
        for stage_num in range(0,4):
            if stage_num != 0:
                transition = getattr(self, f'trans{stage_num + 1}')
                x, l_idx, act_scaling_factor = transition.get_output_max_distribution(x, cluster, n_clusters, self.max_counter, l_idx,
                                                                                      initialized, act_scaling_factor)
            tmp_func = getattr(self, f'stage{stage_num + 1}')
            x, l_idx, act_scaling_factor = tmp_func.get_output_max_distribution(x, cluster, n_clusters, self.max_counter, l_idx, 
                                                                                initialized, act_scaling_factor)




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
        x, act_scaling_factor = self.quant_act1(x, act_scaling_factor, bn_scaling_factor, cluster=cluster)

        x, conv_scaling_factor = self.conv(x, act_scaling_factor)
        x, act_scaling_factor = self.quant_act2(x, act_scaling_factor, conv_scaling_factor, cluster=cluster)

        x, act_scaling_factor = self.pool(x, act_scaling_factor)
        return x, act_scaling_factor

    # def initialize_counter(self, x, n_clusters, zero_counter):
    #     self.zero_counter = zero_counter
        
    #     self.features = nn.Sequential(self.batch_norm, self.act, 
    #                                   self.conv, self.pool)

    #     x, _ = self.features[0](x)
    #     x = self.features[1](x)

    #     n_features = x.view(-1).size(0)
    #     self.zero_counter.append(torch.zeros((n_clusters, n_features), device='cuda'))

    #     x, _ = self.features[2](x)

    #     n_features = x.view(-1).size(0)
    #     self.zero_counter.append(torch.zeros((n_clusters, n_features), device='cuda'))

    #     x = self.features[3](x)
    #     return x

    # def count_zeros_per_index(self, x, layer_idx, cluster):
    #     x, _ = self.features[0](x)
    #     x = self.features[1](x)

    #     layer_idx += 1
    #     n_features = self.zero_counter[layer_idx].size(1)
    #     for idx in range(x.size(0)):
    #         flattened = x[idx].view(-1)
    #         zeros_idx = (flattened == 0.0).nonzero(as_tuple=True)[0]
    #         zeros_idx %= n_features
    #         self.zero_counter[layer_idx][cluster, zeros_idx] += 1

    #     x, _ = self.features[2](x)

    #     layer_idx += 1
    #     n_features = self.zero_counter[layer_idx].size(1)
    #     for idx in range(x.size(0)):
    #         flattened = x[idx].view(-1)
    #         zeros_idx = (flattened == 0.0).nonzero(as_tuple=True)[0]
    #         zeros_idx %= n_features
    #         self.zero_counter[layer_idx][cluster, zeros_idx] += 1
            
    #     x = self.features[3](x)
    #     return x, layer_idx

    def accumulate_output_max_distribution(self, x, cluster, n_clusters, head, l_idx, act_scaling_factor=None):
        x, bn_scaling_factor = self.batch_norm(x, act_scaling_factor)
        x = self.act(x)
        
        ##################################
        l_idx = head.update_max_accumulator(x, cluster, l_idx)
        ##################################
        
        x, act_scaling_factor = self.quant_act1(x, act_scaling_factor, bn_scaling_factor, cluster=cluster)
        x, conv_scaling_factor = self.conv(x, act_scaling_factor)
        
        ##################################
        l_idx = head.update_max_accumulator(x, cluster, l_idx)
        ##################################
            
        x, act_scaling_factor = self.quant_act2(x, act_scaling_factor, conv_scaling_factor, cluster=cluster)
        x, act_scaling_factor = self.pool(x, act_scaling_factor)

        return x, l_idx, act_scaling_factor

    def get_output_max_distribution(self, x, cluster, n_clusters, max_counter, l_idx, initialized, act_scaling_factor=None): 
        if not initialized:
            max_counter.append([[] for _ in range(n_clusters)])
            max_counter.append([[] for _ in range(n_clusters)])

        x, bn_scaling_factor = self.batch_norm(x, act_scaling_factor)
        x = self.act(x)
        
        l_idx += 1
        _max = x.view(x.size(0), -1).max(dim=1).values
        if max_counter[l_idx][cluster] == []:
            max_counter[l_idx][cluster] = _max
        else:
            max_counter[l_idx][cluster] = torch.cat([max_counter[l_idx][cluster], _max])
        
        x, act_scaling_factor = self.quant_act1(x, act_scaling_factor, bn_scaling_factor, cluster=cluster)
        x, conv_scaling_factor = self.conv(x, act_scaling_factor)
        
        l_idx += 1
        _max = x.view(x.size(0), -1).max(dim=1).values
        if max_counter[l_idx][cluster] == []:
            max_counter[l_idx][cluster] = _max
        else:
            max_counter[l_idx][cluster] = torch.cat([max_counter[l_idx][cluster], _max])
            
        x, act_scaling_factor = self.quant_act2(x, act_scaling_factor, conv_scaling_factor, cluster=cluster)
        x, act_scaling_factor = self.pool(x, act_scaling_factor)

        return x, l_idx, act_scaling_factor
    

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
        x, act_scaling_factor = self.quant_act1(x, input_scaling_factor, bn_scaling_factor, cluster=cluster)

        x, weight_scaling_factor = self.quant_convbn(x, act_scaling_factor)
        x = self.act2(x)
        x, act_scaling_factor = self.quant_act2(x, act_scaling_factor, weight_scaling_factor, cluster=cluster)

        x, conv_scaling_factor = self.quant_conv2(x, act_scaling_factor)

        concat_tensor = torch.cat((batch, x), 1)
        x, act_scaling_factor = self.quant_act_output(concat_tensor, act_scaling_factor, conv_scaling_factor, 
                                                concat=True, concat_scaling_factor=input_scaling_factor, cluster=cluster)
        return x, act_scaling_factor

    # def initialize_counter(self, x, n_clusters, zero_counter):
    #     self.zero_counter = zero_counter
        
    #     self.features = nn.Sequential(self.quant_bn1, self.act1,
    #                                   self.quant_convbn, self.act2,
    #                                   self.quant_conv2)

    #     batch = x

    #     x, _ = self.features[0](x)
    #     x = self.features[1](x)

    #     n_features = x.view(-1).size(0)
    #     self.zero_counter.append(torch.zeros((n_clusters, n_features), device='cuda'))

    #     x, _ = self.features[2](x)
    #     x = self.features[3](x)
        
    #     n_features = x.view(-1).size(0)
    #     self.zero_counter.append(torch.zeros((n_clusters, n_features), device='cuda'))

    #     x, _ = self.features[4](x)
    #     x = torch.cat((batch, x), 1)
        
    #     n_features = x.view(-1).size(0)
    #     self.zero_counter.append(torch.zeros((n_clusters, n_features), device='cuda'))

    #     return x

    # def count_zeros_per_index(self, x, layer_idx, cluster):
    #     batch = x

    #     x, _ = self.features[0](x)
    #     x = self.features[1](x)

    #     layer_idx += 1
    #     n_features = self.zero_counter[layer_idx].size(1)
    #     for idx in range(x.size(0)):
    #         flattened = x[idx].view(-1)
    #         zeros_idx = (flattened == 0.0).nonzero(as_tuple=True)[0]
    #         zeros_idx %= n_features
    #         self.zero_counter[layer_idx][cluster, zeros_idx] += 1

    #     x, _ = self.features[2](x)
    #     x = self.features[3](x)
        
    #     layer_idx += 1
    #     n_features = self.zero_counter[layer_idx].size(1)
    #     for idx in range(x.size(0)):
    #         flattened = x[idx].view(-1)
    #         zeros_idx = (flattened == 0.0).nonzero(as_tuple=True)[0]
    #         zeros_idx %= n_features
    #         self.zero_counter[layer_idx][cluster, zeros_idx] += 1

    #     x, _ = self.features[4](x)
    #     x = torch.cat((batch, x), 1)

    #     layer_idx += 1
    #     n_features = self.zero_counter[layer_idx].size(1)
    #     for idx in range(x.size(0)):
    #         flattened = x[idx].view(-1)
    #         zeros_idx = (flattened == 0.0).nonzero(as_tuple=True)[0]
    #         zeros_idx %= n_features
    #         self.zero_counter[layer_idx][cluster, zeros_idx] += 1

    #     return x, layer_idx

    def accumulate_output_max_distribution(self, batch, cluster, n_clusters, head, l_idx, input_scaling_factor=None):
        x, bn_scaling_factor = self.quant_bn1(batch, input_scaling_factor)
        x = self.act1(x)
        
        ##################################
        l_idx = head.update_max_accumulator(x, cluster, l_idx)
        ##################################
        
        x, act_scaling_factor = self.quant_act1(x, input_scaling_factor, bn_scaling_factor, cluster=cluster)
        x, weight_scaling_factor = self.quant_convbn(x, act_scaling_factor)
        x = self.act2(x)
        
        ##################################
        l_idx = head.update_max_accumulator(x, cluster, l_idx)
        ##################################
        
        x, act_scaling_factor = self.quant_act2(x, act_scaling_factor, weight_scaling_factor, cluster=cluster)
        x, conv_scaling_factor = self.quant_conv2(x, act_scaling_factor)

        concat_tensor = torch.cat((batch, x), 1)
        
        ##################################
        l_idx = head.update_max_accumulator(x, cluster, l_idx)
        ##################################
        
        x, act_scaling_factor = self.quant_act_output(concat_tensor, act_scaling_factor, conv_scaling_factor, 
                                                concat=True, concat_scaling_factor=input_scaling_factor, cluster=cluster)
        
        return x, l_idx, act_scaling_factor

    def get_output_max_distribution(self, batch, cluster, n_clusters, max_counter, l_idx, initialized, input_scaling_factor=None): 
        if not initialized:
            max_counter.append([[] for _ in range(n_clusters)])
            max_counter.append([[] for _ in range(n_clusters)])
            max_counter.append([[] for _ in range(n_clusters)])

        x, bn_scaling_factor = self.quant_bn1(batch, input_scaling_factor)
        x = self.act1(x)
        
        l_idx += 1
        _max = x.view(x.size(0), -1).max(dim=1).values
        if max_counter[l_idx][cluster] == []:
            max_counter[l_idx][cluster] = _max
        else:
            max_counter[l_idx][cluster] = torch.cat([max_counter[l_idx][cluster], _max])
        
        x, act_scaling_factor = self.quant_act1(x, input_scaling_factor, bn_scaling_factor, cluster=cluster)
        x, weight_scaling_factor = self.quant_convbn(x, act_scaling_factor)
        x = self.act2(x)
        
        l_idx += 1
        _max = x.view(x.size(0), -1).max(dim=1).values
        if max_counter[l_idx][cluster] == []:
            max_counter[l_idx][cluster] = _max
        else:
            max_counter[l_idx][cluster] = torch.cat([max_counter[l_idx][cluster], _max])
        
        x, act_scaling_factor = self.quant_act2(x, act_scaling_factor, weight_scaling_factor, cluster=cluster)
        x, conv_scaling_factor = self.quant_conv2(x, act_scaling_factor)

        concat_tensor = torch.cat((batch, x), 1)
        
        l_idx += 1
        _max = x.view(x.size(0), -1).max(dim=1).values
        if max_counter[l_idx][cluster] == []:
            max_counter[l_idx][cluster] = _max
        else:
            max_counter[l_idx][cluster] = torch.cat([max_counter[l_idx][cluster], _max])
        
        x, act_scaling_factor = self.quant_act_output(concat_tensor, act_scaling_factor, conv_scaling_factor, 
                                                concat=True, concat_scaling_factor=input_scaling_factor, cluster=cluster)
        
        return x, l_idx, act_scaling_factor



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
            x, act_scaling_factor = function(x, act_scaling_factor, cluster=cluster)
        return x, act_scaling_factor

    # def initialize_counter(self, x, n_clusters, zero_counter):
    #     self.zero_counter = zero_counter
    #     for unit_num in range(self.layers):
    #         function = getattr(self, f'unit{unit_num + 1}')
    #         x = function.initialize_counter(x, n_clusters, self.zero_counter)
    #     return x
        
    # def count_zeros_per_index(self, x, layer_idx, cluster):
    #     for unit_num in range(self.layers):
    #         function = getattr(self, f'unit{unit_num + 1}')
    #         x, layer_idx = function.count_zeros_per_index(x, layer_idx, cluster)
    #     return x, layer_idx

    def accumulate_output_max_distribution(self, x, cluster, n_clusters, head, l_idx, act_scaling_factor=None):
        for unit_num in range(self.layers):
            function = getattr(self, f'unit{unit_num + 1}')
            x, l_idx, act_scaling_factor = function.accumulate_output_max_distribution(x, cluster, n_clusters, head, 
                                                                                       l_idx, act_scaling_factor)
        return x, l_idx, act_scaling_factor

    def get_output_max_distribution(self, x, cluster, n_clusters, max_counter, l_idx, initialized, act_scaling_factor=None): 
        for unit_num in range(self.layers):
            function = getattr(self, f'unit{unit_num + 1}')
            x, l_idx, act_scaling_factor = function.get_output_max_distribution(x, cluster, n_clusters, max_counter, 
                                                                                l_idx, initialized, act_scaling_factor)
        return x, l_idx, act_scaling_factor



def q_densenet(model, num_clusters=None):
    return Q_DenseNet(model, num_clusters)



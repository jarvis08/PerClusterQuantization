"""
    Quantized ResNet for ImageNet-1K, implemented in PyTorch.
    Original paper: 'Deep Residual Learning for Image Recognition,' https://arxiv.org/abs/1512.03385.
"""

import torch
import torch.nn as nn
from ..quantization_utils.quant_modules import *
from pytorchcv.models.common import ConvBlock
from pytorchcv.models.shufflenetv2 import ShuffleUnit, ShuffleInitBlock


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
        self.quant_init_block_convbn.set_param(init_block.conv.conv, init_block.conv.bn)

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
                setattr(self, f"stage{stage_num + 1}.unit{unit_num + 1}", quant_unit)

        self.final_pool = QuantAveragePool2d(kernel_size=7, stride=1)

        self.quant_act_output = QuantAct(num_clusters=num_clusters)

        output = getattr(model, 'output')
        self.quant_output = QuantLinear()
        self.quant_output.is_classifier = True
        self.quant_output.set_param(output)


    def toggle_full_precision(self):
        # print('Model Toggle full precision FUNC')
        self.full_precision = False if self.full_precision is True else True
        for module in self.modules():
            if isinstance(module, (QuantAct, QuantLinear, QuantConv2d, QuantBn, QuantBnConv2d)):
                precision = getattr(module, 'full_precision_flag')
                setattr(module, 'full_precision_flag', not precision)


    def forward(self, x, cluster):
        x, act_scaling_factor = self.quant_input(x, cluster=cluster)

        x, weight_scaling_factor = self.quant_init_block_convbn(x, act_scaling_factor)

        x = self.pool(x)
        x, act_scaling_factor = self.quant_act_int32(x, act_scaling_factor, weight_scaling_factor, cluster=cluster)

        x = self.act(x)

        for stage_num in range(0, 4):
            for unit_num in range(0, self.channel[stage_num]):
                tmp_func = getattr(self, f"stage{stage_num+1}.unit{unit_num+1}")
                x, act_scaling_factor = tmp_func(x, act_scaling_factor, cluster=cluster)

        x = self.final_pool(x, act_scaling_factor)

        x, act_scaling_factor = self.quant_act_output(x, act_scaling_factor, cluster=cluster)
        x = x.view(x.size(0), -1)
        x = self.quant_output(x, act_scaling_factor)

        return x


    def delete_counters(self):
        if hasattr(self, 'zero_counter'):
            del self.zero_counter
        if hasattr(self, 'max_counter'):
            del self.max_counter
        if hasattr(self, 'max_accumulator'):
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
            self.max_accumulator = torch.zeros([17, n_clusters]).cuda()
            self.zero_buffer = torch.zeros(n_clusters).cuda()
            
        x, act_scaling_factor = self.quant_input(x, cluster=cluster)
        x, weight_scaling_factor = self.quant_init_block_convbn(x, act_scaling_factor)
        
        ##################################
        l_idx = self.update_max_accumulator(x, cluster, l_idx)
        ##################################
        
        x = self.pool(x)
        x, act_scaling_factor = self.quant_act_int32(x, act_scaling_factor, weight_scaling_factor, cluster=cluster)
        x = self.act(x)

        for stage_num in range(0, 4):
            for unit_num in range(0, self.channel[stage_num]):
                tmp_func = getattr(self, f"stage{stage_num+1}.unit{unit_num+1}")
                x, l_idx, act_scaling_factor = tmp_func.accumulate_output_max_distribution(x, cluster, n_clusters, self, 
                                                                                           l_idx, act_scaling_factor)

    
    
    # def get_max_activations(self, x, cluster=None):
    #     x, act_scaling_factor = self.quant_input(x, cluster=cluster)

    #     x, weight_scaling_factor = self.quant_init_block_convbn(x, act_scaling_factor)

    #     maxs = torch.amax(x.view(x.size(0), -1), dim=-1, keepdim=True)                                      #####
    #     x = self.pool(x)
    #     x, act_scaling_factor = self.quant_act_int32(x, act_scaling_factor, weight_scaling_factor, cluster=cluster)

    #     x = self.act(x)

    #     for stage_num in range(0, 4):
    #         for unit_num in range(0, self.channel[stage_num]):
    #             tmp_func = getattr(self, f"stage{stage_num+1}.unit{unit_num+1}")
    #             x, act_scaling_factor, maxs = tmp_func.get_max_activations(x, act_scaling_factor, cluster=cluster, maxs=maxs)

    #     x = self.final_pool(x, act_scaling_factor)

    #     x, act_scaling_factor = self.quant_act_output(x, act_scaling_factor, cluster=cluster)
    #     x = x.view(x.size(0), -1)
    #     x = self.quant_output(x, act_scaling_factor)

    #     maxs = torch.cat((maxs, torch.amax(x[0].view(x[0].size(0), -1), dim=-1, keepdim=True)), dim=1)            #####
    #     return maxs


    # def initialize_counter(self, x, n_clusters):
    #     self.zero_counter = []

    #     self.features = nn.Sequential(self.quant_init_block_convbn, self.pool, self.act)
        
    #     x, _ = self.features[0](x)
    #     x = self.features[1](x)
    #     x = self.features[2](x)

    #     n_features = x.view(-1).size(0)
    #     self.zero_counter.append(torch.zeros((n_clusters, n_features), device='cuda'))

    #     for stage_num in range(0,4):
    #         for unit_num in range(0, self.channel[stage_num]):
    #             tmp_func = getattr(self, f'stage{stage_num + 1}.unit{unit_num + 1}')
    #             x = tmp_func.initialize_counter(x, n_clusters, self.zero_counter)
    #             setattr(self, f'stage{stage_num + 1}.unit{unit_num + 1}', tmp_func)


    # def count_zeros_per_index(self, x, cluster, n_clusters):
    #     if not hasattr(self, 'zero_counter'):
    #         self.initialize_counter(x[0].unsqueeze(0), n_clusters)

    #     x, _ = self.features[0](x)
    #     x = self.features[1](x)
    #     x = self.features[2](x)

    #     layer_idx = 0
    #     n_features = self.zero_counter[layer_idx].size(1)
    #     for idx in range(x.size(0)):
    #         flattened = x[idx].view(-1)
    #         zeros_idx = (flattened == 0.0).nonzero(as_tuple=True)[0]
    #         zeros_idx %= n_features
    #         self.zero_counter[layer_idx][cluster, zeros_idx] += 1
    
    #     for stage_num in range(0,4):
    #         for unit_num in range(0, self.channel[stage_num]):
    #             tmp_func = getattr(self, f'stage{stage_num + 1}.unit{unit_num + 1}')
    #             x, layer_idx = tmp_func.count_zeros_per_index(x, layer_idx, cluster, n_clusters)


    def get_output_max_distribution(self, x, cluster, n_clusters):
        initialized = True
        if not hasattr(self, 'max_counter'):
            initialized = False
            self.max_counter = []
            self.max_counter.append([[] for _ in range(n_clusters)])
            
        x, act_scaling_factor = self.quant_input(x, cluster=cluster)
        x, weight_scaling_factor = self.quant_init_block_convbn(x, act_scaling_factor)
        
        l_idx = 0
        _max = x.view(x.size(0), -1).max(dim=1).values
        if self.max_counter[l_idx][cluster] == []:
            self.max_counter[l_idx][cluster] = _max
        else:
            self.max_counter[l_idx][cluster] = torch.cat([self.max_counter[l_idx][cluster], _max])
            
        x = self.pool(x)
        x, act_scaling_factor = self.quant_act_int32(x, act_scaling_factor, weight_scaling_factor, cluster=cluster)
        x = self.act(x)

        for stage_num in range(0, 4):
            for unit_num in range(0, self.channel[stage_num]):
                tmp_func = getattr(self, f"stage{stage_num+1}.unit{unit_num+1}")
                x, l_idx, act_scaling_factor = tmp_func.get_output_max_distribution(x, cluster, n_clusters, self.max_counter, 
                                                                                    l_idx, initialized, act_scaling_factor)


    def get_ema_per_layer(self):
        ema = []
        ema.append(self.quant_act_int32.x_max)
        for stage_num in range(0, 4):
            for unit_num in range(0, self.channel[stage_num]):
                tmp_func = getattr(self, f'stage{stage_num + 1}.unit{unit_num + 1}')
                ema = ema + tmp_func.get_ema_per_layer()
        return torch.stack(ema)

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
                setattr(self, f'stage{stage_num + 1}.unit{unit_num + 1}', quant_unit)

        self.final_pool = QuantAveragePool2d(kernel_size=8, stride=1)

        self.quant_act_output = QuantAct(num_clusters=num_clusters)

        output = getattr(model, 'output')
        self.quant_output = QuantLinear()
        self.quant_output.is_classifier = True
        self.quant_output.set_param(output)


    def toggle_full_precision(self):
        # print('Model Toggle full precision FUNC')
        self.full_precision = False if self.full_precision is True else True
        for module in self.modules():
            if isinstance(module, (QuantAct, QuantLinear, QuantConv2d, QuantBn, QuantBnConv2d)):
                precision = getattr(module, 'full_precision_flag')
                setattr(module, 'full_precision_flag', not precision)


    def forward(self, x, cluster):
        x, act_scaling_factor = self.quant_input(x, cluster=cluster)

        x, weight_scaling_factor = self.quant_init_block_convbn(x, act_scaling_factor)
        x, act_scaling_factor = self.quant_act_int32(x, act_scaling_factor, weight_scaling_factor, cluster=cluster)

        x = self.act(x)

        for stage_num in range(0,3):
            for unit_num in range(0, self.channel[stage_num]):
                tmp_func = getattr(self, f'stage{stage_num + 1}.unit{unit_num + 1}')
                x, act_scaling_factor = tmp_func(x, act_scaling_factor, cluster=cluster)

        x, act_scaling_factor = self.final_pool(x, act_scaling_factor)

        x, act_scaling_factor = self.quant_act_output(x, act_scaling_factor, cluster=cluster)
        x = x.view(x.size(0), -1)
        x = self.quant_output(x, act_scaling_factor)

        return x


    def delete_counters(self):
        if hasattr(self, 'zero_counter'):
            del self.zero_counter
        if hasattr(self, 'max_counter'):
            del self.max_counter
        if hasattr(self, 'max_accumulator'):
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
            self.max_accumulator = torch.zeros([19, n_clusters]).cuda()
            self.zero_buffer = torch.zeros(n_clusters).cuda()
            
        x, act_scaling_factor = self.quant_input(x, cluster=cluster)
        x, weight_scaling_factor = self.quant_init_block_convbn(x, act_scaling_factor)
        
        ##################################
        l_idx = self.update_max_accumulator(x, cluster, l_idx)
        ##################################
            
        x, act_scaling_factor = self.quant_act_int32(x, act_scaling_factor, weight_scaling_factor, cluster=cluster)
        x = self.act(x)

        for stage_num in range(0,3):
            for unit_num in range(0, self.channel[stage_num]):
                tmp_func = getattr(self, f'stage{stage_num + 1}.unit{unit_num + 1}')
                x, l_idx, act_scaling_factor = tmp_func.accumulate_output_max_distribution(x, cluster, n_clusters, self, 
                                                                                           l_idx, act_scaling_factor)
            
            
            
    # def get_max_activations(self, x, cluster=None):
    #     x, act_scaling_factor = self.quant_input(x, cluster=cluster)

    #     x, weight_scaling_factor = self.quant_init_block_convbn(x, act_scaling_factor)
    #     maxs = torch.amax(x.view(x.size(0), -1), dim=-1, keepdim=True)                                      #####
    #     x, act_scaling_factor = self.quant_act_int32(x, act_scaling_factor, weight_scaling_factor, cluster=cluster)

    #     x = self.act(x)

    #     for stage_num in range(0,3):
    #         for unit_num in range(0, self.channel[stage_num]):
    #             tmp_func = getattr(self, f'stage{stage_num + 1}.unit{unit_num + 1}')
    #             x, act_scaling_factor, maxs = tmp_func.get_max_activations(x, act_scaling_factor, cluster=cluster, maxs=maxs)

    #     x, act_scaling_factor = self.final_pool(x, act_scaling_factor)

    #     x, act_scaling_factor = self.quant_act_output(x, act_scaling_factor, cluster=cluster)
    #     x = x.view(x.size(0), -1)
    #     x = self.quant_output(x, act_scaling_factor)

    #     maxs = torch.cat((maxs, torch.amax(x[0].view(x[0].size(0), -1), dim=-1, keepdim=True)), dim=1)            #####
    #     return maxs


    # def initialize_counter(self, x, n_clusters):
    #     self.zero_counter = []

    #     self.features = nn.Sequential(self.quant_init_block_convbn, self.act)
        
    #     x, _ = self.features[0](x)
    #     x = self.features[1](x)

    #     n_features = x.view(-1).size(0)
    #     self.zero_counter.append(torch.zeros((n_clusters, n_features), device='cuda'))

    #     for stage_num in range(0,3):
    #         for unit_num in range(0, self.channel[stage_num]):
    #             tmp_func = getattr(self, f'stage{stage_num + 1}.unit{unit_num + 1}')
    #             x = tmp_func.initialize_counter(x, n_clusters, self.zero_counter)
    #             setattr(self, f'stage{stage_num + 1}.unit{unit_num + 1}', tmp_func)


    # def count_zeros_per_index(self, x, cluster, n_clusters):
    #     if not hasattr(self, 'zero_counter'):
    #         self.initialize_counter(x[0].unsqueeze(0), n_clusters)

    #     x, _ = self.features[0](x)
    #     x = self.features[1](x)

    #     layer_idx = 0
    #     n_features = self.zero_counter[layer_idx].size(1)
    #     for idx in range(x.size(0)):
    #         flattened = x[idx].view(-1)
    #         zeros_idx = (flattened == 0.0).nonzero(as_tuple=True)[0]
    #         zeros_idx %= n_features
    #         self.zero_counter[layer_idx][cluster, zeros_idx] += 1
    
    #     for stage_num in range(0,3):
    #         for unit_num in range(0, self.channel[stage_num]):
    #             tmp_func = getattr(self, f'stage{stage_num + 1}.unit{unit_num + 1}')
    #             x, layer_idx = tmp_func.count_zeros_per_index(x, layer_idx, cluster, n_clusters)


    def get_output_max_distribution(self, x, cluster, n_clusters):
        initialized = True
        if not hasattr(self, 'max_counter'):
            initialized = False
            self.max_counter = []
            self.max_counter.append([[] for _ in range(n_clusters)])

        x, act_scaling_factor = self.quant_input(x, cluster=cluster)

        x, weight_scaling_factor = self.quant_init_block_convbn(x, act_scaling_factor)
        
        l_idx = 0
        _max = x.view(x.size(0), -1).max(dim=1).values
        if self.max_counter[l_idx][cluster] == []:
            self.max_counter[l_idx][cluster] = _max
        else:
            self.max_counter[l_idx][cluster] = torch.cat([self.max_counter[l_idx][cluster], _max])
            
        x, act_scaling_factor = self.quant_act_int32(x, act_scaling_factor, weight_scaling_factor, cluster=cluster)
        x = self.act(x)

        for stage_num in range(0,3):
            for unit_num in range(0, self.channel[stage_num]):
                tmp_func = getattr(self, f'stage{stage_num + 1}.unit{unit_num + 1}')
                x, l_idx, act_scaling_factor = tmp_func.get_output_max_distribution(x, cluster, n_clusters, self.max_counter, l_idx,
                                                                                    initialized, act_scaling_factor)
            

    def get_ema_per_layer(self):
        ema = []
        ema.append(self.quant_act_int32.x_max)
        for stage_num in range(0, 3):
            for unit_num in range(0, self.channel[stage_num]):
                tmp_func = getattr(self, f'stage{stage_num + 1}.unit{unit_num + 1}')
                ema = ema + tmp_func.get_ema_per_layer()
        return torch.stack(ema)


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
        self.quant_init_convbn.set_param(init_block.conv.conv, init_block.conv.bn)

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
                setattr(self, f"stage{stage_num + 1}.unit{unit_num + 1}", quant_unit)

        self.final_pool = QuantAveragePool2d(kernel_size=7, stride=1, padding=0)

        self.quant_act_output = QuantAct(num_clusters=num_clusters)

        output = getattr(model, 'output')
        self.quant_output = QuantLinear()
        self.quant_output.is_classifier = True
        self.quant_output.set_param(output)


    def toggle_full_precision(self):
        # print('Model Toggle full precision FUNC')
        self.full_precision = False if self.full_precision is True else True
        for module in self.modules():
            if isinstance(module, (QuantAct, QuantLinear, QuantConv2d, QuantBn, QuantBnConv2d)):
                precision = getattr(module, 'full_precision_flag')
                setattr(module, 'full_precision_flag', not precision)


    def forward(self, x, cluster):
        x, act_scaling_factor = self.quant_input(x, cluster=cluster)

        x, weight_scaling_factor = self.quant_init_convbn(x, act_scaling_factor)
        
        x = self.pool(x)
        x, act_scaling_factor = self.quant_act_int32(x, act_scaling_factor, weight_scaling_factor, cluster=cluster)
        
        x = self.act(x)

        for stage_num in range(0, 4):
            for unit_num in range(0, self.channel[stage_num]):
                tmp_func = getattr(self, f"stage{stage_num+1}.unit{unit_num+1}")
                x, act_scaling_factor = tmp_func(x, act_scaling_factor, cluster=cluster)

        x, act_scaling_factor = self.final_pool(x, act_scaling_factor)

        x, act_scaling_factor = self.quant_act_output(x, act_scaling_factor, cluster=cluster)
        x = x.view(x.size(0), -1)
        x = self.quant_output(x, act_scaling_factor)

        return x


    def delete_counters(self):
        if hasattr(self, 'zero_counter'):
            del self.zero_counter
        if hasattr(self, 'max_counter'):
            del self.max_counter
        if hasattr(self, 'max_accumulator'):
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
            self.max_accumulator = torch.zeros([49, n_clusters]).cuda()
            self.zero_buffer = torch.zeros(n_clusters).cuda()
            
        x, act_scaling_factor = self.quant_input(x, cluster=cluster)
        x, weight_scaling_factor = self.quant_init_convbn(x, act_scaling_factor)
        
        ##################################
        l_idx = self.update_max_accumulator(x, cluster, l_idx)
        ##################################
        
        x = self.pool(x)
        x, act_scaling_factor = self.quant_act_int32(x, act_scaling_factor, weight_scaling_factor, cluster=cluster)
        x = self.act(x)

        for stage_num in range(0, 4):
            for unit_num in range(0, self.channel[stage_num]):
                tmp_func = getattr(self, f"stage{stage_num+1}.unit{unit_num+1}")
                x, l_idx, act_scaling_factor = tmp_func.accumulate_output_max_distribution(x, cluster, n_clusters, self, 
                                                                                           l_idx, act_scaling_factor)

        
    # def get_max_activations(self, x, cluster=None):
    #     x, act_scaling_factor = self.quant_input(x, cluster=cluster)

    #     x, weight_scaling_factor = self.quant_init_convbn(x, act_scaling_factor)
    #     maxs = torch.amax(x.view(x.size(0), -1), dim=-1, keepdim=True)                                      #####
        
    #     x = self.pool(x)
    #     x, act_scaling_factor = self.quant_act_int32(x, act_scaling_factor, weight_scaling_factor, cluster=cluster)
        
    #     x = self.act(x)

    #     for stage_num in range(0, 4):
    #         for unit_num in range(0, self.channel[stage_num]):
    #             tmp_func = getattr(self, f"stage{stage_num+1}.unit{unit_num+1}")
    #             x, act_scaling_factor, maxs = tmp_func.get_max_activations(x, act_scaling_factor, cluster=cluster, maxs=maxs)

    #     x, act_scaling_factor = self.final_pool(x, act_scaling_factor)

    #     x, act_scaling_factor = self.quant_act_output(x, act_scaling_factor, cluster=cluster)
    #     x = x.view(x.size(0), -1)
    #     x = self.quant_output(x, act_scaling_factor)

    #     maxs = torch.cat((maxs, torch.amax(x[0].view(x[0].size(0), -1), dim=-1, keepdim=True)), dim=1)            #####
    #     return maxs


    # def initialize_counter(self, x, n_clusters):
    #     self.zero_counter = []

    #     self.features = nn.Sequential(self.quant_init_block_convbn, self.pool, self.act)
        
    #     x, _ = self.features[0](x)
    #     x = self.features[1](x)
    #     x = self.features[2](x)

    #     n_features = x.view(-1).size(0)
    #     self.zero_counter.append(torch.zeros((n_clusters, n_features), device='cuda'))

    #     for stage_num in range(0,4):
    #         for unit_num in range(0, self.channel[stage_num]):
    #             tmp_func = getattr(self, f'stage{stage_num + 1}.unit{unit_num + 1}')
    #             x = tmp_func.initialize_counter(x, n_clusters, self.zero_counter)
    #             setattr(self, f'stage{stage_num + 1}.unit{unit_num + 1}', tmp_func)


    # def count_zeros_per_index(self, x, cluster, n_clusters):
    #     if not hasattr(self, 'zero_counter'):
    #         self.initialize_counter(x[0].unsqueeze(0), n_clusters)

    #     x, _ = self.features[0](x)
    #     x = self.features[1](x)
    #     x = self.features[2](x)

    #     layer_idx = 0
    #     n_features = self.zero_counter[layer_idx].size(1)
    #     for idx in range(x.size(0)):
    #         flattened = x[idx].view(-1)
    #         zeros_idx = (flattened == 0.0).nonzero(as_tuple=True)[0]
    #         zeros_idx %= n_features
    #         self.zero_counter[layer_idx][cluster, zeros_idx] += 1

    #     for stage_num in range(0,4):
    #         for unit_num in range(0, self.channel[stage_num]):
    #             tmp_func = getattr(self, f'stage{stage_num + 1}.unit{unit_num + 1}')
    #             x, layer_idx = tmp_func.count_zeros_per_index(x, layer_idx, cluster, n_clusters)


    def get_output_max_distribution(self, x, cluster, n_clusters):
        initialized = True
        if not hasattr(self, 'max_counter'):
            initialized = False
            self.max_counter = []
            self.max_counter.append([[] for _ in range(n_clusters)])
            
        x, act_scaling_factor = self.quant_input(x, cluster=cluster)

        x, weight_scaling_factor = self.quant_init_convbn(x, act_scaling_factor)
        
        l_idx = 0
        _max = x.view(x.size(0), -1).max(dim=1).values
        if self.max_counter[l_idx][cluster] == []:
            self.max_counter[l_idx][cluster] = _max
        else:
            self.max_counter[l_idx][cluster] = torch.cat([self.max_counter[l_idx][cluster], _max])
        
        x = self.pool(x)
        x, act_scaling_factor = self.quant_act_int32(x, act_scaling_factor, weight_scaling_factor, cluster=cluster)
        x = self.act(x)

        for stage_num in range(0, 4):
            for unit_num in range(0, self.channel[stage_num]):
                tmp_func = getattr(self, f"stage{stage_num+1}.unit{unit_num+1}")
                x, l_idx, act_scaling_factor = tmp_func.get_output_max_distribution(x, cluster, n_clusters, self.max_counter, l_idx,
                                                                initialized, act_scaling_factor)
            

    def get_ema_per_layer(self):
        ema = []
        ema.append(self.quant_act_int32.x_max)
        for stage_num in range(0,4):
            for unit_num in range(0, self.channel[stage_num]):
                tmp_func = getattr(self, f'stage{stage_num + 1}.unit{unit_num + 1}')
                ema = ema + tmp_func.get_ema_per_layer()
        return torch.stack(ema)


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
            self.quant_identity_convbn.set_param(unit.identity_conv.conv, unit.identity_conv.bn)

        self.quant_act_int32 = QuantAct(num_clusters=num_clusters)
        self.act3 = nn.ReLU()


    def forward(self, x, scaling_factor_int32=None, cluster=None):
        # forward using the quantized modules
        if self.resize_identity:
            x, act_scaling_factor = self.quant_act(x, scaling_factor_int32, cluster=cluster)
            identity_act_scaling_factor = act_scaling_factor.clone() if act_scaling_factor is not None else None
            identity, identity_weight_scaling_factor = self.quant_identity_convbn(x, act_scaling_factor)
        else:
            identity = x
            x, act_scaling_factor = self.quant_act(x, scaling_factor_int32, cluster=cluster)

        x, weight_scaling_factor = self.quant_convbn1(x, act_scaling_factor)
        x = self.act1(x)
        x, act_scaling_factor = self.quant_act1(x, act_scaling_factor, weight_scaling_factor, cluster=cluster)

        x, weight_scaling_factor = self.quant_convbn2(x, act_scaling_factor)
        x = self.act2(x)
        x, act_scaling_factor = self.quant_act2(x, act_scaling_factor, weight_scaling_factor, cluster=cluster)

        x, weight_scaling_factor = self.quant_convbn3(x, act_scaling_factor)

        x = x + identity

        if self.resize_identity:
            x, act_scaling_factor = self.quant_act_int32(x, act_scaling_factor, weight_scaling_factor, 
                                                         identity, identity_act_scaling_factor, identity_weight_scaling_factor, cluster=cluster)
        else:
            x, act_scaling_factor = self.quant_act_int32(x, act_scaling_factor, weight_scaling_factor, 
                                                         identity, scaling_factor_int32, None, cluster=cluster)

        x = self.act3(x)

        return x, act_scaling_factor


    # def get_max_activations(self, x, scaling_factor_int32=None, cluster=None, maxs=None):
    #     if self.resize_identity:
    #         x, act_scaling_factor = self.quant_act(x, scaling_factor_int32, cluster=cluster)
    #         identity_act_scaling_factor = act_scaling_factor.clone() if act_scaling_factor is not None else None
    #         identity, identity_weight_scaling_factor = self.quant_identity_convbn(x, act_scaling_factor)
    #         maxs = torch.cat((maxs, torch.amax(x.view(x.size(0), -1), dim=-1, keepdim=True)), dim=1)            #####
    #     else:
    #         identity = x
    #         x, act_scaling_factor = self.quant_act(x, scaling_factor_int32, cluster=cluster)

    #     x, weight_scaling_factor = self.quant_convbn1(x, act_scaling_factor)
    #     maxs = torch.cat((maxs, torch.amax(x.view(x.size(0), -1), dim=-1, keepdim=True)), dim=1)            #####
        
    #     x = self.act1(x)
    #     x, act_scaling_factor = self.quant_act1(x, act_scaling_factor, weight_scaling_factor, cluster=cluster)

    #     x, weight_scaling_factor = self.quant_convbn2(x, act_scaling_factor)
    #     maxs = torch.cat((maxs, torch.amax(x.view(x.size(0), -1), dim=-1, keepdim=True)), dim=1)            #####
    #     x = self.act2(x)
    #     x, act_scaling_factor = self.quant_act2(x, act_scaling_factor, weight_scaling_factor, cluster=cluster)

    #     x, weight_scaling_factor = self.quant_convbn3(x, act_scaling_factor)
    #     maxs = torch.cat((maxs, torch.amax(x.view(x.size(0), -1), dim=-1, keepdim=True)), dim=1)            #####

    #     x = x + identity

    #     if self.resize_identity:
    #         x, act_scaling_factor = self.quant_act_int32(x, act_scaling_factor, weight_scaling_factor, 
    #                                                      identity, identity_act_scaling_factor, identity_weight_scaling_factor, cluster=cluster)
    #     else:
    #         x, act_scaling_factor = self.quant_act_int32(x, act_scaling_factor, weight_scaling_factor, 
    #                                                      identity, scaling_factor_int32, None, cluster=cluster)

    #     maxs = torch.cat((maxs, torch.amax(x.view(x.size(0), -1), dim=-1, keepdim=True)), dim=1)            #####
    #     x = self.act3(x)

    #     return x, act_scaling_factor, maxs



    # def initialize_counter(self, x, n_clusters, zero_counter):
    #     self.zero_counter = zero_counter

    #     if self.resize_identity:
    #         self.features = nn.Sequential(self.quant_convbn1, self.act1,
    #                                       self.quant_convbn2, self.act2,
    #                                       self.quant_convbn3, self.act3,
    #                                       self.quant_identity_convbn)

    #         identity, _ = self.features[6](x)
    #     else :
    #         self.features = nn.Sequential(self.quant_convbn1, self.act1, 
    #                                       self.quant_convbn2, self.act2,
    #                                       self.quant_convbn3, self.act3)
    #         identity = x
    #     x, _ = self.features[0](x)
    #     x = self.features[1](x)

    #     n_features = x.view(-1).size(0)
    #     self.zero_counter.append(torch.zeros((n_clusters, n_features), device='cuda'))

    #     x, _ = self.features[2](x)
    #     x = self.features[3](x)

    #     n_features = x.view(-1).size(0)
    #     self.zero_counter.append(torch.zeros((n_clusters, n_features), device='cuda'))

    #     x, _ = self.features[4](x)
    #     x = x + identity
    #     x = self.features[5](x)

    #     n_features = x.view(-1).size(0)
    #     self.zero_counter.append(torch.zeros((n_clusters, n_features), device='cuda'))

    #     return x


    # def count_zeros_per_index(self, x, layer_idx, cluster, n_clusters):
    #     if self.resize_identity:
    #         identity, _ = self.features[6](x)
    #     else:
    #         identity = x

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
    #     x = x + identity
    #     x = self.features[5](x)

    #     layer_idx += 1
    #     n_features = self.zero_counter[layer_idx].size(1)
    #     for idx in range(x.size(0)):
    #         flattened = x[idx].view(-1)
    #         zeros_idx = (flattened == 0.0).nonzero(as_tuple=True)[0]
    #         zeros_idx %= n_features
    #         self.zero_counter[layer_idx][cluster, zeros_idx] += 1

    #     return x, layer_idx
    
    def accumulate_output_max_distribution(self, x, cluster, n_clusters, head, l_idx, scaling_factor_int32=None):
        if self.resize_identity:
            x, act_scaling_factor = self.quant_act(x, scaling_factor_int32, cluster=cluster)
            identity_act_scaling_factor = act_scaling_factor.clone() if act_scaling_factor is not None else None
            identity, identity_weight_scaling_factor = self.quant_identity_convbn(x, act_scaling_factor)
        else:
            identity = x
            x, act_scaling_factor = self.quant_act(x, scaling_factor_int32, cluster=cluster)

        x, weight_scaling_factor = self.quant_convbn1(x, act_scaling_factor)
        x = self.act1(x)
        
        ##################################
        l_idx = head.update_max_accumulator(x, cluster, l_idx)
        ##################################
        
        x, act_scaling_factor = self.quant_act1(x, act_scaling_factor, weight_scaling_factor, cluster=cluster)
        x, weight_scaling_factor = self.quant_convbn2(x, act_scaling_factor)
        x = self.act2(x)
        
        ##################################
        l_idx = head.update_max_accumulator(x, cluster, l_idx)
        ##################################
        
        x, act_scaling_factor = self.quant_act2(x, act_scaling_factor, weight_scaling_factor, cluster=cluster)
        x, weight_scaling_factor = self.quant_convbn3(x, act_scaling_factor)
        x = x + identity
        
        ##################################
        l_idx = head.update_max_accumulator(x, cluster, l_idx)
        ##################################
        
        if self.resize_identity:
            x, act_scaling_factor = self.quant_act_int32(x, act_scaling_factor, weight_scaling_factor, 
                                                        identity, identity_act_scaling_factor, identity_weight_scaling_factor, cluster=cluster)
        else:
            x, act_scaling_factor = self.quant_act_int32(x, act_scaling_factor, weight_scaling_factor, 
                                                        identity, scaling_factor_int32, None, cluster=cluster)

        x = self.act3(x)

        return x, l_idx, act_scaling_factor

    def get_output_max_distribution(self, x, cluster, n_clusters, max_counter, l_idx, initialized, scaling_factor_int32=None):
        if not initialized:
            max_counter.append([[] for _ in range(n_clusters)])
            max_counter.append([[] for _ in range(n_clusters)])
            max_counter.append([[] for _ in range(n_clusters)])

        if self.resize_identity:
            x, act_scaling_factor = self.quant_act(x, scaling_factor_int32, cluster=cluster)
            identity_act_scaling_factor = act_scaling_factor.clone() if act_scaling_factor is not None else None
            identity, identity_weight_scaling_factor = self.quant_identity_convbn(x, act_scaling_factor)
        else:
            identity = x
            x, act_scaling_factor = self.quant_act(x, scaling_factor_int32, cluster=cluster)

        x, weight_scaling_factor = self.quant_convbn1(x, act_scaling_factor)
        x = self.act1(x)
        
        l_idx += 1
        _max = x.view(x.size(0), -1).max(dim=1).values
        if max_counter[l_idx][cluster] == []:
            max_counter[l_idx][cluster] = _max
        else:
            max_counter[l_idx][cluster] = torch.cat([max_counter[l_idx][cluster], _max])
        
        x, act_scaling_factor = self.quant_act1(x, act_scaling_factor, weight_scaling_factor, cluster=cluster)
        x, weight_scaling_factor = self.quant_convbn2(x, act_scaling_factor)
        x = self.act2(x)
        
        l_idx += 1
        _max = x.view(x.size(0), -1).max(dim=1).values
        if max_counter[l_idx][cluster] == []:
            max_counter[l_idx][cluster] = _max
        else:
            max_counter[l_idx][cluster] = torch.cat([max_counter[l_idx][cluster], _max])
            
        x, act_scaling_factor = self.quant_act2(x, act_scaling_factor, weight_scaling_factor, cluster=cluster)
        x, weight_scaling_factor = self.quant_convbn3(x, act_scaling_factor)
        x = x + identity
        
        l_idx += 1
        _max = x.view(x.size(0), -1).max(dim=1).values
        if max_counter[l_idx][cluster] == []:
            max_counter[l_idx][cluster] = _max
        else:
            max_counter[l_idx][cluster] = torch.cat([max_counter[l_idx][cluster], _max])
        
        if self.resize_identity:
            x, act_scaling_factor = self.quant_act_int32(x, act_scaling_factor, weight_scaling_factor, 
                                                        identity, identity_act_scaling_factor, identity_weight_scaling_factor, cluster=cluster)
        else:
            x, act_scaling_factor = self.quant_act_int32(x, act_scaling_factor, weight_scaling_factor, 
                                                        identity, scaling_factor_int32, None, cluster=cluster)

        x = self.act3(x)

        return x, l_idx, act_scaling_factor
            
            

    def get_ema_per_layer(self):
        ema = []
        ema.append(self.quant_act1.x_max)
        ema.append(self.quant_act2.x_max)
        ema.append(self.quant_act_int32.x_max)
        return ema

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
            self.quant_identity_convbn.set_param(unit.identity_conv.conv, unit.identity_conv.bn)

        self.quant_act_int32 = QuantAct(num_clusters=num_clusters)
        self.act2 = nn.ReLU()


    def forward(self, x, scaling_factor_int32=None, cluster=None):
        # forward using the quantized modules
        if self.resize_identity:
            x, act_scaling_factor = self.quant_act(x, scaling_factor_int32, cluster=cluster)
            identity_act_scaling_factor = act_scaling_factor.clone() if act_scaling_factor is not None else None
            identity, identity_weight_scaling_factor = self.quant_identity_convbn(x, act_scaling_factor)
        else:
            identity = x
            x, act_scaling_factor = self.quant_act(x, scaling_factor_int32, cluster=cluster)

        x, weight_scaling_factor = self.quant_convbn1(x, act_scaling_factor)
        x = self.act1(x)
        x, act_scaling_factor = self.quant_act1(x, act_scaling_factor, weight_scaling_factor, cluster=cluster)

        x, weight_scaling_factor = self.quant_convbn2(x, act_scaling_factor)

        x = x + identity

        if self.resize_identity:
            x, act_scaling_factor = self.quant_act_int32(x, act_scaling_factor, weight_scaling_factor, 
                                                         identity, identity_act_scaling_factor, identity_weight_scaling_factor, cluster=cluster)
        else:
            x, act_scaling_factor = self.quant_act_int32(x, act_scaling_factor, weight_scaling_factor, 
                                                         identity, scaling_factor_int32, None, cluster=cluster)

        x = self.act2(x)

        return x, act_scaling_factor


    # def get_max_activations(self, x, scaling_factor_int32=None, cluster=None, maxs=None):
    #     if self.resize_identity:
    #         x, act_scaling_factor = self.quant_act(x, scaling_factor_int32, cluster=cluster)
    #         identity_act_scaling_factor = act_scaling_factor.clone() if act_scaling_factor is not None else None
    #         identity, identity_weight_scaling_factor = self.quant_identity_convbn(x, act_scaling_factor)
    #         maxs = torch.cat((maxs, torch.amax(x.view(x.size(0), -1), dim=-1, keepdim=True)), dim=1)            #####
    #     else:
    #         identity = x
    #         x, act_scaling_factor = self.quant_act(x, scaling_factor_int32, cluster=cluster)

    #     x, weight_scaling_factor = self.quant_convbn1(x, act_scaling_factor)
    #     maxs = torch.cat((maxs, torch.amax(x.view(x.size(0), -1), dim=-1, keepdim=True)), dim=1)            #####
    #     x = self.act1(x)
    #     x, act_scaling_factor = self.quant_act1(x, act_scaling_factor, weight_scaling_factor, cluster=cluster)

    #     x, weight_scaling_factor = self.quant_convbn2(x, act_scaling_factor)
    #     maxs = torch.cat((maxs, torch.amax(x.view(x.size(0), -1), dim=-1, keepdim=True)), dim=1)            #####

    #     x = x + identity

    #     if self.resize_identity:
    #         x, act_scaling_factor = self.quant_act_int32(x, act_scaling_factor, weight_scaling_factor, 
    #                                                      identity, identity_act_scaling_factor, identity_weight_scaling_factor, cluster=cluster)
    #     else:
    #         x, act_scaling_factor = self.quant_act_int32(x, act_scaling_factor, weight_scaling_factor, 
    #                                                      identity, scaling_factor_int32, None, cluster=cluster)

    #     maxs = torch.cat((maxs, torch.amax(x.view(x.size(0), -1), dim=-1, keepdim=True)), dim=1)            #####
    #     x = self.act2(x)

    #     return x, act_scaling_factor, maxs


    # def initialize_counter(self, x, n_clusters, zero_counter):
    #     self.zero_counter = zero_counter

    #     if self.resize_identity:
    #         self.features = nn.Sequential(self.quant_convbn1, self.act1,
    #                                       self.quant_convbn2, self.act2,
    #                                       self.quant_identity_convbn)

    #         identity, _ = self.features[4](x)
    #     else :
    #         self.features = nn.Sequential(self.quant_convbn1, self.act1,
    #                                       self.quant_convbn2, self.act2)
    #         identity = x
    #     x, _ = self.features[0](x)
    #     x = self.features[1](x)

    #     n_features = x.view(-1).size(0)
    #     self.zero_counter.append(torch.zeros((n_clusters, n_features), device='cuda'))

    #     x, _ = self.features[2](x)
    #     x = x + identity
    #     x = self.features[3](x)

    #     n_features = x.view(-1).size(0)
    #     self.zero_counter.append(torch.zeros((n_clusters, n_features), device='cuda'))

    #     return x


    # def count_zeros_per_index(self, x, layer_idx, cluster, n_clusters):
    #     if self.resize_identity:
    #         identity, _ = self.features[4](x)
    #     else:
    #         identity = x

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
    #     x = x + identity
    #     x = self.features[3](x)
        
    #     layer_idx += 1
    #     n_features = self.zero_counter[layer_idx].size(1)
    #     for idx in range(x.size(0)):
    #         flattened = x[idx].view(-1)
    #         zeros_idx = (flattened == 0.0).nonzero(as_tuple=True)[0]
    #         zeros_idx %= n_features
    #         self.zero_counter[layer_idx][cluster, zeros_idx] += 1
        
    #     return x, layer_idx
    
    
    def accumulate_output_max_distribution(self, x, cluster, n_clusters, head, l_idx, scaling_factor_int32=None):      
        if self.resize_identity:
            x, act_scaling_factor = self.quant_act(x, scaling_factor_int32, cluster=cluster)
            identity_act_scaling_factor = act_scaling_factor.clone() if act_scaling_factor is not None else None
            identity, identity_weight_scaling_factor = self.quant_identity_convbn(x, act_scaling_factor)
        else:
            identity = x
            x, act_scaling_factor = self.quant_act(x, scaling_factor_int32, cluster=cluster)

        x, weight_scaling_factor = self.quant_convbn1(x, act_scaling_factor)
        x = self.act1(x)
        
        ##################################
        l_idx = head.update_max_accumulator(x, cluster, l_idx)
        ##################################
            
        x, act_scaling_factor = self.quant_act1(x, act_scaling_factor, weight_scaling_factor, cluster=cluster)
        x, weight_scaling_factor = self.quant_convbn2(x, act_scaling_factor)

        x = x + identity

        ##################################
        l_idx = head.update_max_accumulator(x, cluster, l_idx)
        ##################################
            
        if self.resize_identity:
            x, act_scaling_factor = self.quant_act_int32(x, act_scaling_factor, weight_scaling_factor, 
                                                            identity, identity_act_scaling_factor, identity_weight_scaling_factor, cluster=cluster)
        else:
            x, act_scaling_factor = self.quant_act_int32(x, act_scaling_factor, weight_scaling_factor, 
                                                            identity, scaling_factor_int32, None, cluster=cluster)

        x = self.act2(x)
        
        return x, l_idx, act_scaling_factor


    def get_output_max_distribution(self, x, cluster, n_clusters, max_counter, l_idx, initialized, scaling_factor_int32=None):
        if not initialized:
            max_counter.append([[] for _ in range(n_clusters)])
            max_counter.append([[] for _ in range(n_clusters)])
            
        if self.resize_identity:
            x, act_scaling_factor = self.quant_act(x, scaling_factor_int32, cluster=cluster)
            identity_act_scaling_factor = act_scaling_factor.clone() if act_scaling_factor is not None else None
            identity, identity_weight_scaling_factor = self.quant_identity_convbn(x, act_scaling_factor)
        else:
            identity = x
            x, act_scaling_factor = self.quant_act(x, scaling_factor_int32, cluster=cluster)

        x, weight_scaling_factor = self.quant_convbn1(x, act_scaling_factor)
        x = self.act1(x)
        
        l_idx += 1
        _max = x.view(x.size(0), -1).max(dim=1).values
        if max_counter[l_idx][cluster] == []:
            max_counter[l_idx][cluster] = _max
        else:
            max_counter[l_idx][cluster] = torch.cat([max_counter[l_idx][cluster], _max])
            
        x, act_scaling_factor = self.quant_act1(x, act_scaling_factor, weight_scaling_factor, cluster=cluster)
        x, weight_scaling_factor = self.quant_convbn2(x, act_scaling_factor)

        x = x + identity

        l_idx += 1
        _max = x.view(x.size(0), -1).max(dim=1).values
        if max_counter[l_idx][cluster] == []:
            max_counter[l_idx][cluster] = _max
        else:
            max_counter[l_idx][cluster] = torch.cat([max_counter[l_idx][cluster], _max])
            
        if self.resize_identity:
            x, act_scaling_factor = self.quant_act_int32(x, act_scaling_factor, weight_scaling_factor, 
                                                            identity, identity_act_scaling_factor, identity_weight_scaling_factor, cluster=cluster)
        else:
            x, act_scaling_factor = self.quant_act_int32(x, act_scaling_factor, weight_scaling_factor, 
                                                            identity, scaling_factor_int32, None, cluster=cluster)

        x = self.act2(x)
        
        return x, l_idx, act_scaling_factor
            

    def get_ema_per_layer(self):
        ema = []
        ema.append(self.quant_act1.x_max)
        ema.append(self.quant_act_int32.x_max)
        return ema


def q_resnet18(model, num_clusters=None):
    return Q_ResNet18(model, num_clusters)

def q_resnet20(model, num_clusters=None):
    return Q_ResNet20(model, num_clusters)

def q_resnet50(model, num_clusters=None):
    return Q_ResNet50(model, num_clusters)

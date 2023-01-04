
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

ACCUMULATE = {'sum': torch.add, 'max': torch.maximum}

class Q_AlexNet(nn.Module):
    def __init__(self, model, model_dict=None, num_clusters=1, skt_helper=None):
        super().__init__()
        self.full_precision = False
        self.skt_helper = skt_helper
        features = getattr(model, 'features')

        if skt_helper:
            self.total_ch_cnt = 0

        self.quant_input = QuantAct(num_clusters=num_clusters)

        # stage1
        stage = getattr(features, 'stage1')
        conv_block = getattr(stage, 'unit1')
        conv_block.conv.in_channels = 3
        conv_block.conv.out_channels = 96
        conv_block.conv.stride = (1, 1)
        conv_block.conv.kernel_size = (5, 5)
        conv_block.conv.padding = (2, 2)

        self.conv1 = QuantConv2d(skt_helper=self.skt_helper)
        self.conv1.set_param(conv_block.conv, model_dict, 'features.stage1.unit1.conv')
        self.act1 = nn.ReLU()
        self.quant_act1 = QuantAct(num_clusters=num_clusters)

        self.maxpool1 = QuantMaxPool2d(ceil_mode=True)

        # stage2
        stage = getattr(features, 'stage2')
        conv_block = getattr(stage, 'unit1')
        conv_block.conv.in_channels = 96
        conv_block.conv.out_channel = 256
        conv_block.conv.stride = (1, 1)
        conv_block.conv.kernel_size = (5, 5)
        conv_block.conv.padding = (2, 2)
        self.conv2 = QuantConv2d(skt_helper=self.skt_helper)
        self.conv2.set_param(conv_block.conv, model_dict, 'features.stage2.unit1.conv')
        self.act2 = nn.ReLU()
        self.quant_act2 = QuantAct(num_clusters=num_clusters)

        self.maxpool2 = QuantMaxPool2d(ceil_mode=True)

        # stage3
        stage = getattr(features, 'stage3')
        conv_block = getattr(stage, 'unit1')
        conv_block.conv.in_channels = 256
        conv_block.conv.out_channels = 384
        conv_block.conv.kernel_size = (3, 3)
        conv_block.conv.stride = (1, 1)
        conv_block.conv.padding = (1, 1)
        self.conv3 = QuantConv2d(skt_helper=self.skt_helper)
        self.conv3.set_param(conv_block.conv, model_dict, 'features.stage3.unit1.conv')
        self.act3 = nn.ReLU()
        self.quant_act3 = QuantAct(num_clusters=num_clusters)

        conv_block = getattr(stage, 'unit2')
        conv_block.conv.in_channels  = 384
        conv_block.conv.out_channels = 384
        conv_block.conv.kernel_size = (3, 3)
        conv_block.conv.stride = (1, 1)
        conv_block.conv.padding = (1, 1)
        self.conv4 = QuantConv2d(skt_helper=self.skt_helper)
        self.conv4.set_param(conv_block.conv, model_dict, 'features.stage3.unit2.conv')
        self.act4 = nn.ReLU()
        self.quant_act4 = QuantAct(num_clusters=num_clusters)

        conv_block = getattr(stage, 'unit3')
        conv_block.conv.in_channels = 384
        conv_block.conv.out_channels = 256
        conv_block.conv.kernel_size = (3, 3)
        conv_block.conv.stride = (1, 1)
        conv_block.conv.padding = (1, 1)
        self.conv5 = QuantConv2d(skt_helper=self.skt_helper)
        self.conv5.set_param(conv_block.conv, model_dict, 'features.stage3.unit3.conv')
        self.act5 = nn.ReLU()
        self.quant_act5 = QuantAct(num_clusters=num_clusters)

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
        self.quant_act6 = QuantAct(num_clusters=num_clusters)

        # fc2
        fc_block = getattr(output, 'fc2')
        fc_block.fc.in_features = 4096
        fc_block.fc.out_features = 4096
        self.fc2 = QuantLinear()
        self.fc2.set_param(fc_block.fc, model_dict, 'output.fc2.fc')
        self.act7 = nn.ReLU()
        self.quant_act7 = QuantAct(num_clusters=num_clusters)

        # fc3
        fc = getattr(output, 'fc3')
        fc.in_features = 4096
        fc.out_features = 10
        self.fc3 = QuantLinear()
        self.fc3.is_classifier = True
        self.fc3.set_param(fc, model_dict, 'output.fc3')

    def toggle_full_precision(self):
        # print('Model Toggle full precision FUNC')
        self.full_precision = False if self.full_precision is True else True
        for module in self.modules():
            if isinstance(module, (QuantAct, QuantLinear, QuantConv2d, QuantBn, QuantBnConv2d)):
                precision = getattr(module, 'full_precision_flag')
                setattr(module, 'full_precision_flag', not precision)

    def forward(self, x, cluster):
        x, act_scaling_factor = self.quant_input(x, cluster=cluster)

        x, conv_scaling_factor = self.conv1(x, act_scaling_factor)
        x = self.act1(x)
        x, act_scaling_factor = self.quant_act1(x, act_scaling_factor, conv_scaling_factor, cluster=cluster)

        x, act_scaling_factor = self.maxpool1(x, act_scaling_factor)

        x, conv_scaling_factor = self.conv2(x, act_scaling_factor)
        x = self.act2(x)
        x, act_scaling_factor = self.quant_act2(x, act_scaling_factor, conv_scaling_factor, cluster=cluster)

        x, act_scaling_factor = self.maxpool2(x, act_scaling_factor)

        x, conv_scaling_factor = self.conv3(x, act_scaling_factor)
        x = self.act3(x)
        x, act_scaling_factor = self.quant_act3(x, act_scaling_factor, conv_scaling_factor, cluster=cluster)

        x, conv_scaling_factor = self.conv4(x, act_scaling_factor)
        x = self.act4(x)
        x, act_scaling_factor = self.quant_act4(x, act_scaling_factor, conv_scaling_factor, cluster=cluster)
        
        x, conv_scaling_factor = self.conv5(x, act_scaling_factor)
        x = self.act5(x)
        x, act_scaling_factor = self.quant_act5(x, act_scaling_factor, conv_scaling_factor, cluster=cluster)

        x, act_scaling_factor = self.maxpool3(x, act_scaling_factor)
        x, act_scaling_factor = self.avgpool(x, act_scaling_factor)

        x = x.view(x.size(0), -1)

        x, fc_scaling_factor = self.fc1(x, act_scaling_factor)
        x = self.act6(x)
        x, act_scaling_factor = self.quant_act6(x, act_scaling_factor, fc_scaling_factor, cluster=cluster)

        x, fc_scaling_factor = self.fc2(x, act_scaling_factor)
        x = self.act7(x)
        x, act_scaling_factor = self.quant_act7(x, act_scaling_factor, fc_scaling_factor, cluster=cluster)

        x = self.fc3(x, act_scaling_factor)

        return x


    def get_max_activations(self, x):
        x, act_scaling_factor = self.quant_input(x)

        x, conv_scaling_factor = self.conv1(x, act_scaling_factor)
        maxs = torch.amax(x.view(x.size(0), -1), dim=-1, keepdim=True)                                      #####
        x = self.act1(x)
        x, act_scaling_factor = self.quant_act1(x, act_scaling_factor, conv_scaling_factor)

        x, act_scaling_factor = self.maxpool1(x, act_scaling_factor)

        x, conv_scaling_factor = self.conv2(x, act_scaling_factor)
        maxs = torch.cat((maxs, torch.amax(x.view(x.size(0), -1), dim=-1, keepdim=True)), dim=1)            #####
        x = self.act2(x)
        x, act_scaling_factor = self.quant_act2(x, act_scaling_factor, conv_scaling_factor)

        x, act_scaling_factor = self.maxpool2(x, act_scaling_factor)

        x, conv_scaling_factor = self.conv3(x, act_scaling_factor)
        maxs = torch.cat((maxs, torch.amax(x.view(x.size(0), -1), dim=-1, keepdim=True)), dim=1)            #####
        x = self.act3(x)
        x, act_scaling_factor = self.quant_act3(x, act_scaling_factor, conv_scaling_factor)

        x, conv_scaling_factor = self.conv4(x, act_scaling_factor)
        maxs = torch.cat((maxs, torch.amax(x.view(x.size(0), -1), dim=-1, keepdim=True)), dim=1)            #####
        x = self.act4(x)
        x, act_scaling_factor = self.quant_act4(x, act_scaling_factor, conv_scaling_factor)
        
        x, conv_scaling_factor = self.conv5(x, act_scaling_factor)
        maxs = torch.cat((maxs, torch.amax(x.view(x.size(0), -1), dim=-1, keepdim=True)), dim=1)            #####
        x = self.act5(x)
        x, act_scaling_factor = self.quant_act5(x, act_scaling_factor, conv_scaling_factor)

        x, act_scaling_factor = self.maxpool3(x, act_scaling_factor)
        x, act_scaling_factor = self.avgpool(x, act_scaling_factor)

        x = x.view(x.size(0), -1)

        x, fc_scaling_factor = self.fc1(x, act_scaling_factor)
        maxs = torch.cat((maxs, torch.amax(x.view(x.size(0), -1), dim=-1, keepdim=True)), dim=1)            #####
        x = self.act6(x)
        x, act_scaling_factor = self.quant_act6(x, act_scaling_factor, fc_scaling_factor)

        x, fc_scaling_factor = self.fc2(x, act_scaling_factor)
        maxs = torch.cat((maxs, torch.amax(x.view(x.size(0), -1), dim=-1, keepdim=True)), dim=1)            #####
        x = self.act7(x)
        x, act_scaling_factor = self.quant_act7(x, act_scaling_factor, fc_scaling_factor)

        x = self.fc3(x, act_scaling_factor)
        maxs = torch.cat((maxs, torch.amax(x[0].view(x[0].size(0), -1), dim=-1, keepdim=True)), dim=1)            #####

        return maxs

    # def initialize_counter(self, x, n_clusters):
    #     self.zero_counter = []

    #     self.features = nn.Sequential(self.conv1, self.act1, self.maxpool1,
    #                                   self.conv2, self.act2, self.maxpool2,
    #                                   self.conv3, self.act3,
    #                                   self.conv4, self.act4,
    #                                   self.conv5, self.act5,
    #                                   self.maxpool3, self.avgpool,
    #                                   self.fc1, self.act6,
    #                                   self.fc2, self.act7)
        
    #     x, _ = self.features[0](x)
    #     x = self.features[1](x)

    #     n_features = x.view(-1).size(0)
    #     self.zero_counter.append(torch.zeros((n_clusters, n_features), device='cuda'))
        
    #     x, _ = self.features[2](x)
    #     x, _ = self.features[3](x)
    #     x = self.features[4](x)
        
    #     n_features = x.view(-1).size(0)
    #     self.zero_counter.append(torch.zeros((n_clusters, n_features), device='cuda'))

    #     x, _ = self.features[5](x)
    #     x, _ = self.features[6](x)
    #     x = self.features[7](x)
        
    #     n_features = x.view(-1).size(0)
    #     self.zero_counter.append(torch.zeros((n_clusters, n_features), device='cuda'))

    #     x, _ = self.features[8](x)
    #     x = self.features[9](x)
        
    #     n_features = x.view(-1).size(0)
    #     self.zero_counter.append(torch.zeros((n_clusters, n_features), device='cuda'))

    #     x, _ = self.features[10](x)
    #     x = self.features[11](x)
        
    #     n_features = x.view(-1).size(0)
    #     self.zero_counter.append(torch.zeros((n_clusters, n_features), device='cuda'))

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
        
    #     x, _ = self.features[2](x)
    #     x, _ = self.features[3](x)
    #     x = self.features[4](x)

    #     layer_idx += 1
    #     n_features = self.zero_counter[layer_idx].size(1)
    #     for idx in range(x.size(0)):
    #         flattened = x[idx].view(-1)
    #         zeros_idx = (flattened == 0.0).nonzero(as_tuple=True)[0]
    #         zeros_idx %= n_features
    #         self.zero_counter[layer_idx][cluster, zeros_idx] += 1

    #     x, _ = self.features[5](x)
    #     x, _ = self.features[6](x)
    #     x = self.features[7](x)

    #     layer_idx += 1
    #     n_features = self.zero_counter[layer_idx].size(1)
    #     for idx in range(x.size(0)):
    #         flattened = x[idx].view(-1)
    #         zeros_idx = (flattened == 0.0).nonzero(as_tuple=True)[0]
    #         zeros_idx %= n_features
    #         self.zero_counter[layer_idx][cluster, zeros_idx] += 1

    #     x, _ = self.features[8](x)
    #     x = self.features[9](x)

    #     layer_idx += 1
    #     n_features = self.zero_counter[layer_idx].size(1)
    #     for idx in range(x.size(0)):
    #         flattened = x[idx].view(-1)
    #         zeros_idx = (flattened == 0.0).nonzero(as_tuple=True)[0]
    #         zeros_idx %= n_features
    #         self.zero_counter[layer_idx][cluster, zeros_idx] += 1

    #     x, _ = self.features[10](x)
    #     x = self.features[11](x)

    #     layer_idx += 1
    #     n_features = self.zero_counter[layer_idx].size(1)
    #     for idx in range(x.size(0)):
    #         flattened = x[idx].view(-1)
    #         zeros_idx = (flattened == 0.0).nonzero(as_tuple=True)[0]
    #         zeros_idx %= n_features
    #         self.zero_counter[layer_idx][cluster, zeros_idx] += 1

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
        src = torch.abs(x.view(x.size(0), -1).amax(dim=1) - self.clamp[l_idx][cluster])
        _max = torch.scatter_reduce(self.zero_buffer, 0, cluster, src=src, reduce=self.reduce, include_self=False)
        self.max_accumulator[l_idx] = self.accumulate(self.max_accumulator[l_idx], _max)
        return l_idx + 1
        
    def accumulate_output_max_distribution(self, x, cluster, n_clusters, l_idx=0, reduce='amax', accumulate='max', clamp=None):
        if not hasattr(self, 'max_accumulator'):
            self.reduce = reduce
            self.accumulate = ACCUMULATE[accumulate]
            self.max_accumulator = torch.zeros([7, n_clusters]).cuda()
            self.zero_buffer = torch.zeros(n_clusters).cuda()
            self.clamp = clamp if clamp is not None else torch.zeros_like(self.max_accumulator)
            
        x, act_scaling_factor = self.quant_input(x, cluster=cluster)
        x, conv_scaling_factor = self.conv1(x, act_scaling_factor)
        x = self.act1(x)
        
        ##################################
        l_idx = self.update_max_accumulator(x, cluster, l_idx)
        ##################################
        
        x, act_scaling_factor = self.quant_act1(x, act_scaling_factor, conv_scaling_factor, cluster=cluster)
        x, act_scaling_factor = self.maxpool1(x, act_scaling_factor)
        x, conv_scaling_factor = self.conv2(x, act_scaling_factor)
        x = self.act2(x)
        
        ##################################
        l_idx = self.update_max_accumulator(x, cluster, l_idx)
        ##################################
        
        x, act_scaling_factor = self.quant_act2(x, act_scaling_factor, conv_scaling_factor, cluster=cluster)
        x, act_scaling_factor = self.maxpool2(x, act_scaling_factor)
        x, conv_scaling_factor = self.conv3(x, act_scaling_factor)
        x = self.act3(x)

        ##################################
        l_idx = self.update_max_accumulator(x, cluster, l_idx)
        ##################################
        
        x, act_scaling_factor = self.quant_act3(x, act_scaling_factor, conv_scaling_factor, cluster=cluster)
        x, conv_scaling_factor = self.conv4(x, act_scaling_factor)
        x = self.act4(x)

        ##################################
        l_idx = self.update_max_accumulator(x, cluster, l_idx)
        ##################################

        x, act_scaling_factor = self.quant_act4(x, act_scaling_factor, conv_scaling_factor, cluster=cluster)
        x, conv_scaling_factor = self.conv5(x, act_scaling_factor)
        x = self.act5(x)

        ##################################
        l_idx = self.update_max_accumulator(x, cluster, l_idx)
        ##################################

        x, act_scaling_factor = self.quant_act5(x, act_scaling_factor, conv_scaling_factor, cluster=cluster)
        x, act_scaling_factor = self.maxpool3(x, act_scaling_factor)
        x, act_scaling_factor = self.avgpool(x, act_scaling_factor)

        x = x.view(x.size(0), -1)

        x, fc_scaling_factor = self.fc1(x, act_scaling_factor)
        x = self.act6(x)

        ##################################
        l_idx = self.update_max_accumulator(x, cluster, l_idx)
        ##################################

        x, act_scaling_factor = self.quant_act6(x, act_scaling_factor, fc_scaling_factor, cluster=cluster)
        x, fc_scaling_factor = self.fc2(x, act_scaling_factor)
        x = self.act7(x)

        ##################################
        l_idx = self.update_max_accumulator(x, cluster, l_idx)
        ##################################
        
        

    def get_output_max_distribution(self, x, cluster, n_clusters):
        if not hasattr(self, 'max_counter'):
            self.max_counter = []
            self.max_counter.append([[] for _ in range(n_clusters)])
            self.max_counter.append([[] for _ in range(n_clusters)])
            self.max_counter.append([[] for _ in range(n_clusters)])
            self.max_counter.append([[] for _ in range(n_clusters)])
            self.max_counter.append([[] for _ in range(n_clusters)])
            self.max_counter.append([[] for _ in range(n_clusters)])
            self.max_counter.append([[] for _ in range(n_clusters)])
        
        
        x, act_scaling_factor = self.quant_input(x, cluster=cluster)
        x, conv_scaling_factor = self.conv1(x, act_scaling_factor)
        x = self.act1(x)

        l_idx = 0
        _max = x.view(x.size(0), -1).max(dim=1).values
        if self.max_counter[l_idx][cluster] == []:
            self.max_counter[l_idx][cluster] = _max
        else:
            self.max_counter[l_idx][cluster] = torch.cat([self.max_counter[l_idx][cluster], _max])
            
        x, act_scaling_factor = self.quant_act1(x, act_scaling_factor, conv_scaling_factor, cluster=cluster)
        x, act_scaling_factor = self.maxpool1(x, act_scaling_factor)
        x, conv_scaling_factor = self.conv2(x, act_scaling_factor)
        x = self.act2(x)
        
        l_idx += 1
        _max = x.view(x.size(0), -1).max(dim=1).values
        if self.max_counter[l_idx][cluster] == []:
            self.max_counter[l_idx][cluster] = _max
        else:
            self.max_counter[l_idx][cluster] = torch.cat([self.max_counter[l_idx][cluster], _max])
        
        x, act_scaling_factor = self.quant_act2(x, act_scaling_factor, conv_scaling_factor, cluster=cluster)
        x, act_scaling_factor = self.maxpool2(x, act_scaling_factor)
        x, conv_scaling_factor = self.conv3(x, act_scaling_factor)
        x = self.act3(x)

        l_idx += 1
        _max = x.view(x.size(0), -1).max(dim=1).values
        if self.max_counter[l_idx][cluster] == []:
            self.max_counter[l_idx][cluster] = _max
        else:
            self.max_counter[l_idx][cluster] = torch.cat([self.max_counter[l_idx][cluster], _max])
            
        x, act_scaling_factor = self.quant_act3(x, act_scaling_factor, conv_scaling_factor, cluster=cluster)
        x, conv_scaling_factor = self.conv4(x, act_scaling_factor)
        x = self.act4(x)

        l_idx += 1
        _max = x.view(x.size(0), -1).max(dim=1).values
        if self.max_counter[l_idx][cluster] == []:
            self.max_counter[l_idx][cluster] = _max
        else:
            self.max_counter[l_idx][cluster] = torch.cat([self.max_counter[l_idx][cluster], _max])
            
        x, act_scaling_factor = self.quant_act4(x, act_scaling_factor, conv_scaling_factor, cluster=cluster)
        x, conv_scaling_factor = self.conv5(x, act_scaling_factor)
        x = self.act5(x)

        l_idx += 1
        _max = x.view(x.size(0), -1).max(dim=1).values
        if self.max_counter[l_idx][cluster] == []:
            self.max_counter[l_idx][cluster] = _max
        else:
            self.max_counter[l_idx][cluster] = torch.cat([self.max_counter[l_idx][cluster], _max])
            
        x, act_scaling_factor = self.quant_act5(x, act_scaling_factor, conv_scaling_factor, cluster=cluster)
        x, act_scaling_factor = self.maxpool3(x, act_scaling_factor)
        x, act_scaling_factor = self.avgpool(x, act_scaling_factor)

        x = x.view(x.size(0), -1)

        x, fc_scaling_factor = self.fc1(x, act_scaling_factor)
        x = self.act6(x)

        l_idx += 1
        _max = x.view(x.size(0), -1).max(dim=1).values
        if self.max_counter[l_idx][cluster] == []:
            self.max_counter[l_idx][cluster] = _max
        else:
            self.max_counter[l_idx][cluster] = torch.cat([self.max_counter[l_idx][cluster], _max])
            
        x, act_scaling_factor = self.quant_act6(x, act_scaling_factor, fc_scaling_factor, cluster=cluster)
        x, fc_scaling_factor = self.fc2(x, act_scaling_factor)
        x = self.act7(x)

        l_idx += 1
        _max = x.view(x.size(0), -1).max(dim=1).values
        if self.max_counter[l_idx][cluster] == []:
            self.max_counter[l_idx][cluster] = _max


    def get_ema_per_layer(self):
        ema = [[] for _ in range(7)]
        ema[0] = self.quant_act1.x_max
        ema[1] = self.quant_act2.x_max
        ema[2] = self.quant_act3.x_max
        ema[3] = self.quant_act4.x_max
        ema[4] = self.quant_act5.x_max
        ema[5] = self.quant_act6.x_max
        ema[6] = self.quant_act7.x_max
        return torch.stack(ema)
        

def q_alexnet(model, model_dict=None, num_clusters=None, skt_helper=None):
    return Q_AlexNet(model, model_dict, num_clusters, skt_helper)

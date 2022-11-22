"""
    Quantized MobileNetV2 for ImageNet-1K, implemented in PyTorch.
    Original paper: 'MobileNetV2: Inverted Residuals and Linear Bottlenecks,' https://arxiv.org/abs/1801.04381.
"""

import os
import torch.nn as nn
import torch.nn.init as init
from ..quantization_utils.quant_modules import *


class Q_LinearBottleneck(nn.Module):
    def __init__(self,
                model,
                in_channels,
                out_channels,
                stride,
                expansion,
                num_clusters=1):
        """
        So-called 'Linear Bottleneck' layer. It is used as a quantized MobileNetV2 unit.
        Parameters:
        ----------
        model : nn.Module
            The pretrained floating-point couterpart of this module with the same structure.
        in_channels : int
            Number of input channels.
        out_channels : int
            Number of output channels.
        stride : int or tuple/list of 2 int
            Strides of the second convolution layer.
        expansion : bool
            Whether do expansion of channels.
        """
        super(Q_LinearBottleneck, self).__init__()
        self.residual = (in_channels == out_channels) and (stride == 1)
        mid_channels = in_channels * 6 if expansion else in_channels
        self.activatition_func = nn.ReLU6()

        self.quant_act = QuantAct(num_clusters=num_clusters)

        self.conv1 = QuantBnConv2d()
        self.conv1.set_param(model.conv1.conv, model.conv1.bn)
        self.quant_act1 = QuantAct(num_clusters=num_clusters)

        self.conv2 = QuantBnConv2d()
        self.conv2.set_param(model.conv2.conv, model.conv2.bn)
        self.quant_act2 = QuantAct(num_clusters=num_clusters)

        self.conv3 = QuantBnConv2d()
        self.conv3.set_param(model.conv3.conv, model.conv3.bn)

        self.quant_act_int32 = QuantAct(num_clusters=num_clusters)

    def forward(self, x, scaling_factor_int32=None, cluster=None):
        if self.residual:
            identity = x

        x, act_scaling_factor = self.quant_act(x, scaling_factor_int32, None, None, None, None, cluster=cluster)

        x, weight_scaling_factor = self.conv1(x, act_scaling_factor)
        x = self.activatition_func(x)
        x, self.act_scaling_factor = self.quant_act1(x, act_scaling_factor, weight_scaling_factor, None, None, cluster=cluster)

        x, weight_scaling_factor = self.conv2(x, act_scaling_factor)
        x = self.activatition_func(x)
        x, act_scaling_factor = self.quant_act2(x, act_scaling_factor, weight_scaling_factor, None, None, cluster=cluster)

        x, weight_scaling_factor = self.conv3(x, act_scaling_factor)

        if self.residual:
            x = x + identity
            x, act_scaling_factor = self.quant_act_int32(x, act_scaling_factor, weight_scaling_factor, identity, scaling_factor_int32, None, cluster=cluster)
        else:
            x, act_scaling_factor = self.quant_act_int32(x, act_scaling_factor, weight_scaling_factor, None, None, None, cluster=cluster)

        return x, act_scaling_factor

    def accumulate_output_max_distribution(self, x, cluster, n_clusters, head, l_idx, scaling_factor_int32=None):
        if self.residual:
            identity = x
            
        x, act_scaling_factor = self.quant_act(x, scaling_factor_int32, None, None, None, None, cluster=cluster)
        x, weight_scaling_factor = self.conv1(x, act_scaling_factor)
        x = self.activatition_func(x)
        
        ##################################
        l_idx = head.update_max_accumulator(x, cluster, l_idx)
        ##################################
        
        x, self.act_scaling_factor = self.quant_act1(x, act_scaling_factor, weight_scaling_factor, None, None, cluster=cluster)
        x, weight_scaling_factor = self.conv2(x, act_scaling_factor)
        x = self.activatition_func(x)

        ##################################
        l_idx = head.update_max_accumulator(x, cluster, l_idx)
        ##################################
        
        x, act_scaling_factor = self.quant_act2(x, act_scaling_factor, weight_scaling_factor, None, None, cluster=cluster)
        x, weight_scaling_factor = self.conv3(x, act_scaling_factor)

        if self.residual:
            x = x + identity
            
            ##################################
            l_idx = head.update_max_accumulator(x, cluster, l_idx)
            ##################################
            
            x, act_scaling_factor = self.quant_act_int32(x, act_scaling_factor, weight_scaling_factor, identity, scaling_factor_int32, None, cluster=cluster)
        else:
            ##################################
            l_idx = head.update_max_accumulator(x, cluster, l_idx)
            ##################################
                
            x, act_scaling_factor = self.quant_act_int32(x, act_scaling_factor, weight_scaling_factor, None, None, None, cluster=cluster)
        return x, l_idx, act_scaling_factor


    def get_output_max_distribution(self, x, cluster, n_clusters, max_counter, l_idx, initialized, scaling_factor_int32=None):
        if not initialized:
            max_counter.append([[] for _ in range(n_clusters)])
            max_counter.append([[] for _ in range(n_clusters)])
            max_counter.append([[] for _ in range(n_clusters)])
            
        if self.residual:
            identity = x
            
        x, act_scaling_factor = self.quant_act(x, scaling_factor_int32, None, None, None, None, cluster=cluster)

        x, weight_scaling_factor = self.conv1(x, act_scaling_factor)
        x = self.activatition_func(x)
        
        l_idx += 1
        _max = x.view(x.size(0), -1).max(dim=1).values
        if max_counter[l_idx][cluster] == []:
            max_counter[l_idx][cluster] = _max
        else:
            max_counter[l_idx][cluster] = torch.cat([max_counter[l_idx][cluster], _max])
        
        x, self.act_scaling_factor = self.quant_act1(x, act_scaling_factor, weight_scaling_factor, None, None, cluster=cluster)

        x, weight_scaling_factor = self.conv2(x, act_scaling_factor)
        x = self.activatition_func(x)

        l_idx += 1
        _max = x.view(x.size(0), -1).max(dim=1).values
        if max_counter[l_idx][cluster] == []:
            max_counter[l_idx][cluster] = _max
        else:
            max_counter[l_idx][cluster] = torch.cat([max_counter[l_idx][cluster], _max])
        
        x, act_scaling_factor = self.quant_act2(x, act_scaling_factor, weight_scaling_factor, None, None, cluster=cluster)

        x, weight_scaling_factor = self.conv3(x, act_scaling_factor)

        if self.residual:
            x = x + identity
            
            l_idx += 1
            _max = x.view(x.size(0), -1).max(dim=1).values
            if max_counter[l_idx][cluster] == []:
                max_counter[l_idx][cluster] = _max
            else:
                max_counter[l_idx][cluster] = torch.cat([max_counter[l_idx][cluster], _max])
            
            x, act_scaling_factor = self.quant_act_int32(x, act_scaling_factor, weight_scaling_factor, identity, scaling_factor_int32, None, cluster=cluster)
        else:
            l_idx += 1
            _max = x.view(x.size(0), -1).max(dim=1).values
            if max_counter[l_idx][cluster] == []:
                max_counter[l_idx][cluster] = _max
            else:
                max_counter[l_idx][cluster] = torch.cat([max_counter[l_idx][cluster], _max])
                
            x, act_scaling_factor = self.quant_act_int32(x, act_scaling_factor, weight_scaling_factor, None, None, None, cluster=cluster)

        return x, l_idx, act_scaling_factor


class Q_MobileNetV2(nn.Module):
    """
    Quantized MobileNetV2 model from 'MobileNetV2: Inverted Residuals and Linear Bottlenecks,' https://arxiv.org/abs/1801.04381.
    Parameters:
    ----------
    model : nn.Module
        The pretrained floating-point MobileNetV2.
    channels : list of list of int
        Number of output channels for each unit.
    init_block_channels : int
        Number of output channels for the initial unit.
    final_block_channels : int
        Number of output channels for the final block of the feature extractor.
    in_channels : int, default 3
        Number of input channels.
    in_size : tuple of two ints, default (224, 224)
        Spatial size of the expected input image.
    num_classes : int, default 1000
        Number of classification classes.
    """
    def __init__(self,
                 model,
                 channels,
                 init_block_channels,
                 final_block_channels,
                 in_channels=3,
                 in_size=(224, 224),
                 num_classes=1000,
                 num_clusters=1):
        super(Q_MobileNetV2, self).__init__()
        self.in_size = in_size
        self.num_classes = num_classes
        self.channels = channels
        self.activatition_func = nn.ReLU6()

        # add input quantization
        self.quant_input = QuantAct(num_clusters=num_clusters)

        # change the inital block
        self.add_module("init_block", QuantBnConv2d())

        self.init_block.set_param(model.features.init_block.conv, model.features.init_block.bn)

        self.quant_act_int32 = QuantAct(num_clusters=num_clusters)

        self.features = nn.Sequential()
        # change the middle blocks
        in_channels = init_block_channels
        for i, channels_per_stage in enumerate(channels):
            stage = nn.Sequential()
            cur_stage = getattr(model.features, f'stage{i+1}')
            for j, out_channels in enumerate(channels_per_stage):
                cur_unit = getattr(cur_stage, f'unit{j+1}')

                stride = 2 if (j == 0) and (i != 0) else 1
                expansion = (i != 0) or (j != 0)

                stage.add_module("unit{}".format(j + 1), Q_LinearBottleneck(
                    cur_unit,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=stride,
                    expansion=expansion,
                    num_clusters=num_clusters
                    ))

                in_channels = out_channels
            self.features.add_module("stage{}".format(i + 1), stage)

        # change the final block
        self.quant_act_before_final_block = QuantAct(num_clusters=num_clusters)
        
        self.features.add_module("final_block", QuantBnConv2d())
        self.features.final_block.set_param(model.features.final_block.conv, model.features.final_block.bn)
        self.quant_act_int32_final = QuantAct(num_clusters=num_clusters)

        in_channels = final_block_channels

        self.features.add_module("final_pool", QuantAveragePool2d())
        self.features.final_pool.set_param(model.features.final_pool)
        self.quant_act_output = QuantAct(num_clusters=num_clusters)

        self.output = QuantConv2d()
        self.output.set_param(model.output)

    def forward(self, x, cluster):
        # quantize input
        x, act_scaling_factor = self.quant_input(x, cluster=cluster)

        # the init block
        x, weight_scaling_factor = self.init_block(x, act_scaling_factor)
        x = self.activatition_func(x)
        x, act_scaling_factor = self.quant_act_int32(x, act_scaling_factor, weight_scaling_factor, None, None, cluster=cluster)

        # the feature block
        for i, channels_per_stage in enumerate(self.channels):
            cur_stage = getattr(self.features, f'stage{i+1}')
            for j, out_channels in enumerate(channels_per_stage):
                cur_unit = getattr(cur_stage, f'unit{j+1}')
                x, act_scaling_factor = cur_unit(x, act_scaling_factor, cluster=cluster)

        x, act_scaling_factor = self.quant_act_before_final_block(x, act_scaling_factor, None, None, None, None, cluster=cluster)
        x, weight_scaling_factor = self.features.final_block(x, act_scaling_factor)
        x = self.activatition_func(x)
        x, act_scaling_factor = self.quant_act_int32_final(x, act_scaling_factor, weight_scaling_factor, None, None, None, cluster=cluster)

        # the final pooling
        x = self.features.final_pool(x, act_scaling_factor)

        # the output
        x, act_scaling_factor = self.quant_act_output(x, act_scaling_factor, None, None, None, None, cluster=cluster)
        x, _ = self.output(x, act_scaling_factor)

        x = x.view(x.size(0), -1)

        return x

    def toggle_full_precision(self):
        # print('Model Toggle full precision FUNC')
        for module in self.modules():
            if isinstance(module, (QuantAct, QuantLinear, QuantBnConv2d, QuantBn, QuantConv2d)):
                precision = getattr(module, 'full_precision_flag')
                setattr(module, 'full_precision_flag', not precision)

    def update_max_accumulator(self, x, cluster, l_idx):
        if type(x) is tuple:
            x = x[0]
        _max = torch.scatter_reduce(self.zero_buffer, 0, cluster, src=x.view(x.size(0), -1).max(dim=1).values, reduce=self.reduce)
        self.max_accumulator[l_idx] = self.max_accumulator[l_idx].max(_max)
        return l_idx + 1    
    
    def accumulate_output_max_distribution(self, x, cluster, n_clusters, l_idx=0, reduce='amax'):
        if not hasattr(self, 'max_accumulator'):
            self.reduce = reduce
            self.max_accumulator = torch.zeros([54, n_clusters]).cuda()
            self.zero_buffer = torch.zeros(n_clusters).cuda()
            
        x, act_scaling_factor = self.quant_input(x, cluster=cluster)
        x, weight_scaling_factor = self.init_block(x, act_scaling_factor)
        x = self.activatition_func(x)
        
        ##################################
        l_idx = self.update_max_accumulator(x, cluster, l_idx)
        ##################################
            
        x, act_scaling_factor = self.quant_act_int32(x, act_scaling_factor, weight_scaling_factor, None, None, cluster=cluster)
        
        # the feature block
        for i, channels_per_stage in enumerate(self.channels):
            cur_stage = getattr(self.features, f'stage{i+1}')
            for j, out_channels in enumerate(channels_per_stage):
                cur_unit = getattr(cur_stage, f'unit{j+1}')
                x, l_idx, act_scaling_factor = cur_unit.accumulate_output_max_distribution(x, cluster, n_clusters, self,
                                                                                           l_idx, act_scaling_factor)

        ##################################
        l_idx = self.update_max_accumulator(x, cluster, l_idx)
        ##################################
            
        x, act_scaling_factor = self.quant_act_before_final_block(x, act_scaling_factor, None, None, None, None, cluster=cluster)
        x, weight_scaling_factor = self.features.final_block(x, act_scaling_factor)
        x = self.activatition_func(x)
        
        ##################################
        l_idx = self.update_max_accumulator(x, cluster, l_idx)
        ##################################
            
    
    def get_output_max_distribution(self, x, cluster, n_clusters):
        initialized = True
        if not hasattr(self, 'max_counter'):
            initialized = False
            self.max_counter = []
            self.max_counter.append([[] for _ in range(n_clusters)])
            self.max_counter.append([[] for _ in range(n_clusters)])
            self.max_counter.append([[] for _ in range(n_clusters)])
        
        x, act_scaling_factor = self.quant_input(x, cluster=cluster)
        x, weight_scaling_factor = self.init_block(x, act_scaling_factor)
        x = self.activatition_func(x)
        
        l_idx = 0
        _max = x.view(x.size(0), -1).max(dim=1).values
        if self.max_counter[l_idx][cluster] == []:
            self.max_counter[l_idx][cluster] = _max
        else:
            self.max_counter[l_idx][cluster] = torch.cat([self.max_counter[l_idx][cluster], _max])
            
        x, act_scaling_factor = self.quant_act_int32(x, act_scaling_factor, weight_scaling_factor, None, None, cluster=cluster)
        
        # the feature block
        for i, channels_per_stage in enumerate(self.channels):
            cur_stage = getattr(self.features, f'stage{i+1}')
            for j, out_channels in enumerate(channels_per_stage):
                cur_unit = getattr(cur_stage, f'unit{j+1}')
                x, l_idx, act_scaling_factor = cur_unit.get_output_max_distribution(x, cluster, n_clusters, self.max_counter, l_idx,
                                                                             initialized, act_scaling_factor)

        l_idx += 1
        _max = x.view(x.size(0), -1).max(dim=1).values
        if self.max_counter[l_idx][cluster] == []:
            self.max_counter[l_idx][cluster] = _max
        else:
            self.max_counter[l_idx][cluster] = torch.cat([self.max_counter[l_idx][cluster], _max])
            
        x, act_scaling_factor = self.quant_act_before_final_block(x, act_scaling_factor, None, None, None, None, cluster=cluster)
        x, weight_scaling_factor = self.features.final_block(x, act_scaling_factor)
        x = self.activatition_func(x)
        
        l_idx += 1
        _max = x.view(x.size(0), -1).max(dim=1).values
        if self.max_counter[l_idx][cluster] == []:
            self.max_counter[l_idx][cluster] = _max
        else:
            self.max_counter[l_idx][cluster] = torch.cat([self.max_counter[l_idx][cluster], _max])
            

def q_get_mobilenetv2(model, width_scale, num_clusters=None):
    """
    Create quantized MobileNetV2 model with specific parameters.
    Parameters:
    ----------
    model : nn.Module
        The pretrained floating-point MobileNetV2.
    width_scale : float
        Scale factor for width of layers.
    """

    init_block_channels = 32
    final_block_channels = 1280
    layers = [1, 2, 3, 4, 3, 3, 1]
    downsample = [0, 1, 1, 1, 0, 1, 0]
    channels_per_layers = [16, 24, 32, 64, 96, 160, 320]

    from functools import reduce
    channels = reduce(
        lambda x, y: x + [[y[0]] * y[1]] if y[2] != 0 else x[:-1] + [x[-1] + [y[0]] * y[1]],
        zip(channels_per_layers, layers, downsample),
        [[]])

    if width_scale != 1.0:
        channels = [[int(cij * width_scale) for cij in ci] for ci in channels]
        init_block_channels = int(init_block_channels * width_scale)
        if width_scale > 1.0:
            final_block_channels = int(final_block_channels * width_scale)

    net = Q_MobileNetV2(
        model,
        channels=channels,
        init_block_channels=init_block_channels,
        final_block_channels=final_block_channels,
        num_clusters=num_clusters)

    return net


def q_mobilenetv2_w1(model, num_clusters=None):
    """
    Quantized 1.0 MobileNetV2-224 model from 'MobileNetV2: Inverted Residuals and Linear Bottlenecks,'
    https://arxiv.org/abs/1801.04381.
    Parameters:
    model : nn.Module
        The pretrained floating-point MobileNetV2.
    """
    return q_get_mobilenetv2(model, width_scale=1.0, num_clusters=num_clusters)

from operator import itemgetter
from pickle import FALSE

import torch
import time
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.nn import Module, Parameter
from .quant_utils import *


class QuantLinear(Module):
    """
    Class to quantize weights of given linear layer

    Parameters:
    ----------
    weight_bit : int, default 4
        Bitwidth for quantized weights.
    bias_bit : int, default None
        Bitwidth for quantized bias.
    full_precision_flag : bool, default False
        If True, use fp32 and skip quantization
    quant_mode : 'symmetric' or 'asymmetric', default 'symmetric'
        The mode for quantization.
    per_channel : bool, default False
        Whether to use channel-wise quantization.
    fix_flag : bool, default False
        Whether the module is in fixed mode or not.
    weight_percentile : float, default 0
        The percentile to setup quantization range, 0 means no use of percentile, 99.9 means to cut off 0.1%.
    """

    def __init__(self,
                 weight_bit=4,
                 bias_bit=None,
                 full_precision_flag=False,
                 quant_mode='symmetric',
                 per_channel=False,
                 fix_flag=False,
                 weight_percentile=0):
        super(QuantLinear, self).__init__()
        self.full_precision_flag = full_precision_flag
        self.weight_bit = weight_bit
        self.quant_mode = quant_mode
        self.per_channel = per_channel
        self.fix_flag = fix_flag
        self.weight_percentile = weight_percentile
        self.bias_bit = bias_bit
        self.quantize_bias = (False if bias_bit is None else True)
        self.counter = 0

        self.is_classifier = False

    def __repr__(self):
        s = super(QuantLinear, self).__repr__()
        s = "(" + s + " weight_bit={}, full_precision_flag={}, quantize_fn={})".format(
            self.weight_bit, self.full_precision_flag, self.quant_mode)
        return s

    def set_param(self, linear, model_dict=None, dict_idx=None):
        #self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.register_buffer('fc_scaling_factor', torch.zeros(self.out_features))
        if model_dict is not None :
            self.weight = Parameter(model_dict[dict_idx+'.weight'].data.clone())
        else :
            self.weight = Parameter(linear.weight.data.clone())
        self.register_buffer('weight_integer', torch.zeros_like(self.weight))

        self.register_buffer('bias_integer', torch.zeros_like(linear.bias))
        if model_dict is not None :
            try : 
                self.bias = Parameter(model_dict[dict_idx + '.bias'].data.clone())
            except AttributeError:
                self.bias = None
        else :
            try:
                self.bias = Parameter(linear.bias.data.clone())
            except AttributeError:
                self.bias = None

    def fix(self):
        self.fix_flag = True

    def unfix(self):
        self.fix_flag = False

    def forward(self, x, prev_act_scaling_factor=None):
        """
        using quantized weights to forward activation x
        """
        if type(x) is tuple:
            prev_act_scaling_factor = x[1]
            x = x[0]

        if not self.full_precision_flag:
            if self.quant_mode == "symmetric":
                self.weight_function = SymmetricQuantFunction.apply
            elif self.quant_mode == "asymmetric":
                self.weight_function = AsymmetricQuantFunction.apply
            else:
                raise ValueError("unknown quant mode: {}".format(self.quant_mode))


            w = self.weight
            w_transform = w.data.detach()
            # calculate the quantization range of weights and bias
            if self.per_channel:
                w_min, _ = torch.min(w_transform, dim=1, out=None)
                w_max, _ = torch.max(w_transform, dim=1, out=None)
            else:
                w_min = w_transform.min().expand(1)
                w_max = w_transform.max().expand(1)

            # perform the quantization
            if self.quant_mode == 'symmetric':
                self.fc_scaling_factor = symmetric_linear_quantization_params(self.weight_bit, w_min, w_max,
                                                                                self.per_channel)
                self.weight_integer = self.weight_function(self.weight, self.weight_bit, self.fc_scaling_factor.view(-1, 1))

                bias_scaling_factor =  prev_act_scaling_factor.view(-1, 1) * self.fc_scaling_factor.view(1, -1)
                if self.quantize_bias and (self.bias is not None):
                    self.bias_integer = self.weight_function(self.bias, self.bias_bit, bias_scaling_factor) # TODO
                else:
                    self.bias_integer = None
            else:
                raise Exception('For weight, we only support symmetric quantization.')

            x_int = x / prev_act_scaling_factor.view(-1, 1)

            if not self.is_classifier:
                if self.bias_integer is not None:
                    return (ste_round.apply(F.linear(x_int, weight=self.weight_integer, bias=None) + self.bias_integer) * bias_scaling_factor, 
                            self.fc_scaling_factor)
                else:
                    return (ste_round.apply(F.linear(x_int, weight=self.weight_integer, bias=None)) * bias_scaling_factor, 
                            self.fc_scaling_factor)
            else:
                if self.bias_integer is not None:
                    return (F.linear(x_int, weight=self.weight_integer) + self.bias_integer) * bias_scaling_factor
                else:
                    return F.linear(x_int, weight=self.weight_integer) * bias_scaling_factor
        else:
            return (F.linear(x, weight=self.weight, bias=self.bias), None)


class QuantAct(Module):
    """
    Class to quantize given activations

    Parameters:
    ----------
    activation_bit : int, default 4
        Bitwidth for quantized activations.
    act_range_momentum : float, default 0.95
        Momentum for updating the activation quantization range.
    full_precision_flag : bool, default False
        If True, use fp32 and skip quantization
    running_stat : bool, default True
        Whether to use running statistics for activation quantization range.
    quant_mode : 'symmetric' or 'asymmetric', default 'symmetric'
        The mode for quantization.
    fix_flag : bool, default False
        Whether the module is in fixed mode or not.
    act_percentile : float, default 0
        The percentile to setup quantization range, 0 means no use of percentile, 99.9 means to cut off 0.1%.
    fixed_point_quantization : bool, default False
        Whether to skip deployment-oriented operations and use fixed-point rather than integer-only quantization.
    """

    def __init__(self,
                 activation_bit=4,
                 act_range_momentum=0.95,
                 full_precision_flag=False,
                 running_stat=True,
                 quant_mode="symmetric",
                 fix_flag=False,
                 act_percentile=0,
                 fixed_point_quantization=False,
                 num_clusters=1):
        super(QuantAct, self).__init__()

        self.activation_bit = activation_bit
        self.act_range_momentum = act_range_momentum
        self.full_precision_flag = full_precision_flag
        self.running_stat = running_stat
        self.quant_mode = quant_mode
        self.fix_flag = fix_flag
        self.act_percentile = act_percentile
        self.fixed_point_quantization = fixed_point_quantization
        self.num_clusters = num_clusters
        self.init_record = True

        self.register_buffer('tmp', torch.zeros(self.num_clusters))
        self.register_buffer('min', torch.zeros(self.num_clusters))
        self.register_buffer('max', torch.zeros(self.num_clusters))
        self.register_buffer('x_min', torch.zeros(self.num_clusters))
        self.register_buffer('x_max', torch.zeros(self.num_clusters))
        
        self.register_buffer('initialize', torch.ones(self.num_clusters, dtype=torch.bool))
        
        self.register_buffer('act_scaling_factor', torch.zeros(self.num_clusters))
        self.register_buffer('pre_weight_scaling_factor', torch.ones(1))
        self.register_buffer('identity_weight_scaling_factor', torch.ones(1))
        self.register_buffer('concat_weight_scaling_factor', torch.ones(1))
        # self.register_buffer('isDaq', torch.zeros(1, dtype=torch.bool))

        self.is_classifier = False

    def __repr__(self):
        return "{0}(activation_bit={1}, full_precision_flag={2}, " \
               "quant_mode={3}".format(self.__class__.__name__, self.activation_bit,
                                       self.full_precision_flag, self.quant_mode)

    def fix(self):
        """
        fix the activation range by setting running stat to False
        """
        self.running_stat = False
        self.fix_flag = True

    def unfix(self):
        """
        unfix the activation range by setting running stat to True
        """
        self.running_stat = True
        self.fix_flag = False

    def init_records(self):
        self.init_record = False
        self.initialize = ~self.initialize
        self.x_min.zero_()
        self.x_max.zero_()
        self.min.zero_()
        self.max.zero_()

    def forward(self, x, pre_act_scaling_factor=None, pre_weight_scaling_factor=None, identity=None,
                identity_scaling_factor=None, identity_weight_scaling_factor=None, concat=None,
                concat_scaling_factor=None, concat_weight_scaling_factor=None, cluster=None, candidate_channels=None):
        """
        x: the activation that we need to quantize
        pre_act_scaling_factor: the scaling factor of the previous activation quantization layer
        pre_weight_scaling_factor: the scaling factor of the previous weight quantization layer
        identity: if True, we need to consider the identity branch
        identity_scaling_factor: the scaling factor of the previous activation quantization of identity
        identity_weight_scaling_factor: the scaling factor of the weight quantization layer in the identity branch

        Note that there are two cases for identity branch:
        (1) identity branch directly connect to the input featuremap
        (2) identity branch contains convolutional layers that operate on the input featuremap
        """
        if type(x) is tuple:
            if len(x) == 4:
                cluster = x[2]
                channel_num = x[3]
            elif len(x) == 3:
                cluster = x[2]
            pre_act_scaling_factor = x[1]
            x = x[0]

        # perform the quantization
        if not self.full_precision_flag:
            if self.quant_mode == "symmetric":
                self.act_function = SymmetricQuantFunction.apply
            elif self.quant_mode == "asymmetric":
                self.act_function = AsymmetricQuantFunction.apply
            else:
                raise ValueError("unknown quant mode: {}".format(self.quant_mode))

            if not torch.is_tensor(cluster):
                cluster = torch.empty(x.shape[0]).fill_(cluster).type(torch.long)

            uniques = torch.unique(cluster)
            # calculate the quantization range of the activations
            if self.running_stat or self.init_record:
                if self.act_percentile == 0:
                    data = x.view(x.size(0), -1).clone().detach()
                    self.min = torch.scatter_reduce(self.tmp, 0, cluster, src=data.amin(dim=1), reduce="amin", include_self=False)
                    self.max = torch.scatter_reduce(self.tmp, 0, cluster, src=data.amax(dim=1), reduce="amax", include_self=False)
                        
                elif self.quant_mode == 'symmetric':        # TODO
                    x_min, x_max = get_percentile_min_max(x.detach().view(-1), 100 - self.act_percentile,
                                                    self.act_percentile, output_tensor=True)
                # Note that our asymmetric quantization is implemented using scaled unsigned integers without zero_points,
                # that is to say our asymmetric quantization should always be after ReLU, which makes
                # the minimum value to be always 0. As a result, if we use percentile mode for asymmetric quantization,
                # the lower_percentile will be set to 0 in order to make sure the final x_min is 0.
                elif self.quant_mode == 'asymmetric':       # TODO
                    x_min, x_max = get_percentile_min_max(x.detach().view(-1), 0, self.act_percentile, output_tensor=True)                # Initialization
                if (non_initialized := torch.flatten(self.initialize.nonzero())).size(0):
                    tmp_min = torch.index_select(self.min, 0, non_initialized)
                    tmp_max = torch.index_select(self.max, 0, non_initialized)
                    self.x_min.index_add_(0, non_initialized, tmp_min)
                    self.x_max.index_add_(0, non_initialized, tmp_max)
                    self.initialize.index_fill_(0, uniques, False)
                
                if self.act_range_momentum == -1:
                    self.x_min.index_copy_(0, uniques, torch.index_select(torch.minimum(self.x_min, self.min), 0, uniques).view(-1, 1))
                    self.x_max.index_copy_(0, uniques, torch.index_select(torch.maximum(self.x_max, self.max), 0, uniques).view(-1, 1))
                else:
                    tmp_min = torch.index_select(self.x_min * self.act_range_momentum + self.min * (1 - self.act_range_momentum), 0, uniques).view(-1, 1)
                    tmp_max = torch.index_select(self.x_max * self.act_range_momentum + self.max * (1 - self.act_range_momentum), 0, uniques).view(-1, 1)
                    self.x_min.view(-1, 1).index_copy_(0, uniques, tmp_min)
                    self.x_max.view(-1, 1).index_copy_(0, uniques, tmp_max)

                    
            if self.quant_mode == 'symmetric':
                self.act_scaling_factor = symmetric_linear_quantization_params(self.activation_bit,
                                                                           self.x_min, self.x_max, (self.num_clusters != 1))
            # Note that our asymmetric quantization is implemented using scaled unsigned integers
            # without zero_point shift. As a result, asymmetric quantization should be after ReLU,
            # and the self.act_zero_point should be 0.
            else:
                self.act_scaling_factor, self.act_zero_point = asymmetric_linear_quantization_params(
                    self.activation_bit, self.x_min, self.x_max, True)
                
            act_scaling_factor = self.act_scaling_factor[cluster]
            if (pre_act_scaling_factor is None) or (self.fixed_point_quantization == True):
                # this is for the case of input quantization,
                # or the case using fixed-point rather than integer-only quantization
                quant_act_int = self.act_function(x, self.activation_bit, act_scaling_factor.view(-1, 1, 1, 1))
            elif type(pre_act_scaling_factor) is list:       # TODO
                # this is for the case of multi-branch quantization
                branch_num = len(pre_act_scaling_factor)
                quant_act_int = x
                start_channel_index = 0
                for i in range(branch_num):
                    quant_act_int[:, start_channel_index: start_channel_index + channel_num[i], :, :] \
                        = fixedpoint_fn.apply(x[:, start_channel_index: start_channel_index + channel_num[i], :, :],
                                              self.activation_bit, self.quant_mode, act_scaling_factor, 0,
                                              pre_act_scaling_factor[i],
                                              self.pre_weight_scaling_factor)
                    start_channel_index += channel_num[i]
            else:
                if identity is None and concat is None:
                    if pre_weight_scaling_factor is None:
                        pre_weight_scaling_factor = self.pre_weight_scaling_factor
                    quant_act_int = fixedpoint_fn.apply(x, self.activation_bit, self.quant_mode,
                                                        act_scaling_factor, 0, pre_act_scaling_factor,
                                                        pre_weight_scaling_factor)
                elif identity is not None:
                    if identity_weight_scaling_factor is None:
                        identity_weight_scaling_factor = self.identity_weight_scaling_factor
                    quant_act_int = fixedpoint_fn.apply(x, self.activation_bit, self.quant_mode,
                                                        act_scaling_factor, 1, pre_act_scaling_factor,
                                                        pre_weight_scaling_factor,
                                                        identity, identity_scaling_factor,
                                                        identity_weight_scaling_factor)
                else:
                    if concat_weight_scaling_factor is None:
                        concat_weight_scaling_factor = self.concat_weight_scaling_factor
                    quant_act_int = fixedpoint_fn.apply(x, self.activation_bit, self.quant_mode,
                                                         act_scaling_factor, 2, pre_act_scaling_factor,
                                                         pre_weight_scaling_factor,
                                                         concat, concat_scaling_factor,
                                                         concat_weight_scaling_factor)
            
            if len(quant_act_int.shape) == 4:
                correct_output_scale = act_scaling_factor.view(-1, 1, 1, 1)
            else:
                correct_output_scale = act_scaling_factor.view(-1, 1)
            return (quant_act_int * correct_output_scale, act_scaling_factor)
        else:
            return (x, None)


class QuantBnConv2d(Module):
    """
    Class to quantize given convolutional layer weights, with support for both folded BN and separate BN.

    Parameters:
    ----------
    weight_bit : int, default 4
        Bitwidth for quantized weights.
    bias_bit : int, default None
        Bitwidth for quantized bias.
    full_precision_flag : bool, default False
        If True, use fp32 and skip quantization
    quant_mode : 'symmetric' or 'asymmetric', default 'symmetric'
        The mode for quantization.
    per_channel : bool, default False
        Whether to use channel-wise quantization.
    fix_flag : bool, default False
        Whether the module is in fixed mode or not.
    weight_percentile : float, default 0
        The percentile to setup quantization range, 0 means no use of percentile, 99.9 means to cut off 0.1%.
    fix_BN : bool, default False
        Whether to fix BN statistics during training.
    fix_BN_threshold: int, default None
        When to start training with folded BN.
    """

    def __init__(self,
                 weight_bit=4,
                 bias_bit=None,
                 full_precision_flag=False,
                 quant_mode="symmetric",
                 per_channel=False,
                 fix_flag=False,
                 weight_percentile=0,
                 fix_BN=False,
                 fix_BN_threshold=None,
                 skt_helper=None,
                 is_identity=False):
        super(QuantBnConv2d, self).__init__()
        self.weight_bit = weight_bit
        self.full_precision_flag = full_precision_flag
        self.per_channel = per_channel
        self.fix_flag = fix_flag
        self.weight_percentile = weight_percentile
        self.bias_bit = bias_bit
        self.quantize_bias = False if bias_bit is None else True
        self.quant_mode = quant_mode
        self.fix_BN = fix_BN
        self.training_BN_mode = fix_BN
        self.fix_BN_threshold = fix_BN_threshold
        self.counter = 1
        self.skt_helper = skt_helper
        self.mixed_precision = False
        self.init_record = False
        self.is_identity = is_identity

    def set_param(self, conv, bn):
        self.in_channels = conv.in_channels
        self.out_channels = conv.out_channels
        self.register_buffer('conv_scaling_factor', torch.zeros(self.out_channels))
        self.register_buffer('convbn_scaling_factor', torch.zeros(self.out_channels))
        self.register_buffer('weight_integer', torch.zeros_like(conv.weight.data))
        self.register_buffer('bias_integer', torch.zeros_like(bn.bias))

        self.conv = conv
        self.bn = bn
        self.bn.momentum = 0.99
        self.in_channels = self.conv.in_channels

        if self.skt_helper:
            self.mixed_precision = self.init_record = True
            self.register_buffer('low_group', torch.tensor([], dtype=torch.int64))
            self.register_buffer('high_group', torch.arange(self.in_channels, dtype=torch.int64))
            self.register_buffer('input_range', torch.zeros((2, self.in_channels), device='cuda'))
            self.register_buffer('apply_ema', torch.zeros(1, dtype=torch.bool, device='cuda'))
            self.register_buffer('prev_mask', torch.zeros(self.in_channels, dtype=torch.bool, device='cuda'))
            self.element_size = self.conv.weight.size(0) * self.conv.weight.size(2) * self.conv.weight.size(3)

    def __repr__(self):
        conv_s = super(QuantBnConv2d, self).__repr__()
        s = "({0}, weight_bit={1}, bias_bit={2}, groups={3}, wt-channel-wise={4}, wt-percentile={5}, quant_mode={6})".format(
            conv_s, self.weight_bit, self.bias_bit, self.conv.groups, self.per_channel, self.weight_percentile,
            self.quant_mode)
        return s

    def fix(self):
        """
        fix the BN statistics by setting fix_BN to True
        """
        self.fix_flag = True
        self.fix_BN = True

    def unfix(self):
        """
        change the mode (fixed or not) of BN statistics to its original status
        """
        self.fix_flag = False
        self.fix_BN = self.training_BN_mode

    def init_records(self):
        self.init_record = False
        self.input_range.zero_()
        self.apply_ema = ~self.apply_ema

    def reset_input_range(self):
        self.input_range.zero_()
        self.apply_ema = ~self.apply_ema

    @torch.no_grad()
    def record_range(self, x):
        if self.init_record:
            self.element_size += x.size(2) * x.size(3)
        x_transform = x.transpose(1, 0).reshape(x.size(1), -1)
        input_min = x_transform.min(dim=1).values
        input_max = x_transform.max(dim=1).values

        if self.apply_ema:
            updated_min = self.input_range[0] * self.act_range_momentum + input_min * (
                    1 - self.act_range_momentum)
            updated_max = self.input_range[1] * self.act_range_momentum + input_max * (
                    1 - self.act_range_momentum)
            self.input_range[0], self.input_range[1] = updated_min, updated_max
        else:
            self.input_range[0], self.input_range[1] = input_min, input_max
            self.apply_ema = ~self.apply_ema

    def forward(self, x, pre_act_scaling_factor=None):
        """
        x: the input activation
        pre_act_scaling_factor: the scaling factor of the previous activation quantization layer

        """
        if type(x) is tuple:
            pre_act_scaling_factor = x[1]
            x = x[0]

        if not self.full_precision_flag:
            if self.quant_mode == "symmetric":
                if self.skt_helper:
                    self.weight_function = MixedSymmetricQuantFunction.apply
                else:
                    self.weight_function = SymmetricQuantFunction.apply
                if self.quantize_bias:
                    self.bias_function = SymmetricQuantFunction.apply
            elif self.quant_mode == "asymmetric":
                if self.skt_helper:
                    self.weight_function = MixedAsymmetricQuantFunction.apply
                else:
                    self.weight_function = AsymmetricQuantFunction.apply
                if self.quantize_bias:
                    self.bias_function = AsymmetricQuantFunction.apply
            else:
                raise ValueError("unknown quant mode: {}".format(self.quant_mode))

            # determine whether to fold BN or not
            # if self.fix_flag == False:
            #     self.counter += 1
            #     if (self.fix_BN_threshold == None) or (self.counter < self.fix_BN_threshold):
            #         self.fix_BN = self.training_BN_mode
            #     else:
            #         if self.counter == self.fix_BN_thrQeshold:
            #             print("Start Training with Folded BN")
            #         self.fix_BN = True

            if not self.fix_flag or self.init_record:
                if self.mixed_precision:
                    self.record_range(x)

            weight = self.conv.weight
            # gradient manipulation
            if self.mixed_precision and not self.fix_flag:
                 x = SKT_GRAD.apply(x, self.skt_helper, self.quant_mode)
                 weight = SKT_GRAD.apply(self.conv.weight, self.skt_helper, self.quant_mode)

            # run the forward without folding BN
            if self.fix_BN == False:
                if self.per_channel:
                    w_transform = weight.data.contiguous().view(self.conv.out_channels, -1)

                    if self.weight_percentile == 0:
                        w_min = w_transform.min(dim=1).values
                        w_max = w_transform.max(dim=1).values
                    else:
                        lower_percentile = 100 - self.weight_percentile
                        upper_percentile = self.weight_percentile
                        input_length = w_transform.shape[1]

                        lower_index = math.ceil(input_length * lower_percentile * 0.01)
                        upper_index = math.ceil(input_length * upper_percentile * 0.01)

                        w_min = torch.kthvalue(w_transform, k=lower_index, dim=1).values
                        w_max = torch.kthvalue(w_transform, k=upper_index, dim=1).values
                else:
                    if self.weight_percentile == 0:
                        w_min = weight.data.min()
                        w_max = weight.data.max()
                    else:
                        w_min, w_max = get_percentile_min_max(weight.view(-1), 100 - self.weight_percentile,
                                                            self.weight_percentile, output_tensor=True)

                self.conv_scaling_factor = symmetric_linear_quantization_params(self.weight_bit, w_min, w_max, self.per_channel)
                if self.skt_helper:
                    self.weight_integer = self.weight_function(weight, self.low_group, self.high_group,
                                                               self.conv_scaling_factor.view(1, -1, 1, 1))
                else:
                    self.weight_integer = self.weight_function(weight, self.weight_bit,
                                                               self.conv_scaling_factor.view(-1, 1, 1, 1))

                bias_scaling_factor = pre_act_scaling_factor.view(-1, 1) * self.conv_scaling_factor.view(1, -1)
                if self.quantize_bias and (self.conv.bias is not None):
                    self.bias_integer = self.weight_function(self.conv.bias, self.bias_bit, bias_scaling_factor)
                    self.bias_integer = self.bias_integer.view(self.bias_integer.size(0), self.bias_integer.size(1), 1, 1)
                else:
                    self.bias_integer = None

                x_int = x / pre_act_scaling_factor.view(-1, 1, 1, 1)

                if self.low_group.size(0):
                    # truncate the rightmost 3 bits
                    with torch.no_grad():
                        x_int[:, self.low_group] = torch.round(x_int[:, self.low_group] / 8) * 8
                    x_int[:, self.low_group] = torch.clamp(x_int[:, self.low_group], -64, 63)

                correct_output_scale = bias_scaling_factor.view(bias_scaling_factor.size(0), bias_scaling_factor.size(1), 1, 1)

                if self.bias_integer is not None:
                    conv_output = (F.conv2d(x_int, self.weight_integer, None, self.conv.stride, self.conv.padding,
                                        self.conv.dilation, self.conv.groups) + self.bias_integer) * correct_output_scale
                else:
                    conv_output = F.conv2d(x_int, self.weight_integer, None, self.conv.stride, self.conv.padding,
                                        self.conv.dilation, self.conv.groups) * correct_output_scale

                batch_mean = torch.mean(conv_output, dim=(0, 2, 3))
                batch_var = torch.var(conv_output, dim=(0, 2, 3))

                # update mean and variance in running stats
                self.bn.running_mean = self.bn.running_mean.detach() * self.bn.momentum + (
                        1 - self.bn.momentum) * batch_mean
                self.bn.running_var = self.bn.running_var.detach() * self.bn.momentum + (
                            1 - self.bn.momentum) * batch_var

                scaled_weight = self.bn.weight / torch.sqrt(batch_var + self.bn.eps)
                scaled_bias = self.bn.bias - batch_mean * scaled_weight

                # if not self.fix_flag:
                #     # update mean and variance in running stats
                #     self.bn.running_mean = self.bn.running_mean.detach() * self.bn.momentum + (
                #                     1 - self.bn.momentum) * batch_mean
                #     self.bn.running_var = self.bn.running_var.detach() * self.bn.momentum + (1 - self.bn.momentum) * batch_var
                #
                #     scaled_weight = self.bn.weight / torch.sqrt(batch_var + self.bn.eps)
                #     scaled_bias = self.bn.bias - batch_mean * scaled_weight
                # else:
                #     scaled_weight = self.bn.weight / torch.sqrt(self.bn.running_var + self.bn.eps)
                #     scaled_bias = self.bn.bias - self.bn.running_mean * scaled_weight

                output = scaled_weight.view(1, -1, 1, 1) * conv_output + scaled_bias.view(1, -1, 1, 1)

                return (output, self.conv_scaling_factor * scaled_weight)
            # fold BN and fix running statistics
            else:
                running_std = torch.sqrt(self.bn.running_var.detach() + self.bn.eps)
                scale_factor = self.bn.weight / running_std
                scaled_weight = weight * scale_factor.reshape([self.conv.out_channels, 1, 1, 1])

                if self.conv.bias is not None:
                    scaled_bias = self.conv.bias
                else:
                    scaled_bias = torch.zeros_like(self.bn.running_mean)
                scaled_bias = (scaled_bias - self.bn.running_mean.detach()) * scale_factor + self.bn.bias

                if self.per_channel:
                    w_transform = scaled_weight.data.contiguous().view(self.conv.out_channels, -1)

                    if self.weight_percentile == 0:
                        w_min = w_transform.min(dim=1).values
                        w_max = w_transform.max(dim=1).values
                    else:
                        lower_percentile = 100 - self.weight_percentile
                        upper_percentile = self.weight_percentile
                        input_length = w_transform.shape[1]

                        lower_index = math.ceil(input_length * lower_percentile * 0.01)
                        upper_index = math.ceil(input_length * upper_percentile * 0.01)

                        w_min = torch.kthvalue(w_transform, k=lower_index, dim=1).values
                        w_max = torch.kthvalue(w_transform, k=upper_index, dim=1).values
                else:
                    if self.weight_percentile == 0:
                        w_min = scaled_weight.data.min()
                        w_max = scaled_weight.data.max()
                    else:
                        w_min, w_max = get_percentile_min_max(scaled_weight.view(-1), 100 - self.weight_percentile,
                                                            self.weight_percentile, output_tensor=True)

                if self.quant_mode == 'symmetric':
                    self.convbn_scaling_factor = symmetric_linear_quantization_params(self.weight_bit,
                                                                                    w_min, w_max, self.per_channel)
                    if self.skt_helper:
                        self.weight_integer = self.weight_function(scaled_weight, self.low_group, self.high_group,
                                                                   self.convbn_scaling_factor.view(1, -1, 1, 1))
                    else:
                        self.weight_integer = self.weight_function(scaled_weight, self.weight_bit,
                                                                   self.convbn_scaling_factor.view(-1, 1, 1, 1))
                    bias_scaling_factor = pre_act_scaling_factor.view(-1, 1) * self.convbn_scaling_factor.view(1, -1)
                    if self.quantize_bias:
                        self.bias_integer = self.bias_function(scaled_bias, self.bias_bit, bias_scaling_factor)
                        self.bias_integer = self.bias_integer.view(self.bias_integer.size(0), self.bias_integer.size(1), 1, 1)
                else:
                    raise Exception('For weight, we only support symmetric quantization.')

                x_int = x / pre_act_scaling_factor.view(-1, 1, 1, 1)

                if self.low_group.size(0):
                    # truncate the rightmost 3 bits
                    with torch.no_grad():
                        x_int[:, self.low_group] = torch.round(x_int[:, self.low_group] / 8) * 8
                    x_int[:, self.low_group] = torch.clamp(x_int[:, self.low_group], -64, 63)

                correct_output_scale = bias_scaling_factor.view(bias_scaling_factor.size(0), bias_scaling_factor.size(1), 1, 1)

                return ((F.conv2d(x_int, self.weight_integer, None, self.conv.stride, self.conv.padding,
                                self.conv.dilation, self.conv.groups) + self.bias_integer) * correct_output_scale, self.convbn_scaling_factor)
        else:
            conv_output = F.conv2d(x, self.conv.weight, self.conv.bias, self.conv.stride, self.conv.padding, self.conv.dilation,
                                   self.conv.groups)
            if self.fix_BN == False:
                batch_mean = torch.mean(conv_output, dim=(0, 2, 3))
                batch_var = torch.var(conv_output, dim=(0, 2, 3))

                self.bn.running_mean = self.bn.running_mean.detach() * self.bn.momentum + (1 - self.bn.momentum) * batch_mean
                self.bn.running_var = self.bn.running_var.detach() * self.bn.momentum + (1 - self.bn.momentum) * batch_var
                    
            else:
                batch_mean = self.bn.running_mean.detach()
                batch_var = self.bn.running_var.detach()

            output_factor = self.bn.weight.view(1, -1, 1, 1) / torch.sqrt(batch_var + self.bn.eps).view(1, -1, 1, 1)
            return (output_factor * (conv_output - batch_mean.view(1, -1, 1, 1)) + self.bn.bias.view(1, -1, 1, 1), None)

class QuantBn(Module):
    """
    Quantized Batch Normalization Layer

    Parameters:
    """
    def __init__(self,
                 weight_bit=16,
                 bias_bit=None,
                 full_precision_flag=False,
                 quant_mode="symmetric",
                 per_channel=False,
                 fix_flag=False,
                 weight_percentile=0,
                 fix_BN=False,
                 fix_BN_threshold=None,
                 runtime_helper=None):
        super(QuantBn, self).__init__()
        self.weight_bit = weight_bit
        self.quant_mode = quant_mode
        self.full_precision_flag = full_precision_flag
        self.per_channel = per_channel
        self.fix_flag = fix_flag
        self.weight_percentile = weight_percentile
        self.bias_bit = bias_bit
        self.quantize_bias = False if bias_bit is None else True
        self.fix_BN = fix_BN
        self.training_BN_mode = fix_BN
        self.fix_BN_threshold = fix_BN_threshold
        self.counter = 1
        self.runtime_helper = runtime_helper


    def set_param(self, bn):
        self.bn = bn
        self.bn.momentum = 0.99

        self.register_buffer('bn_scaling_factor', torch.zeros(1))
        self.register_buffer('weight_integer', torch.zeros_like(self.bn.weight))
        self.register_buffer('bias_integer', torch.zeros_like(self.bn.bias))

    def __repr__(self):
        s = super(QuantBn, self).__repr__()
        s = "({0}, weight_bit={1}, bias_bit={2}, wt-percentile={3}, quant_mode={4}, fix_BN={5})".format(
            s, self.weight_bit, self.bias_bit, self.weight_percentile, self.quant_mode, self.fix_BN)
        
        return s

    def fix(self):
        """
        fix the BN statistics by setting fix_BN to True
        """
        self.fix_flag = True
        self.fix_BN = True

    def unfix(self):
        """
        change the mode (fixed or not) of BN statistics to its original status
        """
        self.fix_flag = False
        self.fix_BN = self.training_BN_mode

    def forward(self, x, pre_act_scaling_factor=None):
        if type(x) is tuple:
            x_scaling_factor = x[1]
            x = x[0]

        if not self.full_precision_flag:
            if self.quant_mode == "symmetric":
                self.weight_function = SymmetricQuantFunction.apply
            elif self.quant_mode == "asymmetric":
                self.weight_function = AsymmetricQuantFunction.apply
            else:
                raise ValueError("unknown quant mode: {}".format(self.quant_mode))
            
            # determine whether to fold BN or not
            # if self.fix_flag == False:
            #     self.counter += 1
            #     if (self.fix_BN_threshold == None) or (self.counter < self.fix_BN_threshold):
            #         self.fix_BN = self.training_BN_mode
            #     else:
            #         if self.counter == self.fix_BN_threshold:
            #             print("Start Training with Folded BN")
            #         self.fix_BN = True
            #         self.training_BN_mode = True

            if self.fix_BN is False:
                batch_mean = torch.mean(x, dim=(0, 2, 3))
                batch_var = torch.var(x, dim=(0, 2, 3))

                # update mean and variance in running stats
                self.bn.running_mean = self.bn.running_mean.detach() * self.bn.momentum + (1 - self.bn.momentum) * batch_mean
                self.bn.running_var = self.bn.running_var.detach() * self.bn.momentum + (1 - self.bn.momentum) * batch_var

                scaled_weight = self.bn.weight / torch.sqrt(batch_var + self.bn.eps)
                scaled_bias = self.bn.bias - batch_mean * scaled_weight
            else :
                scaled_weight = self.bn.weight / torch.sqrt(self.bn.running_var.detach() + self.bn.eps)
                scaled_bias = self.bn.bias - self.bn.running_mean.detach() * scaled_weight

            if self.per_channel:
                w_transform = scaled_weight.data.contiguous().view(self.bn.num_features, -1)

                if self.weight_percentile == 0:
                    w_min = w_transform.min(dim=1).values
                    w_max = w_transform.max(dim=1).values
                else:
                    lower_percentile = 100 - self.weight_percentile
                    upper_percentile = self.weight_percentile
                    input_length = w_transform.shape[1]

                    lower_index = math.ceil(input_length * lower_percentile * 0.01)
                    upper_index = math.ceil(input_length * upper_percentile * 0.01)

                    w_min = torch.kthvalue(w_transform, k=lower_index, dim=1).values
                    w_max = torch.kthvalue(w_transform, k=upper_index, dim=1).values
            else:
                if self.weight_percentile == 0:
                    w_min = scaled_weight.data.min()
                    w_max = scaled_weight.data.max()
                else:
                    w_min, w_max = get_percentile_min_max(scaled_weight.view(-1), 100 - self.weight_percentile,
                                                    self.weight_percentile, output_tensor=True)

            if self.quant_mode == 'symmetric':
                self.bn_scaling_factor = symmetric_linear_quantization_params(self.weight_bit, w_min, w_max, self.per_channel)
                self.weight_integer = self.weight_function(scaled_weight, self.weight_bit, 
                                                        self.bn_scaling_factor)
                bias_scaling_factor = pre_act_scaling_factor.view(-1, 1) * self.bn_scaling_factor.view(1, -1)
                if self.quantize_bias and scaled_bias is not None:
                    self.bias_integer = self.weight_function(scaled_bias, self.bias_bit, bias_scaling_factor)
                    self.bias_integer = self.bias_integer.view(self.bias_integer.size(0), self.bias_integer.size(1), 1, 1)
            else:
                raise Exception('For weight, we only support symmetric quantization.')

            x_int = x / pre_act_scaling_factor.view(-1, 1, 1, 1)
            correct_output_scale = bias_scaling_factor.view(bias_scaling_factor.size(0), bias_scaling_factor.size(1), 1, 1)

            output = self.weight_integer.view(1, -1, 1, 1) * x_int + self.bias_integer

            return (output * correct_output_scale, self.bn_scaling_factor)
        else :
            if self.fix_BN == False:
                batch_mean = torch.mean(x, dim=(0, 2, 3))
                batch_var = torch.var(x, dim=(0, 2, 3))

                self.bn.running_mean = self.bn.running_mean.detach() * self.bn.momentum + (1 - self.bn.momentum) * batch_mean
                self.bn.running_var = self.bn.running_var.detach() * self.bn.momentum + (1 - self.bn.momentum) * batch_var
                output_factor = self.bn.weight.view(1, -1, 1, 1) / torch.sqrt(batch_var + self.bn.eps).view(1, -1, 1, 1)
                return (output_factor * (x - batch_mean.view(1, -1, 1, 1)) + self.bn.bias.view(1, -1, 1, 1), None)
            else:
                batch_mean = self.bn.running_mean.detach()
                batch_var = self.bn.running_var.detach()

                output_factor = self.bn.weight.view(1, -1, 1, 1) / torch.sqrt(batch_var + self.bn.eps).view(1, -1, 1, 1)
                return (output_factor * (x - batch_mean.view(1, -1, 1, 1)) + self.bn.bias.view(1, -1, 1, 1), None)


class QuantMaxPool2d(Module):
    """
    Quantized MaxPooling Layer

    Parameters:
    ----------
    kernel_size : int, default 3
        Kernel size for max pooling.
    stride : int, default 2
        stride for max pooling.
    padding : int, default 0
        padding for max pooling.
    """

    def __init__(self,
                 kernel_size=3,
                 stride=2,
                 padding=0,
                 ceil_mode=False):
        super(QuantMaxPool2d, self).__init__()

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.pool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding, ceil_mode=self.ceil_mode)

    def forward(self, x, x_scaling_factor=None):
        if type(x) is tuple:
            if len(x) == 3:
                cluster = x[2]
            x_scaling_factor = x[1]
            x = x[0]

            x = self.pool(x)
            return (x, x_scaling_factor, cluster)

        x = self.pool(x)
        return (x, x_scaling_factor)


class QuantDropout(Module):
    """
    Quantized Dropout Layer

    Parameters:
    ----------
    p : float, default 0
        p is the dropout ratio.
    """

    def __init__(self, p=0):
        super(QuantDropout, self).__init__()

        self.dropout = nn.Dropout(p)

    def forward(self, x, x_scaling_factor=None):
        if type(x) is tuple:
            x_scaling_factor = x[1]
            x = x[0]

        x = self.dropout(x)

        return (x, x_scaling_factor)


class QuantAveragePool2d(Module):
    """
    Quantized Average Pooling Layer

    Parameters:
    ----------
    kernel_size : int, default 7
        Kernel size for average pooling.
    stride : int, default 1
        stride for average pooling.
    padding : int, default 0
        padding for average pooling.
    """

    def __init__(self,
                 kernel_size=7,
                 stride=1,
                 padding=0,
                 output=None):
        super(QuantAveragePool2d, self).__init__()

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        if output is None:
            self.final_pool = nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=padding)
        else:
            self.final_pool = nn.AdaptiveAvgPool2d(output_size=output)

    def set_param(self, pool):
        self.final_pool = pool

    def forward(self, x, x_scaling_factor=None):
        if type(x) is tuple:
            x_scaling_factor = x[1]
            x = x[0]

        if x_scaling_factor is None:
            return (self.final_pool(x), None)

        correct_scaling_factor = x_scaling_factor.view(-1, 1, 1, 1)

        x_int = x / correct_scaling_factor
        x_int = ste_round.apply(x_int)
        x_int = self.final_pool(x_int)

        x_int = transfer_float_averaging_to_int_averaging.apply(x_int)

        return (x_int * correct_scaling_factor, x_scaling_factor)


class QuantConv2d(Module):
    """
    Class to quantize weights of given convolutional layer

    Parameters:
    ----------
    weight_bit : int, default 4
        Bitwidth for quantized weights.
    bias_bit : int, default None
        Bitwidth for quantized bias.
    full_precision_flag : bool, default False
        If True, use fp32 and skip quantization
    quant_mode : 'symmetric' or 'asymmetric', default 'symmetric'
        The mode for quantization.
    per_channel : bool, default False
        Whether to use channel-wise quantization.
    fix_flag : bool, default False
        Whether the module is in fixed mode or not.
    weight_percentile : float, default 0
        The percentile to setup quantization range, 0 means no use of percentile, 99.9 means to cut off 0.1%.
    """

    def __init__(self,
                 weight_bit=4,
                 bias_bit=None,
                 full_precision_flag=False,
                 quant_mode="symmetric",
                 per_channel=False,
                 fix_flag=False,
                 weight_percentile=0,
                 skt_helper=None):
        super(QuantConv2d, self).__init__()
        self.full_precision_flag = full_precision_flag
        self.weight_bit = weight_bit
        self.quant_mode = quant_mode
        self.per_channel = per_channel
        self.fix_flag = fix_flag
        self.weight_percentile = weight_percentile
        self.bias_bit = bias_bit
        self.quantize_bias = (False if bias_bit is None else True)
        self.skt_helper = skt_helper
        self.mixed_precision = False
        self.init_record = False

    def __repr__(self):
        s = super(QuantConv2d, self).__repr__()
        s = "({0}, weight_bit={1}, bias_bit={2}, groups={3}, wt-channel-wise={4}, wt-percentile={5}, quant_mode={6})".format(
            s, self.weight_bit, self.bias_bit, self.conv.groups, self.per_channel, self.weight_percentile,
            self.quant_mode)
        return s

    def set_param(self, conv, model_dict=None, dict_idx=None):
        self.in_channels = conv.in_channels
        self.out_channels = conv.out_channels
        self.kernel_size = conv.kernel_size
        self.stride = conv.stride
        self.padding = conv.padding
        self.dilation = conv.dilation
        self.groups = conv.groups
        self.conv = conv
        self.register_buffer('conv_scaling_factor', torch.zeros(self.out_channels))

        if self.skt_helper:
            self.mixed_precision = self.init_record = True
            self.register_buffer('low_group', torch.tensor([], dtype=torch.int64))
            self.register_buffer('high_group', torch.arange(self.in_channels, dtype=torch.int64))
            self.register_buffer('input_range', torch.zeros((2, self.in_channels), device='cuda'))
            self.register_buffer('apply_ema', torch.zeros(1, dtype=torch.bool, device='cuda'))
            self.register_buffer('prev_mask', torch.zeros(self.in_channels, dtype=torch.bool, device='cuda'))
            self.element_size = self.conv.weight.size(0) * self.conv.weight.size(2) * self.conv.weight.size(3)

        if model_dict is not None :
            self.weight = Parameter(model_dict[dict_idx+'.weight'].data.clone())
        else :
            self.weight = Parameter(conv.weight.data.clone())
        self.register_buffer('weight_integer', torch.zeros_like(self.weight))

        if model_dict is not None :
            try : 
                self.bias = Parameter(model_dict[dict_idx + '.bias'].data.clone())
            except AttributeError:
                self.bias = None
        else :
            try:
                self.bias = Parameter(conv.bias.data.clone())
            except AttributeError:
                self.bias = None

    def fix(self):
        self.fix_flag = True

    def unfix(self):
        self.fix_flag = False

    def init_records(self):
        self.init_record = False
        self.input_range.zero_()
        self.apply_ema = ~self.apply_ema

    def reset_input_range(self):
        self.input_range.zero_()
        self.apply_ema = ~self.apply_ema

    @torch.no_grad()
    def record_range(self, x):
        if self.init_record:
            self.element_size += x.size(2) * x.size(3)
        x_transform = x.transpose(1, 0).reshape(x.size(1), -1)
        input_min = x_transform.min(dim=1).values
        input_max = x_transform.max(dim=1).values

        if self.apply_ema:
            updated_min = self.input_range[0] * self.act_range_momentum + input_min * (
                    1 - self.act_range_momentum)
            updated_max = self.input_range[1] * self.act_range_momentum + input_max * (
                    1 - self.act_range_momentum)
            self.input_range[0], self.input_range[1] = updated_min, updated_max
        else:
            self.input_range[0], self.input_range[1] = input_min, input_max
            self.apply_ema = ~self.apply_ema


    def forward(self, x, pre_act_scaling_factor=None):
        if type(x) is tuple:
            pre_act_scaling_factor = x[1]
            x = x[0]

        if not self.full_precision_flag:
            if self.quant_mode == "symmetric":
                if self.skt_helper:
                    self.weight_function = MixedSymmetricQuantFunction.apply
                else:
                    self.weight_function = SymmetricQuantFunction.apply
                if self.quantize_bias:
                    self.bias_function = SymmetricQuantFunction.apply
            elif self.quant_mode == "asymmetric":
                if self.skt_helper:
                    self.weight_function = MixedAsymmetricQuantFunction.apply
                else:
                    self.weight_function = AsymmetricQuantFunction.apply
                if self.quantize_bias:
                    self.bias_function = AsymmetricQuantFunction.apply
            else:
                raise ValueError("unknown quant mode: {}".format(self.quant_mode))

            if not self.fix_flag or self.init_record:
                if self.mixed_precision:
                    self.record_range(x)

            weight = self.weight
            # gradient manipulation
            if self.mixed_precision and not self.fix_flag:
                x = SKT_GRAD.apply(x, self.skt_helper, self.quant_mode)
                weight = SKT_GRAD.apply(self.weight, self.skt_helper, self.quant_mode)

            if self.per_channel:
                w_transform = weight.data.contiguous().view(self.out_channels, -1)

                if self.weight_percentile == 0:
                    w_min = w_transform.min(dim=1).values
                    w_max = w_transform.max(dim=1).values
                else:
                    lower_percentile = 100 - self.weight_percentile
                    upper_percentile = self.weight_percentile
                    input_length = w_transform.shape[1]

                    lower_index = math.ceil(input_length * lower_percentile * 0.01)
                    upper_index = math.ceil(input_length * upper_percentile * 0.01)

                    w_min = torch.kthvalue(w_transform, k=lower_index, dim=1).values
                    w_max = torch.kthvalue(w_transform, k=upper_index, dim=1).values

            else:
                if self.weight_percentile == 0:
                    w_min = weight.data.min()
                    w_max = weight.data.max()
                else:
                    w_min, w_max = get_percentile_min_max(weight.view(-1), 100 - self.weight_percentile,
                                                        self.weight_percentile, output_tensor=True)

            self.conv_scaling_factor = symmetric_linear_quantization_params(self.weight_bit, w_min, w_max,
                                                                            self.per_channel)
            if self.skt_helper:
                self.weight_integer = self.weight_function(weight, self.low_group, self.high_group, self.conv_scaling_factor.view(1, -1, 1, 1))
            else:
                self.weight_integer = self.weight_function(weight, self.weight_bit, self.conv_scaling_factor.view(-1, 1, 1, 1))

            bias_scaling_factor = pre_act_scaling_factor.view(-1, 1) * self.conv_scaling_factor.view(1, -1)
            if self.quantize_bias and (self.bias is not None):
                self.bias_integer = self.bias_function(self.bias, self.bias_bit, bias_scaling_factor)
                self.bias_integer = self.bias_integer.view(self.bias_integer.size(0), self.bias_integer.size(1), 1, 1)
            else:
                self.bias_integer = None
            
            x_int = x / pre_act_scaling_factor.view(-1, 1, 1, 1)
            if self.low_group.size(0):
                with torch.no_grad():
                    # truncate the rightmost 3 bits
                    x_int[:, self.low_group] = torch.round(x_int[:, self.low_group] / 8) * 8
                x_int[:, self.low_group] = torch.clamp(x_int[:, self.low_group], -64, 63)

            correct_output_scale = bias_scaling_factor.view(bias_scaling_factor.size(0), bias_scaling_factor.size(1), 1, 1)

            if self.bias_integer is not None:
                return ((F.conv2d(x_int, self.weight_integer, None, self.conv.stride, self.conv.padding,
                                self.conv.dilation, self.conv.groups) + self.bias_integer) * correct_output_scale, self.conv_scaling_factor)
            else:
                return (F.conv2d(x_int, self.weight_integer, None, self.conv.stride, self.conv.padding,
                                self.conv.dilation, self.conv.groups) * correct_output_scale, self.conv_scaling_factor)
        return (F.conv2d(x, self.weight, self.bias, self.conv.stride, self.conv.padding, self.conv.dilation, self.conv.groups), None)


def freeze_model(model):
    """
    freeze the activation range
    """
    if type(model) == QuantAct:
        model.fix()
    elif type(model) == QuantConv2d:
        model.fix()
    elif type(model) == QuantBn:
        model.fix()
    elif type(model) == QuantLinear:
        model.fix()
    elif type(model) == QuantBnConv2d:
        model.fix()
    elif type(model) == nn.Sequential:
        for n, m in model.named_children():
            freeze_model(m)
    else:
        for attr in dir(model):
            mod = getattr(model, attr)
            if isinstance(mod, nn.Module) and 'norm' not in attr:
                freeze_model(mod)


def unfreeze_model(model):
    """
    unfreeze the activation range
    """
    if type(model) == QuantAct:
        model.unfix()
    elif type(model) == QuantConv2d:
        model.unfix()
    elif type(model) == QuantBn:
        model.unfix()
    elif type(model) == QuantLinear:
        model.unfix()
    elif type(model) == QuantBnConv2d:
        model.unfix()
    elif type(model) == nn.Sequential:
        for n, m in model.named_children():
            unfreeze_model(m)
    else:
        for attr in dir(model):
            mod = getattr(model, attr)
            if isinstance(mod, nn.Module) and 'norm' not in attr:
                unfreeze_model(mod)

"""
    Quantized InceptionV3 for ImageNet-1K, implemented in PyTorch.
    Original paper: 'Rethinking the Inception Architecture for Computer Vision,'
    https://arxiv.org/abs/1512.00567.
"""

__all__ = ['Q_InceptionV3', 'q_inceptionv3']

import os
import torch
import torch.nn as nn
import torch.nn.init as init
from ..quantization_utils.quant_modules import *


class Q_InceptConv(nn.Module):
    """
    Quantized InceptionV3 specific convolution block.

    Note that all other parameters are cloned from model.
    Parameters:
    ----------
    model : nn.Module
        The pretrained floating-point couterpart of this module with the same structure.
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int or tuple/list of 2 int
        Convolution window size.
    stride : int or tuple/list of 2 int
        Strides of the convolution.
    padding : int or tuple/list of 2 int
        Padding value for convolution layer.
    """
    def __init__(self,
                 model,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 num_clusters=1):
        super(Q_InceptConv, self).__init__()

        self.q_convbn = QuantBnConv2d()
        self.q_convbn.set_param(model.conv, model.bn)

        self.relu = nn.ReLU(inplace=True)
        self.q_activ = QuantAct(num_clusters=num_clusters)

    def forward(self, x):
        assert (type(x) is tuple)
        cluster = x[2]
        a_sf = x[1]
        (x, w_sf) = self.q_convbn(x)
        x = self.relu(x)
        (x, a_sf) = self.q_activ(x, a_sf, w_sf, cluster=cluster)
        return (x, a_sf, cluster)


def q_incept_conv1x1(model,
                     in_channels,
                     out_channels,
                     num_clusters=1):
    """
    1x1 version of the quantized InceptionV3 specific convolution block.

    Parameters:
    ----------
    model : nn.Module
        The pretrained floating-point couterpart of this module with the same structure.
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    """
    return Q_InceptConv(
        model=model,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        num_clusters=num_clusters)


class Q_Concurrent(nn.Sequential):
    """
    A container for concatenation of modules on the base of the sequential container.

    Parameters:
    ----------
    axis : int, default 1
        The axis on which to concatenate the outputs.
    stack : bool, default False
        Whether to concatenate tensors along a new dimension.
    """
    def __init__(self,
                 axis=1,
                 stack=False):
        super(Q_Concurrent, self).__init__()
        self.axis = axis
        self.stack = stack

    def forward(self, x):
        out = []
        scaling_factor = []
        channel_num = []
        cluster = x[2]
        for module in self._modules.values():
            branch_out = module(x)
            if type(branch_out) is tuple:
                out.append(branch_out[0])
                scaling_factor.append(branch_out[1])
                channel_num.append(branch_out[0].shape[1])
            else:
                out.append(branch_out)
        if self.stack:
            out = torch.stack(tuple(out), dim=self.axis)
        else:
            out = torch.cat(tuple(out), dim=self.axis)

        assert (type(out) is not tuple)
        return (out, scaling_factor, cluster, channel_num)


class Q_MaxPoolBranch(nn.Module):
    """
    Quantized InceptionV3 specific max pooling branch block.
    """
    def __init__(self, num_clusters=1):
        super(Q_MaxPoolBranch, self).__init__()
        self.q_input_act = QuantAct(num_clusters=num_clusters)
        self.q_pool = QuantMaxPool2d(
            kernel_size=3,
            stride=2,
            padding=0)

    def forward(self, x):
        cluster = x[2]
        (x, a_sf) = self.q_input_act(x)
        (x, a_sf) = self.q_pool((x, a_sf))
        return (x, a_sf, cluster)


class Q_AvgPoolBranch(nn.Module):
    """
    Quantized InceptionV3 specific average pooling branch block.

    Parameters:
    ----------
    model : nn.Module
        The pretrained floating-point couterpart of this module with the same structure.
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    """
    def __init__(self,
                 model,
                 in_channels,
                 out_channels,
                 num_clusters=1):
        super(Q_AvgPoolBranch, self).__init__()
        self.q_input_act = QuantAct(num_clusters=num_clusters)
        self.q_pool = QuantAveragePool2d(
            kernel_size=3,
            stride=1,
            padding=1)
        self.q_pool_act = QuantAct(num_clusters=num_clusters)
        self.q_conv = q_incept_conv1x1(
            model=model.conv,
            in_channels=in_channels,
            out_channels=out_channels,
            num_clusters=num_clusters)

    def forward(self, x):
        cluster = x[2]
        (x, a_sf) = self.q_input_act(x)
        (x, a_sf) = self.q_pool((x, a_sf))
        (x, a_sf) = self.q_pool_act((x, a_sf, cluster))
        (x, a_sf, _) = self.q_conv((x, a_sf, cluster))

        return (x, a_sf, cluster)


class Q_Conv1x1Branch(nn.Module):
    """
    Quantized InceptionV3 specific convolutional 1x1 branch block.

    Parameters:
    ----------
    model : nn.Module
        The pretrained floating-point couterpart of this module with the same structure.
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    """
    def __init__(self,
                 model,
                 in_channels,
                 out_channels,
                 num_clusters=1):
        super(Q_Conv1x1Branch, self).__init__()
        self.q_input_act = QuantAct(num_clusters=num_clusters)
        self.q_conv = q_incept_conv1x1(
            model=model.conv,
            in_channels=in_channels,
            out_channels=out_channels,
            num_clusters=num_clusters)

    def forward(self, x):
        cluster = x[2]
        (x, a_sf) = self.q_input_act(x)
        (x, a_sf, _) = self.q_conv((x, a_sf, cluster))
        return (x, a_sf, cluster)


class Q_ConvSeqBranch(nn.Module):
    """
    Quantized InceptionV3 specific convolutional sequence branch block.

    Parameters:
    ----------
    model : nn.Module
        The pretrained floating-point couterpart of this module with the same structure.
    in_channels : int
        Number of input channels.
    out_channels_list : list of tuple of int
        List of numbers of output channels.
    kernel_size_list : list of tuple of int or tuple of tuple/list of 2 int
        List of convolution window sizes.
    strides_list : list of tuple of int or tuple of tuple/list of 2 int
        List of strides of the convolution.
    padding_list : list of tuple of int or tuple of tuple/list of 2 int
        List of padding values for convolution layers.
    """
    def __init__(self,
                 model,
                 in_channels,
                 out_channels_list,
                 kernel_size_list,
                 strides_list,
                 padding_list,
                 num_clusters=1):
        super(Q_ConvSeqBranch, self).__init__()
        assert (len(out_channels_list) == len(kernel_size_list))
        assert (len(out_channels_list) == len(strides_list))
        assert (len(out_channels_list) == len(padding_list))

        self.q_input_act = QuantAct(num_clusters=num_clusters)
        self.q_conv_list = nn.Sequential()
        for i, (out_channels, kernel_size, strides, padding) in enumerate(zip(
                out_channels_list, kernel_size_list, strides_list, padding_list)):
            self.q_conv_list.add_module("q_conv{}".format(i + 1), Q_InceptConv(
                model=getattr(model.conv_list, "conv{}".format(i + 1)),
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=strides,
                padding=padding,
                num_clusters=num_clusters))
            in_channels = out_channels

    def forward(self, x):
        cluster = x[2]
        (x, a_sf) = self.q_input_act(x)
        (x, a_sf, _) = self.q_conv_list((x, a_sf, cluster))

        return (x, a_sf, cluster)


class Q_ConvSeq3x3Branch(nn.Module):
    """
    Quantized InceptionV3 specific convolutional sequence branch block with splitting by 3x3.

    Parameters:
    ----------
    model : nn.Module
        The pretrained floating-point couterpart of this module with the same structure.
    in_channels : int
        Number of input channels.
    out_channels_list : list of tuple of int
        List of numbers of output channels.
    kernel_size_list : list of tuple of int or tuple of tuple/list of 2 int
        List of convolution window sizes.
    strides_list : list of tuple of int or tuple of tuple/list of 2 int
        List of strides of the convolution.
    padding_list : list of tuple of int or tuple of tuple/list of 2 int
        List of padding values for convolution layers.
    """
    def __init__(self,
                 model,
                 in_channels,
                 out_channels_list,
                 kernel_size_list,
                 strides_list,
                 padding_list,
                 num_clusters=1):
        super(Q_ConvSeq3x3Branch, self).__init__()

        self.q_input_act = QuantAct(num_clusters=num_clusters)
        self.q_conv_list = nn.Sequential()
        for i, (out_channels, kernel_size, strides, padding) in enumerate(zip(
                out_channels_list, kernel_size_list, strides_list, padding_list)):
            self.q_conv_list.add_module("q_conv{}".format(i + 1), Q_InceptConv(
                model=getattr(model.conv_list, "conv{}".format(i + 1)),
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=strides,
                padding=padding,
                num_clusters=num_clusters))
            in_channels = out_channels
        self.q_conv1x3 = Q_InceptConv(
            model=model.conv1x3,
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=(1, 3),
            stride=1,
            padding=(0, 1),
            num_clusters=num_clusters)
        self.q_conv3x1 = Q_InceptConv(
            model=model.conv3x1,
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=(3, 1),
            stride=1,
            padding=(1, 0),
            num_clusters=num_clusters)
        self.q_rescaling_activ = QuantAct(num_clusters=num_clusters)

    def forward(self, x):
        cluster = x[2]
        (x, a_sf) = self.q_input_act(x)
        (x, a_sf, _) = self.q_conv_list((x, a_sf, cluster))
        y1, scaling_factor1, _ = self.q_conv1x3((x, a_sf, cluster))
        y2, scaling_factor2, _ = self.q_conv3x1((x, a_sf, cluster))
        channel_num = [y1.shape[1], y2.shape[1]]
        x = torch.cat((y1, y2), dim=1)
        x, a_sf = self.q_rescaling_activ((x, [scaling_factor1, scaling_factor2], cluster, channel_num))
        return (x, a_sf, cluster)


class Q_InceptionAUnit(nn.Module):
    """
    Quantized InceptionV3 type Inception-A unit.

    Parameters:
    ----------
    model : nn.Module
        The pretrained floating-point couterpart of this module with the same structure.
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    """
    def __init__(self,
                 model,
                 in_channels,
                 out_channels,
                 num_clusters=1):
        super(Q_InceptionAUnit, self).__init__()
        assert (out_channels > 224)
        pool_out_channels = out_channels - 224

        self.branches = Q_Concurrent()
        self.branches.add_module("branch1", Q_Conv1x1Branch(
            model=model.branches.branch1,
            in_channels=in_channels,
            out_channels=64,
            num_clusters=num_clusters))
        self.branches.add_module("branch2", Q_ConvSeqBranch(
            model=model.branches.branch2,
            in_channels=in_channels,
            out_channels_list=(48, 64),
            kernel_size_list=(1, 5),
            strides_list=(1, 1),
            padding_list=(0, 2),
            num_clusters=num_clusters))
        self.branches.add_module("branch3", Q_ConvSeqBranch(
            model=model.branches.branch3,
            in_channels=in_channels,
            out_channels_list=(64, 96, 96),
            kernel_size_list=(1, 3, 3),
            strides_list=(1, 1, 1),
            padding_list=(0, 1, 1),
            num_clusters=num_clusters))
        self.branches.add_module("branch4", Q_AvgPoolBranch(
            model=model.branches.branch4,
            in_channels=in_channels,
            out_channels=pool_out_channels,
            num_clusters=num_clusters))
        self.q_rescaling_activ = QuantAct(num_clusters=num_clusters)

    def forward(self, x):
        cluster = x[2]
        x = self.branches(x)
        x, a_sf = self.q_rescaling_activ(x)
        return (x, asf, cluster)


class Q_ReductionAUnit(nn.Module):
    """
    Quantized InceptionV3 type Reduction-A unit.

    Parameters:
    ----------
    model : nn.Module
        The pretrained floating-point couterpart of this module with the same structure.
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    """
    def __init__(self,
                 model,
                 in_channels,
                 out_channels,
                 num_clusters=1):
        super(Q_ReductionAUnit, self).__init__()
        assert (in_channels == 288)
        assert (out_channels == 768)

        self.branches = Q_Concurrent()
        self.branches.add_module("branch1", Q_ConvSeqBranch(
            model=model.branches.branch1,
            in_channels=in_channels,
            out_channels_list=(384,),
            kernel_size_list=(3,),
            strides_list=(2,),
            padding_list=(0,),
            num_clusters=num_clusters))
        self.branches.add_module("branch2", Q_ConvSeqBranch(
            model=model.branches.branch2,
            in_channels=in_channels,
            out_channels_list=(64, 96, 96),
            kernel_size_list=(1, 3, 3),
            strides_list=(1, 1, 2),
            padding_list=(0, 1, 0),
            num_clusters=num_clusters))
        self.branches.add_module("branch3", Q_MaxPoolBranch(
            num_clusters=num_clusters))
        self.q_rescaling_activ = QuantAct(num_clusters=num_clusters)

    def forward(self, x):
        cluster = x[2]
        x = self.branches(x)
        x, a_sf = self.q_rescaling_activ(x)
        return (x, asf, cluster)


class Q_InceptionBUnit(nn.Module):
    """
    Quantized InceptionV3 type Inception-B unit.

    Parameters:
    ----------
    model : nn.Module
        The pretrained floating-point couterpart of this module with the same structure.
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    mid_channels : int
        Number of output channels in the 7x7 branches.
    """
    def __init__(self,
                 model,
                 in_channels,
                 out_channels,
                 mid_channels,
                 num_clusters=1):
        super(Q_InceptionBUnit, self).__init__()
        assert (in_channels == 768)
        assert (out_channels == 768)

        self.branches = Q_Concurrent()
        self.branches.add_module("branch1", Q_Conv1x1Branch(
            model=model.branches.branch1,
            in_channels=in_channels,
            out_channels=192,
            num_clusters=num_clusters))
        self.branches.add_module("branch2", Q_ConvSeqBranch(
            model=model.branches.branch2,
            in_channels=in_channels,
            out_channels_list=(mid_channels, mid_channels, 192),
            kernel_size_list=(1, (1, 7), (7, 1)),
            strides_list=(1, 1, 1),
            padding_list=(0, (0, 3), (3, 0)),
            num_clusters=num_clusters))
        self.branches.add_module("branch3", Q_ConvSeqBranch(
            model=model.branches.branch3,
            in_channels=in_channels,
            out_channels_list=(mid_channels, mid_channels, mid_channels, mid_channels, 192),
            kernel_size_list=(1, (7, 1), (1, 7), (7, 1), (1, 7)),
            strides_list=(1, 1, 1, 1, 1),
            padding_list=(0, (3, 0), (0, 3), (3, 0), (0, 3)),
            num_clusters=num_clusters))
        self.branches.add_module("branch4", Q_AvgPoolBranch(
            model=model.branches.branch4,
            in_channels=in_channels,
            out_channels=192,
            num_clusters=num_clusters))
        self.q_rescaling_activ = QuantAct(num_clusters=num_clusters)

    def forward(self, x):
        cluster = x[2]
        x = self.branches(x)
        x, a_sf = self.q_rescaling_activ(x)
        return (x, asf, cluster)


class Q_ReductionBUnit(nn.Module):
    """
    Quantized InceptionV3 type Reduction-B unit.

    Parameters:
    ----------
    model : nn.Module
        The pretrained floating-point couterpart of this module with the same structure.
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    """
    def __init__(self,
                 model,
                 in_channels,
                 out_channels,
                 num_clusters=1):
        super(Q_ReductionBUnit, self).__init__()
        assert (in_channels == 768)
        assert (out_channels == 1280)

        self.branches = Q_Concurrent()
        self.branches.add_module("branch1", Q_ConvSeqBranch(
            model=model.branches.branch1,
            in_channels=in_channels,
            out_channels_list=(192, 320),
            kernel_size_list=(1, 3),
            strides_list=(1, 2),
            padding_list=(0, 0),
            num_clusters=num_clusters))
        self.branches.add_module("branch2", Q_ConvSeqBranch(
            model=model.branches.branch2,
            in_channels=in_channels,
            out_channels_list=(192, 192, 192, 192),
            kernel_size_list=(1, (1, 7), (7, 1), 3),
            strides_list=(1, 1, 1, 2),
            padding_list=(0, (0, 3), (3, 0), 0),
            num_clusters=num_clusters))
        self.branches.add_module("branch3", Q_MaxPoolBranch(
            num_clusters=num_clusters))
        self.q_rescaling_activ = QuantAct(num_clusters=num_clusters)

    def forward(self, x):
        cluster = x[2]
        x = self.branches(x)
        x, a_sf = self.q_rescaling_activ(x)
        return (x, asf, cluster)


class Q_InceptionCUnit(nn.Module):
    """
    Quantized InceptionV3 type Inception-C unit.

    Parameters:
    ----------
    model : nn.Module
        The pretrained floating-point couterpart of this module with the same structure.
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    """
    def __init__(self,
                 model,
                 in_channels,
                 out_channels,
                 num_clusters=1):
        super(Q_InceptionCUnit, self).__init__()
        assert (out_channels == 2048)

        self.branches = Q_Concurrent()
        self.branches.add_module("branch1", Q_Conv1x1Branch(
            model=model.branches.branch1,
            in_channels=in_channels,
            out_channels=320,
            num_clusters=num_clusters))
        self.branches.add_module("branch2", Q_ConvSeq3x3Branch(
            model=model.branches.branch2,
            in_channels=in_channels,
            out_channels_list=(384,),
            kernel_size_list=(1,),
            strides_list=(1,),
            padding_list=(0,),
            num_clusters=num_clusters))
        self.branches.add_module("branch3", Q_ConvSeq3x3Branch(
            model=model.branches.branch3,
            in_channels=in_channels,
            out_channels_list=(448, 384),
            kernel_size_list=(1, 3),
            strides_list=(1, 1),
            padding_list=(0, 1),
            num_clusters=num_clusters))
        self.branches.add_module("branch4", Q_AvgPoolBranch(
            model=model.branches.branch4,
            in_channels=in_channels,
            out_channels=192,
            num_clusters=num_clusters))
        self.q_rescaling_activ = QuantAct(num_clusters=num_clusters)

    def forward(self, x):
        cluster = x[2]
        x = self.branches(x)
        x, a_sf = self.q_rescaling_activ(x)
        return (x, asf, cluster)


class Q_InceptInitBlock(nn.Module):
    """
    Quantized InceptionV3 specific initial block.

    Parameters:
    ----------
    model : nn.Module
        The pretrained floating-point couterpart of this module with the same structure.
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    """
    def __init__(self,
                 model,
                 in_channels,
                 out_channels,
                 num_clusters=1):
        super(Q_InceptInitBlock, self).__init__()
        assert (out_channels == 192)

        self.q_input_activ = QuantAct(num_clusters=num_clusters)
        self.q_conv1 = Q_InceptConv(
            model=model.conv1,
            in_channels=in_channels,
            out_channels=32,
            kernel_size=3,
            stride=2,
            padding=0,
            num_clusters=num_clusters)
        self.q_conv2 = Q_InceptConv(
            model=model.conv2,
            in_channels=32,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=0,
            num_clusters=num_clusters)
        self.q_conv3 = Q_InceptConv(
            model=model.conv3,
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            num_clusters=num_clusters)
        self.q_pool1 = QuantMaxPool2d(
            kernel_size=3,
            stride=2,
            padding=0)
        self.q_conv4 = Q_InceptConv(
            model=model.conv4,
            in_channels=64,
            out_channels=80,
            kernel_size=1,
            stride=1,
            padding=0,
            num_clusters=num_clusters)
        self.q_conv5 = Q_InceptConv(
            model=model.conv5,
            in_channels=80,
            out_channels=192,
            kernel_size=3,
            stride=1,
            padding=0,
            num_clusters=num_clusters)
        self.q_pool2 = QuantMaxPool2d(
            kernel_size=3,
            stride=2,
            padding=0)

    def forward(self, x):
        cluster = x[1]
        (x, a_sf) = self.q_input_activ(x, cluster=cluster)
        x = self.q_conv1((x, a_sf, cluster))
        x = self.q_conv2(x)
        x = self.q_conv3(x)
        (x, a_sf) = self.q_pool1(x)
        x = self.q_conv4((x, a_sf, cluster))
        x = self.q_conv5(x)
        (x, a_sf) = self.q_pool2(x)
        return (x, a_sf, cluster)


class Q_InceptionV3(nn.Module):
    """
    Quantized InceptionV3 model from 'Rethinking the Inception Architecture for Computer Vision,'
    https://arxiv.org/abs/1512.00567.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    init_block_channels : int
        Number of output channels for the initial unit.
    b_mid_channels : list of int
        Number of middle channels for each Inception-B unit.
    model : nn.Module
        The pretrained floating-point InceptionV3.
    dropout_rate : float, default 0.0
        Fraction of the input units to drop. Must be a number between 0 and 1.
    in_channels : int, default 3
        Number of input channels.
    in_size : tuple of two ints, default (299, 299)
        Spatial size of the expected input image.
    num_classes : int, default 1000
        Number of classification classes.
    """
    def __init__(self,
                 channels,
                 init_block_channels,
                 b_mid_channels,
                 model,
                 dropout_rate=0.5,
                 in_channels=3,
                 in_size=(299, 299),
                 num_classes=1000,
                 num_clusters=1):
        super(Q_InceptionV3, self).__init__()
        self.in_size = in_size
        self.num_classes = num_classes
        normal_units = [Q_InceptionAUnit, Q_InceptionBUnit, Q_InceptionCUnit]
        reduction_units = [Q_ReductionAUnit, Q_ReductionBUnit]

        self.features = nn.Sequential()
        self.features.add_module("q_init_block", Q_InceptInitBlock(
            model=model.features.init_block,
            in_channels=in_channels,
            out_channels=init_block_channels,
            num_clusters=num_clusters))
        in_channels = init_block_channels

        for i, channels_per_stage in enumerate(channels):
            stage = nn.Sequential()
            for j, out_channels in enumerate(channels_per_stage):
                if (j == 0) and (i != 0):
                    unit = reduction_units[i - 1]
                else:
                    unit = normal_units[i]

                model_stage = getattr(model.features, "stage{}".format(i + 1))

                if unit == Q_InceptionBUnit:
                    stage.add_module("unit{}".format(j + 1), unit(
                        model=getattr(model_stage, "unit{}".format(j + 1)),
                        in_channels=in_channels,
                        out_channels=out_channels,
                        mid_channels=b_mid_channels[j - 1],
                        num_clusters=num_clusters))
                else:
                    stage.add_module("unit{}".format(j + 1), unit(
                        model=getattr(model_stage, "unit{}".format(j + 1)),
                        in_channels=in_channels,
                        out_channels=out_channels,
                        num_clusters=num_clusters))
                in_channels = out_channels
            self.features.add_module("stage{}".format(i + 1), stage)

        self.features.add_module("q_final_pool", QuantAveragePool2d(kernel_size=8, stride=1))

        self.q_concat_activ = QuantAct(num_clusters=num_clusters)

        self.output = nn.Sequential()
        self.output.add_module("q_dropout", QuantDropout(p=dropout_rate))

        q_fc = QuantLinear()
        q_fc.is_classifier = True
        q_fc.set_param(model.output.fc)
        self.output.add_module("q_fc", q_fc)

    def _init_params(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    init.constant_(module.bias, 0)

    def forward(self, x, cluster):
        (x, a_sf) = self.features((x, cluster))
        (x, a_sf) = self.q_concat_activ((x, a_sf, cluster))
        x = x.view(x.size(0), -1)
        x = self.output((x, a_sf))
        return x


def q_inceptionv3(model=None, num_clusters=None):

    init_block_channels = 192
    channels = [[256, 288, 288],
                [768, 768, 768, 768, 768],
                [1280, 2048, 2048]]
    b_mid_channels = [128, 160, 160, 192]

    net = Q_InceptionV3(
        channels=channels,
        init_block_channels=init_block_channels,
        b_mid_channels=b_mid_channels,
        model=model,
        num_clusters=num_clusters)

    return net


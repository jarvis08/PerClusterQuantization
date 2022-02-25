# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn

# from .models import fake_quantize, calc_qparams
from .quantization_utils import fake_quantize, calc_qparams

def _quant_noise(module, p, block_size, q_max):
    """
    Wraps modules and applies quantization noise to the weights for
    subsequent quantization with Iterative Product Quantization as
    described in "Training with Quantization Noise for Extreme Model Compression"

    Args:
        - module: nn.Module
        - p: amount of Quantization Noise
        - block_size: size of the blocks for subsequent quantization with iPQ

    Remarks:
        - Module weights must have the right sizes wrt the block size
        - Only Linear, Embedding and Conv2d modules are supported for the moment
        - For more detail on how to quantize by blocks with convolutional weights,
          see "And the Bit Goes Down: Revisiting the Quantization of Neural Networks"
        - We implement the simplest form of noise here as stated in the paper
          which consists in randomly dropping blocks
    """

    # if no quantization noise, don't register hook
    if p <= 0:
        return module

    # supported modules
    assert isinstance(module, (nn.Linear, nn.Embedding, nn.Conv2d))

    # test whether module.weight has the right sizes wrt block_size
    is_conv = module.weight.ndim == 4

    # 2D matrix
    if not is_conv:
        assert (
            module.weight.size(1) % block_size == 0
        ), "Input features must be a multiple of block sizes"

    # 4D matrix
    else:
        # 1x1 convolutions
        if module.kernel_size == (1, 1):
            assert (
                module.in_channels % block_size == 0
            ), "Input channels must be a multiple of block sizes"
        # regular convolutions
        else:
            k = module.kernel_size[0] * module.kernel_size[1]
            assert k % block_size == 0, "Kernel size must be a multiple of block size"

    def _forward_pre_hook(mod, input):
        # no noise for evaluation
        s = None
        z = None
        with torch.autograd.profiler.profile(use_cuda=True) as prof:
           if mod.training:
               # P의 확률로 quantize,
               if not is_conv: # Fullty Connected
                   # #gather weight and sizes
                   # weight = mod.weight
                   # in_features = weight.size(1)
                   # out_features = weight.size(0)
                   #
                   # # split weight matrix into blocks and randomly drop selected blocks
                   # mask = torch.zeros(
                   #     in_features // block_size * out_features, device=weight.device
                   # )
                   # mask.bernoulli_(p)
                   # mask = mask.repeat_interleave(block_size, -1).view(-1, in_features)

                   ### Customize Fully Connected
                   weight = mod.weight
                   in_features = weight.size(1)
                   out_features = weight.size(0)

                   # fake quantize 하기 위해 필요 요소 구하기
                   w_min = torch.min(weight.data)
                   w_max = torch.max(weight.data)
                   s, z = calc_qparams(w_min, w_max, q_max)

                   mask = torch.zeros(
                       in_features // block_size * out_features, device=weight.device
                   )
                   mask.bernoulli_(p)
                   mask = mask.repeat_interleave(block_size, -1).view(-1, in_features).cuda()
                   unmask = torch.ones(mask.shape).cuda() - mask
                   quantized_weight = weight * mask
                   unquantized_weight = weight * unmask


                   # fake quantization 진행. >> fake vector로하기 for loop X
                   # to_shape = quantized_weight.shape
                   # quantized_weight = quantized_weight.view(-1)
                   # for i in range(len(quantized_weight)):
                   #     quantized_weight[i] = fake_quantize(quantized_weight[i], s, z)
                   # quantized_weight.view(to_shape)


               else:           # Convolution
                   # # gather weight and sizes
                   # weight = mod.weight
                   # in_channels = mod.in_channels
                   # out_channels = mod.out_channels
                   #
                   # # split weight matrix into blocks and randomly drop selected blocks
                   # if mod.kernel_size == (1, 1):
                   #     mask = torch.zeros(
                   #         int(in_channels // block_size * out_channels),
                   #         device=weight.device,
                   #     )
                   #     mask.bernoulli_(p)
                   #     mask = mask.repeat_interleave(block_size, -1).view(-1, in_channels)
                   # else:
                   #     mask = torch.zeros(
                   #         weight.size(0), weight.size(1), device=weight.device
                   #     )
                   #     mask.bernoulli_(p)
                   #     mask = (
                   #         mask.unsqueeze(2)
                   #         .unsqueeze(3)
                   #         .repeat(1, 1, mod.kernel_size[0], mod.kernel_size[1])
                   #     )

                   ### Customize Convolution
                   weight = mod.weight
                   in_channels = mod.in_channels
                   out_channels = mod.out_channels


                   w_min = torch.min(weight.data)
                   w_max = torch.max(weight.data)
                   s, z = calc_qparams(w_min, w_max, q_max)


                   if mod.kernel_size == (1, 1):
                       mask = torch.zeros( int(in_channels // block_size * out_channels), device=weight.device)
                       mask.bernoulli_(p)
                       mask = mask.repeat_interleave(block_size, -1).view(-1, in_channels)
                       # adding custom
                       mask = (
                           mask.unsqueeze(2)
                               .unsqueeze(3)
                               .repeat(1, 1, mod.kernel_size[0], mod.kernel_size[1])
                       )
                       mask.cuda()
                       quantized_weight = weight.cuda() * mask
                       unquantized_weight = weight.cuda() * (torch.ones(mask.shape).cuda() - mask)

                   else:
                       mask = torch.zeros(weight.size(0), weight.size(1), device=weight.device)
                       mask.bernoulli_(p)
                       mask = (
                           mask.unsqueeze(2)
                           .unsqueeze(3)
                           .repeat(1, 1, mod.kernel_size[0], mod.kernel_size[1])
                       )
                       mask.cuda()

                       quantized_weight = weight.cuda() * mask
                       unquantized_weight = weight.cuda() * (torch.ones(mask.shape).cuda() - mask)

               # scale weights and apply mask
               # mask = mask.to(
               #     torch.bool
               # )  # x.bool() is not currently supported in TorchScript
               # s = 1 / (1 - p)
               # mod.weight.data = s * weight.masked_fill(mask, 0)

               quantized_weight = fake_quantize(quantized_weight, s, z, q_max=q_max, use_ste=True)
               mod.weight.data = nn.Parameter(quantized_weight + unquantized_weight)
    module.register_forward_pre_hook(_forward_pre_hook)
    return module

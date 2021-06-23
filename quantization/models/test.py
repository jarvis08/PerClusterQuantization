import torch
import torch.nn as nn

import re
import numpy as np
import os
import tensorflow as tf




from quantization.layers.fused_conv import *
from quantization.layers.fused_linear import *
from quantization.quantization_utils import *


class TorchFusedTest(nn.Module):
    def __init__(self):
        super(TorchFusedTest, self).__init__()
        self.bit = 8
        self.smooth = 8
        self.conv = FusedConv2d(1, 4, kernel_size=3, stride=1, padding=2, bias=True, bit=8, smooth=0.999, bn=False, relu=True)
        self.fc = FusedLinear(3136, 10, smooth=0.999, bit=8, relu=True)

    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

model = TorchFusedTest()

tf_path = os.path.abspath("./qat_test.ckpt")
init_vars = tf.train.list_variables(tf_path)
tf_vars = []
for name, shape in init_vars:
    print("Loading TF weight {} with shape {}".format(name, shape))
    array = tf.train.load_variable(tf_path, name)
    tf_vars.append((name, array.squeeze()))

# FOr each variable in the PyTorch model
for name, array in tf_vars:
    # skip the prefix ('model/') and split the path-like variable name in a list of sub-path
    name = name[6:].split('/')

    # Initiate the pointer from the main model class
    pointer = model

    # We iterate along the scopes and move our pointer accordingly
    for m_name in name:
        # we take special care of the `h0`, `h1`... paths and split them in `h` + the number
        if re.fullmatch(r'[A-Za-z]+\d+', m_name):
            l = re.split(r'(\d+)', m_name)
        else:
            l = [m_name]

        # Convert parameters final names to the PyTorch modules equivalent names
        if l[0] == 'w' or l[0] == 'g':
            pointer = getattr(pointer, 'weight')
        elif l[0] == 'b':
            pointer = getattr(pointer, 'bias')
        elif l[0] == 'wpe' or l[0] == 'wte':
            pointer = getattr(pointer, l[0])
            pointer = getattr(pointer, 'weight')
        else:
            pointer = getattr(pointer, l[0])

        # If we had a `hXX` name, let's access the sub-module with the right number
        if len(l) >= 2:
            num = int(l[1])
            pointer = pointer[num]
    try:
        assert pointer.shape == array.shape  # Catch error if the array shapes are not identical
    except AssertionError as e:
        e.args += (pointer.shape, array.shape)
        raise

    print("Initialize PyTorch weight {}".format(name))
    pointer.data = torch.from_numpy(array)
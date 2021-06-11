import torch
import numpy as np


def load_preprocessed_cifar10_from_darknet():
    input = torch.tensor(np.fromfile("result/darknet/cifar_test_dataset.bin", dtype='float32').reshape((10000, 1, 3, 32, 32)))
    target = torch.tensor(np.fromfile("result/darknet/cifar_test_target.bin", dtype='int32').reshape((10000, 1)), dtype=torch.long)
    return input, target

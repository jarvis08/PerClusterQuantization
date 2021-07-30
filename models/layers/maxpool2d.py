from operator import itemgetter

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..quantization_utils import *


class QuantizedMaxPool2d(nn.MaxPool2d):
    def __init__(self, kernel_size, stride=1, padding=0, arg_dict=None):
        super(QuantizedMaxPool2d, self).__init__(kernel_size, stride, padding)
        self.layer_type = 'QuantizedMaxPool2d'     
        self.num_clusters, self.runtime_helper = itemgetter('cluster', 'runtime_helper')(arg_dict)

        self.kernel_size = kernel_size
        self.stride = stride
        self.maxpool = nn.MaxPool2d(self.kernel_size, self.stride, 0)

        t_init = list(range(self.num_clusters)) if self.num_clusters > 1 else 0
        self.zero_point = nn.Parameter(torch.tensor(t_init, dtype=torch.int32), requires_grad=False)

    def forward(self, x):
        if self.runtime_helper.batch_cluster is not None and self.padding > 0:
            return self.pcq(x)
        else:
            return self.general(x)

    def pcq(self, x):
        done = 0
        padded = torch.zeros((x.shape[0], x.shpae[1], x.shape[2] + self.padding * 2, x.shape[3] + self.padding * 2))
        for i in range(self.runtime_helper.batch_cluster.shape[0]):
            c = self.runtime_helper.batch_cluster[i][0].item()
            n = self.runtime_helper.batch_cluster[i][1].item()
            padded[done:done + n] = F.pad(x[done:done + n], (self.padding, self.padding, self.padding, self.padding),
                                          mode='constant', value=self.zero_point[c].item())
            done += n
        return self.maxpool(padded)

    def general(self, x):
        if self.padding > 0:
            x = F.pad(x, (self.padding, self.padding, self.padding, self.padding), mode='constant', value=self.zero_point.item())
        return self.maxpool(x)


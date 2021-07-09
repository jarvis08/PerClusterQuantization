import torch
import torch.nn as nn
import torch.nn.functional as F
from ..quantization_utils import *


class QuantizedMaxPool2d(nn.MaxPool2d):
    def __init__(self, kernel_size, stride=1, padding=0, num_clusters=1):
        super(QuantizedMaxPool2d, self).__init__(kernel_size, stride, padding)
        self.layer_type = 'QuantizedMaxPool2d'     
        self.kernel_size = kernel_size
        self.stride = stride
        self.maxpool = nn.MaxPool2d(self.kernel_size, self.stride, 0)
        t_init = list(range(num_clusters)) if num_clusters > 1 else 0
        self.zero_point = nn.Parameter(torch.tensor(t_init, dtype=torch.int32), requires_grad=False)
        self.num_clusters = num_clusters

    def forward(self, x, cluster_info):
        if cluster_info is not None and self.padding > 0:
            return self.pcq(x, cluster_info)
        else:
            return self.general(x)

    def pcq(self, x, cluster_info):
        done = 0
        padded = torch.zeros((x.shape[0], x.shpae[1], x.shape[2] + self.padding * 2, x.shape[3] + self.padding * 2))
        for i in range(cluster_info.shape[0]):
            c = cluster_info[i][0].item()
            n = cluster_info[i][1].item()
            padded[done:done + n] = F.pad(x[done:done + n], (self.padding, self.padding, self.padding, self.padding), mode='constant', value=self.zero_point[c].item())
            done += n
        return self.maxpool(padded)

    def general(self, x):
        if self.padding > 0:
            x = F.pad(x, (self.padding, self.padding, self.padding, self.padding), mode='constant', value=self.zero_point.item())
        return self.maxpool(x)

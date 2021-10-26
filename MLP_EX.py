import torch
import torch.nn as nn
from typing import Any
import numpy as np
from ctypes import *

import int_quantization

class MLP(nn.Module):
    def __init__(self, num_classes:int = 10) -> None:
        super(MLP, self).__init__()
        self.L1 = nn.Linear(2048, 2048)
        self.L2 = nn.Linear(2048, 2048)
        self.L3 = nn.Linear(2048, 2048)
        self.L4 = nn.Linear(2048, num_classes)

    def forward(self, x:torch.Tensor):
        out = self.L1(x)
        out = self.L2(out)
        out = self.L3(out)
        out = self.L4(out)
        return out

class MLP_cublas(nn.Module):
    def __init__(self, num_classes: int = 10, c=None, d=None, e=None, f=None) -> None:
        super(MLP_cublas, self).__init__()
        self.L1 = nn.Linear(2048, 2048)
        self.L2 = nn.Linear(2048, 2048)
        self.L3 = nn.Linear(2048, 2048)
        self.L4 = nn.Linear(2048, num_classes)

        self.L1.weight = nn.Parameter(torch.randint(0, 16, (2048, 2048), dtype=torch.int8), requires_grad=False)
        self.L2.weight = nn.Parameter(torch.randint(0, 16, (2048, 2048), dtype=torch.int8), requires_grad=False)
        self.L3.weight = nn.Parameter(torch.randint(0, 16, (2048, 2048), dtype=torch.int8), requires_grad=False)
        self.L4.weight = nn.Parameter(torch.randint(0, 16, (2048, 10), dtype=torch.int8), requires_grad=False)

        self.torch_L1 = nn.Linear(2048, 2048)
        self.torch_L2 = nn.Linear(2048, 2048)
        self.torch_L3 = nn.Linear(2048, 2048)
        self.torch_L4 = nn.Linear(2048, num_classes)
        self.torch_L1.weight = nn.Parameter(torch.tensor(self.L1.weight.clone().detach(), dtype=torch.float32), requires_grad=False)
        self.torch_L2.weight = nn.Parameter(torch.tensor(self.L2.weight.clone().detach(), dtype=torch.float32), requires_grad=False)
        self.torch_L3.weight = nn.Parameter(torch.tensor(self.L3.weight.clone().detach(), dtype=torch.float32), requires_grad=False)
        self.torch_L4.weight = nn.Parameter(torch.tensor(self.L4.weight.clone().detach(), dtype=torch.float32), requires_grad=False)

        self.c = c
        self.d = d
        self.e = e
        self.f = f

    def forward(self, x):
        m = x.size(0)
        k = self.L1.weight.shape[1]
        n = self.L1.weight.shape[0]
        b = self.L1.weight
        int_quantization.cublasGemm(1, 0, n, m, k, 1,
                                    b, k,
                                    x, k,
                                    0,
                                    self.c,
                                    n)
        #data = torch.tensor(x.clone().detach(), dtype=torch.float32)
        #torch_out = self.torch_L1(data)
        #print('Layer 1')
        #print('Cublas Out\n',self.c)
        #print('Torch out\n', torch_out)
        
        #print('===================================')
        #self.c = self.c.type(torch.int8)
        #print('Cublas out type cast to int8\n', self.c)
        #print('===================================')
        """
        print('Torch data\n', data)
        print('Cublas data\n', x)
        print('******************************************************')

        print('Torch weight\n', self.torch_L1.weight.data)
        #print('Cublas weight\n', self.L1.weight.data)
        print('Cublas weight\n', b)
        print('******************************************************')

        print(torch_out)
        print('==================================')
        print(self.c)
        """
        m = c.size(0)
        k = self.L2.weight.shape[1]
        n = self.L2.weight.shape[0]
        int_quantization.cublasGemm(1, 0, n, m, k, 1,
                                    b, k,
                                    x, k,
                                    1,
                                    self.d, n)

        m = c.size(0)
        k = self.L3.weight.shape[1]
        n = self.L3.weight.shape[0]
        b = self.L3.weight
        int_quantization.cublasGemm(1, 0, n, m, k, 1,
                                    b, k,
                                    x, k,
                                    1,
                                    self.e, n)

        m = c.size(0)
        k = self.L4.weight.shape[1]
        n = self.L4.weight.shape[0]
        b = self.L4.weight
        int_quantization.cublasGemm(1, 0, n, m, k, 1,
                                    b, k,
                                    x, k,
                                    1,
                                    self.f, n)
        return self.f


"""
class MLP_cublas(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super(MLP_cublas, self).__init__()
        self.L1 = nn.Linear(4, 3)
        #self.L1 = nn.Linear(32, 128)
        self.L2 = nn.Linear(128, 2048)
        self.L3 = nn.Linear(2048, 2048)
        self.L4 = nn.Linear(2048, 10)

        self.L1.weight = nn.Parameter(torch.randint(0, 5, (3, 4), dtype=torch.int8), requires_grad=False)
        self.torch_L1 = nn.Linear(4,3)
        self.torch_L1.weight = nn.Parameter(torch.tensor(self.L1.weight.clone().detach(), dtype=torch.float32), requires_grad=False)

    def forward(self, x: torch.Tensor):
        m = x.size(0)
        k = self.L1.weight.size(1)
        n = self.L1.weight.size(0)
        print('input size : {}, weight size :{}'.format((x.size(0), x.size(1)), (self.L1.weight.size(0), self.L1.weight.size(1))))
        print('m:{}, n:{}, k:{}, a shape:{}'.format(m, n, k, x.shape))
        a = x
        b = self.L1.weight
        c = torch.zeros((m, n), dtype=torch.float32)
        c = c.cuda()       # accumluator

        print('Torch Linear matmul')
        #print(torch.nn.functional.linear(a, b))
        float_data = torch.tensor(x.clone().detach(), dtype=torch.float32)
        print('Float data-{}\n'.format(float_data.shape), float_data)
        print('Torch Layer-{}\n'.format(self.torch_L1.weight.shape), self.torch_L1.weight)
        torch_output = self.torch_L1(float_data)
        #print(torch_output)


        #a = torch.transpose(a, 0, 1).contiguous()
        #b = torch.transpose(b, 0, 1).contiguous()
        #a = a.view(-1).cuda()
        #b = b.view(-1).cuda()
        #c = c.view(-1).cuda()
        print('m:{}, n:{}, k:{}, a shape:{}'.format(m, n, k, a.shape))
        print('a shape:{}, b shape:{}'.format(a.shape, b.shape))
        int_quantization.cublasGemm(1, 0, n, m, k, 1,
                                    b, k,
                                    a, k,
                                    1,
                                    c, n)
        print('Cublas Output\n', c)
        print('==========================================')
        print('Torch data\n', float_data)
        print('Cublas data\n', x)
        print('Torch weight\n', self.torch_L1.weight)
        print('Cublas weight\n', self.L1.weight)
        print('==========================================')
        print('Torch Output\n', torch_output)
        print('Cublas Output\n', c)
        exit(1)
        
        #Version1
        int_quantization.cublasGemm(1, 1, m, n, k, 1,
                                    a, k,
                                    b, n,
                                    1,
                                    c, m)
        #Origin
        int_quantization.cublasGemm(0, 0, n, m, k, 1,
                                    b, n,
                                    a, k,
                                    1,
                                    c, n)
        
        print('Cublas output accumulator\n',c)
        int_quantization.distict_gemm(x, b, c)
        int_quantization.cublasGemm(0, 1, m, n, k, 1, a, k, b, k, c, n)
        a = c, b = self.L2.weight, c = self.L2.weight.shape[1]
        int_quantization.cublasGemm(0, 1, m, n, k, 1, a, k, b, k, c, n)
        a = c, b = self.L3.weight, c = self.L3.weight.shape[1]
        int_quantization.cublasGemm(0, 1, m, n, k, 1, a, k, b, k, c, n)
        a = c, b = self.L4.weight, c = self.L4.weight.shape[1]
        int_quantization.cublasGemm(0, 1, m, n, k, 1, a, k, b, k, c, n)

        return c
"""

if __name__ == '__main__':
    # m = batch, k=inputs, n=outputs
    # a = input / b = weight / c = output / c32 = output accumulator / bias = bias
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MLP()
    
    c = torch.zeros(256, 2048, dtype=torch.float32).cuda()
    d = torch.zeros(256, 2048, dtype=torch.float32).cuda()
    e = torch.zeros(256, 2048, dtype=torch.float32).cuda()
    f = torch.zeros(256, 2048, dtype=torch.float32).cuda()
    # model_cublas = MLP_cublas()
    model_cublas = MLP_cublas(c=c, d=d, e=e, f=f)
    data = torch.randint(0, 3,(256, 2048), dtype=torch.int8).cuda()
    #data = torch.randint(0, 3,(256, 2048), dtype=torch.float).cuda()
    #data = torch.randint(0, 3,(3, 4), dtype=torch.int8).cuda()
    #print('Data type:', type(data.data[0][0].item()))
    model.to(device)
    model_cublas.to(device)

    time_arr = []
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for _ in range(50):
        #model(data)
        model_cublas(data)
        time_arr.append(0)
        time_arr.append(0)

    with torch.no_grad():
        for index in range(20):
            starter.record()
            _ = model_cublas(data)
            #_ = model(data)
            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            time_arr[index] = curr_time

    with open('./cublas_forward_inference_time.txt', 'w') as f:
        for time in time_arr:
            f.write('{}\n'.format(time))

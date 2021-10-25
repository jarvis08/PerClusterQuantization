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
        data = torch.tensor(x.clone().detach(), dtype=torch.float32)
        torch_out = self.torch_L1(data)

        print(torch_out)
        print('==================================')
        print(self.c)
        exit(1)

        m = c.size(0)
        k = self.L2.weight.shape[1]
        n = self.L2.weight.shape[0]
        b = self.L2.weight
        self.c = torch.tensor(self.c.clone().detach(), dtype=torch.uint8)
        int_quantization.cublasGemm(1, 0, n, m, k, 1,
                                    b, k,
                                    self.c, k,
                                    1,
                                    self.d, n)
        torch_out = self.torch_L2(torch_out)

        m = c.size(0)
        k = self.L3.weight.shape[1]
        n = self.L3.weight.shape[0]
        b = self.L3.weight
        self.d = torch.tensor(self.d.clone().detach(), dtype=torch.uint8)
        int_quantization.cublasGemm(1, 0, n, m, k, 1,
                                    b, k,
                                    self.d, k,
                                    1,
                                    self.e, n)
        torch_out = self.torch_L3(torch_out)

        m = c.size(0)
        k = self.L4.weight.shape[1]
        n = self.L4.weight.shape[0]
        b = self.L4.weight
        self.e = torch.tensor(self.e.clone().detach(), dtype=torch.uint8)
        int_quantization.cublasGemm(1, 0, n, m, k, 1,
                                    b, k,
                                    self.e, k,
                                    1,
                                    self.f, n)
        torch_out = self.torch_L4(torch_out)
        self.c = torch.zeros(256, 2048, dtype=torch.int).cuda()
        self.d = torch.zeros(256, 2048, dtype=torch.int).cuda()
        self.e = torch.zeros(256, 2048, dtype=torch.int).cuda()

        print(torch_out)
        return self.f


# class MLP_cublas(nn.Module):
#     def __init__(self, num_classes: int = 10) -> None:
#         super(MLP_cublas, self).__init__()
#         self.L1 = nn.Linear(3, 4)
#         # self.L1 = nn.Linear(32, 128)
#         # self.L2 = nn.Linear(128, 1024)
#         # self.L3 = nn.Linear(1024, 1024)
#         # self.L4 = nn.Linear(1024, 10)
#         # self.weight = torch.Tensor(torch.ones(10, 16))
#         self.weight = torch.randint(0, 5, (3,4), dtype=torch.float)
#
#     def forward(self, x: torch.Tensor):
#         # m = x.size(0)
#         # k = self.L1.weight.size(1)
#         # n = self.L1.weight.size(0)
#         # print('input size : {}, weight size :{}'.format((x.size(0), x.size(1)), (self.L1.weight.size(0), self.L1.weight.size(1))))
#
#         m = x.size(0)
#         k = self.weight.size(1)
#         n = self.weight.size(0)
#         exit(1)
#         a = x
#         # b = self.L1.weight
#         b = self.weight
#
#         # Version1
#         # c = torch.zeros((n, m))
#
#         c = torch.zeros((m, n))
#
#         print('Weight {}\n'.format(self.weight.shape), self.weight)
#
#         print('Torch Linear matmul')
#         print(torch.nn.functional.linear(a, b))
#
#         a = a.cuda()        # input
#         b = b.cuda()        # weight
#         c = c.cuda()        # accumluator
#         # a = torch.transpose(a, 0, 1).contiguous()
#         # b = torch.transpose(b, 0, 1).contiguous()
#         # a = a.view(-1).cuda()
#         # b = b.view(-1).cuda()
#         # c = c.view(-1).cuda()
#         print('m:{}, n:{}, k:{}, a shape:{}'.format(m, n, k, a.shape))
#
#         int_quantization.cublasGemm(1, 0, n, m, k, 1,
#                                     b, k,
#                                     a, k,
#                                     1,
#                                     c, n)
#         # Version1
#         # int_quantization.cublasGemm(1, 1, m, n, k, 1,
#         #                             a, k,
#         #                             b, n,
#         #                             1,
#         #                             c, m)
#
#         # Origin
#         # int_quantization.cublasGemm(0, 0, n, m, k, 1,
#         #                             b, n,
#         #                             a, k,
#         #                             1,
#         #                             c, n)
#         print('Cublas output accumulator\n',c)
#
#         # int_quantization.distict_gemm(x, b, c)
#         # int_quantization.cublasGemm(0, 1, m, n, k, 1, a, k, b, k, c, n)
#         #
#         # a = c, b = self.L2.weight, c = self.L2.weight.shape[1]
#         # int_quantization.cublasGemm(0, 1, m, n, k, 1, a, k, b, k, c, n)
#         #
#         # a = c, b = self.L3.weight, c = self.L3.weight.shape[1]
#         # int_quantization.cublasGemm(0, 1, m, n, k, 1, a, k, b, k, c, n)
#         #
#         # a = c, b = self.L4.weight, c = self.L4.weight.shape[1]
#         # int_quantization.cublasGemm(0, 1, m, n, k, 1, a, k, b, k, c, n)
#
#         return c

if __name__ == '__main__':
    # m = batch, k=inputs, n=outputs
    # a = input / b = weight / c = output / c32 = output accumulator / bias = bias
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MLP()
    c = torch.zeros(256, 2048, dtype=torch.int).cuda()
    d = torch.zeros(256, 2048, dtype=torch.int).cuda()
    e = torch.zeros(256, 2048, dtype=torch.int).cuda()
    f = torch.zeros(256, 2048, dtype=torch.int).cuda()
    model_cublas = MLP_cublas(c=c, d=d, e=e, f=f)
    data = torch.randint(0, 3,(256, 2048), dtype=torch.int8).cuda()
    # print('Data type:', type(data.data[0][0].item()))
    model.to(device)
    model_cublas.to(device)

    time_arr = []
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for _ in range(10):
        # model(data)
        model_cublas(data)
        time_arr.append(0)
        time_arr.append(0)

    with torch.no_grad():
        for index in range(20):
            starter.record()
            _ = model_cublas(data)
            print(_)
            exit(1)
            model.f = torch.zeros(256, 2048, dtype=torch.int).cuda()
            # _ = model(data)
            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            time_arr[index] = curr_time

    for time in time_arr:
        with open('./cublas_forward_inference_time.txt', 'a') as f:
            f.write('{}\n'.format(time))

import torch
from torch import nn


class STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx):
        pass

    @staticmethod
    def backward(ctx, grad):
        return grad, None


def pre_hook(module, grad_in, grad_out):
    print('backward 시작!!')
    print(grad_in, grad_out)
    grad_out[0].zero_()
    print(grad_in, grad_out)
#
#
# def hook(module, input, output):
#     return output + 10000  # output 값에 10000 더해서 반환
#
#
linear1 = nn.Linear(2, 2)
# linear1.register_full_backward_hook(pre_hook)
linear2 = nn.Linear(2, 2)
# linear2.register_full_backward_hook(pre_hook)

out = linear1(torch.Tensor([[1, 2], [3, 4]]))
out = linear2(out)

out.mean().backward()


# x = torch.autograd.Variable(torch.tensor([1], dtype=torch.float), requires_grad=True)
#
# y = x * 2
#
# z = y * 5
#
# z.backward()




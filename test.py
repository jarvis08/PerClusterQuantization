import torch

a = torch.rand((2, 2, 2, 2))
#a = torch.rand((2, 3, 2, 2))
print(a)
print()
print(torch.min(a, 3))

a = [3, 2, 1, 2]
torch.unsqueeze(x, 0)

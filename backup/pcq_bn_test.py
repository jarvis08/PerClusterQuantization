import torch
import torch.nn as nn

class PCQBnReLU(nn.Module):
    def __init__(self, num_features):
        super(PCQBnReLU, self).__init__()
        self.layer_type = 'PCQBnReLU'

        self.num_features = num_features

        # self.weight = nn.Parameter(torch.ones(num_features), requires_grad=True)
        # self.bias = nn.Parameter(torch.zeros(num_features), requires_grad=True)
        # self.running_mean = torch.zeros(num_features)
        # self.running_var = torch.ones(num_features)

        # Example with Cluster 4
        self.weight = nn.Parameter(torch.ones(4, num_features), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(4, num_features), requires_grad=True)
        self.running_mean = torch.zeros(4, num_features)
        self.running_var = torch.ones(4, num_features)
        self.eps = torch.tensor(1e-05)
        self.momentum = torch.tensor(0.1)

    def forward(self, x):
        bc = torch.tensor([0, 0, 0]).type(torch.LongTensor)
        # bc = torch.tensor([0, 0, 1]).type(torch.LongTensor)
        exists = torch.unique(bc)

        _means = torch.zeros(4, self.num_features)
        _vars = torch.zeros(4, self.num_features)
        _n = torch.zeros(4, self.num_features)
        for c in exists:
            indices = (bc == c).nonzero(as_tuple=True)[0]
            inputs_of_cluster = x[indices]
            _means[c] = inputs_of_cluster.mean(dim=(0, 2, 3))
            _vars[c] = inputs_of_cluster.var(dim=(0, 2, 3), unbiased=False)
            _n[c] = inputs_of_cluster.numel() / inputs_of_cluster.size(1)
        with torch.no_grad():
            # numel : total number of elements in the input tensor.
            # size  : Returns the size of the self tensor. The returned value is a subclass of tuple
            self.running_mean[exists] = self.running_mean[exists] * (1 - self.momentum) + _means[exists] * self.momentum
            self.running_var[exists] = self.running_var[exists] * (1 - self.momentum) + _vars[exists] * self.momentum * _n[exists] / (_n[exists] - 1)

        # Module has weight per cluster, but only use and update cluster 1's params
        w = torch.index_select(self.weight, 0, bc)
        b = torch.index_select(self.bias, 0, bc)
        m = torch.index_select(_means, 0, bc)
        v = torch.index_select(_vars, 0, bc)
        out = (x - m[:, :, None, None]) / (torch.sqrt(v[:, :, None, None] + self.eps))
        return out * w[:, :, None, None] + b[:, :, None, None]


class PCQBnReLU2(nn.Module):
    def __init__(self, num_features):
        super(PCQBnReLU2, self).__init__()
        self.layer_type = 'PCQBnReLU'

        self.num_features = num_features

        # self.weight = nn.Parameter(torch.ones(num_features), requires_grad=True)
        # self.bias = nn.Parameter(torch.zeros(num_features), requires_grad=True)
        # self.running_mean = torch.zeros(num_features)
        # self.running_var = torch.ones(num_features)

        # Example with Cluster 4
        self.weight = nn.Parameter(torch.ones(4, num_features), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(4, num_features), requires_grad=True)
        self.running_mean = torch.zeros(4, num_features)
        self.running_var = torch.ones(4, num_features)
        self.eps = torch.tensor(1e-05)
        self.momentum = torch.tensor(0.1)

    def forward(self, x):
        bc = torch.tensor([0, 0, 1]).type(torch.LongTensor)
        exists = torch.unique(bc)

        _means1 = x[:2].mean(dim=(0, 2, 3))
        _means2 = x[2:].mean(dim=(0, 2, 3))
        _vars1 = x[:2].var(dim=(0, 2, 3), unbiased=False)
        _vars2 = x[2:].var(dim=(0, 2, 3), unbiased=False)
        n1 = x[:2].numel() / x.size(1)
        n2 = x[2:].numel() / x.size(1)
        with torch.no_grad():
            # numel : total number of elements in the input tensor.
            # size  : Returns the size of the self tensor. The returned value is a subclass of tuple
            self.running_mean[0] = self.running_mean[0] * (1 - self.momentum) + _means1 * self.momentum
            self.running_var[0] = self.running_var[0] * (1 - self.momentum) + _vars1 * self.momentum * n1 / (n1 - 1)
            self.running_mean[1] = self.running_mean[1] * (1 - self.momentum) + _means2 * self.momentum
            self.running_var[1] = self.running_var[1] * (1 - self.momentum) + _vars2 * self.momentum * n2 / (n2 - 1)

        # Module has weight per cluster, but only use and update cluster 1's params
        # w = torch.index_select(self.weight, 0, bc)
        # b = torch.index_select(self.bias, 0, bc)
        out = (x - torch.cat([_means1.unsqueeze(0), _means1.unsqueeze(0), _means2.unsqueeze(0)])[:, :, None, None]) / (torch.sqrt(torch.cat([_vars1.unsqueeze(0), _vars1.unsqueeze(0), _vars2.unsqueeze(0)])[:, :, None, None] + self.eps))
        # return out * w[:, :, None, None] + b[:, :, None, None]
        return out * torch.cat([self.weight[0].unsqueeze(0), self.weight[0].unsqueeze(0), self.weight[1].unsqueeze(0)])[:, :, None, None] + torch.cat([self.bias[0].unsqueeze(0), self.bias[0].unsqueeze(0), self.bias[1].unsqueeze(0)])[:, :, None, None]


custom = PCQBnReLU(2)
bn = nn.BatchNorm2d(2)
# bn = PCQBnReLU2(2)
custom_optimizer = torch.optim.SGD(custom.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0)
torch_optimizer = torch.optim.SGD(bn.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0)
criterion = torch.nn.CrossEntropyLoss().cuda()

# for i in range(3):
for i in range(20):
    x = torch.rand((3, 2, 1, 1))
    y = torch.rand(3).type(torch.LongTensor).abs()


    custom_out = custom(x)
    torch_out = bn(x)
    tmp = [torch.flatten(custom_out[0]), torch.flatten(custom_out[1]), torch.flatten(custom_out[2])]
    tmp = torch.cat(tmp).reshape(3, -1).abs()
    custom_loss = criterion(tmp, y)
    custom_optimizer.zero_grad()
    custom_loss.backward()
    custom_optimizer.step()

    tmp = [torch.flatten(torch_out[0]), torch.flatten(torch_out[1]), torch.flatten(torch_out[2])]
    tmp = torch.cat(tmp).reshape(3, -1).abs()
    torch_loss = criterion(tmp, y)
    torch_optimizer.zero_grad()
    torch_loss.backward()
    torch_optimizer.step()
    print("Step: {}".format(i))
    print(">>> Custom")
    print(custom_out)
    # print(custom.running_mean[0])
    # print(custom.running_var[0])
    # print(custom.weight[0])
    # print(custom.bias[0])
    print(custom.running_mean)
    print(custom.running_var)
    print(custom.weight)
    print(custom.bias)
    print()
    print(">>> Torch")
    print(torch_out)
    # print(bn.running_mean)
    # print(bn.running_var)
    # print(bn.weight)
    # print(bn.bias)
    print(bn.running_mean)
    print(bn.running_var)
    print(bn.weight)
    print(bn.bias)
    print()
    print()


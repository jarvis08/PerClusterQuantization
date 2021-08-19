import torch

# dense layer 1 -> 2

# input = torch.rand(8,64,256,256)
#
# norm1 = torch.nn.BatchNorm2d(64)
# relu1 = torch.nn.ReLU()
# conv1 = torch.nn.Conv2d(64, 128, kernel_size=1, stride=1, bias=False)
# norm2 = torch.nn.BatchNorm2d(128)
# relu2 = torch.nn.ReLU()
# conv2 = torch.nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1, bias=False)
#
# norm = torch.nn.BatchNorm2d(32)
#
# l = [input]
# out = conv2(relu2(norm2(conv1(relu1(norm1(input))))))
# l.append(out)
# next_in = torch.cat(l,1)
#
# x = relu1(norm1(input))
# res_dense = torch.nn.ReLU()(x)
#
# tmp = relu1(norm1(input))
#
# l2 = [tmp]
# out2 = relu2(norm(conv2(relu2(norm2(conv1(tmp))))))
# l2.append(out2)
#
# res_ours = torch.cat(l2,1)
#
# print(res_dense)
# print(res_ours)




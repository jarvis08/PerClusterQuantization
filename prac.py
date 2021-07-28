import torch.nn as nn
import torch
import torch.nn.functional as F
import time

# q1 = torch.randint(0,9,(2,3,3,3))
# q2 = torch.randint(0,9,(2,3,2,2))
# sum_a1 = F.conv2d(q1, q2, None)
# sum_a2 = F.conv2d(q1, q2, None)
#
# print(q1)
# print(q2)
# print(sum_a1)
#
# start = time.time()
# for o_col in range(sum_a1.shape[3]):
#     for o_row in range(sum_a1.shape[3]):
#         col_st, col_end = o_col, o_col + 2
#         row_st, row_end = o_row, o_row + 2
#         sum_a1[:, :, o_col, o_row] = torch.sum(q1[:, :, col_st: col_end, row_st: row_end], (1, 2, 3)).mul(2)
#         sum_a2[:]
# print(sum_a1)
# print("\nmul z2\t", time.time() - start, "\n")

q1 = torch.ones(2,3,3,3)
print(q1.shape)
print(q1)
print(q1[:,:,0,0])
print(torch.sum(q1[:,0,0]))
print(q1[:,0,0].shape)
# print(torch.sum(q1[:,:,0,0], (1,2,3)))
# print(torch.sum(q1[:,:,0,0], (1, 2, 3)))

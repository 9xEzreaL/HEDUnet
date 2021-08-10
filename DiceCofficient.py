import torch
import torch.nn as nn
import numpy as np

smooth=0.0001
gt = torch.Tensor([[
        [[0, 1, 1, 0],
         [1, 0, 0, 1],
         [1, 0, 0, 1],
         [0, 1, 1, 0]],
        [[0, 0, 0, 0],
         [0, 0, 0, 0],
         [0, 0, 0, 0],
         [0, 0, 0, 0]],
        [[1, 0, 0, 1],
         [0, 1, 1, 0],
         [0, 1, 1, 0],
         [1, 0, 0, 1]]]])



pred = torch.Tensor([[
        [[0, 1, 1, 0],
         [1, 0, 0, 1],
         [1, 0, 0, 1],
         [0, 1, 1, 0]]
 ]])
print(pred.shape)

b=torch.argmax(pred,dim=1)
print("b shape:", b.shape)
print("b:/n ", b )
# bladder = pred[:, 0:1, :]
# print("bladder: ", bladder.shape)
# print("blad: ", bladder)
#
# N = pred.size(0)
# print("N: ",N)
# pred_flat = pred.view(N, -1)
# print("pred_flat: ",pred_flat.shape)
# gt_flat = gt.view(N, -1)
#
# intersection = (pred_flat * gt_flat).sum(1)
# print("intersection: ",intersection)
# unionset = pred_flat.sum(1) + gt_flat.sum(1)
# print("uni: ", unionset)
# loss = 2 * (intersection + smooth) / (unionset + smooth)
# print("loss: ",loss)

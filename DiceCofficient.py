import torch
import torch.nn as nn
import numpy as np


def accuracy(target, y_hat):
    seg_pred = torch.argmax(y_hat[:, 1:], dim=1)
    seg_acc = (seg_pred == target[:, 1]).float().mean()

    edge_pred = (y_hat[:, 0] > 0).float()
    edge_acc = (edge_pred == target[:, 0]).float().mean()

    return seg_acc , edge_acc

# dice cofficient for BCE
def BCEDiceCofficient(target , y_hat):
    smooth = 0.00001
    edge_target = target[:, 1]
    N = edge_target.size(0)
    edge_pred = (y_hat[:, 1:] > 0).float()
    edge_pred_flat = edge_pred.view(N, -1)
    edge_target_flat = edge_target.view(N, -1)

    seg_target = target[:, 0]
    n = seg_target.size(0)
    seg_pred = (y_hat[:, 0] > 0).float()
    seg_pred_flat = seg_pred.view(n, -1)
    seg_target_flat = seg_target.view(n, -1)

    edge_intersection = (edge_pred_flat * edge_target_flat).sum(1)
    edge_unionset = edge_pred_flat.sum(1) + edge_target_flat.sum(1)
    edge_acc = (2 * (edge_intersection + smooth) / (edge_unionset + smooth)).mean()

    seg_intersection = (seg_pred_flat * seg_target_flat).sum(1)
    seg_unionset = seg_pred_flat.sum(1) + seg_target_flat.sum(1)
    seg_acc = (2 * (seg_intersection + smooth) / (seg_unionset + smooth)).mean()

    return seg_acc , edge_acc

def oldCEDiceCofficient(target , y_hat):
    n = int(y_hat.shape[0]) # 2
    seg_target = target[:n, 0, ::]  # [256,224]
    edge_target = target[:n, 1, ::]
    seg_target = seg_target.view(n,-1)
    edge_target = edge_target.view(n,-1)
    seg_pred = y_hat[:n, :2, ::]  # [8,2,256,224]
    edge_pred = y_hat[:n, 2:, ::]

    softmax_func = nn.Softmax(dim=1)
    seg_pred = softmax_func(seg_pred)
    edge_pred = softmax_func(edge_pred)
    seg_pred = torch.max(seg_pred, 1)[1]
    edge_pred = torch.max(edge_pred, 1)[1]
    seg_pred = seg_pred.view(n,-1)  # (2, 256*224)
    edge_pred = edge_pred.view(n,-1)

    # seg_target = seg_target.reshape(seg_target.shape[0] * seg_target.shape[1])
    # edge_target = edge_target.reshape(edge_target.shape[0] * edge_target.shape[1])

    dice_tp_seg = (seg_pred * seg_target).sum()
    dice_div_seg = seg_pred.sum() + seg_target.sum()
    seg_acc = ((2 * dice_tp_seg) / dice_div_seg).mean()

    dice_tp_edge = (edge_pred * edge_target).sum()
    dice_div_edge = edge_pred.sum() + edge_target.sum()
    edge_acc = ((2 * dice_tp_edge) / dice_div_edge).mean()

    return seg_acc, edge_acc


# dice cofficient for CE
def CEDiceCofficient(target, y_hat):
    smooth = 0.00001
    edge_target = target[:, 1]
    n = edge_target.size(0)
# for edge
    edge_target = target[:, 1]  # [8,256,224]
    edge_pred = (y_hat[:, 3] > 0).float()
    edge_pred = edge_pred.view(n ,-1)
    edge_target = edge_target.view(n, -1)

    dice_tp_edge = (edge_pred * edge_target).sum(1)
    dice_div_edge = edge_pred.sum(1) + edge_target.sum(1)
    edge_acc = (2 * (dice_tp_edge +smooth) / (dice_div_edge + smooth)).mean()
# for seg
    seg_target = target[:, 0, ::]  # [8,256,224]
    seg_pred = (y_hat[:, 1] > 0).float()
    seg_pred = seg_pred.view(n, -1)
    seg_target = seg_target.view(n, -1)
    dice_tp_seg = (seg_pred * seg_target).sum(1)
    dice_div_seg = seg_pred.sum(1) + seg_target.sum(1)
    seg_acc = (2 * (dice_tp_seg + smooth) / (dice_div_seg +smooth)).mean()

    return seg_acc, edge_acc




import re
import torch
import torch.nn as nn
import torch.nn.functional as F

# torch.nn.CrossEntropyLoss  # target = Long, 1D
# torch.nn.BCELoss
# torch.nnBCEWithLogitsLoss (This loss combines a Sigmoid layer and the BCELoss in one single class)

def get_loss(loss_args):
    loss_type = loss_args['type']
    functional_style = re.search(r'(\w+)\((\w+)\)', loss_type)
    args = dict()
    if functional_style:
        func, arg = functional_style.groups()
        new_args = dict(loss_args)
        if func == 'Summed':
            new_args['type'] = arg
            return sum_loss(get_loss(new_args))
    if loss_type == 'BCEwLogits':
        loss_class = nn.BCEWithLogitsLoss
        if 'pos_weight' in loss_args:
            args['pos_weight'] = loss_args['pos_weight'] * torch.ones([])
    elif loss_type == 'CE':
        loss_class = nn.CrossEntropyLoss
    elif loss_type == 'FocalLoss':
        return focal_loss_with_logits
    elif loss_type == 'AutoBCE':
        return auto_weight_bce
    else:
        raise ValueError(f"No Loss of type {loss_type} known")

    return loss_class(**args)


def focal_loss_with_logits(y_hat_log, y, gamma=2):
    log0 = F.logsigmoid(-y_hat_log)
    log1 = F.logsigmoid(y_hat_log)

    gamma0 = torch.pow(torch.abs(1 - y - torch.exp(log0)), gamma)
    gamma1 = torch.pow(torch.abs(y - torch.exp(log1)), gamma)

    return torch.mean(-(1 - y) * gamma0 * log0 - y * gamma1 * log1)


def auto_weight_bce(y_hat_log, y):
    with torch.no_grad():
        beta = y.mean(dim=[2, 3], keepdims=True)
    logit_1 = F.logsigmoid(y_hat_log)
    logit_0 = F.logsigmoid(-y_hat_log)
    loss = -(1 - beta) * logit_1 * y \
           - beta * logit_0 * (1 - y)
    return loss.mean()


def sum_loss(loss_fn):
    def loss(prediction, target):
        if type(prediction) is list:
            losses = torch.stack([loss_fn(p, t) for p, t in zip(prediction, target)])
            return torch.sum(losses)
        else:
            return loss_fn(prediction, target)
    return loss


def get_sobel():
    SOBEL = nn.Conv2d(1, 2, 1, padding=1, padding_mode='replicate', bias=False)
    SOBEL.weight.requires_grad = False
    SOBEL.weight.set_(torch.Tensor([[
        [-1,  0,  1],
        [-2,  0,  2],
        [-1,  0,  1]],
       [[-1, -2, -1],
        [ 0,  0,  0],
        [ 1,  2,  1]
    ]]).reshape(2, 1, 3, 3))
    #SOBEL = SOBEL.to(dev)
    return SOBEL


def get_pyramid(mask, stack_height=1, hed=False):
    SOBEL = get_sobel()

    with torch.no_grad():
        masks = [mask]
        ## Build mip-maps
        for _ in range(stack_height):
            # Pretend we have a batch
            big_mask = masks[-1]
            small_mask = nn.functional.avg_pool2d(big_mask, 2)
            masks.append(small_mask)


        targets = []
        for mask in masks:
            sobel = torch.any(SOBEL(mask) != 0, dim=1, keepdims=True).float()
            if hed:
                targets.append(sobel)
            else:
                targets.append(torch.cat([mask, sobel], dim=1))

    return targets


class HedUnetLoss(torch.nn.Module):
    def __init__(self):
        super(HedUnetLoss, self).__init__()
        # loss function
        config = {'type': 'AutoBCE'}
        self.loss_function = get_loss(config)

    def forward(self, y_hat, y_hat_levels, target):
        target = get_pyramid(target)
        # loss
        loss_levels = []
        for y_hat_el, y in zip(y_hat_levels, target):
            loss_levels.append(self.loss_function(y_hat_el, y))

        loss_final = self.loss_function(y_hat, target[0])
        # Pyramid Losses (Deep Supervision)
        loss_deep_super = torch.sum(torch.stack(loss_levels))
        loss = loss_final + loss_deep_super
        return loss
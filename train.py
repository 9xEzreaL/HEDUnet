"""
Usecase 3 Training Script

Usage:
    train.py [options]

Options:
    -h --help          Show this screen
    --summary          Only print model summary and return (Requires the torchsummary package)
    --resume=CKPT      Resume from checkpoint
    --config=CONFIG    Specify run config to use [default: config.yml]
"""
import sys, shutil, random, yaml, os
from datetime import datetime
from pathlib import Path
from docopt import docopt
from tqdm import tqdm
from data_loading import get_dataset

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
from einops import reduce

try:
    from apex.optimizers import FusedAdam as Adam
except ModuleNotFoundError as e:
    from torch.optim import Adam

from deep_learning import get_loss, get_model, Metrics, flatui_cmap
from deep_learning.utils.data import Augment


def showexample(idx, img, target, prediction):
    m = 0.02
    gridspec_kw = dict(left=m, right=1 - m, top=1 - m, bottom=m,
                       hspace=m, wspace=m)
    fig, ax = plt.subplots(2, 3, figsize=(9, 6), gridspec_kw=gridspec_kw)
    heatmap_seg  = dict(cmap='gray', vmin=0, vmax=1)
    heatmap_edge = dict(cmap=flatui_cmap('Clouds', 'Midnight Blue'), vmin=0, vmax=1)
    # Clear all axes
    for axis in ax.flat:
        axis.imshow(np.ones([1, 1, 3]))
        axis.axis('off')

    rgb = (1. + img.cpu().numpy()) / 2.
    ax[0, 0].imshow(np.clip(rgb.transpose(1, 2, 0), 0, 1))
    ax[0, 1].imshow(target[0].cpu(), **heatmap_seg)
    ax[1, 1].imshow(target[1].cpu(), **heatmap_edge)

    seg_pred, edge_pred = torch.sigmoid(prediction)
    ax[0, 2].imshow(seg_pred.cpu(), **heatmap_seg)
    ax[1, 2].imshow(edge_pred.cpu(), **heatmap_edge)

    filename = log_dir / 'figures' / f'{idx:03d}_{epoch}.jpg'
    filename.parent.mkdir(exist_ok=True)
    plt.savefig(filename, bbox_inches='tight')
    plt.close()


def get_sobel():
    # new   ==========
    cuda = True if torch.cuda.is_available() else False
    dev = torch.device("cpu") if not cuda else torch.device("cuda")

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
    SOBEL = SOBEL.to(dev)
    return SOBEL


def get_pyramid(mask, stack_height=1, hed=False):
    SOBEL = get_sobel()

    with torch.no_grad():
        masks = [mask]
        ## Build mip-maps
        for _ in range(stack_height):
            # Pretend we have a batch
            big_mask = masks[-1]
            small_mask = F.avg_pool2d(big_mask, 2)
            masks.append(small_mask)

        targets = []
        for mask in masks:
            sobel = torch.any(SOBEL(mask) != 0, dim=1, keepdims=True).float()
            if hed:
                targets.append(sobel)
            else:
                targets.append(torch.cat([mask, sobel], dim=1))

    return targets

def norm_0_1(x0):
    x = 1 * x0
    x = x - x.min()
    return x / x.max()

def full_forward(model, img, target, metrics):
    # new   ========
    cuda = True if torch.cuda.is_available() else False
    dev = torch.device("cpu") if not cuda else torch.device("cuda")
    loss_args = {"type":"BCE"}
    loss_function = get_loss(loss_args=loss_args)

    img = img.to(dev)
    target = target.to(dev)
    y_hat, y_hat_levels = model(img)
    target = get_pyramid(target)
    loss_levels = []
    
    for y_hat_el, y in zip(y_hat_levels, target):
        loss_levels.append(loss_function(y_hat_el, y))
    # Overall Loss
    loss_final = loss_function(y_hat, target[0])
    # Pyramid Losses (Deep Supervision)
    loss_deep_super = torch.sum(torch.stack(loss_levels))
    loss = loss_final + loss_deep_super


    # dice cofficient
    smooth = 0.0001
    target = target[0]

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

    seg_intersection = (seg_pred_flat * seg_target_flat).sum(1)
    seg_unionset = seg_pred_flat.sum(1) + seg_target_flat.sum(1)
    seg_acc = (2 * (seg_intersection + smooth) / (seg_unionset + smooth)).mean()

    edge_intersection = (edge_pred_flat * edge_target_flat).sum(1)
    edge_unionset = edge_pred_flat.sum(1) + edge_target_flat.sum(1)
    edge_acc = (2 * (edge_intersection + smooth) / (edge_unionset + smooth)).mean()


    # target = target[0]
    # seg_pred = torch.argmax(y_hat[:, 1:], dim=1)
    # seg_acc = (seg_pred == target[:, 1]).float().mean()
    #
    # edge_pred = (y_hat[:, 0] > 0).float()
    # edge_acc = (edge_pred == target[:, 0]).float().mean()
    # print("target shape: ", target.shape)
    # print("target[:, 1] shape: ", target[:, 1].shape)
    # print("target[:, 1] mean: ", target[:, 1].mean())
    # print("target[:, 1] max: ", target[:, 1].max())
    # print(" seg_pred: ",  seg_pred.shape)
    # print(" seg_pred max: ", seg_pred.max())
    # print(" seg_pred min: ", seg_pred.min())
    # print(" seg_acc: ", seg_acc)
    # print("  y_hat[:, 1:]: ", y_hat[:, 1:].shape)
    # print("  y_hat[:, 1:] mean: ", y_hat[:, 1:].mean())
    # print("  y_hat[:, 1:] max: ", y_hat[:, 1:].max())
    # print("  y_hat[:, 1:] min: ", y_hat[:, 1:].min())
    # print("equal: ",seg_pred == target[:, 1])
    # print("  y_hat[:, 0] max: ", y_hat[:, 0].max())
    # print("  y_hat[:, 0] min: ", y_hat[:, 0].min())

    # print("edge_pred shape: ", edge_pred.shape)
    # print("edge_pred max: ", edge_pred.max())
    # print("edge_pred =1: ", edge_pred.sum())
    # print("edge_pred min: ", edge_pred.min())
    # #print("edge : ", edge_pred)
    # print("target[:, 0] shape: ", target[:, 0].shape)
    # print("target[:, 0] =3: ", ((target[:, 0]==3).sum())/3)
    # print("target[:, 0] =2: ", ((target[:, 0] == 2).sum())/2)
    # print("target[:, 0] =1: ", (target[:, 0] == 1).sum())
    # print("target[:, 0] =0: ", (target[:, 0] == 0).sum())
    # print("target[:, 0] max: ", target[:, 0].max())
    # print("target[:, 0] min: ", target[:, 0].min())

    metrics.step(Loss=loss, SegAcc=seg_acc, EdgeAcc=edge_acc)

    return dict(
        img=img,
        target=target,
        y_hat=y_hat,
        loss=loss,
        loss_final=loss_final,
        loss_deep_super=loss_deep_super
    )


def train(dataset):
    global epoch
    # Training step

    data_loader = DataLoader(dataset,
        batch_size=config['batch_size'],
        shuffle=True, num_workers=config['data_threads'],
        pin_memory=True
    )

    epoch += 1
    model.train(True)
    prog = tqdm(data_loader)
    for i, (img, target) in enumerate(prog): 
        for param in model.parameters():
            param.grad = None
        res = full_forward(model, img, target, metrics)
        res['loss'].backward()
        opt.step()

        if (i+1) % 1000 == 0:
            prog.set_postfix(metrics.peek())

    metrics_vals = metrics.evaluate()
    logstr = f'Epoch {epoch:02d} - Train: ' \
           + ', '.join(f'{key}: {val:.3f}' for key, val in metrics_vals.items())
    print(logstr)
    #with (log_dir / 'metrics.txt').open('a+') as f:
    #    print(logstr, file=f)

    # Save model Checkpoint
    # torch.save(model.state_dict(), checkpoints / f'{epoch:02d}.pt')
    torch.save(model, checkpoints / f'{epoch:02d}.pt')

@torch.no_grad()
def val(dataset):
    # Validation step
    data_loader = DataLoader(dataset,
        batch_size=config['batch_size'],
        shuffle=False, num_workers=config['data_threads'],
        pin_memory=True
    )

    model.train(False)

    idx = 0
    for img, target in tqdm(data_loader):
        B = img.shape[0]
        res = full_forward(model, img, target, metrics)

        if 0:  # SHOWING EXAMPLES HAVE PROBLEMS
            for i in range(B):
                if idx+i in config['visualization_tiles']:
                    showexample(idx+i, img[i], res['target'][i], res['y_hat'][i])
            idx += B

    metrics_vals = metrics.evaluate()
    logstr = f'Epoch {epoch:02d} - Val: ' \
           + ', '.join(f'{key}: {val:.3f}' for key, val in metrics_vals.items())
    print(logstr)
    with (log_dir / 'metrics.txt').open('a+') as f:
        print(logstr, file=f)


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True

    # configurations
    cli_args = docopt(__doc__, version="Usecase 2 Training Script 1.0")
    config_file = Path(cli_args['--config'])
    config = yaml.load(config_file.open(), Loader=yaml.SafeLoader)
    print(config)

    # models
    modelclass = get_model(config['model'])
    model = modelclass(**config['model_args'])

    # resume training from checkpoints
    if cli_args['--resume']:
        config['resume'] = cli_args['--resume']

    if 'resume' in config and config['resume']:
        checkpoint = Path(config['resume'])
        if not checkpoint.exists():
            raise ValueError(f"There is no Checkpoint at {config['resume']} to resume from!")
        if checkpoint.is_dir():
            # Load last checkpoint in run dir
            ckpt_nums = [int(ckpt.stem) for ckpt in checkpoint.glob('checkpoints/*.pt')]
            last_ckpt = max(ckpt_nums)
            config['resume'] = checkpoint / 'checkpoints' / f'{last_ckpt:02d}.pt'
        print(f"Resuming training from checkpoint {config['resume']}")
        model.load_state_dict(torch.load(config['resume']))

    # ude gpu (cuda)
    cuda = True if torch.cuda.is_available() else False
    dev = torch.device("cpu") if not cuda else torch.device("cuda")
    print(f'Training on {dev} device')
    model = model.to(dev)

    epoch = 0
    metrics = Metrics()

    lr = config['learning_rate']
    opt = Adam(model.parameters(), lr)

    stack_height = 1 if 'stack_height' not in config['model_args'] else \
            config['model_args']['stack_height']

    log_dir = Path('logs') / datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_dir.mkdir(exist_ok=False, parents=True)

    shutil.copy(config_file, log_dir / 'config.yml')

    checkpoints = log_dir / 'checkpoints'
    checkpoints.mkdir()

    # DATASET
    if config['dataset'] == 'original':
        trnval = get_dataset('train')
        indices = list(range(len(trnval)))
        val_filter = lambda x: x % 10 == 0

        val_indices = list(filter(val_filter, indices))
        trn_indices = list(filter(lambda x: not val_filter(x), indices))

        trn_dataset = Augment(Subset(trnval, trn_indices))
        val_dataset = Subset(trnval, val_indices)
    elif config['dataset'] == 'knee':
        from loaders.loader_imorphics import LoaderImorphics

        args_d = {'mask_name': 'bone_resize_B_crop_00',
                  'data_path': os.getenv("HOME") + '/Dataset/OAI_DESS_segmentation/',
                  'mask_used': [['femur', 'tibia']] ,# [1], [2, 3]],  # ,
                  'scale': 0.5,
                  'interval': 1,
                  'thickness': 0,
                  'method': 'automatic'}

        # datasets
        trn_dataset = LoaderImorphics(args_d, subjects_list=list(range(10, 71)))
        print(len(trn_dataset))
        val_dataset = LoaderImorphics(args_d, subjects_list=list(range(1, 10)) + list(range(71, 89)))
        print(len(val_dataset))

    # loss function
    loss_function = get_loss(config['loss_args'])
    if type(loss_function) is torch.nn.Module:
        loss_function = loss_function.to(dev)

    for _ in range(config['epochs']):
        train(trn_dataset)
        val(val_dataset)

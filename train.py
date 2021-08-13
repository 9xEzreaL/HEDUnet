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
from DiceCofficient import BCEDiceCofficient, accuracy, CEDiceCofficient
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

cli_args = docopt(__doc__, version="Usecase 2 Training Script 1.0")
config_file = Path(cli_args['--config'])
config = yaml.load(config_file.open(), Loader=yaml.SafeLoader)


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
    loss_args = {"type":"CE"}
    loss_function = get_loss(loss_args=loss_args)

    img = img.to(dev)
    target = target.to(dev)
    y_hat, y_hat_levels = model(img)
    target = get_pyramid(target)
    loss_levels = []
    loss_levels_seg = []
    loss_levels_edge = []

    # loss
    if config['loss_args']['type'] == 'CE':
        y_hat_levels = [x.view(x.shape[0], x.shape[1], x.shape[2]*x.shape[3]) for x in y_hat_levels]
        n = y_hat.size(0)
        y_hat = y_hat.view(n, 4, -1)
        Target = [x.view(x.shape[0], x.shape[1], x.shape[2]*x.shape[3])
                      .type(torch.cuda.LongTensor) for x in target]

        for y_hat_el, y in zip(y_hat_levels, Target):
            print('#')
            print(y_hat_el.shape)
            print(y.shape)
            y_hat_levels_seg = y_hat_el[:, :2, :]
            y_hat_levels_edge = y_hat_el[:, 2:, :]
            target_seg = (y[:, :1, :]).squeeze(axis = 1)  #
            target_edge = (y[:, 1:, :]).squeeze(axis = 1)  #
            loss_levels_seg.append(loss_function(y_hat_levels_seg, target_seg))
            loss_levels_edge.append(loss_function(y_hat_levels_edge, target_edge))

        loss_deep_super_seg = torch.sum(torch.stack(loss_levels_seg))
        loss_deep_super_edge = torch.sum(torch.stack(loss_levels_edge))
        loss_deep_super = loss_deep_super_seg + loss_deep_super_edge
        y_hat_seg = y_hat[:, :2, :]
        y_hat_edge = y_hat[:, 2:, :]

        target_seg = (Target[0][:, :1, :]).squeeze(axis = 1)  #
        target_edge = (Target[0][:, 1:, :]).squeeze(axis = 1) #
        loss_final_seg = loss_function(y_hat_seg, target_seg)
        loss_final_edge = loss_function(y_hat_edge, target_edge)
        loss_final = loss_final_seg + loss_final_edge

    else:
        # Overall Loss
        loss_final = loss_function(y_hat, target[0])
        # Pyramid Losses (Deep Supervision)
        for y_hat_el, y in zip(y_hat_levels, target):
            loss_levels.append(loss_function(y_hat_el, y))
        loss_deep_super = torch.sum(torch.stack(loss_levels))

    loss = loss_final + loss_deep_super

    # dice cofficient
    if config['loss_args']['type'] == 'CE':
        seg_acc , edge_acc = CEDiceCofficient(target= target, y_hat= y_hat)
    else:
        seg_acc, edge_acc = BCEDiceCofficient(target= target, y_hat= y_hat)
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

        # creating visualization of results
        if i == 1:
            targets = torch.cat([norm_0_1(res['target'][0, i, ::]) for i in range(res['target'].shape[1])], 1).detach().cpu()
            y_hats = torch.cat([norm_0_1(res['y_hat'][0, i, ::]) for i in range(res['target'].shape[1])], 1).detach().cpu()
            y_hats_p = torch.cat([(res['y_hat'][0, i, ::] > 0) for i in range(res['target'].shape[1])], 1).detach().type(torch.FloatTensor).cpu()
            all = torch.cat([targets, y_hats, y_hats_p], 0)
            imagesc(all, show=False, save='sample_visualization.png')

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
    basic_output = 2
    if config['loss_args']['type'] == 'CE':
        output_channels = basic_output * 2
    else:
        output_channels = basic_output
    model = modelclass(**config['model_args'], output_channels=output_channels)

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

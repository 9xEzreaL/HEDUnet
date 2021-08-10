import os, glob, torch, time
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


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
    #SOBEL = SOBEL.cuda()
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

def to_8bit(x):
    if type(x) == torch.Tensor:
        x = (x / x.max() * 255).numpy().astype(np.uint8)
    else:
        x = (x / x.max() * 255).astype(np.uint8)

    if len(x.shape) == 2:
        x = np.concatenate([np.expand_dims(x, 2)]*3, 2)
    return x


def imagesc(x, show=True, save=None):
    if isinstance(x, list):
        x = [to_8bit(y) for y in x]
        x = np.concatenate(x, 1)
        x = Image.fromarray(x)
    else:
        x = x - x.min()
        x = Image.fromarray(to_8bit(x))
    if show:
        x.show()
    if save:
        x.save(save)


def append_dict(x):
    return [j for i in x for j in i]


def resize_and_crop(pilimg, scale):
    w = pilimg.size[0]
    h = pilimg.size[1]
    newW = int(w * scale)
    newH = int(h * scale)
    pilimg = pilimg.resize((newW, newH))

    dx = 32
    w0 = pilimg.size[0]//dx * dx
    h0 = pilimg.size[1]//dx * dx
    pilimg = pilimg.crop((0, 0, w0, h0))

    return pilimg


class imorphics_masks():
    def __init__(self, adapt=None):
        self.adapt = adapt

    def load_masks(self, id, dir, fmt, scale):
        if self.adapt is not None:
            id = str(self.adapt.index((int(id.split('/')[1]), id.split('/')[0])) + 1) + '_' + str(int(id.split('/')[2]))
        raw_masks = []
        for d in dir:
            temp = []
            for m in d:
                x = Image.open(os.path.join(m, id + fmt))  # PIL
                x = resize_and_crop(x, scale=scale)  # PIL
                x = np.array(x)  # np.int32
                temp.append(x.astype(np.float32))  # np.float32

            raw_masks.append(temp)

        out = np.expand_dims(self.assemble_masks(raw_masks), 0)
        return out

    def assemble_masks(self, raw_masks):
        converted_masks = np.zeros(raw_masks[0][0].shape, np.long)
        for i in range(len(raw_masks)):
            for j in range(len(raw_masks[i])):
                converted_masks[raw_masks[i][j] == 1] = i + 1

        return converted_masks


class LoaderImorphics(Dataset):
    def __init__(self, args_d, subjects_list):
        #  Folder of the images
        dir_img = os.path.join(args_d['data_path'], args_d['mask_name'], 'original/')
        #  Folder of the masks
        dir_mask = [[os.path.join(args_d['data_path'], args_d['mask_name'],
                                  'train_masks/' + str(y) + '/') for y in x] for x in
                    args_d['mask_used']]

        self.dir_img = dir_img
        self.fmt_img = glob.glob(self.dir_img+'/*')[0].split('.')[-1]
        self.dir_mask = dir_mask

        # Assemble the masks from the folders
        self.masks = imorphics_masks(adapt=None)

        # Picking  subjects
        ids = sorted(glob.glob(self.dir_mask[0][0] + '*'))  # scan the first mask foldr
        ids = [x.split(self.dir_mask[0][0])[-1].split('.')[0] for x in ids]  # get the ids
        self.ids = [x for x in ids if int(x.split('_')[0]) in subjects_list]  # subject name belongs to subjects_list

        # Rescale the images
        self.scale = args_d['scale']

    def load_imgs(self, id):
        x = Image.open(self.dir_img + id + '.' + self.fmt_img)
        x = resize_and_crop(x, self.scale)
        x = np.expand_dims(np.array(x), 0)  # to numpy
        return x

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        id = self.ids[idx]

        # load image
        img = self.load_imgs(id)

        # load mask
        mask = self.masks.load_masks(id, self.dir_mask, '.png', scale=self.scale)
        mask = mask.astype(np.float32)

        # normalization
        img = torch.from_numpy(img)
        img = img.type(torch.float32)
        img = img / img.max()

        img = torch.cat([1*img, 1*img, 1*img], 0)

        return img, mask


if __name__ == '__main__':
    args_d = {'mask_name': 'bone_resize_B_crop_00',
              'data_path': os.getenv("HOME") + '/Dataset/OAI_DESS_segmentation/',
              'mask_used': [['femur', 'tibia'], [1], [2, 3]],  # ,
              'scale': 0.5,
              'interval': 1,
              'thickness': 0,
              'method': 'automatic'}

    def imorphics_split():
        train_00 = list(range(10, 71))
        eval_00 = list(range(1, 10)) + list(range(71, 89))
        train_01 = list(range(10+88, 71+88))
        eval_01 = list(range(1+88, 10+88)) + list(range(71+88, 89+88))
        return train_00, eval_00, train_01, eval_01

    train_00, eval_00, train_01, eval_01 = imorphics_split()

    # datasets
    train_set = LoaderImorphics(args_d, subjects_list=train_00)
    img, target = train_set.__getitem__(100)

    target = torch.from_numpy(target == 1).type(torch.float32)
    (img, target) = (x.unsqueeze(0) for x in (img, target))

    # data -> model -> loss
    # model
    from deep_learning.models import HEDUNet
    model = HEDUNet(input_channels=3)

    # looking for forward
    y_hat, y_hat_levels = model(img)
    print('y_hat size:')
    print(y_hat.shape)
    print('y_hat_levels size:')
    for i in y_hat_levels:
        print(i.shape)

    # loss
    from deep_learning.metrics_hedunet import get_loss
    loss_args = {'type': 'BCE'}
    loss_function = get_loss(loss_args=loss_args)

    # output -> loss
    target = get_pyramid(target)

    loss_levels = []
    for y_hat_el, y in zip(y_hat_levels, target):
        loss_levels.append(loss_function(y_hat_el, y))
    # Overall Loss
    loss_final = loss_function(y_hat, target[0])
    # Pyramid Losses (Deep Supervision)
    loss_deep_super = torch.sum(torch.stack(loss_levels))
    loss = loss_final + loss_deep_super
    print('loss :')
    print(loss)

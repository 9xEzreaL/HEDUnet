from torch.utils.data import DataLoader
from deep_learning import get_loss, get_model, Metrics, flatui_cmap

from loaders.loader_imorphics import LoaderImorphics
from train import full_forward, showexample
from PIL import Image
from torchvision import transforms
from docopt import docopt
import os
import torch


def norm_0_1(x0):
    x = 1 * x0
    x = x - x.min()
    return x / x.max()

def validation(model, metrics, batch_size, dataset, destination):
    # Validation step
    data_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             shuffle=False, num_workers=4,
                             pin_memory=True
                             )

    model.train(False)

    for idx, (img, target) in enumerate(data_loader):

        res = full_forward(model, img, target, metrics)
        # print(res['target'].shape[1])   # 2
        if idx == 10:
            COL = 2
            ROW = 3
            UNIT_HEIGHT_SIZE = 256
            UNIT_WIDTH_SIZE = 224
            # create new picture
            pic = Image.new(
                'RGB', (UNIT_WIDTH_SIZE * COL, UNIT_HEIGHT_SIZE * ROW))


            # B = batch_size
            B = img.shape[0]
            for j in range(res['target'].shape[1]):
                x = img[0]
                # print("j: ",j)
                y = res['target'][0, j, ::]
                z = norm_0_1(res['y_hat'][0, j, ::])


                # from tensor to img
                unloader = transforms.ToPILImage()
                x = x.cpu().clone()
                x = unloader(x)
                y = y.cpu().clone()
                y = unloader(y)
                z = z.cpu().clone()
                z = unloader(z)

                pic.paste(x,
                          (0 + UNIT_WIDTH_SIZE * j, 0 + UNIT_HEIGHT_SIZE * 0))
                pic.paste(y,
                          (0 + UNIT_WIDTH_SIZE * j, 0 + UNIT_HEIGHT_SIZE * 1))
                pic.paste(z,
                          (0 + UNIT_WIDTH_SIZE * j, 0 + UNIT_HEIGHT_SIZE * 2))
            name = i.replace(".pt", "")
            pic.save(destination + name + '.jpg')





if __name__ == "__main__":

    args_d = {'mask_name': 'bone_resize_B_crop_00',
              'data_path': os.getenv("HOME") + '/Dataset/OAI_DESS_segmentation/',
              'mask_used': [['femur', 'tibia'], [1], [2, 3]],
              'scale': 0.5,
              'interval': 1,
              'thickness': 0,
              'method': 'automatic'}
    val_dataset = LoaderImorphics(
        args_d, subjects_list=list(range(1, 10)) + list(range(71, 89)))
    print(len(val_dataset))

    # ude gpu (cuda)
    cuda = True if torch.cuda.is_available() else False
    dev = torch.device("cpu") if not cuda else torch.device("cuda")

    # loss
    loss_args = {"type": "BCE"}
    loss_function = get_loss(loss_args=loss_args)

    pathlist = os.listdir(
        '/home/ghc/PycharmProjects/HED-UNet-zhi/logs/2021-08-08_22-55-19/checkpoints')
    pathlist = sorted(pathlist)
    for i in pathlist:
        path = os.path.join(
            '/home/ghc/PycharmProjects/HED-UNet-zhi/logs/2021-08-08_22-55-19/checkpoints', i)
        model = torch.load(path)
        metrics = Metrics()
        validation(model,
                   metrics,
                   batch_size=8,
                   dataset=val_dataset,
                   destination='/home/ghc/PycharmProjects/HED-UNet-zhi/logs/2021-08-08_22-55-19/figures/')


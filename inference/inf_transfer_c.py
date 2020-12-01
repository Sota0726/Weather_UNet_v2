import argparse
import os
import sys

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import shutil
from glob import glob
from torchvision.utils import save_image

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, default='3')
parser.add_argument('--image_root', type=str,
                    default='/mnt/fs2/2018/matsuzaki/dataset_fromnitta/Image/')
parser.add_argument('--pkl_path', type=str,
                    default='/mnt/fs2/2019/Takamuro/m2_research/i2w/sepalated_data.pkl')
parser.add_argument('--output_dir', '-o', type=str,
                    default='/mnt/fs2/2019/Takamuro/m2_research/weather_transfer/results/eval_class_transfer/'
                    'cUNet_w-c-res101-0317_img-i2w_train-D1T1_aug_supervised_shuffle_adam-b1-09_wloss_CrossEnt_e0035_s132000/out')
parser.add_argument('--cp_path', type=str,
                    default='/mnt/fs2/2019/Takamuro/m2_research/weather_transfer/cp/transfer/'
                    # 'cUNet_w-c-res101-0317_img-flicker-200k_aug_shuffle_adam-b1-09_wloss-CrossEnt/cUNet_w-c-res101-0317_img-flicker-200k_aug_shuffle_adam-b1-09_wloss-CrossEnt_e0025_s324000.pt')
                    'cUNet_w-c-res101-0317_img-i2w_train-D1T1_adam-b1-00_aug_sampler_supervised_wloss-CrossEnt/cUNet_w-c-res101-0317_img-i2w_train-D1T1_aug_supervised_shuffle_adam-b1-09_wloss_CrossEnt_e0035_s132000.pt')
parser.add_argument('--classifer_path', type=str,
                    default='/mnt/fs2/2019/Takamuro/m2_research/weather_transfer/cp/classifier/i2w-classifier-res101-train-2020317/better_resnet101_epoch15_step59312.pt')
parser.add_argument('--input_size', type=int, default=224)
parser.add_argument('--batch_size', type=int, default=5)
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--num_classes', type=int, default=5)
parser.add_argument('--image_only', action='store_true')
args = parser.parse_args()

os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset

sys.path.append(os.getcwd())
from dataset import ClassImageLoader, ImageLoader
from cunet import Conditional_UNet
from ops import make_table_img


if __name__ == '__main__':

    transform = transforms.Compose([
        transforms.Resize((args.input_size,)*2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    if args.image_only:
        sep_data = glob(os.path.join(args.image_root, '*.png'))
        print('loaded {} data'.format(len(sep_data)))

        dataset = ImageLoader(paths=sep_data, transform=transform, inf=True)
    else:
        s_li = ['sunny', 'cloudy', 'rain', 'snow', 'foggy']
        os.makedirs(args.output_dir, exist_ok=True)
        sep_data = pd.read_pickle(args.pkl_path)
        sep_data = sep_data['test']
        print('loaded {} data'.format(len(sep_data)))

        dataset = ClassImageLoader(paths=sep_data, transform=transform, inf=True)

    loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            drop_last=True
            )
    random_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            drop_last=True
            )

    # load model
    transfer = Conditional_UNet(num_classes=args.num_classes)
    sd = torch.load(args.cp_path)
    transfer.load_state_dict(sd['inference'])

    # classifer = torch.load(args.classifer_path)
    # classifer.eval()

    # if args.gpu > 0:
    transfer.cuda()
    # classifer.cuda()

    bs = args.batch_size

    labels = torch.as_tensor(np.arange(args.num_classes, dtype=np.int64))
    onehot = torch.eye(args.num_classes)[labels].to('cuda')

    cls_li = []
    vec_li = []
    out_li = []

    for k, (data, rnd) in tqdm(enumerate(zip(loader, random_loader)), total=len(sep_data)//bs):
        batch = data[0].to('cuda')
        ori_label = data[1]
        r_batch = rnd[0].to('cuda')
        path = data[2]
        # [save_image(img, os.path.join(args.output_dir, _.split('/')[-1]), normalize=True) for _, img in zip(path, batch)]
        for i in range(bs):
            with torch.no_grad():
                ref_labels_expand = torch.cat([onehot[i]] * bs).view(-1, args.num_classes)
                out = transfer(batch, ref_labels_expand)
                out_li.append(out)
                [save_image(output, os.path.join(args.output_dir,
                 '{}_{}_{}'.format(s_li[ori_label[j]], path[j].split('/')[-1].split('.')[0], s_li[torch.argmax(onehot[i]).to('cpu')]) + '.jpg'), normalize=True)
                 for j, output in enumerate(out)]
        # res = make_table_img(batch, r_batch, out_li)
        # save_image(res, os.path.join(args.output_dir, 'summary_results_{}.jpg'.format(str(k))), normalize=True)
        out_li = []

    # for i, (data, rnd) in tqdm(enumerate(zip(loader, random_loader)), total=len(sep_data)//bs):
    #     batch = data[0].to('cuda')
    #     r_batch = rnd[0].to('cuda')
    #     c_batch = rnd[1].to('cuda')
    #     r_cls = c_batch
    #     c_batch = F.one_hot(c_batch, args.num_classes).float()
    #     # r_cls = torch.argmax(classifer(r_batch).detach(), 1)
    #     out = transfer(batch, c_batch)

    #     # for check output
    #     # add "return image, target, self.paths[idx]" to __getitem__ of ClassImageLoader
    #     path = data[2]
    #     # for _ in path:
    #     #     shutil.copy(_, args.output_dir)

    #     [shutil.copy(_, args.output_dir) for _ in path]
    #     # _ = [shutil.copy(_, os.path.join(args.output_dir, 'out')) for _ in path]
    #     [save_image(output, os.path.join(args.output_dir, '{}_'.format(path[j].split('/')[-1].split('.')[0]) + s_li[r_cls[j]] + '.jpg'), normalize=True)
    #      for j, output in enumerate(out)]
    #     # [save_image(out[(r_cls == j)], os.path.join(args.output_dir, 'out', s_li[j]+'_{}.png'.format(i)), normalize=True) for j in range(5) if len((r_cls == j).nonzero()) != 0]
    #     # if i>20: exit()

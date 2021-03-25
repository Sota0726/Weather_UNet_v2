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
parser.add_argument('--image_root', type=str, default='/mnt/HDD8T/takamuro/dataset/img_align_celeba')
parser.add_argument('--pkl_path', type=str, default='/mnt/fs2/2019/Takamuro/db/CelebA/Anno/list_attr_celeba_for_attr_inf.pkl')
parser.add_argument('--cp_path', type=str,
                    default='/mnt/fs2/2019/Takamuro/m2_research//weather_transferV2/cp/transfer/'
                    # 'cUNet_w-c-res101-0317_img-flicker-200k_aug_shuffle_adam-b1-09_wloss-CrossEnt/cUNet_w-c-res101-0317_img-flicker-200k_aug_shuffle_adam-b1-09_wloss-CrossEnt_e0025_s324000.pt')
                    'celeba/CelebA_1220_cUNet_clelbA_cls-mobileV2-1217e15-TDR1_wl-x2_D-SNDisc_loss_BCE_b1-0.5_b2-0.9_GDratio-8_lr-1.5*0.0001_bs-24_ne-150/cUNet_cls_e0143_s848000.pt')
parser.add_argument('--classifier_path', type=str,
                    default='/mnt/fs2/2019/Takamuro/m2_research/weather_transferV2/cp/classifier_celeba/1211_CelebA_cls_mobilenetV2_train50780/resnet101_epoch80_step32076.pt'
                    )
parser.add_argument('--input_size', type=int, default=256)
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--num_workers', type=int, default=8)
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
from dataset import CelebALoader
from cunet import Conditional_UNet
from ops import make_table_img


if __name__ == '__main__':
    transform = nn.Sequential(
        transforms.CenterCrop((178, 178)),
        transforms.Resize((args.input_size,) * 2),
        transforms.ConvertImageDtype(torch.float32),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    )

    df = pd.read_pickle(args.pkl_path)
    df_ = df[df['mode'] == 'test']
    print('{} data loaded'.format(len(df_)))
    dataset = CelebALoader(args.image_root, df_, transform=transform, inf=True)

    bs = args.batch_size
    num_classes = dataset.num_classes
    cols = dataset.cols

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=bs,
        num_workers=args.num_workers,
        drop_last=True
    )
    random_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=bs,
        num_workers=args.num_workers,
        drop_last=True
    )

    # load model
    transfer = Conditional_UNet(num_classes=num_classes)
    sd = torch.load(args.cp_path)
    transfer.load_state_dict(sd['inference'])
    transfer.eval()
    transfer.to('cuda')
    # classifer = torch.load(args.classifer_path)
    # classifer.eval()

    # if args.gpu > 0:
    # classifer.cuda()

    labels = torch.as_tensor(np.arange(num_classes, dtype=np.int64))
    onehot = torch.eye(num_classes)[labels].to('cuda')
    out_li = []
    attr_li = [4, 5, 8, 9, 15, 17, 23, 26, 31]

    cp_name = args.cp_path.split('/')[-1].split('.')[0]
    save_path = './temp_celeba'
    #save_path = os.path.join('/mnt/fs2/2019/Takamuro/m2_research/weather_transferV2/results/eval_transfer', 'CelebA',
    #                         args.cp_path.split('/')[-2],
    #                         cp_name.split('_')[-2] + cp_name.split('_')[-1], 'attr_inf')
    print(save_path)
    print('If you have done to confirm save_path, please push enter')
    input()
    os.makedirs(save_path, exist_ok=True)
    for k, (data, rnd) in tqdm(enumerate(zip(loader, random_loader)), total=len(df_)//bs):
        batch = data[0].to('cuda')
        batch = dataset.transform(batch)
        label = data[1]
        path = data[2]

        r_label = rnd[1]
        r_path = rnd[2]
        blank = torch.zeros_like(batch[0]).unsqueeze(0)
        # for i in range(bs):
        # for i in range(num_classes):
        for i in attr_li:
            with torch.no_grad():
                label_ = label.clone()
                label_[:, i] = torch.ones(bs)
                # ref_labels_expand = torch.cat([torch.maximum(label[:, i], torch.ones(bs))] * bs).view(-1, num_classes).to('cuda')
                out = transfer(batch, label_.to('cuda'))
                out_li.append(out)
                # [save_image(output, os.path.join(save_path,
                #  '{}_{}'.format(path[j].split('/')[-1].split('.')[0], cols[i]) + '.jpg'), normalize=True, scale_each=True)
                #  for j, output in enumerate(out)]
        # io_im_l = [torch.cat([batch] + out_li[i: i+10], dim=3).to('cpu') for i in np.arange(0, num_classes, 10)]
        io_im = torch.cat([batch] + out_li, dim=3).to('cpu')
        # io_im = torch.cat(io_im_l, dim=2)
        [save_image(_, fp=os.path.join(save_path, path[i]), normalize=True, scale_each=True) for i, _ in enumerate(io_im)]
        out_li = []

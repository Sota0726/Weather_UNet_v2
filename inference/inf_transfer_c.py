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
parser.add_argument('--cp_path', type=str,
                    default='/mnt/fs2/2019/Takamuro/m2_research/weather_transferV2/cp/transfer/cls/1203_Flickr_cUNet_w-c_res101-1122e15_data-WoPerson_sky-10_L-05_SNdisc_sampler-True_loss_lamda-c1-w1-CE_b1-0.5_b2-0.9_GDratio-8_amp-True_MGpu-False_lr-1.5*0.0001_bs-24_ne-150/cUNet_cls_e0056_s600000.pt')
parser.add_argument('--classifer_path', type=str,
                    default='/mnt/fs2/2019/Takamuro/m2_research/weather_transferV2/cp/classifier/cls_res101_1122_NotPreTrain/resnet101_epoch15_step59312.pt')
parser.add_argument('--input_size', type=int, default=224)
parser.add_argument('--batch_size', type=int, default=5)
parser.add_argument('--num_workers', type=int, default=8)
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
    bs = args.batch_size
    s_li = ['sunny', 'cloudy', 'rain', 'snow', 'foggy']
    num_classes = len(s_li)

    transform = transforms.Compose([
        transforms.Resize((args.input_size,) * 2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    if args.image_only:
        sep_data = glob(os.path.join(args.image_root, '*.png'))
        print('loaded {} data'.format(len(sep_data)))

        dataset = ImageLoader(paths=sep_data, transform=transform, inf=True)
    else:
        # os.makedirs(args.output_dir, exist_ok=True)
        sep_data = pd.read_pickle(args.pkl_path)
        sep_data = sep_data['test']
        print('loaded {} data'.format(len(sep_data)))

        dataset = ClassImageLoader(paths=sep_data, transform=transform, inf=True)

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

    # classifer = torch.load(args.classifer_path)
    # classifer.eval()

    # if args.gpu > 0:
    transfer.cuda()
    # classifer.cuda()

    labels = torch.as_tensor(np.arange(num_classes, dtype=np.int64))
    onehot = torch.eye(num_classes)[labels].to('cuda')

    cls_li = []
    vec_li = []
    out_li = []

    cp_name = args.cp_path.split('/')[-1].split('.')[0]
    save_path = './temp'
    # save_path = os.path.join('/mnt/fs2/2019/Takamuro/m2_research/weather_transferV2/results/eval_transfer', 'cls',
    #                          args.cp_path.split('/')[-2],
    #                          cp_name.split('_')[-2] + cp_name.split('_')[-1], 'out_img')
    print(save_path)
    print('If you have done to confirm save_path, please push enter')
    input()
    os.makedirs(save_path, exist_ok=True)
    for k, (data, rnd) in tqdm(enumerate(zip(loader, random_loader)), total=len(sep_data)//bs):
        batch = data[0].to('cuda')
        # batch = dataset.transform(batch)
        ori_label = data[1]
        # r_batch = rnd[0].to('cuda')
        path = data[2]
        # [save_image(img, os.path.join(args.output_dir, _.split('/')[-1]), normalize=True) for _, img in zip(path, batch)]
        for i in range(bs):
            with torch.no_grad():
                ref_labels_expand = torch.cat([onehot[i]] * bs).view(-1, num_classes)
                out = transfer(batch, ref_labels_expand)
                out_li.append(out)
                [save_image(output, os.path.join(save_path,
                 '{}_before-{}_after-{}'.format(path[j].split('/')[-1].split('.')[0], s_li[ori_label[j]], s_li[torch.argmax(onehot[i]).to('cpu')]) + '.jpg'), normalize=True, scale_each=True)
                 for j, output in enumerate(out)]
        # res = make_table_img(batch, r_batch, out_li)
        # save_image(res, os.path.join(args.output_dir, 'summary_results_{}.jpg'.format(str(k))), normalize=True)
        out_li = []

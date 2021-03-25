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
parser.add_argument('--gpu', type=str, default='2')
parser.add_argument('--image_root', type=str,
                    default='/mnt/HDD8T/takamuro/dataset/photos_usa_224_2016-2017')
parser.add_argument('--s_pkl_path', type=str,
                    default='/mnt/fs2/2019/Takamuro/m2_research/flicker_data/wwo/2016_17/lambda_0/outdoor_all_dbdate_wwo_weather_2016_17_delnoise_WoPerson_sky-10_L-05_50testImgs.pkl')
parser.add_argument('--r_pkl_path', type=str)
parser.add_argument('--cp_path', type=str,
                    default='/mnt/fs2/2019/Takamuro/m2_research/weather_transferV2/cp/transfer/est/1204_cUNet_w-e_res101-1203L05e15_SNDisc_sampler-False_GDratio1-8_adam-b10.5-b20.9_lr-0.0001_bs-24_ne-150/cUNet_est_e0134_s1432000.pt')
parser.add_argument('--estimator_path', type=str,
                    default='/mnt/fs2/2019/Takamuro/m2_research/weather_transferV2/cp/estimator/'
                    'est_res101-1203_sampler_pre_WoPerson_sky-10_L-05/est_resnet101_15_step62240.pt')
parser.add_argument('--input_size', type=int, default=224)
parser.add_argument('--batch_size', type=int, default=50)
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--image_only', action='store_true')
# args = parser.parse_args(['--pkl_path', '/mnt/fs2/2019/Takamuro/m2_research/flicker_data/wwo/2016_17/lambda_0/for_inf_10imgs.pkl'])
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
from dataset import ImageLoader, FlickrDataLoader
from cunet import Conditional_UNet
from sampler import ImbalancedDatasetSampler
from ops import make_table_img


if __name__ == '__main__':
    transform = nn.Sequential(
        # transforms.Resize((args.input_size,) * 2),
        transforms.ConvertImageDtype(torch.float32),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    )
    s_bs = args.batch_size
    r_bs = args.batch_size

    if args.image_only:
        sep_data = glob(os.path.join(args.image_root, '*.png'))
        print('loaded {} data'.format(len(sep_data)))

        dataset = ImageLoader(paths=sep_data, transform=transform, inf=True)
    else:
        cols = ['tempC', 'uvIndex', 'visibility', 'windspeedKmph', 'cloudcover', 'humidity', 'pressure', 'DewPointC']

        # get norm paramator
        temp = pd.read_pickle('/mnt/fs2/2019/Takamuro/m2_research/flicker_data/wwo/2016_17/lambda_0/outdoor_all_dbdate_wwo_weather_2016_17_delnoise_WoPerson_sky-10_L-05.pkl')
        df_ = temp.loc[:, cols].fillna(0)
        df_mean = df_.mean()
        df_std = df_.std()

        df = pd.read_pickle(args.s_pkl_path)
        df_.loc[:, cols] = (df_.loc[:, cols].fillna(0) - df_mean) / df_std
        df.loc[:, cols] = (df.loc[:, cols].fillna(0) - df_mean) / df_std
        df_sep = df[df['mode'] == 'test']
        print('loaded {} signals data'.format(len(df_sep)))
        del df, df_, temp
        dataset = FlickrDataLoader(args.image_root, df_sep, cols, bs=args.batch_size, transform=transform, inf=True)

    loader = torch.utils.data.DataLoader(
        dataset,
        # sampler=ImbalancedDatasetSampler(dataset),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=True,
        # shuffle=True
    )
    random_loader = torch.utils.data.DataLoader(
        dataset,
        # sampler=ImbalancedDatasetSampler(dataset),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=True,
        # shuffle=True
    )

    if args.r_pkl_path:
        del random_loader
        r_df = pd.read_pickle(args.r_pkl_path)
        print(r_df)
        r_df.loc[:, cols] = (r_df.loc[:, cols].fillna(0) - df_mean) / df_std
        r_dataset = FlickrDataLoader(args.image_root, r_df, cols, transform=transform, inf=True)
        random_loader = torch.utils.data.DataLoader(
            r_dataset,
            # sampler=ImbalancedDatasetSampler(dataset),
            batch_size=10,
            num_workers=args.num_workers,
            drop_last=True,
        )
        r_bs = 10

    # load model
    transfer = Conditional_UNet(len(cols))
    sd = torch.load(args.cp_path)
    transfer.load_state_dict(sd['inference'])
    transfer.eval()

    estimator = torch.load(args.estimator_path)
    estimator.eval()

    # if args.gpu > 0:
    transfer.cuda()
    estimator.cuda()

    out_li = []
    save_path = './temp_transfer_e'
    #save_path = os.path.join('/mnt/fs2/2019/Takamuro/m2_research/weather_transferV2/results/eval_transfer', 'seq',
    #                         args.cp_path.split('/')[-2],
    #                         args.cp_path.split('/')[-1].split('.pt')[0], '{}x{}'.format(len(df_sep), len(df_sep)))
    os.makedirs(save_path, exist_ok=True)
    for k, (data, rnd) in tqdm(enumerate(zip(loader, random_loader)), total=len(df_sep)//s_bs):
        batch = data[0].to('cuda')
        batch = dataset.transform(batch)
        sig = data[1].to('cuda')
        b_photos = data[2]

        r_batch = rnd[0].to('cuda')
        r_batch = dataset.transform(r_batch)
        r_sig = rnd[1].to('cuda')
        r_photos = rnd[2]

        blank = torch.zeros_like(batch[0]).unsqueeze(0)
        ref_imgs = torch.cat([blank] + list(torch.split(r_batch, 1)), dim=3).to('cpu')
        # save_image(ref_imgs, fp=os.path.join(save_path, 'ref.jpg'), normalize=True, scale_each=True, nrow=1)
        out_l = []
        for i in tqdm(range(r_bs)):
            with torch.no_grad():
                ref_labels_expand = torch.cat([r_sig[i]] * s_bs).view(-1, len(cols))
                out = transfer(batch, ref_labels_expand)
                out_l.append(out)
                [save_image(output, os.path.join(save_path,
                 '{}-{}_r-{}.jpg'.format('gt', b_photos[j], r_photos[i])), normalize=True, scale_each=True)
                 for j, output in enumerate(out)]

        io_im = torch.cat([batch] + out_l, dim=3).to('cpu')
        [save_image(_, fp=os.path.join(save_path, 'row_{}.jpg'.format(i)), normalize=True, scale_each=True) for i, _ in enumerate(io_im)]
        tab_im = torch.cat([ref_imgs, io_im], dim=0)
        save_image(tab_im, fp=os.path.join(save_path, '{}.jpg'.format(k)), normalize=True, scale_each=True, nrow=1)

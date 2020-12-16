import argparse
import os
import sys

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import shutil
from glob import glob
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--image_root', type=str,
                    default='/mnt/HDD8T/takamuro/dataset/photos_usa_224_2016-2017')
parser.add_argument('--pkl_path', type=str,
                    default='/mnt/fs2/2019/Takamuro/m2_research/flicker_data/wwo/2016_17/lambda_0/outdoor_all_dbdate_wwo_weather_2016_17_delnoise_WoPerson_sky-10_L-05.pkl')
parser.add_argument('--cp_path', type=str,
                    default='/mnt/fs2/2019/Takamuro/m2_research/weather_transferV2/cp/transfer/est/1204_cUNet_w-e_res101-1203L05e15_SNDisc_sampler-False_GDratio1-8_adam-b10.5-b20.9_lr-0.0001_bs-24_ne-150/cUNet_est_e0134_s1432000.pt')
parser.add_argument('--input_size', type=int, default=224)
parser.add_argument('--batch_size', type=int, default=5)
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--csv_root', type=str, default='/mnt/fs2/2019/Takamuro/db/wwo_weather_data/usa/2016_2017')
parser.add_argument('--city', type=str, default='Abram')
parser.add_argument('--date', nargs=4, required=True)
# args = parser.parse_args()
args = parser.parse_args(args=['--date', '2016', '2', '21', '10'])

os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
import torchvision.transforms as transforms
from torchvision.utils import save_image

sys.path.append(os.getcwd())
from dataset import FlickrDataLoader, SensorLoader
from cunet import Conditional_UNet


def make_vid(save_path, p_name, start, end):
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    start = start.replace(':', '-')
    start = start.replace('-', '')
    start = start[:-8]
    end = end.replace(':', '-')
    end = end.replace('-', '')
    end = end[:-8]
    vid_name = os.path.join(save_path, p_name, '{}_{}.mp4'.format(start, end))
    video = cv2.VideoWriter(vid_name, fourcc, 8, (args.input_size, )*2)
    img_paths = glob(os.path.join(save_path, p_name, '*.jpg'))
    for img_path in img_paths:
        img = cv2.imread(img_path)
        video.write(img)
    video.release()
    return


if __name__ == '__main__':

    transform = nn.Sequential(
        # transforms.Resize((args.input_size,) * 2),
        transforms.ConvertImageDtype(torch.float32),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    )

    cols = ['tempC', 'uvIndex', 'visibility', 'windspeedKmph', 'cloudcover', 'humidity', 'pressure', 'DewPointC']

    # get norm paramator
    temp = pd.read_pickle('/mnt/fs2/2019/Takamuro/m2_research/flicker_data/wwo/2016_17/lambda_0/outdoor_all_dbdate_wwo_weather_2016_17_delnoise_WoPerson_sky-10_L-05.pkl')
    df_ = temp.loc[:, cols].fillna(0)
    df_mean = df_.mean()
    df_std = df_.std()

    df = pd.read_pickle(args.pkl_path)
    df_.loc[:, cols] = (df_.loc[:, cols].fillna(0) - df_mean) / df_std
    df.loc[:, cols] = (df.loc[:, cols].fillna(0) - df_mean) / df_std
    df_sep = df[df['mode'] == 'test']

    del df, df_, temp
    im_dataset = FlickrDataLoader(args.image_root, df_sep, cols, transform=transform, inf=True)
    sig_dateset = SensorLoader(args.csv_root, date=args.date, city=args.city, cols=cols, df_std=df_std, df_mean=df_mean)

    im_loader = torch.utils.data.DataLoader(
        im_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=True
    )

    sig_loader = torch.utils.data.DataLoader(
        sig_dateset,
        batch_size=24,
        num_workers=args.num_workers,
        drop_last=True
    )

    # load model
    transfer = Conditional_UNet(len(cols))
    sd = torch.load(args.cp_path)
    transfer.load_state_dict(sd['inference'])

    # if args.gpu > 0:
    transfer.cuda()

    bs = args.batch_size
    out_li = []

    save_path = os.path.join('/mnt/fs2/2019/Takamuro/m2_research/weather_transferV2/results/eval_transfer', 'est',
                             args.cp_path.split('/')[-2].split('_GDratio')[0],
                             args.cp_path.split('/')[-1].split('.pt')[0], 'video')
    os.makedirs(save_path, exist_ok=True)
    for i, data in tqdm(enumerate(im_loader), total=len(df_sep)//bs):
        batch = data[0].to('cuda')
        p_name = data[2]
        batch = im_dataset.transform(batch)

        for j in tqdm(range(bs)):
            out_l = []
            date_l = []
            for sig_data in (sig_loader):
                sig = sig_data[0]
                sig = sig.to('cuda')
                date = sig_data[1]
                with torch.no_grad():
                    batch_ = torch.unsqueeze(batch[j], dim=0)
                    batch_expand = torch.cat([batch_] * 24, dim=0)

                    out = transfer(batch_expand, sig)
                    out_l.append(out)
                    date_l.extend(date)
            outs = torch.cat(out_l, dim=0)
            os.makedirs(os.path.join(save_path, p_name[j]), exist_ok=True)
            [save_image(out, fp=os.path.join(save_path, p_name[j], args.city + '_' + date.replace(':', '-') + '.jpg'), normalize=True)
                for out, date in zip(outs, date_l)]
            make_vid(save_path, p_name[j], date_l[0], date_l[-1])

import argparse
import os
import shutil
import sys
from glob import glob

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--image_root', type=str,
                    default='/mnt/HDD8T/takamuro/dataset/photos_usa_224_2016-2017')
parser.add_argument('--pkl_path', type=str,
                    default='/mnt/fs2/2019/Takamuro/m2_research/flicker_data/wwo/2016_17/lambda_0/outdoor_all_dbdate_wwo_weather_2016_17_delnoise_WoPerson_sky-10_L-05.pkl')
parser.add_argument('--cp_path', type=str,
                    default='/mnt/fs2/2019/Takamuro/m2_research/weather_transferV2/cp/transfer/est/1204_cUNet_w-e_res101-1203L05e15_SNDisc_sampler-False_GDratio1-8_adam-b10.5-b20.9_lr-0.0001_bs-24_ne-150/cUNet_est_e0134_s1432000.pt')
parser.add_argument('--estimator_path', type=str,
                    default='/mnt/fs2/2019/Takamuro/m2_research/weather_transferV2/cp/estimator/'
                    '1209_est_res101_val_WoPerson_ss-10_L05/est_resnet101_10_step42790.pt')
parser.add_argument('--input_size', type=int, default=224)
parser.add_argument('--batch_size', type=int, default=5)
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--csv_root', type=str, default='/mnt/HDD8T/takamuro/dataset/wwo/2016_2017')
parser.add_argument('--city', type=str, default='Abram')
parser.add_argument('--date', nargs=4, required=True)
parser.add_argument('-g', '--generator', type=str, default='cUNet')
args = parser.parse_args()
# args = parser.parse_args(args=['--date', '2016', '2', '21', '10'])

os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
import torchvision.transforms as transforms
from torchvision.utils import save_image

sys.path.append(os.getcwd())
from cunet import Conditional_UNet, Conditional_UNet_V2
from dataset import FlickrDataLoader, SensorLoader


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
    df_max = df_.max()
    df_min = df_.min()
    del temp

    df = pd.read_pickle(args.pkl_path)
    df_.loc[:, cols] = (df_.loc[:, cols].fillna(0) - df_mean) / df_std
    df.loc[:, cols] = (df.loc[:, cols].fillna(0) - df_mean) / df_std
    df_sep = df[df['mode'] == 'test']

    del df, df_
    im_dataset = FlickrDataLoader(args.image_root, df_sep, cols, bs=args.batch_size, transform=transform, inf=True)
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
    if args.generator == 'cUNet':
        transfer = Conditional_UNet(len(cols))
    elif args.generator == 'cUNetV2':
        transfer = Conditional_UNet_V2(len(cols))
    else:
        print('{} is invalid generator'.format(args.generator))
        exit()

    sd = torch.load(args.cp_path)
    transfer.load_state_dict(sd['inference'])

    estimator = torch.load(args.estimator_path)
    estimator.eval()

    # if args.gpu > 0:
    transfer.cuda()
    estimator.cuda()

    bs = args.batch_size
    y_true_l = []
    y_pred_l = []
    for i, data in tqdm(enumerate(im_loader), total=len(df_sep)//bs):
        batch = data[0].to('cuda')
        batch = im_dataset.transform(batch)
        sig = data[1].to('cuda')
        for j in tqdm(range(bs)):

            for sig_data in (sig_loader):
                sig = sig_data[0].to('cuda')
                date = sig_data[1]
                with torch.no_grad():
                    batch_ = torch.unsqueeze(batch[j], dim=0)
                    batch_expand = torch.cat([batch_] * 24, dim=0)
                    out = transfer(batch_expand, sig)

                    out_sig = estimator(out)
                y_true_l.append(sig.cpu())
                y_pred_l.append(out_sig.cpu())
    y_true = torch.cat(y_true_l, dim=0).numpy()
    y_pred = torch.cat(y_pred_l, dim=0).numpy()

    y_true_n = (y_true * df_std.values + df_mean.values) / df_max.values
    y_pred_n = (y_pred * df_std.values + df_mean.values) / df_max.values
    mae = mean_absolute_error(y_true_n, y_pred_n)
    mse = mean_squared_error(y_true_n, y_pred_n)

    print("mae score = {}".format(mae))
    print("mse score = {}".format(mse))

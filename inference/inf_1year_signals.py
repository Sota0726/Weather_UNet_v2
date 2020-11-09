import argparse
import os
import sys

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from datetime import datetime
from torchvision.utils import save_image

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, default='3')
parser.add_argument('--image_root', type=str,
                    default="/mnt/HDD8T/takamuro/dataset/photos_usa_2016/")
parser.add_argument('--pkl_path', type=str,
                    default='/mnt/fs2/2019/Takamuro/m2_research/flicker_data/from_nitta/param03/sepalated_data_wo-outlier.pkl')
parser.add_argument('--output_dir', '-o', type=str,
                    default='/mnt/fs2/2019/Takamuro/m2_research/weather_transfer/results/eval_est_transfer/'
                    'cUNet_w-e_res101-0408_train-D1T1_adam_b1-00_aug_wloss-mse_train200k-test500/e23_322k')
parser.add_argument('--cp_path', type=str,
                    default='/mnt/fs2/2019/Takamuro/m2_research/weather_transfer/cp/transfer/'
                    'cUNet_w-e_res101-0408_train-D1T1_adam_b1-00_aug_wloss-mse_train200k-test500/cUNet_w-e_res101-0408_train-D1T1_adam_b1-00_aug_wloss-mse_train200k-test500_e0023_s322000.pt')
parser.add_argument('--classifer_path', type=str,
                    default='/mnt/fs2/2019/Takamuro/m2_research/weather_transfer/cp/estimator/'
                            'est_res101_flicker-p03th01-WoOutlier_sep-train_aug_pre_loss-mse-reduction-none-grad-all-1/est_resnet101_20_step22680.pt')
parser.add_argument('--photo_id', type=str, default='23787626109')
parser.add_argument('--city_name', type=str)
parser.add_argument('--input_size', type=int, default=224)
parser.add_argument('--batch_size', type=int, default=12)
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
from dataset import OneYearWeatherSignals, FlickrDataLoader
from cunet import Conditional_UNet
from sampler import ImbalancedDatasetSampler
from ops import make_table_img


if __name__ == '__main__':

    transform = transforms.Compose([
        transforms.Resize((args.input_size,)*2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    cols = ['clouds', 'temp', 'humidity', 'pressure', 'windspeed']

    df = pd.read_pickle(args.pkl_path)

    temp = pd.read_pickle(args.pkl_path)
    df_ = temp.loc[:, cols].fillna(0)
    df_mean = df_.mean()
    df_std = df_.std()

    df.loc[:, cols] = (df.loc[:, cols].fillna(0) - df_mean) / df_std

    del temp

    oneyear_dataset = OneYearWeatherSignals(args.image_root, df, cols, args.photo_id, transform, args.city_name)

    signal_loader = torch.utils.data.DataLoader(
            oneyear_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            drop_last=True
            )

    # load model
    transfer = Conditional_UNet(len(cols))
    sd = torch.load(args.cp_path)
    transfer.load_state_dict(sd['inference'])

    classifer = torch.load(args.classifer_path)
    classifer.eval()

    # if args.gpu > 0:
    transfer.cuda()
    classifer.cuda()

    bs = args.batch_size
    # out_li = []

    os.makedirs(args.output_dir, exist_ok=True)
    for k, data in tqdm(enumerate(signal_loader)):
        batch = data[0].to('cuda')
        sig = data[1].to('cuda')
        s_time = data[2]

        with torch.no_grad():
            out = transfer(batch, sig)
            [save_image(output, os.path.join(args.output_dir,
            '{}_{}.{}'.format(args.photo_id, str(datetime.utcfromtimestamp(s_time[j])).replace(' ', '-').replace(':', '-'), 'jpg')), normalize=True)
            for j, output in enumerate(out)]

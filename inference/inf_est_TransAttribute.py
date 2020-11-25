import argparse
import os
import sys

import pandas as pd
import numpy as np
from tqdm import tqdm
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='/mnt/HDD8T/takamuro/dataset/photos_usa_2016-2017')
parser.add_argument('--save_path', type=str, default='/mnt/fs2/2019/Takamuro/m2_research/flicker_data/wwo/2016_17/PhotoTime_correspondance_check/18-uvind-1_TA-est')
parser.add_argument('--pkl_path', type=str,
                    default='/mnt/fs2/2019/Takamuro/m2_research/flicker_data/wwo/2016_17/equal_con-cnn-mlp/outdoor_all_dbdate_wwo_weather_selected_ent_owner_2016_17_delnoise_addpred_equal_con-cnn-mlp.pkl')
parser.add_argument('--estimator_path', type=str, default='./cp/estimator_Transient_Attributes/est_res101_TA_day-night-sunset-dawn/est_resnet101_15_step2112.pt')
parser.add_argument('--gpu', type=str, default='1')
parser.add_argument('--input_size', type=int, default=224)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--wd', type=float, default=1e-5)
parser.add_argument('--batch_size', '-bs', type=int, default=51)
parser.add_argument('--num_workers', type=int, default=8)
args = parser.parse_args()
# args = parser.parse_args(['--pre_trained'])

os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

import torch
import torch.nn as nn
import torchvision as tv
import torchvision.transforms as transforms
import torchvision.models as models

sys.path.append(os.getcwd())
from dataset import ImageLoader


if __name__ == '__main__':
    os.makedirs(os.path.join(args.save_path, 'positive'), exist_ok=True)
    os.makedirs(os.path.join(args.save_path, 'negative'), exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((args.input_size,)*2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    cols = ['daylight', 'night', 'sunrisesunset', 'dawndusk']

    df = pd.read_pickle(args.pkl_path)
    # df['orig_date_h'] = df['orig_date']
    # temp = df['orig_date_h'].str.split(':', expand=True)
    # temp_ = temp[0].str.split('T', expand=True)
    # df.orig_date_h = temp_[1].astype(int)
    # del temp, temp_
    # df = df[(df.orig_date_h == 18) & (df.uvIndex == 1)]
    # df = df[(df['mode'] == 'train') | (df['mode'] == 'val') | (df['mode'] == 'test')]

    for col in cols:
        df['pred_{}'.format(col)] = -99

    print('{} data were loaded'.format(len(df)))

    photo_list = df['photo'].to_list()
    photo_list = [os.path.join(args.root_path, p + '.jpg') for p in photo_list]

    dataset = ImageLoader(photo_list, transform)

    loader = torch.utils.data.DataLoader(
            dataset,
            shuffle=True,
            drop_last=True,
            batch_size=args.batch_size,
            num_workers=args.num_workers)

    num_classes = len(cols)
    bs = args.batch_size

    estimator = torch.load(args.estimator_path)
    estimator.eval()
    estimator.to('cuda')

    for i, data in tqdm(enumerate(loader), total=len(df) // bs):
        inputs, paths = data
        inputs = inputs.to('cuda')

        output = estimator(inputs).detach()

        for j in range(bs):
            _ = output[j]
            path = paths[j].split('/')[-1].split('.')[0]
            ind_df = df[df.photo == path].index[0]
            for k, col in enumerate(cols):
                df.loc[ind_df, 'pred_{}'.format(col)] = _[k].item()
            # if _[0].item() > 0.6 or _[1].item() > 0.6:
            #     file_name = paths[j].split('/')[-1]
            #     shutil.copyfile(paths[j], os.path.join(args.save_path, 'positive', file_name))
            # else:
            #     shutil.copyfile(paths[j], os.path.join(args.save_path, 'negative', file_name))

    df.to_pickle(args.pkl_path.split('.')[0] + '_add_est-TransAttribute.pkl')
    print('done')

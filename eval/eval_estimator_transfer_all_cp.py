import argparse
import pickle
import os
import sys
from glob import glob

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import trange, tqdm
from torchvision.utils import save_image
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--pkl_path', type=str,
                    default='/mnt/fs2/2019/Takamuro/m2_research/flicker_data/wwo/2016_17/lambda_0/outdoor_all_dbdate_wwo_weather_2016_17_delnoise_WoPerson_sky-10_L-05_50testImgs.pkl')
parser.add_argument('--image_root', type=str, default='/mnt/HDD8T/takamuro/dataset/photos_usa_224_2016-2017')
parser.add_argument('--cp_dir', type=str,
                    default='cp/transfer/1204_cUNet_w-e_res101-1203L05e15_SNDisc_sampler-False_GDratio1-8_adam-b10.5-b20.9_lr-0.0001_bs-24_ne-150')
parser.add_argument('--estimator_path', type=str,
                    default='/mnt/fs2/2019/Takamuro/m2_research/weather_transferV2/cp/estimator/'
                            '1209_est_res101_val_WoPerson_ss-10_L05/est_resnet101_15_step62240.pt')
parser.add_argument('--input_size', type=int, default=224)
parser.add_argument('--batch_size', type=int, default=20)
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
from dataset import FlickrDataLoader
from cunet import Conditional_UNet

if __name__ == '__main__':

    # os.makedirs(os.path.join(save_path, 'out'), exist_ok=True)
    cols = ['tempC', 'uvIndex', 'visibility', 'windspeedKmph', 'cloudcover', 'humidity', 'pressure', 'DewPointC']

    transform = nn.Sequential(
            # transforms.Resize((args.input_size,) * 2),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            )

    temp = pd.read_pickle('/mnt/fs2/2019/Takamuro/m2_research/flicker_data/wwo/2016_17/lambda_0/outdoor_all_dbdate_wwo_weather_2016_17_delnoise_WoPerson_sky-10_L-05.pkl')
    df_ = temp.loc[:, cols].fillna(0)
    df_mean = df_.mean()
    df_std = df_.std()
    df_max = df_.max()
    df_min = df_.min()

    df = pd.read_pickle(args.pkl_path)
    df.loc[:, cols] = (df.loc[:, cols].fillna(0) - df_mean) / df_std

    df_sep = df[df['mode'] == 'test']

    del df, df_, temp
    print('loaded {} data'.format(len(df_sep)))

    dataset = FlickrDataLoader(args.image_root, df_sep, cols, transform=transform)

    loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers
            )
    random_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            )
    estimator = torch.load(args.estimator_path)
    estimator.eval()
    estimator.to('cuda')

    bs = args.batch_size
    cp_paths = sorted(glob(os.path.join(args.cp_dir, '*.pt')), reverse=True)
    for cp_path in tqdm(cp_paths):
        save_path = os.path.join('/mnt/fs2/2019/Takamuro/m2_research/weather_transferV2/results/eval_transfer/est',
                                 cp_path.split('/')[-2], 'sig_pred_hist')
        os.makedirs(save_path, exist_ok=True)
        cp_name = cp_path.split('/')[-1].split('.pt')[0]
        # load model
        transfer = Conditional_UNet(num_classes=len(cols))
        sd = torch.load(cp_path)
        transfer.load_state_dict(sd['inference'])
        transfer.to('cuda')

        pred_li = []
        sig_li = []
        for i, (data, rnd) in enumerate(zip(loader, random_loader)):
            batch = data[0].to('cuda')
            batch = dataset.transform(batch)
            b_sig = data[1].to('cuda')
            # r_batch  = rnd[0].to('cuda')
            r_sig = rnd[1].to('cuda')

            for j in range(bs):
                with torch.no_grad():
                    ref_sig_expand = torch.cat([r_sig[j]] * bs).view(-1, len(cols))
                    out = transfer(batch, ref_sig_expand)
                    fake_c_out = estimator(out)

                    sig_li.extend(ref_sig_expand.to('cpu').clone().numpy())
                    pred_li.extend(fake_c_out.to('cpu').clone().numpy())

        pred_li = np.array(pred_li).T
        sig_li = np.array(sig_li).T
        l1 = np.abs(sig_li - pred_li)
        ave_l1 = np.mean(l1, axis=1)
        std_l1 = np.std(l1, axis=1)

        pred_li_ = np.array([pred * std + mean for pred, std, mean in zip(pred_li, df_std, df_mean)])
        sig_li_ = np.array([sig * std + mean for sig, std, mean in zip(sig_li, df_std, df_mean)])

        l1_ = np.abs(sig_li_ - pred_li_)
        ave_l1_ = np.mean(l1_, axis=1)
        std_l1_ = np.std(l1_, axis=1)

        fig = plt.figure(figsize=(60, 30))
        for i, data in enumerate(zip(cols, sig_li_, pred_li_), 1):
            col = data[0]
            sig = data[1]
            pred = data[2]
            ax = fig.add_subplot(3, 4, i)

            sig_min = df_min[i - 1]
            sig_max = df_max[i - 1]
            h = ax.hist2d(sig, pred, bins=(10, 50), range=[[sig_min, sig_max], [sig_min, sig_max]], cmap='magma')

            ax.set_xlabel('gt_{}'.format(col))
            ax.set_ylabel('pred_{}'.format(col))
            fig.colorbar(h[3], ax=ax)
        ax = fig.add_subplot(3, 4, len(cols)+1)
        ax.bar(np.arange(len(cols)), ave_l1_, yerr=std_l1_, tick_label=cols, align='center')
        plt.xticks(rotation=90)
        fig.savefig(os.path.join(save_path, '{}.png'.format(cp_name)))
        # fig.savefig('temp.png')
        plt.clf()
        plt.close()

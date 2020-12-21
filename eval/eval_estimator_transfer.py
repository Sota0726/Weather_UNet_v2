import argparse
import pickle
import os
import sys


import numpy as np
import pandas as pd
from PIL import Image
from tqdm import trange, tqdm
from torchvision.utils import save_image
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str)
parser.add_argument('--pkl_path', type=str,
                    default='/mnt/fs2/2019/okada/from_nitta/parm_0.3/50test_high-consis_10images-each-con2.pkl')
parser.add_argument('--image_root', type=str, default='/mnt/HDD8T/takamuro/dataset/photos_usa_224_2016-2017')
parser.add_argument('--cp_path', type=str,
                    default='/mnt/fs2/2019/Takamuro/m2_research/weather_transfer/cp/transfer/'
                    'cUNet_w-e_res101-0408_train-D1T1_adam_b1-00_aug_wloss-mse_train200k-test500/cUNet_w-e_res101-0408_train-D1T1_adam_b1-00_aug_wloss-mse_train200k-test500_e0023_s322000.pt')
parser.add_argument('--estimator_path', type=str,
                    default='/mnt/fs2/2019/Takamuro/m2_research/weather_transferV2/cp/estimator/'
                            '1209_est_res101_val_WoPerson_ss-10_L05/est_resnet101_15_step62240.pt')
parser.add_argument('--input_size', type=int, default=224)
parser.add_argument('--batch_size', type=int, default=5)
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


def eval_est_trasnfer(batch, r_sig):
    y_true = []
    y_pred = []
    for j in range(bs):
        with torch.no_grad():
            ref_sig_expand = torch.cat([r_sig[j]] * bs).view(-1, len(cols))
            out = transfer(batch, ref_sig_expand)

            y_true.append(ref_sig_expand.to('cpu'))
            pred_sig = estimator(out)
            y_pred.append(pred_sig.to('cpu'))
            # l1 = torch.abs(pred_sig - ref_sig_expand)
            # l1 = pred_sig - ref_sig_expand
            # mse = F.mse_loss(pred_sig, ref_sig_expand, reduction='none')
            # l1_li = np.append(l1_li, torch.mean(l1, dim=0).cpu().numpy().reshape(1, -1), axis=0)
            # mse_li = np.appenc(mse_li, torch.mean(mse, dim=0).cpu.numpy().reshape(1, -1), axis=0)
            # [save_image(output, os.path.join(args.output_dir,
            #     '{}_t-{}_r-{}.jpg'.format('gt', b_photos[j], r_photos[i])), normalize=True)
            #     for j, output in enumerate(out)]
    y_true_ = torch.cat(y_true, dim=0)
    y_pred_ = torch.cat(y_pred, dim=0)
    return y_true_, y_pred_


if __name__ == '__main__':
    save_path = os.path.join('/mnt/fs2/2019/Takamuro/m2_research/weather_transfer/results/eval_est_transfer',
                             args.cp_path.split('/')[-2],
                             args.cp_path.split('/')[-1].split('_')[-2])

    os.makedirs(save_path, exist_ok=True)
    # os.makedirs(os.path.join(save_path, 'out'), exist_ok=True)
    cols = ['tempC', 'uvIndex', 'visibility', 'windspeedKmph', 'cloudcover', 'humidity', 'pressure', 'DewPointC']

    transform = nn.Sequential(
            # transforms.Resize((args.input_size,) * 2),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            )

    df = pd.read_pickle(args.pkl_path)
    df_ = df.loc[:, cols].fillna(0)
    df_mean = df_.mean()
    df_std = df_.std()

    df.loc[:, cols] = (df.loc[:, cols].fillna(0) - df_mean) / df_std

    df_sep = df[df['mode'] == 'test']

    print('loaded {} data'.format(len(df_sep)))

    dataset = FlickrDataLoader(args.image_root, df, cols, transform=transform)

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

    # load model
    transfer = Conditional_UNet(num_classes=len(cols))
    sd = torch.load(args.cp_path)
    transfer.load_state_dict(sd['inference'])
    transfer.eval()

    estimator = torch.load(args.estimator_path)
    estimator.eval()

    transfer.cuda()
    estimator.cuda()

    bs = args.batch_size
    y_true_l = []
    y_pred_l = []
    for i, (data, rnd) in tqdm(enumerate(zip(loader, random_loader)), total=len(df)//bs):
        batch = data[0].to('cuda')
        batch = dataset.transform(batch)
        # b_sig = data[1].to('cuda')
        # r_batch  = rnd[0].to('cuda')
        r_sig = rnd[1].to('cuda')

        y_true, y_pred = eval_est_trasnfer(batch, r_sig)
        y_true_l.append(y_true)
        y_pred_l.append(y_pred)

    y_true_l = torch.cat(y_true_l, dim=0).numpy()
    y_pred_l = torch.cat(y_pred_l, dim=0).numpy()

    r2 = r2_score(y_true_l, y_pred_l)
    mae = mean_absolute_error(y_true_l, y_pred_l)
    mse = mean_squared_error(y_true_l, y_pred_l)

    print("r2 score = {}".format(r2))
    print("mae score = {}".format(mae))
    print("mse score = {}".format(mse))

    # ave_l1 = np.mean(l1_li, axis=0)
    # std_l1 = np.std(l1_li, axis=0)
    # ave_mse = np.mean(mse_li, axis=0)

    # print(cols)
    # print('l1')
    # print(ave_l1)
    # print((ave_l1 * df_std))
    # print('l1 std')
    # print(std_l1)
    # print((std_l1 * df_std))

    # print('each mse')
    # print(ave_mse)
    # print('all mean mse')
    # print(np.mean(mse_li))

    # fig = plt.figure(figsize=(40, 10))
    # for i, col in enumerate(cols):
    #     ax = fig.add_subplot(2, 4, i)
    #     ax.hist(l1_li[:, i], bins=100)
    #     ax.set_xlabel('{}_L1'.format(col))

    # fig.savefig(os.path.join(save_path, '{}_eval_result.jpg'.format(col)))
    # plt.clf()

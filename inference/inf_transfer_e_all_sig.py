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
                    default='/mnt/HDD8T/takamuro/dataset/photos_usa_224_2016-2017')
parser.add_argument('--pkl_path', type=str,
                    default='/mnt/fs2/2019/Takamuro/m2_research/flicker_data/wwo/2016_17/lambda_0/outdoor_all_dbdate_wwo_weather_2016_17_delnoise_WoPerson_sky-10_L-05_50testImgs.pkl')
# parser.add_argument('--output_dir', '-o', type=str,
#                     default='/mnt/fs2/2019/Takamuro/m2_research/weather_transfer/results/eval_est_transfer/'
#                     'cUNet_w-e_res101-0408_train-D1T1_adam_b1-00_aug_wloss-mse_train200k-test500/e23_322k')
parser.add_argument('--cp_path', type=str,
                    default='cp/transfer/1204_cUNet_w-e_res101-1203L05e15_SNDisc_sampler-False_GDratio1-8_adam-b10.5-b20.9_lr-0.0001_bs-24_ne-150/cUNet_est_e0110_s1168000.pt')
parser.add_argument('--estimator_path', type=str,
                    default='/mnt/fs2/2019/Takamuro/m2_research/weather_transferV2/cp/estimator/'
                    'est_res101-1203_sampler_pre_WoPerson_sky-10_L-05/est_resnet101_15_step62240.pt')
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

        df = pd.read_pickle(args.pkl_path)
        df_.loc[:, cols] = (df_.loc[:, cols].fillna(0) - df_mean) / df_std
        df.loc[:, cols] = (df.loc[:, cols].fillna(0) - df_mean) / df_std
        df_sep = df[df['mode'] == 'test']

        sig_step_num = 10
        sig_l = [np.linspace(df_.loc[:, col].min(), df_.loc[:, col].max(), sig_step_num) for col in cols]

        print('loaded {} signals data'.format(len(df_sep)))
        del df, df_, temp
        dataset = FlickrDataLoader(args.image_root, df_sep, cols, transform=transform, inf=True)

    loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            drop_last=True
            )

    # load model
    transfer = Conditional_UNet(len(cols))
    sd = torch.load(args.cp_path)
    transfer.load_state_dict(sd['inference'])

    estimator = torch.load(args.estimator_path)
    estimator.eval()

    # if args.gpu > 0:
    transfer.cuda()
    estimator.cuda()

    bs = args.batch_size
    out_li = []

    save_path = os.path.join('/mnt/fs2/2019/Takamuro/m2_research/weather_transferV2/results/eval_est_transfer',
                             args.cp_path.split('/')[-2],
                             'e' + args.cp_path.split('/')[-1].split('_')[-2])
    os.makedirs(save_path, exist_ok=True)
    for k, data in tqdm(enumerate(loader), total=len(df_sep)//bs):
        batch = data[0].to('cuda')
        p_name = data[2]
        batch = dataset.transform(batch)

        sig = data[1].to('cuda')
        b_photos = data[2]

        blank = torch.zeros_like(batch[0]).unsqueeze(0)


        # pred_sig = estimator(r_batch).detach()
        # with torch.no_grad():
        #     out = transfer(batch, pred_sig)
        #     # out_ = transfer(batch, r_sig)
        #     [save_image(output, os.path.join(args.output_dir,
        #          '{}_{}'.format('pred', photos[j])), normalize=True)
        #          for j, output in enumerate(out)]
        #     [save_image(output, os.path.join(args.output_dir,
        #         '{}_{}'.format('rand', photos[j])), normalize=True)
        #         for j, output in enumerate(out_)]

        for i in tqdm(range(bs)):
            with torch.no_grad():
                out_l = []
                for j, col in enumerate(cols):
                    ref_labels_expand = torch.cat([sig[i]] * sig_step_num).view(-1, len(cols))
                    ref_labels_expand[:, j] = torch.from_numpy(sig_l[j]).float().to('cuda')
                    batch_ = torch.unsqueeze(batch[i], dim=0)
                    batch_expand = torch.cat([batch_] * sig_step_num, dim=0)
                    out = transfer(batch_expand, ref_labels_expand)
                    out_l.append(out)
                out_ = torch.cat(out_l, dim=2)
                save_image(out_, fp=os.path.join(save_path, p_name[i] + '.jpg'), normalize=True, scale_each=True, nrow=sig_step_num)

                # [save_image(output, os.path.join(save_path,
                #  '{}-{}_r-{}.jpg'.format('gt', b_photos[j], r_photos[i])), normalize=True, scale_each=True)
                #  for j, output in enumerate(out)]
                # [save_image(output, os.path.join(args.output_dir,
                #  '{}_{}'.format('rand', photos[j]), normalize=True)
                #  for j, output in enumerate(out_)]
        #     out_li.append(out)
        # ref_img = torch.cat([blank] + list(torch.split(r_batch, 1)), dim=3)
        # in_out_img = torch.cat([batch] + out_li, dim=3)
        # res_img = torch.cat([ref_img, in_out_img], dim=0)
        # save_image(ref_img, os.path.join(args.output_dir, '0ref.jpg'), normalize=True)
        # [save_image(out, os.path.join(args.output_dir, '{}_in_out.jpg'.format(i)), normalize=True) for i, out in enumerate(in_out_img)]
        # save_image(res_img, os.path.join(args.output_dir, '0summury.jpg'), normalize=True)

        # res = make_table_img(batch, r_batch, out_li)
        # save_image(res, os.path.join(args.output_dir, 'summary_results_{}.jpg'.format(str(k))), normalize=True)


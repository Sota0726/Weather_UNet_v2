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
                    default="/mnt/8THDD/takamuro/dataset/photos_usa_2016/")
parser.add_argument('--pkl_path', type=str,
                    default='/mnt/fs2/2019/Takamuro/m2_research/flicker_data/from_nitta/param03/50test_high-consis_from-10images-each-con2.pkl')
parser.add_argument('--output_dir', '-o', type=str,
                    default='/mnt/fs2/2019/Takamuro/m2_research/weather_transfer/results/eval_est_transfer/'
                    'cUNet_w-e_res101-0408_train-D1T1_adam_b1-00_aug_wloss-mse_train200k-test500/e23_322k')
parser.add_argument('--cp_path', type=str,
                    default='/mnt/fs2/2019/Takamuro/m2_research/weather_transfer/cp/transfer/'
                    'cUNet_w-e_res101-0408_train-D1T1_adam_b1-00_aug_wloss-mse_train200k-test500/cUNet_w-e_res101-0408_train-D1T1_adam_b1-00_aug_wloss-mse_train200k-test500_e0023_s322000.pt')
parser.add_argument('--classifer_path', type=str,
                    default='/mnt/fs2/2019/Takamuro/m2_research/weather_transfer/cp/estimator/'
                            'est_res101_flicker-p03th01-WoOutlier_sep-train_aug_pre_loss-mse-reduction-none-grad-all-1/est_resnet101_20_step22680.pt')
parser.add_argument('--input_size', type=int, default=224)
parser.add_argument('--batch_size', type=int, default=50)
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

    transform = transforms.Compose([
        transforms.Resize((args.input_size,)*2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    if args.image_only:
        sep_data = glob(os.path.join(args.image_root, '*.png'))
        print('loaded {} data'.format(len(sep_data)))

        dataset = ImageLoader(paths=sep_data, transform=transform, inf=True)
    else:
        cols = ['clouds', 'temp', 'humidity', 'pressure', 'windspeed']

        df = pd.read_pickle(args.pkl_path)

        temp = pd.read_pickle('/mnt/fs2/2019/okada/from_nitta/parm_0.3/sepalated_data_wo-outlier.pkl')
        df_ = temp.loc[:, cols].fillna(0)
        df_mean = df_.mean()
        df_std = df_.std()

        df.loc[:, cols] = (df.loc[:, cols].fillna(0) - df_mean) / df_std

        df_sep = df[df['mode'] == 'test']
        print('loaded {} signals data'.format(len(df_sep)))
        del df, temp
        dataset = FlickrDataLoader(args.image_root, df_sep, cols, transform=transform)

    loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            drop_last=True
            )
    random_loader = torch.utils.data.DataLoader(
            dataset,
            # sampler=ImbalancedDatasetSampler(dataset),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            drop_last=True
            # shuffle=True
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
    out_li = []

    save_path = os.path.join('/mnt/fs2/2019/Takamuro/m2_research/weather_transfer/results/eval_est_transfer',
                             args.cp_path.split('/')[-2],
                             'e' + args.cp_path.split('/')[-1].split('_')[-2])
    os.makedirs(save_path, exist_ok=True)
    for k, (data, rnd) in tqdm(enumerate(zip(loader, random_loader)), total=len(df_sep)//bs):
        batch = data[0].to('cuda')
        r_batch = rnd[0].to('cuda')
        r_sig = rnd[1].to('cuda')

        b_photos = data[2]
        r_photos = rnd[2]

        blank = torch.zeros_like(batch[0]).unsqueeze(0)
        # pred_sig = classifer(r_batch).detach()
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
                ref_labels_expand = torch.cat([r_sig[i]] * bs).view(-1, len(cols))
                out = transfer(batch, ref_labels_expand)

                [save_image(output, os.path.join(args.output_dir,
                 '{}-{}_r-{}.jpg'.format('gt', b_photos[j], r_photos[i])), normalize=True)
                 for j, output in enumerate(out)]
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


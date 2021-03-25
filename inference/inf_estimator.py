import argparse
import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
from glob import glob

# Dataloder でPID kill error が出るときは画像を読み込めてない可能性が高いので，まずはパスをチェックする．
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, default='2')
parser.add_argument('--pkl_path', type=str,
                    default='/mnt/fs2/2019/Takamuro/m2_research/flicker_data/wwo/2016_17/equal_con-cnn-mlp/outdoor_all_dbdate_wwo_weather_selected_ent_owner_2016_17_delnoise_addpred_equal_con-cnn-mlp.pkl')
parser.add_argument('--image_root', type=str, 
                    default='/mnt/HDD8T/takamuro/dataset/photos_usa_224_2016-2017/')
parser.add_argument('--estimator_path', type=str,
                    default='/mnt/fs2/2019/Takamuro/m2_research/weather_transferV2/cp/estimator/'
                    'est_res101-1121_data-equal-mlp-con-gt_time-6-18/est_resnet101_15_step77424.pt'
                    )
parser.add_argument('--input_size', type=int, default=224)
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--save_path', type=str)
# parser.add_argument('--mode', type=str, default='test')
args = parser.parse_args()
# args = parser.parse_args(['--gpu', '0', '--estimator_path', './cp/estimator/est_res101-1112/est_resnet101_40_step43296.pt'])


# GPU Setting
os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
import torch.nn.functional as F
import torch.nn as nn
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

sys.path.append(os.getcwd())
from dataset import ImageLoader


def plot_hist(dict_result, dir_list):
    for dir_ in dir_list:
        preds = np.array(dict_result[dir_]).T
        for i, pred in enumerate(preds):
            fig = plt.figure(figsize=(40, 10))
            x_ = [i for i in range(len(pred))]
            plt.plot(x_, pred)
            plt.xlabel(cols[i])
            fig.savefig(os.path.join(save_path, dir_, '{}.png'.format(cols[i])))
            plt.clf()
            plt.close()


if __name__ == '__main__':
    # save_path = os.path.join('/mnt/fs2/2019/Takamuro/m2_research/weather_transferV2/results/eval_estimator',
    #                          args.estimator_path.split('/')[-2],
    #                          'e' + args.estimator_path.split('/')[-1].split('_')[-2], mode)
    save_path = args.save_path
    os.makedirs(save_path, exist_ok=True)
    # os.makedirs(os.path.join(save_path, 'input_imgs'), exist_ok=True)

    df = pd.read_pickle(args.pkl_path)
    cols = ['tempC', 'uvIndex', 'visibility', 'windspeedKmph', 'cloudcover', 'humidity', 'precipMM', 'pressure', 'DewPointC']

    num_classes = len(cols)

    df_ = df.loc[:, cols].fillna(0)
    df_mean = df_.mean()
    df_std = df_.std()
    df.loc[:, cols] = (df.loc[:, cols] - df_mean) / df_std

    del df_
    # print('loaded {} data'.format(len(df)))
    dict_result = {}
    dir_list = os.listdir(args.image_root)
    for dir_ in dir_list:
        dict_result[dir_] = []

    photo_list = sorted(glob(os.path.join(args.image_root, '*', '*.jpg'), recursive=True))
    print('{} loaded'.format(len(photo_list)))
    # for FlickerDataLoader
    # transform = nn.Sequential(
    #     transforms.ConvertImageDtype(torch.float32),
    #     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    #     )

    # for ImageLoader
    transform = transforms.Compose([
        transforms.Resize((args.input_size,)*2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    dataset = ImageLoader(photo_list, transform=transform)

    loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            drop_last=True
            )

    # load model
    estimator = torch.load(args.estimator_path)
    estimator.eval()
    estimator.to('cuda')

    bs = args.batch_size

    font = ImageFont.truetype("meiryo.ttc", 11)
    # vec_li = []
    for i, data in tqdm(enumerate(loader), total=len(photo_list) // bs):
        batch = data[0].to('cuda')
        photo_paths = data[1]
        # torch >= 1.7
        # batch = dataset.transform(batch)

        preds = estimator(batch).detach()
        batch = batch.cpu()

        for j in range(bs):
            pred = preds[j].to('cpu')
            s_photo_path_l = photo_paths[j].split('/')

            pred_img = Image.new('RGB', (args.input_size,) * 2, (0, 0, 0))

            draw_pred = ImageDraw.Draw(pred_img)
            draw_pred.text((0, 0), 'pred signal', font=font, fill=(200, 200, 200))

            preds_ = []
            for k in range(num_classes):
                pred_ = pred[k].item() * df_std[k] + df_mean[k]
                pred_text = '{} = {}'.format(cols[k], pred_)

                k_ = k + 1
                draw_pred.text((0, k_ * 14), pred_text, font=font, fill=(200, 200, 200))

                preds_.append(pred_)

            t_pred_img = transform(pred_img)
            output = torch.cat([batch[j], t_pred_img], dim=2)

            fp = os.path.join(save_path, s_photo_path_l[-2], s_photo_path_l[-1])
            os.makedirs(os.path.join(save_path, s_photo_path_l[-2]), exist_ok=True)
            save_image(output, fp=fp, normalize=True)

            dict_result[s_photo_path_l[-2]].append(preds_)

    plot_hist(dict_result, dir_list)

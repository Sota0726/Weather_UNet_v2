import argparse
import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# Dataloder でPID kill error が出るときは画像を読み込めてない可能性が高いので，まずはパスをチェックする．
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--pkl_path', type=str,
                    default='/mnt/fs2/2019/Takamuro/m2_research/flicker_data/wwo/2016_17/lambda_0/outdoor_all_dbdate_wwo_weather_2016_17_delnoise_WoPerson_sky-10_L-05.pkl')
parser.add_argument('--image_root', type=str,
                    default='/mnt/HDD8T/takamuro/dataset/photos_usa_224_2016-2017')
parser.add_argument('--estimator_path', type=str,
                    default='/mnt/fs2/2019/Takamuro/m2_research/weather_transferV2/cp/estimator/'
                    'est_res101-1203_sampler_pre_WoPerson_sky-10_L-05/est_resnet101_15_step62240.pt')
parser.add_argument('--input_size', type=int, default=224)
parser.add_argument('--batch_size', type=int, default=25)
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--mode', type=str, default='test')
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
from dataset import FlickrDataLoader


def make_matricx_img(df, pred, col):
    dir_name = args.image_root
    temp_df = df
    temp = pred
    # print(temp.shape)
    print('{} gt range is {} ~ {}'.format(col, min(df[col]), max(df[col])))
    print()
    print('not stdrange is {} ~ {}'.format(np.min(temp) * df_std + df_mean, np.max(temp) * df_std + df_mean))
    print('range is {} ~ {}'.format(np.min(temp), np.max(temp)))
    bins = np.linspace(np.min(temp), np.max(temp), 11)
    print('bins is')
    print(bins)

    # img_size = 64
    # bin_num = 10
    # img_num = 24
    # dst = Image.new('RGB', (img_size*bin_num, img_size*img_num))
    # for i in range(bin_num):
    #     temp_df2 = temp_df[(temp_df.clouds>=bins[i])&(temp_df.clouds<bins[i+1])]
    #     sample_num = min(img_num, len(temp_df2))
    #     photo_id = temp_df2.sample(n=sample_num)['photo'].tolist()
    #     # print(photo_id)
    #     for j, p in enumerate(photo_id):
    #         im = Image.open(dir_name+p+".jpg")
    #         im_resize = im.resize((img_size,img_size))
    #         dst.paste(im_resize, (i*img_size, j*img_size))
    dst = 0
    return dst


def plot_hist(col, df, l1, pred, df_std, df_mean):
    # gt = df[col].tolist()
    gt = df[col].values
    # gt = gt * df_std[col] + df_mean[col]
    pred = pred * df_std[col] + df_mean[col]
    l1 = l1 * df_std[col]
    l1_abs = np.abs(l1)
    fig = plt.figure(figsize=(40, 10))

    if col == 'uvIndex' or col == 'visibility':
        ax = fig.add_subplot(1, 4, 1)
        ax.hist(gt, bins=range(0, 11))
        ax.set_xlabel('{}_gt'.format(col))
    else:
        ax = fig.add_subplot(1, 4, 1)
        ax.hist(gt, bins=100)
        ax.set_xlabel('{}_gt'.format(col))

    if col == 'uvIndex' or col == 'visibility':
        ax = fig.add_subplot(1, 4, 2)
        ax.hist(pred, bins=range(0, 11))
        ax.set_xlabel('{}_pred'.format(col))
    else:
        ax = fig.add_subplot(1, 4, 2)
        ax.hist(pred, bins=np.arange(np.min(pred), np.max(pred), 0.25))
        ax.set_xlabel('{}_pred'.format(col))

    ax = fig.add_subplot(1, 4, 3)
    ax.hist(l1, bins=100)
    ax.set_xlabel('{}_L1'.format(col))

    ax = fig.add_subplot(1, 4, 4)
    ax.hist(l1_abs, bins=100)
    ax.set_xlabel('{}_AbsL1'.format(col))

    fig.savefig(os.path.join(save_path, '{}_eval_result.jpg'.format(col)))


if __name__ == '__main__':

    mode = args.mode
    save_path = './temp_eval_est/'
    # save_path = os.path.join('/mnt/fs2/2019/Takamuro/m2_research/weather_transferV2/results/eval_estimator',
    #                          args.estimator_path.split('/')[-2],
    #                          'e' + args.estimator_path.split('/')[-1].split('_')[-2], mode)
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(os.path.join(save_path, 'input_imgs'), exist_ok=True)

    df = pd.read_pickle(args.pkl_path)

    cols = ['tempC', 'uvIndex', 'visibility', 'windspeedKmph', 'cloudcover', 'humidity', 'pressure', 'DewPointC']
    # cols = ['tempC', 'uvIndex', 'visibility', 'windspeedKmph', 'cloudcover', 'humidity', 'precipMM', 'pressure', 'DewPointC']

    num_classes = len(cols)

    df_ = df.loc[:, cols].fillna(0)
    df_mean = df_.mean()
    df_std = df_.std()
    df_max = df_.max()
    df_min = df_.min()
    df.loc[:, cols] = (df.loc[:, cols] - df_mean) / df_std

    if not mode == 'all':
        df_['mode'] = df['mode']
        df = df[df['mode'] == mode]
        df_ = df_[df_['mode'] == mode]
    # 推定結果を記録するようのcolumnsを初期化
    for col in cols:
        df['pred_{}'.format(col)] = '-99'

    # df = df[df['mode'] == 'test']

    # for col in cols:
    #     tab_img = make_matricx_img(df, df[col].tolist(), col)
        # tab_img.save('gt_{}.jpg'.format(col))

    print('loaded {} data'.format(len(df)))
    # torch >= 1.7
    transform = nn.Sequential(
        transforms.ConvertImageDtype(torch.float32),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    )

    transform_rgb = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    dataset = FlickrDataLoader(args.image_root, df, cols, bs=args.batch_size, transform=transform, inf=True)

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

    y_true_l = []
    y_pred_l = []

    font = ImageFont.truetype("meiryo.ttc", 11)
    # vec_li = []
    for i, data in tqdm(enumerate(loader), total=len(df) // bs):
        batch = data[0].to('cuda')
        signals = data[1].to('cuda')
        photo_ids = data[2]
        # torch >= 1.7
        batch = dataset.transform(batch)

        preds = estimator(batch).detach()
        batch = batch.cpu()

        signals = signals.to('cpu')
        preds = preds.to('cpu')
        y_true_l.append(signals)
        y_pred_l.append(preds)

        # 出力画像を保存したければ，forの中のコメントアウトを外す
        for j in range(bs):
            signal = signals[j]
            pred = preds[j]

            # gt_img = Image.new('RGB', (args.input_size,) * 2, (0, 0, 0))
            # pred_img = Image.new('RGB', (args.input_size,) * 2, (0, 0, 0))

            # draw_gt = ImageDraw.Draw(gt_img)
            # draw_pred = ImageDraw.Draw(pred_img)
            # draw_gt.text((0, 0), 'gt signal', font=font, fill=(200, 200, 200))
            # draw_pred.text((0, 0), 'pred signal', font=font, fill=(200, 200, 200))

            ind_df = df[df.photo == photo_ids[j]].index[0]

            for k in range(num_classes):
                signal_ = signal[k].item() * df_std[k] + df_mean[k]
                pred_ = pred[k].item() * df_std[k] + df_mean[k]
                # gt_text = '{} = {}'.format(cols[k], signal_)
                # pred_text = '{} = {}'.format(cols[k], pred_)

                # k_ = k + 1
                # draw_gt.text((0, k_ * 14), gt_text, font=font, fill=(200, 200, 200))
                # draw_pred.text((0, k_ * 14), pred_text, font=font, fill=(200, 200, 200))

                df.loc[ind_df, 'pred_{}'.format(cols[k])] = pred_

            # t_gt_img = transform_rgb(gt_img)
            # t_pred_img = transform_rgb(pred_img)
            # output = torch.cat([batch[j], t_gt_img, t_pred_img], dim=2)

            # fp = os.path.join(save_path, 'input_imgs', photo_ids[j] + '.jpg')
            # save_image(output, fp=fp, normalize=True)

    y_true_l = torch.cat(y_true_l, dim=0).numpy()
    y_pred_l = torch.cat(y_pred_l, dim=0).numpy()
    y_true_l_n = (y_true_l * df_std.values + df_mean.values) / df_max.values
    y_pred_l_n = (y_pred_l * df_std.values + df_mean.values) / df_max.values

    l1_li = y_true_l - y_pred_l

    ave_l1 = np.mean(l1_li, axis=0)
    std_l1 = np.std(l1_li, axis=0)

    df.loc[:, cols] = df_.loc[:, cols]
    df.to_pickle(os.path.join(save_path, mode + '_result.pkl'))

    print(cols)
    print('l1')
    print(ave_l1)
    print((ave_l1 * df_std))
    print('l1 std')
    print(std_l1)
    print((std_l1 * df_std))

    for i, col in enumerate(cols):
    #     tab_img = make_matricx_img(df, y_pred_l[:, i], col)
    #     # tab_img.save(os.path.join(save_path, 'est_{}.jpg'.format(col)))
        plot_hist(col, df, l1_li[:, i], y_pred_l[:, i], df_std, df_mean)

    r2 = r2_score(y_true_l_n, y_pred_l_n)
    mae = mean_absolute_error(y_true_l_n, y_pred_l_n)
    mse = mean_squared_error(y_true_l_n, y_pred_l_n)

    print("r2 score = {}".format(r2))
    print("mae score = {}".format(mae))
    print("mse score = {}".format(mse))

    l1 = np.abs(y_true_l_n - y_pred_l_n)
    mae_l = np.mean(l1, axis=0)
    mae_std_l = np.std(l1, axis=0)

    plt.bar(np.arange(len(cols)), mae_l, yerr=mae_std_l, tick_label=cols, align='center')
    plt.xticks(rotation=90)
    plt.savefig(os.path.join(save_path, 'mae.png'))


# train
# r2 score = 0.149372071422682
# mae score = 0.160504865099651
# mse score = 0.05649023232962158
# val
# epoch 15
# r2 score = 0.18266121390842013
# mae score = 0.15324589499927965
# mse score = 0.05277723799014609

# epoch 10
# r2 score = 0.20227510886747546
# mae score = 0.15185067555927037
# mse score = 0.052515542568275564

# R^E(T, \hat{S})
# r2 score = 0.015976026891002706
# mae score = 0.17013560287450424
# mse score = 0.058921149095193665
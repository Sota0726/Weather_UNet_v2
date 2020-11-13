import argparse
import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt

# Dataloder でPID kill error が出るときは画像を読み込めてない可能性が高いので，まずはパスをチェックする．
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, default='2')
parser.add_argument('--pkl_path', type=str,
                    default='/mnt/fs2/2019/Takamuro/m2_research/flicker_data/wwo/2016_17/lambda_06/outdoor_all_dbdate_wwo_weather_selected_ent_owner_2016_17_WO-outlier-gray_duplicate.pkl')
parser.add_argument('--image_root', type=str, 
                    default='/mnt/HDD8T/takamuro/dataset/photos_usa_224_2016-2017')
parser.add_argument('--estimator_path', type=str,
                    default='/mnt/fs2/2019/Takamuro/m2_research/weather_transfer/cp/estimator/'
                    'est_res101_flicker-p03th01-WoOutlier_sep-train_aug_pre_loss-mse-reduction-none-grad-all-1/est_resnet101_20_step22680.pt'
                    )
parser.add_argument('--input_size', type=int, default=224)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--num_workers', type=int, default=8)
# parser.add_argument('--num_classes', type=int, default=6)
# args = parser.parse_args()
args = parser.parse_args(['--gpu', '0', '--estimator_path', './cp/estimator/est_res101-1112/est_resnet101_40_step43296.pt'])


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
    dir_name = '/mnt/fs2/2019/Takamuro/db/photos_usa_2016/'
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


def plot_hist(col, df, l1, pred):
    gt = df[col].tolist()

    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(1, 3, 1)
    ax.hist(gt)
    ax.set_xlabel('{}_gt'.format(col))

    ax = fig.add_subplot(1, 3, 2)
    ax.hist(pred, bins=np.arange(np.min(pred), np.max(pred), 0.25))
    ax.set_xlabel('{}_pred'.format(col))

    ax = fig.add_subplot(1, 3, 3)
    ax.hist(l1)
    ax.set_xlabel('{}_l1'.format(col))

    fig.savefig(os.path.join(save_path, '{}_eval_result.jpg'.format(col)))


if __name__ == '__main__':

    save_path = os.path.join('/mnt/fs2/2019/Takamuro/m2_research/weather_transfer/results/eval_estimator',
                             args.estimator_path.split('/')[-2],
                             'e' + args.estimator_path.split('/')[-1].split('_')[-2])
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(os.path.join(save_path, 'input_imgs'), exist_ok=True)

    df = pd.read_pickle(args.pkl_path)
    print('{} data were loaded'.format(len(df)))

    # cols = ['tempC', 'uvIndex', 'visibility', 'windspeedKmph', 'cloudcover', 'humidity', 'pressure', 'HeatIndexC', 'FeelsLikeC', 'DewPointC']
    cols = ['tempC', 'uvIndex', 'visibility', 'windspeedKmph', 'cloudcover', 'humidity', 'pressure', 'FeelsLikeC', 'DewPointC']
    num_classes = len(cols)

    df_ = df.loc[:, cols].fillna(0)
    df_mean = df_.mean()
    df_std = df_.std()
    df.loc[:, cols] = (df.loc[:, cols] - df_mean) / df_std
    del df_

    df = df[df['mode'] == 'test']

    for col in cols:
        tab_img = make_matricx_img(df, df[col].tolist(), col)
        # tab_img.save('gt_{}.jpg'.format(col))

    print('loaded {} data'.format(len(df)))


    transform = nn.Sequential(
        # transforms.Resize((args.input_size,)*2),
        transforms.ConvertImageDtype(torch.float32),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        )

    transform_ = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    dataset = FlickrDataLoader(args.image_root, df, cols, transform=transform, inf=True)

    loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            drop_last=True
            )

    # load model
    estimator = torch.load(args.estimator_path)
    estimator.eval()
    estimator.cuda()

    bs = args.batch_size

    l1_li = np.empty((0, len(cols)))
    pred_li = np.empty((0, len(cols)))
    mse_li = np.empty((0, len(cols)))


    font = ImageFont.truetype("meiryo.ttc", 11)
    # vec_li = []
    for i, data in tqdm(enumerate(loader), total=len(df) // bs):
        batch = data[0].to('cuda')
        signals = data[1].to('cuda')
        photo_ids = data[2]
        batch = dataset.transform(batch)

        preds = estimator(batch).detach()
        batch = batch.cpu()

        # l1_ = F.l1_loss(pred, signals)
        mse = F.mse_loss(preds, signals, reduction='none')
        # l1 = torch.mean(torch.abs(pred - signals), dim=0)
        # l1 = torch.abs(pred - signals)
        l1 = preds - signals
        if len(cols) == 1:
            pred_li = np.append(pred_li, preds.cpu().numpy().reshape(bs, -1))
            l1_li = np.append(l1_li, l1.cpu().numpy().reshape(bs, -1))
            mse_li = np.append(mse_li, mse.cpu().numpy().reshape(bs, -1))
        else:
            pred_li = np.append(pred_li, preds.cpu().numpy().reshape(bs, -1), axis=0)
            l1_li = np.append(l1_li, l1.cpu().numpy().reshape(bs, -1), axis=0)
            mse_li = np.append(mse_li, mse.cpu().numpy().reshape(bs, -1), axis=0)

        for i in range(bs):
            signal = signals[i]
            pred = preds[i]

            gt_img = Image.new('RGB', (args.input_size,)*2, (0, 0, 0))
            pred_img = Image.new('RGB', (args.input_size,)*2, (0, 0, 0))

            draw_gt = ImageDraw.Draw(gt_img)
            draw_pred = ImageDraw.Draw(pred_img)
            draw_gt.text((0, 0), 'gt signal', font=font, fill=(200, 200, 200))
            draw_pred.text((0, 0), 'pred signal', font=font, fill=(200, 200, 200))

            for j in range(num_classes):
                gt_text = '{} = {}'.format(cols[j], signal[j].item())
                pred_text = '{} = {}'.format(cols[j], pred[j].item())

                j_ = j + 1
                draw_gt.text((0, j_ * 14), gt_text, font=font, fill=(200, 200, 200))
                draw_pred.text((0, j_ * 14), pred_text, font=font, fill=(200, 200, 200))

            t_gt_img = transform_(gt_img)
            t_pred_img = transform_(pred_img)
            output = torch.cat([batch[i], t_gt_img, t_pred_img], dim=2)

            fp = os.path.join(save_path, 'input_imgs', photo_ids[i] + '.jpg')
            save_image(output, fp=fp, normalize=True)

    ave_l1 = np.mean(l1_li, axis=0)
    std_l1 = np.std(l1_li, axis=0)
    ave_mse = np.mean(mse_li, axis=0)
    # with open(os.path.join(save_path, 'l1.pkl'), 'wb') as f:
    #     pickle.dump(l1_li, f)
    # with open(os.path.join(save_path, 'pred.pkl'), 'wb') as f:
    #     pickle.dump(pred_li, f)
    # with open(os.path.join(save_path, 'mse.pkl'), 'wb') as f:
    #     pickle.dump(mse_li, f)
    # with open('.pkl', 'wb') as f:
    #     pickle.dump(df, f)

    print(cols)
    print('l1')
    print(ave_l1)
    print((ave_l1 * df_std))
    print('l1 std')
    print(std_l1)
    print((std_l1 * df_std))
    print('mse')
    print(ave_mse)

    # tab_img = make_matricx_img(df, pred_li[:,0])
    if len(cols) == 1:
        tab_img = make_matricx_img(df, pred_li, cols[0])
        tab_img.save(os.path.join(save_path, 'est_{}.jpg'.format(cols[0])))
        plot_hist(cols[0], df, l1_li, pred_li)
    else:
        for i, col in enumerate(cols):
            tab_img = make_matricx_img(df, pred_li[:, i], col)
            # tab_img.save(os.path.join(save_path, 'est_{}.jpg'.format(col)))
            plot_hist(col, df, l1_li[:, i], pred_li[:, i])
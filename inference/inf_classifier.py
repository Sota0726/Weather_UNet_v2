# to add classifier label to flicker data
import argparse
import os
import sys

import numpy as np
import pandas as pd
from glob import glob
# from PIL import Image
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=2)
parser.add_argument('--image_root', type=str,
                    default='/mnt/HDD8T/takamuro/dataset/photos_usa_224_2016-2017')
                    # default='/mnt/fs2/2019/Takamuro/m2_research/weather_transfer/results/c_UNet/inf/cUNet_w-c-i2w-res101_img-i2w-train_sampler_D1T1_supervised_wloss-crossent_e0000_s1000/i2w')
parser.add_argument('--pkl_path', type=str,
                    default='/mnt/fs2/2019/Takamuro/m2_research/flicker_data/wwo/2016_17/equal_con-cnn-mlp/outdoor_all_dbdate_wwo_weather_selected_ent_owner_2016_17_delnoise_addpred_equal_con-cnn-mlp.pkl')
parser.add_argument('--output_dir', type=str,
                    default='/mnt/fs2/2019/Takamuro/m2_research/flicker_data/from_nitta/param03/')
# imb mean imbalanced
parser.add_argument('--classifer_path', type=str,
                    default='/mnt/fs2/2019/Takamuro/m2_research/weather_transfer/cp/classifier/i2w_classifier-res101-train-2020317/better_resnet101_epoch15_step59312.pt')
# parser.add_argument('--classifer_path', type=str, default='cp/classifier/res_aug_5_cls/resnet101_95.pt')
parser.add_argument('--input_size', type=int, default=224)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--num_classes', type=int, default=5)
parser.add_argument('--image_only', action='store_true')
parser.add_argument('--dataset', type=str, default='flicker')
args = parser.parse_args()

os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

import torch
import torchvision.transforms as transforms
import torch.nn as nn

sys.path.append(os.getcwd())
from dataset import FlickrDataLoader, ClassImageLoader, ImageLoader
# from sampler import ImbalancedDatasetSampler


if __name__ == '__main__':
    os.makedirs(args.output_dir, exist_ok=True)
    # load model
    classifer = torch.load(args.classifer_path)
    classifer = nn.Sequential(
                        classifer,
                        nn.Softmax(dim=1)
                    )
    classifer.eval()

    if args.gpu > 0:
        classifer.to('cuda')

    cls_li = ['Clear', 'Clouds', 'Rain', 'Snow', 'Mist']
    if args.image_only:
        transform = transforms.Compose([
            transforms.Resize((args.input_size,)*2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        paths = sorted(glob(os.path.join(args.image_root, '*.jpg')))
        print('loaded {} data'.format(len(paths)))

        df = pd.DataFrame.from_dict({'paths': [path.split('/')[-1] for path in paths]})
        df['pred_condition'] = None
        dataset = ImageLoader(paths=paths, transform=transform)

    if args.dataset == 'i2w':
        transform = transforms.Compose([
            transforms.Resize((args.input_size,)*2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        cls_li = ['sunny', 'cloudy', 'rain', 'snow', 'foggy']
        df = pd.read_pickle(args.pkl_path)
        df = df['test']
        print('loaded {} data'.format(len(df)))

        df['pred_condition'] = None
        dataset = ClassImageLoader(paths=df, transform=transform)
    elif args.dataset == 'flicker':
        transform = nn.Sequential(
            # transforms.Resize((args.input_size,) * 2),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        )
        df = pd.read_pickle(args.pkl_path)
        df = df[df['mode'] == 'train']
        cols = ['tempC', 'uvIndex', 'visibility', 'windspeedKmph', 'cloudcover', 'humidity', 'precipMM', 'pressure', 'DewPointC']
        print('loaded {} data'.format(len(df)))

        df['pred_condition'] = None
        dataset = FlickrDataLoader(args.image_root, df, cols, bs=args.batch_size, transform=transform, class_id=True)

    loader = torch.utils.data.DataLoader(
            dataset,
            # sampler=ImbalancedDatasetSampler(dataset),
            batch_size=args.batch_size,
            drop_last=False,
            num_workers=args.num_workers)

    bs = args.batch_size

    if len(df) % bs != 0:
        print('not divisible by batch size')
        sys.exit()

    res_li = []
    print('{} itetates'.format(len(df) / bs))
    for i, data in tqdm(enumerate(loader)):
        batch = data[0].to('cuda')
        ind_batch = data[1].to('cuda')
        if args.dataset == 'flicker':
            batch = dataset.transform(batch)
        pred_ = classifer(batch).detach()
        pred = torch.argmax(pred_, 1)

        res_li.append(pred.int().cpu().view(bs, -1))
    all_res = torch.cat(res_li, 0).numpy()
    for i, res in enumerate(all_res):
        df.iat[i, df.columns.get_loc('pred_condition')] = cls_li[res[0]]
    print(df.info())
    print(df['condition2'].value_counts())
    print(df['pred_condition'].value_counts())
    # df.to_pickle(os.path.join(args.output_dir, 'sep_with_est-label.pkl'))
    df.to_pickle(os.path.join(args.output_dir, 'check_result.pkl'))

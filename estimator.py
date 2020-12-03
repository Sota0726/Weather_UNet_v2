import argparse
import os

import pandas as pd
import numpy as np
from tqdm import trange
from collections import OrderedDict

parser = argparse.ArgumentParser()
parser.add_argument('--image_root', type=str,
                    default='/mnt/HDD8T/takamuro/dataset/photos_usa_224_2016-2017')
parser.add_argument('--pkl_path', type=str,
                    default='/mnt/fs2/2019/Takamuro/m2_research/flicker_data/wwo/2016_17/equal_con-cnn-mlp/outdoor_all_dbdate_wwo_weather_selected_ent_owner_2016_17_delnoise_addpred_equal_con-cnn-mlp-time6-18_Wo-person-animals.pkl')
parser.add_argument('--save_path', type=str, default='cp/estimator')
parser.add_argument('--name', type=str, default='noname-estimator')
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--input_size', type=int, default=224)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--wd', type=float, default=1e-5)
parser.add_argument('--num_epoch', type=int, default=25)
parser.add_argument('--batch_size', '-bs', type=int, default=16)
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--mode', type=str, default='T', help='T(Train data) or E(Evaluate data)')
parser.add_argument('--pre_trained', action='store_true')
parser.add_argument('--sampler', action='store_true')
parser.add_argument('--amp', action='store_true')
args = parser.parse_args()
# args = parser.parse_args(['--amp', '--pre_trained', '--gpu', '1', '--name', 'debug'])

os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

import torch
import torch.nn as nn
import torchvision as tv
import torchvision.transforms as transforms
import torchvision.models as models

from torch.utils.tensorboard import SummaryWriter

if args.amp:
    from apex import amp, optimizers

from dataset import FlickrDataLoader
from sampler import ImbalancedDatasetSampler
from ops import l1_loss, adv_loss  # , soft_transform


if __name__ == '__main__':

    name = '{}_amp-{}_sampler-{}_PreTrained-{}'.format(args.name, args.amp, args.sampler, args.pre_trained)

    comment = '_lr-{}_bs-{}_ne-{}_x{}_name-{}'.format(args.lr,
                                                      args.batch_size,
                                                      args.num_epoch,
                                                      args.input_size,
                                                      name)
    writer = SummaryWriter(comment=comment)

    save_dir = os.path.join(args.save_path, args.name)
    os.makedirs(save_dir, exist_ok=True)

    # torch >= 1.7
    train_transform = nn.Sequential(
        transforms.RandomRotation(10),
        # transforms.RandomResizedCrop(args.input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(
                brightness=0.5,
                contrast=0.3,
                saturation=0.3,
                hue=0
            ),
        # torch >= 1.7
        transforms.ConvertImageDtype(torch.float32),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    )

    # torch >= 1.7
    test_transform = nn.Sequential(
        # transforms.Resize((args.input_size,) * 2),
        # torch >= 1.7
        transforms.ConvertImageDtype(torch.float32),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    )

    transform = {'train': train_transform, 'test': test_transform}

    df = pd.read_pickle(args.pkl_path)
    print('{} data were loaded'.format(len(df)))

    # cols = ['tempC', 'uvIndex', 'visibility', 'windspeedKmph', 'cloudcover', 'humidity', 'pressure', 'HeatIndexC', 'FeelsLikeC', 'DewPointC']
    cols = ['tempC', 'uvIndex', 'visibility', 'windspeedKmph', 'cloudcover', 'humidity', 'pressure', 'DewPointC']

    # standandelize
    df_ = df.loc[:, cols].fillna(0)
    df_mean = df_.mean()
    df_std = df_.std()
    df.loc[:, cols] = (df.loc[:, cols] - df_mean) / df_std

    # time cut
    # df['orig_date_h'] = df['orig_date']
    # temp = df['orig_date_h'].str.split(':', expand=True)
    # temp_ = temp[0].str.split('T', expand=True)
    # df['orig_date_h'] = temp_[1].astype(int)
    # df = df[(df.orig_date_h >= 6) & (df.orig_date_h <= 18)]

    # t_train    279424
    # val         56162
    # train       56160
    # test          250

    if args.mode == 'T':
        df_sep = {'train': df[df['mode'] == 'train'],
                  'test': df[df['mode'] == 'test']}
    elif args.mode == 'E':  # for evaluation
        df_sep = {'train': df[df['mode'] == 'val'],
                  'test': df[df['mode'] == 'test']}
    else:
        raise NotImplementedError

    del df, df_
    print('{} train data were loaded'.format(len(df_sep['train'])))

    loader = lambda s: FlickrDataLoader(args.image_root, df_sep[s], cols, transform=transform[s])

    train_set = loader('train')
    test_set = loader('test')

    if args.sampler:
        train_loader = torch.utils.data.DataLoader(
                train_set,
                sampler=ImbalancedDatasetSampler(train_set),
                drop_last=True,
                batch_size=args.batch_size,
                num_workers=args.num_workers)
    else:
        train_loader = torch.utils.data.DataLoader(
                train_set,
                shuffle=True,
                drop_last=True,
                batch_size=args.batch_size,
                num_workers=args.num_workers)

    test_loader = torch.utils.data.DataLoader(
            test_set,
            # sampler=ImbalancedDatasetSampler(test_set),
            drop_last=True,
            shuffle=True,
            batch_size=args.batch_size,
            num_workers=args.num_workers)

    num_classes = train_set.num_classes

    if not args.pre_trained:
        model = models.resnet101(pretrained=False, num_classes=num_classes)
    else:
        model = models.resnet101(pretrained=True)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)

    model.to('cuda')

    # train setting
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    if args.amp:
        model, opt = amp.initialize(model, opt, opt_level='O1')

    criterion = nn.MSELoss()
    # criterion = nn.L1Loss()

    eval_per_iter = 500
    save_per_epoch = 5
    global_step = 0

    tqdm_iter = trange(args.num_epoch, desc='Training', leave=True)
    for epoch in tqdm_iter:
        loss_li = []
        diff_mse_li = []
        diff_l1_li = []
        for i, data in enumerate(train_loader, start=0):
            inputs, labels = (d.to('cuda') for d in data)
            # torch >= 1.7
            inputs = train_set.transform(inputs)
            tqdm_iter.set_description('Training [ {} step ]'.format(global_step))

            # optimize
            opt.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            if args.amp:
                with amp.scale_loss(loss, opt) as scale_loss:
                    scale_loss.backward()
            else:
                loss.backward()

            opt.step()

            diff_l1 = l1_loss(outputs.detach(), labels)
            diff_mse = adv_loss(outputs.detach(), labels)

            diff_mse_li.append(diff_mse.item())
            diff_l1_li.append(diff_l1.item())

            if global_step % eval_per_iter == 0:
                diff_mse_li_ = []
                diff_l1_li_ = []
                for j, data_ in enumerate(test_loader, start=0):
                    with torch.no_grad():
                        inputs_, labels_ = (d.to('cuda') for d in data_)
                        # torch >= 1.7
                        inputs_ = train_set.transform(inputs_)
                        outputs_ = model(inputs_).detach()
                        diff_mse_ = adv_loss(outputs_, labels_)
                        diff_l1_ = l1_loss(outputs_, labels_)
                        diff_mse_li_.append(diff_mse_.item())
                        diff_l1_li_.append(diff_l1_.item())

                # write summary
                train_mse = np.mean(diff_mse_li)
                train_diff_l1 = np.mean(diff_l1_li)
                test_mse = np.mean(diff_mse_li_)
                test_diff_l1 = np.mean(diff_l1_li_)
                tqdm_iter.set_postfix(OrderedDict(train_l1=train_diff_l1, test_l1=test_diff_l1))
                writer.add_scalars('mse_loss', {'train': train_mse,
                                                'test': test_mse}, global_step)
                writer.add_scalars('l1_loss', {'train': train_diff_l1,
                                               'test': test_diff_l1}, global_step)
                diff_mse_li = []
                diff_l1_li = []

            global_step += 1

        if epoch % save_per_epoch == 0:
            out_path = os.path.join(save_dir, 'est_resnet101_' + str(epoch) + '_step' + str(global_step) + '.pt')
            torch.save(model, out_path)

    print('Done: training')

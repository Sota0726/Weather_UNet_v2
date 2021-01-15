import pickle
import os
import pandas as pd

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models

from dataset import ClassImageLoader, FlickrDataLoader, CelebALoader
from ops import *


class Predictor(object):
    def __init__(self, args, _type):
        self.args = args
        self.type = _type
        if self.type == 'cls':
            self.save_dir = os.path.join(args.cls_save_dir, args.dataset, args.name)
        if self.type == 'est':
            self.save_dir = os.path.join(args.est_save_dir, args.dataset, args.name)
        os.makedirs(self.save_dir, exist_ok=True)

        # buid transform
        if args.dataset == 'i2w':
            self.train_transform = transforms.Compose([
                transforms.RandomRotation(10),
                transforms.RandomResizedCrop(args.input_size),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(
                    brightness=0.5,
                    contrast=0.3,
                    saturation=0.3,
                    hue=0
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
            self.test_transform = transforms.Compose([
                transforms.Resize((args.input_size,) * 2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        elif args.dataset == 'flickr':
            self.train_transform = nn.Sequential(
                transforms.RandomRotation(10),
                transforms.RandomResizedCrop(args.input_size),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(
                    brightness=0.5,
                    contrast=0.3,
                    saturation=0.3,
                    hue=0
                ),
                transforms.ConvertImageDtype(torch.float32),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            )
            self.test_transform = nn.Sequential(
                transforms.Resize((args.input_size,) * 2),
                transforms.ConvertImageDtype(torch.float32),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            )
        elif args.dataset == 'celebA':
            self.train_transform = nn.Sequential(
                transforms.CenterCrop((178, 178)),
                transforms.Resize((args.input_size,) * 2),
                transforms.RandomRotation(10),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(
                    brightness=0.5,
                    contrast=0.3,
                    saturation=0.3,
                    hue=0
                ),
                transforms.ConvertImageDtype(torch.float32),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            )
            self.test_transform = nn.Sequential(
                transforms.CenterCrop((178, 178)),
                transforms.Resize((args.input_size,) * 2),
                transforms.ConvertImageDtype(torch.float32),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            )

        transform = {'train': self.train_transform, 'test': self.test_transform}
        ##########

        # build dataloader
        if args.dataset == 'flickr':
            df = pd.read_pickle(args.pkl_path)
            print('{} data were loaded'.format(len(df)))
            cols = ['tempC', 'uvIndex', 'visibility', 'windspeedKmph', 'cloudcover', 'humidity', 'pressure', 'DewPointC']
            # standandelize
            df_ = df.loc[:, cols].fillna(0)
            self.df_mean = df_.mean()
            self.df_std = df_.std()
            df.loc[:, cols] = (df.loc[:, cols] - self.df_mean) / self.df_std
            if args.data_mode == 'T':
                df_sep = {
                    # 'train': df[df['mode'] == 'train'],
                    'train': df[(df['mode'] == 'train') | (df['mode'] == 't_train')],
                    'test': df[df['mode'] == 'test']
                }
            elif args.mode == 'E':  # for evaluation
                df_sep = {
                    'train': df[df['mode'] == 'val'],
                    'test': df[df['mode'] == 'test']
                }
            else:
                raise NotImplementedError
            del df, df_
            print('{} train data were loaded'.format(len(df_sep['train'])))
            if self.type == 'cls':
                loader = lambda s: FlickrDataLoader(args.image_root, df_sep[s], cols, transform=transform[s], class_id=True)
            elif self.type == 'est':
                loader = lambda s: FlickrDataLoader(args.image_root, df_sep[s], cols, transform=transform[s])

        elif args.dataset == 'i2w':
            with open(args.i2w_pkl_path, 'rb') as f:
                sep_data = pickle.load(f)
            if args.data_mode == 'E':
                sep_data['train'] = sep_data['val']
            print('{} train data were loaded'.format(len(sep_data['train'])))
            loader = lambda s: ClassImageLoader(paths=sep_data[s], transform=transform[s])

        elif args.dataset == 'celebA':
            df = pd.read_pickle(args.celebA_pkl_path)
            if args.data_mode == 'T':
                num_train_data = len(df[df['mode'] == 'train'])
                df_sep = {
                    'train': df[df['mode'] == 'train'].iloc[:int(num_train_data * args.train_data_ratio)],
                    'test': df[df['mode'] == 'test']
                }
            elif args.data_mode == 'E':  # for evaluation
                df_sep = {
                    'train': df[df['mode'] == 'val'],
                    'test': df[df['mode'] == 'test']
                }
            print('{} train data were loaded'.format(len(df_sep['train'])))
            loader = lambda s: CelebALoader(root_path=args.celebA_root, df=df_sep[s], transform=transform[s])
        self.train_set = loader('train')
        self.test_set = loader('test')
        self.train_loader = make_dataloader(self.train_set, args)
        self.test_loader = make_dataloader(self.test_set, args)
        ##############

        # build network
        self.num_classes = self.train_set.num_classes
        if args.predictor == 'resnet':
            model = models.resnet101(pretrained=args.pre_trained)
            num_features = model.fc.in_features
            model.fc = nn.Linear(num_features, self.num_classes, bias=True)
        elif args.predictor == 'mobilenet':
            model = models.mobilenet_v2(pretrained=args.pre_trained)
            num_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(num_features, self.num_classes, bias=True)
        elif args.predictor == 'resnext':
            model = models.resnext50_32x4d(pretrained=args.pre_trained)
            num_features = model.fc.in_features
            model.fc = nn.Linear(num_features, self.num_classes, bias=True)
        elif args.predictor == 'wideresnet':
            model = models.wide_resnet50_2(pretrained=args.pre_trained)
            num_features = model.fc.in_features
            model.fc = nn.Linear(num_features, self.num_classes, bias=True)
        self.model = model
        ######################

        # build criterion
        if args.dataset == 'flickr' and self.type == 'est':
            self.criterion = nn.MSELoss()
        elif args.dataset == 'flickr' and self.type == 'cls':
            self.criterion = nn.CrossEntropyLoss()
            self.eval_metric = 'precision'
        elif args.dataset == 'i2w':
            self.criterion = nn.CrossEntropyLoss()
            self.eval_metric = 'precision'
        elif args.dataset == 'celebA':
            self.criterion = nn.BCEWithLogitsLoss()
            self.eval_metric = 'acc'
        ######################

    def get_accuracy(self, outputs, labels):
        result = outputs > 0.5
        correct = (result == labels).sum().item()
        acc = correct / (self.num_classes * self.args.batch_size)
        return acc.item()

    def get_precision(self, outputs, labels):
        out = torch.argmax(outputs, dim=1)
        return torch.eq(out, labels).float().mean().item()

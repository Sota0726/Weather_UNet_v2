import os

import pandas as pd
import torch
import numpy as np

from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import DatasetFolder
from torchvision.datasets.folder import default_loader


def _collate_fn(batch):
    data = [v[0] for v in batch]
    target = [v[1] for v in batch]
    target = torch.FloatTensor(target)
    return [data, target]


def get_class_id_from_string(string):
    s_li = ['sunny', 'cloudy', 'rain', 'snow', 'foggy']
    if not string in s_li: raise
    else:
        return s_li.index(string)


class FlickrDataLoader(Dataset):
    def __init__(self, image_root, df, columns, transform=None, class_id=False, imbalance=None, inf=False):
        super(FlickrDataLoader, self).__init__()
        # init
        self.root = image_root
        self.columns = columns
        self.photo_id = df['photo'].to_list()
        self.class_id = class_id
        # df_ = df.loc[:, columns].fillna(0)
        # self.conditions = (df_ - df_.mean()) / df_.std()
        self.conditions = df.loc[:, columns]
        if imbalance:
            # --- master --- #
            # self.labels = df['w_condition']
            # -------------- #
            # --- experiment2 --- #
            self.labels = df['condition2']
            # ------------------- #
        else:
            self.labels = df['condition2']
        # self.cls_li = sorted(self.labels.unique())
        self.cls_li = ['Clear', 'Clouds', 'Rain', 'Snow', 'Mist']
        self.num_classes = len(columns)
        self.transform = transform
        del df  # , df_
        self.inf = inf

    def __len__(self):
        return len(self.photo_id)

    def get_class(self, idx):
        w_cls = self.labels.iloc[idx]
        cls_id = self.cls_li.index(w_cls)
        return cls_id

    def get_signal(self, idx):
        sig = self.conditions.iloc[idx].fillna(0).to_list()
        sig_tensor = torch.from_numpy(np.array(sig)).float()
        del sig
        return sig_tensor

    def __getitem__(self, idx):

        # --- GET IMAGE ---#
        try:
            image = Image.open(os.path.join(self.root, self.photo_id[idx] + '.jpg'))
        except:
            return self.__getitem__(idx)
        image = image.convert('RGB')
        if self.transform:
            image = self.transform(image)

        # --- GET LABEL ---#
        if not self.class_id:
            label = self.get_signal(idx)
        else:
            label = self.get_class(idx)

        if not self.inf:
            return image, label
        elif self.inf:
            return image, label, self.photo_id[idx]


class ImageLoader(Dataset):
    def __init__(self, paths, transform=None):
        super(ImageLoader, self).__init__()
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        try:
            image = Image.open(self.paths[idx])
        except:
            print('load error{}'.format(self.paths[idx]))
            return self.__getitem__(idx)
        image = image.convert('RGB')
        if self.transform:
            image = self.transform(image)
        # for train.py
        return image, True
        # for inception_score.py
        # return image


class ClassImageLoader(Dataset):
    def __init__(self, paths, transform=None, inf=False):
        # without z-other
        paths = [p for p in paths if 'z-other' not in p]
        # count dirs on root
        path = os.path.commonpath(paths)
        files = os.listdir(path)
        files_dir = [f for f in files if os.path.isdir(os.path.join(path, f)) if 'z-other' not in f]
        # init
        self.paths = paths
        self.classes = files_dir
        self.num_classes = len(files_dir)
        self.transform = transform
        self.inf = inf

    def get_class(self, idx):
        string = self.paths[idx].split('/')[-2]
        return get_class_id_from_string(string)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        image = Image.open(self.paths[idx])
        image = image.convert('RGB')
        target = self.get_class(idx)
        if self.transform:
            image = self.transform(image)

        if not self.inf:
            return image, target
        elif self.inf:
            return image, target, self.paths[idx]


class ImageFolder(DatasetFolder):
    def __init__(self, root, transform=None, loader=default_loader):
        super(ImageFolder, self).__init__(root,
                transform=transform,
                extensions='jpg'
            )

    def __getitem__(self, ind):
        path, target = self.samples[ind]
        image = Image.open(path)
        image = image.convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, target


class OneYearWeatherSignals(Dataset):
    def __init__(self, image_root, df, columns, photo_id, transform=None, name=None):
        super(OneYearWeatherSignals, self).__init__()
        # init
        self.root = image_root
        self.columns = columns
        self.photo_id = photo_id
        self.transform = transform
        
        if name is not None:
            self.name = name
        else:
            self.name = df[df['photo'] == self.photo_id]['name'].to_list()[0]

        self.num_classes = len(columns)
        self.conditions, self.s_times = self.get_oneyear_data(self.name, df)

        del df

        try:
            self.image = Image.open(os.path.join(self.root, self.photo_id + '.jpg'))
        except:
            print('no photo id')
            exit()
        self.image = self.image.convert('RGB')
        if self.transform:
            self.image = self.transform(self.image)

    def __len__(self):
        return len(self.conditions)

    def get_oneyear_data(self, name, df):
        df = df[df['name'] == name].drop_duplicates(subset=['s_unixtime'])
        df = df.sort_values('s_unixtime', ascending=False).reset_index()
        s_times = df['s_unixtime']
        df = df.loc[:, self.columns]

        return df, s_times

    def get_condition(self, idx):
        c = self.conditions.iloc[idx].fillna(0).to_list()
        c_tensor = torch.from_numpy(np.array(c)).float()
        del c
        return c_tensor

    def __getitem__(self, idx):
        sig = self.get_condition(idx)
        s_time = self.s_times[idx]
        return self.image, sig, s_time

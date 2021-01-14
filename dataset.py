import os
import random

import pandas as pd
import torch
import numpy as np

from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import DatasetFolder
from torchvision.datasets.folder import default_loader
from torchvision.io import read_image


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


class ImageFolder(DatasetFolder):
    def __init__(self, root, transform=None, loader=default_loader):
        super(ImageFolder, self).__init__(
            root,
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
        return image, self.paths[idx]
        # for inception_score.py
        # return image


class FlickrDataLoader(Dataset):
    def __init__(self, image_root, df, columns, transform=None, class_id=False, inf=False):
        super(FlickrDataLoader, self).__init__()
        # init
        self.root = image_root
        self.columns = columns
        self.photo_id = df['photo'].to_list()
        self.class_id = class_id
        self.conditions = df.loc[:, columns]
        self.labels = df['condition']

        df['orig_date_h'] = df['orig_date']
        temp = df['orig_date_h'].str.split(':', expand=True)
        temp = temp[0].str.split('T', expand=True)
        df.orig_date_h = temp[1].astype(int)
        del temp
        self.time_list = df['orig_date_h'].to_list()

        self.cls_li = ['Clear', 'Clouds', 'Rain', 'Snow', 'Mist']
        self.num_classes = len(columns)
        # torch >= 1.7
        self.transform = transform.to('cuda')
        del df
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

    def get_time(self, idx):
        time = self.time_list[idx]
        time_ = int(time // 6)
        return time_

    def __getitem__(self, idx):
        # --- GET IMAGE ---#
        # torch > 1.7
        try:
            image = read_image(os.path.join(self.root, self.photo_id[idx] + '.jpg'))
        except:
            return self.__getitem__(idx)

        # --- GET LABEL ---#
        if not self.class_id:
            label = self.get_signal(idx)
        else:
            label = self.get_class(idx)

        if not self.inf:
            return image, label
        elif self.inf:
            return image, label, self.photo_id[idx]


# FlickerDataLoaderから継承するように書き換える
class SequenceFlickrDataLoader(Dataset):
    def __init__(self, image_root, csv_root, df, columns, df_mean, df_std,
                 transform=None, inf=False):
        super(SequenceFlickrDataLoader, self).__init__()
        from glob import glob
        self.root = image_root
        self.csvs = glob(os.path.join(csv_root, '*.csv'))
        self.columns = columns
        self.cities = df['location'].sample(frac=1).to_list()
        self.utc_dates = df['utc_date'].sample(frac=1).to_list()
        self.photo_id = df['photo'].to_list()
        self.conditions = df.loc[:, columns]
        self.mean = df_mean
        self.std = df_std
        self.seq_len = 12
        self.seq_step = int(24 // self.seq_len)
        # self.labels = df['condition']

        df['orig_date_h'] = df['orig_date']
        temp = df['orig_date_h'].str.split(':', expand=True)
        temp = temp[0].str.split('T', expand=True)
        df.orig_date_h = temp[1].astype(int)
        del temp
        self.time_list = df['orig_date_h'].to_list()

        self.cls_li = ['Clear', 'Clouds', 'Rain', 'Snow', 'Mist']
        self.num_classes = len(columns)
        # torch >= 1.7
        self.transform = transform
        self.inf = inf
        del df

    def __len__(self):
        return len(self.photo_id)

    def get_signal(self, idx):
        sig = self.conditions.iloc[idx].fillna(0).to_list()
        sig_tensor = torch.from_numpy(np.array(sig)).float()
        del sig
        return sig_tensor

    def get_seqence(self, idx):
        csv_path = [i for i in self.csvs if self.cities[idx] in i]
        random.shuffle(csv_path)
        df_ = pd.read_csv(csv_path[0])
        date_num = len(df_)
        date = self.utc_dates[idx]
        try:
            idx_ = int(df_[df_.utc_date == date].index[0])
        except:
            print(date)
            idx_ = random.randint(0, date_num - 24)
        df_seq = df_.iloc[idx_: idx_ + 24: self.seq_step]
        if len(df_seq) != self.seq_len:
            idx_ = random.randint(0, date_num - 24)
            df_seq = df_.iloc[idx_: idx_ + 24: self.seq_step]
        df_seq = df_seq.loc[:, self.columns]
        df_seq = (df_seq.fillna(0) - self.mean) / self.std
        del df_

        seq = df_seq.values
        seq_tensor = torch.from_numpy(np.array(seq)).float()
        del seq
        return seq_tensor

    def get_time(self, idx):
        time = self.time_list[idx]
        time_ = int(time // 3)
        if time_ > 2 and time_ < 6:
            time_ = 3
        elif time_ >= 6:
            time_ -= 2
        return time_

    def __getitem__(self, idx):
        # --- GET IMAGE ---#
        # torch > 1.7
        try:
            image = read_image(os.path.join(self.root, self.photo_id[idx] + '.jpg'))
        except:
            return self.__getitem__(idx)
        # --- GET LABEL ---#
        label = self.get_signal(idx)
        seq_label = self.get_seqence(idx)

        if not self.inf:
            return image, label, seq_label
        elif self.inf:
            return image, label, seq_label, self.photo_id[idx]


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


class SensorLoader(Dataset):
    def __init__(self, csv_root, date, city, cols, df_std, df_mean):
        from glob import glob
        # from datetime import datetime
        year = int(date[0])
        month = int(date[1])
        day = int(date[2])
        target = '{y}-{m:02}-{d:02}'.format(y=year, m=month, d=day)

        self.csvs = glob(os.path.join(csv_root, '*.csv'))
        self.csv = [i for i in self.csvs if city in i][0]

        df = pd.read_csv(self.csv)

        temp = df['local_date'].str.split(':', expand=True)[0]
        df['local_date_'] = temp.str.split('T', expand=True)[0]
        del temp

        df.loc[:, cols] = (df.loc[:, cols] - df_mean) / df_std
        df_ind = df[df.local_date_ == target].index[0]

        self.df_sig = df.loc[df_ind: df_ind + (int(date[3]) * 24) - 1, cols + ['local_date']]
        del df

    def get_signal(self, idx):
        sig = self.df_sig.iloc[idx, :-1].fillna(0).to_list()
        sig_tensor = torch.from_numpy(np.array(sig)).float()
        del sig
        return sig_tensor

    def __len__(self):
        return len(self.df_sig)

    def __getitem__(self, idx):
        sig = self.get_signal(idx)
        date = self.df_sig.iloc[idx, -1]

        return sig, date


class TransientAttributes(Dataset):
    def __init__(self, image_root, df, columns, transform=None):
        super(TransientAttributes, self).__init__()
        # init
        # ['dirty', 'daylight', 'night', 'sunrisesunset', 'dawndusk', 'sunny', 'clouds', 'fog', 'storm', 'snow', 
        # 'warm', 'cold', 'busy', 'beautiful', 'flowers', 'spring', 'summer', 'autumn', 'winter', 'glowing', 
        # 'colorful', 'dull', 'rugged', 'midday', 'dark', 'bright', 'dry', 'moist', 'windy', 'rain', 'ice', 'cluttered', 
        # 'soothing', 'stressful', 'exciting', 'sentimental', 'mysterious', 'boring', 'gloomy', 'lush']
        self.root = os.path.join(image_root, 'imageLD')
        self.photos = df['photo'].to_list()
        self.classes = df.loc[:, columns]
        self.num_classes = len(columns)
        self.transform = transform
        # self.cls_li = ['daylight', 'night', 'sunrisesunset', 'dawndusk', 'midday']

    def get_class(self, idx):
        w_cls = self.classes.iloc[idx].to_list()
        w_cls_ = [float(x.split(',')[0]) for x in w_cls]
        cls_tensor = torch.from_numpy(np.array(w_cls_)).float()
        del w_cls, w_cls_
        return cls_tensor

    def __len__(self):
        return len(self.photos)

    def __getitem__(self, idx):
        # --- GET IMAGE ---#
        try:
            image = Image.open(os.path.join(self.root, self.photos[idx]))
        except:
            return self.__getitem__(idx)
        image = image.convert('RGB')
        if self.transform:
            image = self.transform(image)

        labels_tensor = self.get_class(idx)

        return image, labels_tensor


class CelebALoader(Dataset):
    def __init__(self, root_path, df, transform=None, inf=False):
        # attributes
        self.cols = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive',
                     'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips', 'Big_Nose',
                     'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows',
                     'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair',
                     'Heavy_Makeup', 'High_Cheekbones', 'Male', 'Mouth_Slightly_Open',
                     'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin',
                     'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns',
                     'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings',
                     'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace',
                     'Wearing_Necktie', 'Young']
        # init
        self.photo_ids = df['image_id'].to_list()
        self.root = root_path
        self.classes = df.loc[:, self.cols]
        self.num_classes = len(self.cols)
        self.transform = transform.to('cuda')
        self.inf = inf

    def get_class(self, idx):
        class_list = self.classes.iloc[idx].to_list()
        cls_tensor = torch.from_numpy(np.array(class_list)).float()
        del class_list
        return cls_tensor

    def __len__(self):
        return len(self.photo_ids)

    def __getitem__(self, idx):
        try:
            image = read_image(os.path.join(self.root, self.photo_ids[idx]))
        except:
            return self.__getitem__(idx)

        target = self.get_class(idx)
        if not self.inf:
            return image, target
        elif self.inf:
            return image, target, self.photo_ids[idx]

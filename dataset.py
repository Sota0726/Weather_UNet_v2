import os
import random

import pandas as pd
import torch
import numpy as np
from glob import glob

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
    def __init__(self, image_root, df, columns, bs, transform=None, class_id=False, inf=False):
        super(FlickrDataLoader, self).__init__()
        # init
        self.root = image_root
        self.columns = columns
        self.photo_id = df['photo'].to_list()
        self.class_id = class_id
        self.conditions = df.loc[:, columns]
        # self.labels = df['condition']

        # df['orig_date_h'] = df['orig_date']
        # temp = df['orig_date_h'].str.split(':', expand=True)
        # temp = temp[0].str.split('T', expand=True)
        # df.orig_date_h = temp[1].astype(int)
        # del temp
        # self.time_list = df['orig_date_h'].to_list()

        self.cls_li = ['Clear', 'Clouds', 'Rain', 'Snow', 'Mist']
        self.num_classes = len(columns)
        self.bs = bs
        # torch >= 1.7
        self.transform = transform
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
            image = read_image(os.path.join(self.root, str(self.photo_id[idx]) + '.jpg'))
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
    def __init__(self, image_root, csv_root, df, columns, df_mean, df_std, bs, seq_len,
                 transform=None, inf=False, mode='train'):
        super(SequenceFlickrDataLoader, self).__init__()
        self.root = image_root
        self.csv_root = csv_root
        self.inf = inf
        self.columns = columns
        if mode == 'train':
            self.locations = df[['location', 'lon', 'lat']].sample(frac=1)
            self.utc_dates = df['utc_date'].sample(frac=1).to_list()
        elif mode == 'test':
            self.locations = df[['location', 'lon', 'lat']]
            self.utc_dates = df['utc_date'].to_list()
            self.inf = True
        self.photo_id = df['photo'].to_list()
        self.conditions = df.loc[:, columns]
        self.mean = df_mean
        self.std = df_std
        self.seq_len = seq_len
        self.bs = bs
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
        del df

    def __len__(self):
        return len(self.photo_id)

    def get_signal(self, idx):
        sig = self.conditions.iloc[idx].fillna(0).to_list()
        sig_tensor = torch.from_numpy(np.array(sig)).float()
        del sig
        return sig_tensor

    def get_seqence(self, idx):
        locations = self.locations.iloc[idx]
        city = locations['location']
        lon = round(locations['lon'], 6)
        lat = round(locations['lat'], 6)
        csv = '{}({},{}).csv'.format(city, str(lon), str(lat))
        df_ = pd.read_csv(os.path.join(self.csv_root, csv))
        csv_date_num = len(df_)
        date = self.utc_dates[idx]
        try:
            idx_ = int(df_[df_.utc_date == date].index[0])
        except:
            print(date)
            # idx_ = random.randint(0, csv_date_num - 24)
            idx_ = random.randint(0, csv_date_num - self.seq_len)
        df_seq = df_.iloc[idx_: idx_ + self.seq_len]
        if len(df_seq) != self.seq_len:
            idx_ = random.randint(0, csv_date_num - self.seq_len)
            df_seq = df_.iloc[idx_: idx_ + self.seq_len]
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
            image = read_image(os.path.join(self.root, str(self.photo_id[idx]) + '.jpg'))
        except:
            return self.__getitem__(idx)
        # --- GET LABEL ---#
        label = self.get_signal(idx)
        seq_label = self.get_seqence(idx)

        if not self.inf:
            return image, label, seq_label
        elif self.inf:
            return image, label, seq_label, str(self.photo_id[idx])


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


class TimeLapseLoader(Dataset):
    def __init__(self, vid_root, bs, seq_len, transform=None):
        self.vid_root = vid_root
        self.vid24s = os.listdir(vid_root)
        self.bs = bs
        self.seq_len = seq_len
        self.transform = transform
        if transform is not None:
            self.transform = transform

    def __len__(self):
        return len(self.vid24s)

    def read_vid(self, vid_name):
        frame_paths = glob(os.path.join(self.vid_root, vid_name, '*.jpg'))
        num_frames = len(frame_paths)
        step = int(num_frames // 24)
        # remain = int(num_frames % 24)
        start_frame = random.randrange(0, step)
        frame_paths_ = frame_paths[start_frame: num_frames: step]
        ind = random.randrange(0, 24 - self.seq_len)
        frames = [read_image(frame_path).unsqueeze(0) for frame_path in frame_paths_[ind: ind + self.seq_len]]
        frames = torch.cat(frames, dim=0)
        return frames

    def __getitem__(self, idx):
        vid_name = self.vid24s[idx]
        timelapse = self.read_vid(vid_name)

        return timelapse


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
        self.transform = transform
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

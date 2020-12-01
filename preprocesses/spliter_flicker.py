import os
import argparse
import pickle
import glob
import random
import pandas as pd
from collections import defaultdict


def split_flicker(df, ent_th):
    # train:val:test = 2:2:1
    df = df[df['ent_label'] < ent_th]
    df.sample(frac=1)
    cls_li = ['Clear', 'Clouds', 'Rain', 'Snow', 'Mist']
    df['mode'] = 'train'

    temp = pd.DataFrame()
    for cls in cls_li:
        df_ = df[df['condition2'] == cls]
        num = len(df[df['condition2'] == cls])
        df_.iloc[int(num*0.4): int(num*0.8), -1] = 'val'
        df_.iloc[int(num*0.8): num, -1] = 'test'
        temp = pd.concat([temp, df_])
    temp = temp[temp['ent_label'] < ent_th]

    return temp


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pickle_path', type=str, default='/mnt/fs2/2019/okada/from_nitta/parm_0.3/outdoor_all_dbdate_withweather_selected_ent.pkl')
    parser.add_argument('--out_path', type=str, default='sepalated_data.pkl')
    parser.add_argument('--ent_th', type=float, default=1.0, help='ent_label threshold')
    args = parser.parse_args()
    print('Start')
    df = pd.read_pickle(args.pickle_path)

    split_df = split_flicker(df, args.ent_th)
    print('data num :{}'.format(len(split_df)))
    with open(os.path.join(args.pickle_path.rsplit('/', 1)[0], args.out_path), "wb") as f:
        pickle.dump(split_df, f)

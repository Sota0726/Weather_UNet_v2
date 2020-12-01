import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

save_path = os.path.join('temp-check_est', 'humi_only-est')
df = pd.read_pickle('/mnt/fs2/2019/okada/from_nitta/parm_0.3/sepalated_data_wo-outlier.pkl')
l1 = pd.read_pickle('/home/sota/data/m2/weather-Unet/l1.pkl')
pred = pd.read_pickle('/home/sota/data/m2/weather-Unet/pred.pkl')

os.makedirs(save_path, exist_ok=True)

df = df[df['mode'] == 'train']
# cols = ['clouds', 'temp', 'humidity', 'pressure', 'windspeed']
cols = ['temp']

df_ = df.loc[:, cols].fillna(0)
df_mean = df_.mean()
df_std = df_.std()
# df = df_ori[df_ori['mode'] == 'test']
df.loc[:, cols] = (df_ - df_mean) / df_std

for i, col in enumerate(cols):

    gt = df[col].tolist()

    plt.figure()
    plt.hist(gt)
    plt.savefig(os.path.join(save_path, '{}_gt_hist.jpg'.format(col)))

    if len(cols) == 1:
        plt.figure()
        plt.hist(l1)
        plt.savefig(os.path.join(save_path, '{}_l1_hist.jpg'.format(col)))

        plt.figure()
        plt.hist(pred, bins=np.arange(np.min(pred), np.max(pred), 0.25))
        plt.savefig(os.path.join(save_path, '{}_pred_hist.jpg'.format(col)))
    else:
        plt.figure()
        plt.hist(l1[i])
        plt.savefig(os.path.join(save_path, '{}_l1_hist.jpg'.format(col)))

        plt.figure()
        plt.hist(pred[i], bins=np.arange(np.min(pred[i]), np.max(pred[i]), 0.25))
        plt.savefig(os.path.join(save_path, '{}_pred_hist.jpg'.format(col)))

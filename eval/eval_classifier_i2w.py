import argparse
import os
import sys

import numpy as np
import pandas as pd
# from PIL import Image
from tqdm import tqdm
import shutil

from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, default=1)
parser.add_argument('--image_root', type=str,
                    default='/mnt/fs2/2018/matsuzaki/dataset_fromnitta/Image/')
parser.add_argument('--pkl_path', type=str,
                    default='/mnt/fs2/2019/Takamuro/db/i2w/sepalated_data.pkl')
# imb mean imbalanced
parser.add_argument('--classifier_path', type=str,
                    default='/mnt/fs2/2019/Takamuro/m2_research/weather_transferV2/cp/classifier/cls_res101-1119_I2W-train/resnet101_epoch10_step40777.pt')
# parser.add_argument('--classifer_path', type=str, default='cp/classifier/res_aug_5_cls/resnet101_95.pt')
parser.add_argument('--input_size', type=int, default=224)
parser.add_argument('--batch_size', type=int, default=10)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--num_classes', type=int, default=5)
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

import torch
import torchvision.transforms as transforms
import torch.nn as nn

sys.path.append(os.getcwd())
from dataset import ClassImageLoader
# from sampler import ImbalancedDatasetSampler


if __name__ == '__main__':
    save_path = os.path.join('/mnt/fs2/2019/Takamuro/m2_research/weather_transferV2/results/eval_classifier',
                        args.classifier_path.split('/')[-2],
                        args.classifier_path.split('/')[-1].split('_')[-2])
    os.makedirs(save_path, exist_ok=True)
    sep_data = pd.read_pickle(args.pkl_path)
    sep_data = sep_data['test']
    print('loaded {} data'.format(len(sep_data)))

    transform = transforms.Compose([
        transforms.Resize((args.input_size,)*2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    dataset = ClassImageLoader(paths=sep_data, transform=transform, inf=True)

    loader = torch.utils.data.DataLoader(
            dataset,
            # sampler=ImbalancedDatasetSampler(dataset),
            batch_size=args.batch_size,
            drop_last=True,
            num_workers=args.num_workers)

    # load model
    classifer = torch.load(args.classifier_path)
    classifer = nn.Sequential(
                        classifer,
                        nn.Softmax(dim=1)
                    )
    classifer.eval()

    classifer.to('cuda')

    bs = args.batch_size
    s_li = ['sunny', 'cloudy', 'rain', 'snow', 'foggy']

    res_li = []
    path_li = []
    for i, data in tqdm(enumerate(loader)):
        batch = data[0].to('cuda')
        c_batch = data[1].to('cuda')
        path = data[2]
        # print(classifer(batch).detach())
        pred = torch.argmax(classifer(batch).detach(), 1)

        path_li.extend(path)
        res_li.append(torch.cat([pred.int().cpu().view(bs, -1),
                                c_batch.int().cpu().view(bs, -1)], 1))

    all_path = np.array(path_li)
    all_res = torch.cat(res_li, 0).numpy()
    y_true, y_pred = (all_res[:, 1], all_res[:, 0])

    error_paths = all_path[y_true != y_pred]
    error_yt = y_true[y_true != y_pred]
    error_yp = y_pred[y_true != y_pred]
    os.makedirs(os.path.join(save_path, 'error_imgs'), exist_ok=True)
    [shutil.copyfile(_, os.path.join(save_path, 'error_imgs',
                     '{}-ture_{}_{}-pred.jpg'.format(s_li[error_yt[i]], _.split('/')[-1], s_li[error_yp[i]]))) for i, _ in enumerate(error_paths)]

    table = classification_report(y_true, y_pred)

    print(table)

    matrix = confusion_matrix(y_true, y_pred, labels=np.arange(len(s_li)))

    df = pd.DataFrame(data=matrix, index=s_li, columns=s_li)
    df.to_pickle(os.path.join(save_path, 'cm.pkl'))

    plot = sns.heatmap(df, square=True, annot=True, fmt='d')

    fig = plot.get_figure()
    fig.savefig(os.path.join(save_path, 'pr_table.png'))

import argparse
import os
import sys

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from torchvision.utils import save_image

from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=1)
parser.add_argument('--image_root', type=str,
                    default='/mnt/fs2/2018/matsuzaki/dataset_fromnitta/Image/')
parser.add_argument('--pkl_path', type=str,
                    default='/mnt/fs2/2019/Takamuro/m2_research/i2w/sepalated_data.pkl')
parser.add_argument('--cp_path', type=str,
                    default='/mnt/fs2/2019/Takamuro/m2_research/weather_transferV2/cp/transfer/cls/1203_Flickr_cUNet_w-c_res101-1122e15_data-WoPerson_sky-10_L-05_SNdisc_sampler-True_loss_lamda-c1-w1-CE_b1-0.5_b2-0.9_GDratio-8_amp-True_MGpu-False_lr-1.5*0.0001_bs-24_ne-150/cUNet_cls_e0056_s600000.pt')
parser.add_argument('--classifer_path', type=str,
                    default='/mnt/fs2/2019/Takamuro/m2_research/weather_transfer/cp/classifier/cls_res101_i2w_sep-val_aug_20200408/resnet101_epoch15_step59312.pt')
parser.add_argument('--input_size', type=int, default=224)
parser.add_argument('--batch_size', type=int, default=5)
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--num_classes', type=int, default=5)

args = parser.parse_args()

os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset

sys.path.append(os.getcwd())
from dataset import ClassImageLoader
from cunet import Conditional_UNet

if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Resize((args.input_size,)*2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    s_li = ['sunny', 'cloudy', 'rain', 'snow', 'foggy']
    sep_data = pd.read_pickle(args.pkl_path)
    sep_data = sep_data['test']
    # sep_data = [p for p in sep_data if 'foggy' in p]
    print('loaded {} data'.format(len(sep_data)))

    dataset = ClassImageLoader(paths=sep_data, transform=transform, inf=True)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=True
    )
    random_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=True
    )

    # load model
    transfer = Conditional_UNet(num_classes=args.num_classes)
    sd = torch.load(args.cp_path)
    transfer.load_state_dict(sd['inference'])
    transfer.eval()

    classifer = torch.load(args.classifer_path)
    classifer = nn.Sequential(
        classifer,
        nn.Softmax(dim=1)
    )
    classifer.eval()

    transfer.cuda()
    classifer.cuda()

    bs = args.batch_size
    labels = torch.as_tensor(np.arange(args.num_classes, dtype=np.int64))
    onehot = torch.eye(args.num_classes)[labels].to('cuda')

    cls_li = []
    vec_li = []

    cp_name = args.cp_path.split('/')[-1].split('.')[0]
    save_path = './temp_eval_class_transfer'
    #save_path = os.path.join('/mnt/fs2/2019/Takamuro/m2_research/weather_transferV2/results/eval_transfer', 'cls',
    #                         args.cp_path.split('/')[-2],
    #                         cp_name.split('_')[-2] + cp_name.split('_')[-1])
    print(save_path)
    print('If you have done to confirm save_path, please push enter')
    input()
    os.makedirs(save_path, exist_ok=True)

    for data, rnd in tqdm(zip(loader, random_loader), total=len(sep_data) // bs):
        batch = data[0].to('cuda')
        # r_batch = rnd[0].to('cuda')
        # c_batch = rnd[1].to('cuda')
        # r_cls = c_batch
        # c_batch = F.one_hot(c_batch, args.num_classes).float()
        for i in range(bs):
            with torch.no_grad():
                ref_labels_expand = torch.cat([onehot[i]] * bs).view(-1, args.num_classes)
                out = transfer(batch, ref_labels_expand)

                c_preds = torch.argmax(classifer(out).detach(), 1)
                r_cls = torch.argmax(ref_labels_expand, 1)

                cls_li.append(torch.cat([r_cls.int().cpu().view(r_cls.size(0), -1),
                              c_preds.int().cpu().view(c_preds.size(0), -1)], 1))

                # cls_li.append(torch.cat([r_cls.int().cpu().view(r_cls.size(0), -1),
                #                 c_preds.int().cpu().view(c_preds.size(0), -1)], 1))

    all_res = torch.cat(cls_li, 0).numpy()
    y_true, y_pred = (all_res[:, 0], all_res[:, 1])
    table = classification_report(y_true, y_pred)
    print(table)

    matrix = confusion_matrix(y_true, y_pred, labels=np.arange(len(s_li)))
    df = pd.DataFrame(data=matrix, index=s_li, columns=s_li)
    df.to_pickle(os.path.join(save_path, 'cm.pkl'))

    sns.set(font_scale=1.5)
    plot = sns.heatmap(df, square=True, annot=True, annot_kws={'size': 20}, fmt='d', vmax=500)
    fig = plot.get_figure()
    fig.savefig(os.path.join(save_path, 'cm.png'))

import argparse
import sys
import os


import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--image_root', type=str, default='/mnt/fs2/2019/Takamuro/db/CelebA/Img/img_align_celeba')
parser.add_argument('--pkl_path', type=str, default='/mnt/fs2/2019/Takamuro/db/CelebA/Anno/list_attr_celeba_add-mode.pkl')
parser.add_argument('--cp_path', type=str, default='cp/classifier_celeba/1211_CelebA_wideres50_train50780/resnet101_epoch80_step64233.pt')
parser.add_argument('--gpu', type=str, default='2')
parser.add_argument('--input_size', type=int, default=224)
parser.add_argument('--batch_size', type=int, default=20)
args = parser.parse_args()

os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models

sys.path.append(os.getcwd())
from dataset import CelebALoader

if __name__ == '__main__':
    save_path = os.path.join('/mnt/fs2/2019/Takamuro/m2_research/weather_transferV2/results/eval_classifier', 'CelebA',
                             args.cp_path.split('/')[-2],
                             args.cp_path.split('/')[-1].split('_')[-2])
    print(save_path)
    os.makedirs(save_path, exist_ok=True)

    df = pd.read_pickle(args.pkl_path)
    df = df[df['mode'] == 'test']
    print('{} train data were loaded'.format(len(df)))

    # torch >= 1.7
    transform = nn.Sequential(
        transforms.Resize((args.input_size,) * 2),
        # torch >= 1.7
        transforms.ConvertImageDtype(torch.float32),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    )

    dataset = CelebALoader(root_path=args.image_root, df=df, transform=transform)

    loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            drop_last=True,
            num_workers=8)

    num_classes = dataset.num_classes
    cols = dataset.cols

    model = torch.load(args.cp_path)
    model.eval()
    model.to('cuda')

    acc_li = []

    for i, data in enumerate(tqdm(loader), start=0):
        inputs, labels = (d.to('cuda') for d in data)
        # torch >= 1.7
        inputs = dataset.transform(inputs)

        outputs = model(inputs).detach()
        outputs = torch.sigmoid(outputs)
        result = outputs > 0.5
        correct = (result == labels).sum(dim=0)
        acc = correct / args.batch_size
        acc_ = acc.to('cpu').clone().numpy()
        acc_li.append(acc_)

    result = np.mean(acc_li, axis=0)

    fig = plt.figure()
    plt.bar(np.arange(num_classes), result, width=0.5, tick_label=cols, align='center')
    plt.xticks(rotation=90)
    fig.savefig(os.path.join(save_path, 'each_class_acc.jpg'))
    print('Done: training')

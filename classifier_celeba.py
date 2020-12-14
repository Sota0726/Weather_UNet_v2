import argparse
import pickle
import os

import numpy as np
from tqdm import trange
from collections import OrderedDict
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--image_root', type=str, default='/mnt/fs2/2019/Takamuro/db/CelebA/Img/img_align_celeba')
parser.add_argument('--pkl_path', type=str, default='/mnt/fs2/2019/Takamuro/db/CelebA/Anno/list_attr_celeba_add-mode.pkl')
parser.add_argument('--name', type=str, default='clelba_classifier')
parser.add_argument('--save_path', type=str, default='cp/classifier_celeba')
parser.add_argument('--gpu', type=str, default='2')
parser.add_argument('--input_size', type=int, default=224)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--num_epoch', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--mode', type=str, default='T')
parser.add_argument('--pre_trained', action='store_true')
parser.add_argument('--model', type=str, default='mobilenet')
parser.add_argument('--amp', action='store_true')
args = parser.parse_args()

os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models

if args.amp:
    from apex import amp, optimizers

from torch.utils.tensorboard import SummaryWriter

from dataset import CelebALoader
from sampler import ImbalancedDatasetSampler

save_dir = os.path.join(args.save_path, args.name)
os.makedirs(save_dir, exist_ok=True)


def validation(model, validloader):
    model.eval()
    validation_loss = 0.0
    correct = 0
    for data, target in validloader:
        data = data.to('cuda')
        target = target.to('cuda')

        data = test_set.transform(data)
        output = model(data)
        validation_loss += criterion(output, target).item()

        output = torch.sigmoid(output)
        result = output > 0.5
        correct += (result == target).sum().item()

    validation_loss /= len(validloader)
    acc = correct / (len(validloader) * num_classes * args.batch_size)

    return acc, validation_loss


if __name__ == '__main__':
    df = pd.read_pickle(args.pkl_path)
    if args.mode == 'T':
        df_sep = {'train': df[df['mode'] == 'train'].iloc[:50780],
                  'test': df[df['mode'] == 'test']}
    elif args.mode == 'E':  # for evaluation
        df_sep = {'train': df[df['mode'] == 'val'],
                  'test': df[df['mode'] == 'test']}
    print('{} train data were loaded'.format(len(df_sep['train'])))

    # torch >= 1.7
    train_transform = nn.Sequential(
        transforms.RandomRotation(10),
        transforms.RandomResizedCrop(args.input_size),
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
        transforms.Resize((args.input_size,) * 2),
        # torch >= 1.7
        transforms.ConvertImageDtype(torch.float32),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    )

    transform = {'train': train_transform, 'test': test_transform}

    loader = lambda s: CelebALoader(root_path=args.image_root, df=df_sep[s], transform=transform[s])

    train_set = loader('train')
    test_set = loader('test')

    train_loader = torch.utils.data.DataLoader(
            train_set,
            shuffle=True,
            batch_size=args.batch_size,
            drop_last=True,
            num_workers=8)

    test_loader = torch.utils.data.DataLoader(
            test_set,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=8)

    num_classes = train_set.num_classes

    # modify exist resnet101 model
    if args.model == 'resnet':
        model = models.resnet101(pretrained=args.pre_trained)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes, bias=True)
    elif args.model == 'mobilenet':
        model = models.mobilenet_v2(pretrained=args.pre_trained)
        num_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_features, num_classes, bias=True)
    elif args.model == 'resnext':
        model = models.resnext50_32x4d(pretrained=args.pre_trained)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes, bias=True)
    elif args.model == 'wideresnet':
        model = models.wide_resnet50_2(pretrained=args.pre_trained)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes, bias=True)

    model.to('cuda')

    # train setting
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    # criterion = nn.BCELoss()
    criterion = nn.BCEWithLogitsLoss()
    global_step = 0
    eval_per_iter = 250
    save_per_epoch = 5
    tqdm_iter = trange(args.num_epoch, desc='Training', leave=True)

    comment = '_lr-{}_bs-{}_ne-{}_x{}_name-{}_pre-train-{}_amp-{}'.format(args.lr, args.batch_size, args.num_epoch, args.input_size, args.name, args.pre_trained, args.amp)
    writer = SummaryWriter(comment=comment)

    if args.amp:
        model, opt = amp.initialize(model, opt, opt_level='O1')

    loss_li = []
    prec_li = []
    acc_li = []
    for epoch in tqdm_iter:

        for i, data in enumerate(train_loader, start=0):
            tqdm_iter.set_description('Training [ {} step ]'.format(global_step))
            inputs, labels = (d.to('cuda') for d in data)
            # torch >= 1.7
            inputs = train_set.transform(inputs)
            opt.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss_li.append(loss.item())

            outputs = torch.sigmoid(outputs)
            result = outputs > 0.5
            correct = (result == labels).sum().item()
            acc = correct / (num_classes * args.batch_size)
            acc_li.append(acc)

            if args.amp:
                with amp.scale_loss(loss, opt) as scale_loss:
                    scale_loss.backward()
            else:
                loss.backward()

            opt.step()

            if global_step % eval_per_iter == 0:
                train_loss = np.mean(loss_li)
                train_acc = np.mean(acc_li)

                val_acc, val_loss = validation(model, test_loader)

                tqdm_iter.set_postfix(OrderedDict(train_loss=train_loss, test_loss=val_loss, train_acc=train_acc, test_acc=val_acc))

                writer.add_scalars('loss', {
                    'train': train_loss,
                    'test': val_loss
                    }, global_step)
                writer.add_scalars('acc', {
                    'train': train_acc,
                    'test': val_acc
                    }, global_step)
                loss_li = []
                prec_li = []
                acc_li = []

            global_step += 1

        if epoch % save_per_epoch == 0:
            out_path = os.path.join(save_dir, '{}_epoch'.format(args.model) + str(epoch) + '_step' + str(global_step) + '.pt')
            torch.save(model, out_path)

    print('Done: training')

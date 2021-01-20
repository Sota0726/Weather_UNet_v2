import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from sampler import ImbalancedDatasetSampler, TimeImbalancedDatasetSampler
# import cupy as cp

xp = np  # cpu or gpu
# if os.environ.get('CUDA_VISIBLE_DEVICES') is not None: xp = cp


def soft_transform(x, std=0.05):
    dist = torch.zeros_like(x).normal_(0, std=std)
    return x + dist


def adv_loss(a, b):
    assert a.size() == b.size(), 'The size of a and b is different.{}!={}'.format(a.size(), b.size())
    return F.mse_loss(a, b)


def l1_loss(a, b):
    assert a.size() == b.size(), 'The size of a and b is different.{}!={}'.format(a.size(), b.size())
    return F.l1_loss(a, b)


def feat_loss(a, b):
    return torch.mean(torch.stack([F.l1_loss(a_, b_) for a_, b_ in zip(a, b)]))


def uvIndex_loss(preds, labels):
    criterion = nn.MSELoss()
    loss = criterion(preds[:, 1] - labels[:, 1])
    return loss


def pred_loss(preds, labels, l_type='mse', weight=[1,1,1,1,1,1,1,1]):
    if l_type == 'CE':
        criterion = nn.CrossEntropyLoss()
        # one-hot to 0~4 label
        labels_ = torch.argmax(labels, dim=1)
        loss = criterion(preds, labels_)
    elif l_type == 'L1':
        criterion = nn.L1Loss()
        loss = criterion(preds, labels)
    elif l_type == 'weightedMSE':
        loss = torch.sum(weight * (preds - labels) ** 2)
        return loss
    elif l_type == 'BCE':
        criterion = nn.BCEWithLogitsLoss()
        loss = criterion(preds, labels)
    else:
        criterion = nn.MSELoss()
        loss = criterion(preds, labels)
    return loss


def dis_hinge(dis_fake, dis_real):
    loss = torch.mean(torch.relu(1. - dis_real)) +\
        torch.mean(torch.relu(1. + dis_fake))
    return loss


def gen_hinge(dis_fake):
    return torch.mean(-dis_fake)


def seq_loss(fake_seq, real_seq, seq_len):
    loss = []
    fake_seq = fake_seq.reshape(real_seq.size())
    bs = real_seq.size(0)
    loss = [(fake_seq[i, j] - fake_seq[i, j+1]) - (real_seq[i, j] - real_seq[i, j+1])
            for i in range(bs) for j in range(seq_len - 1)]
    return torch.mean(torch.cat(loss))


def vector_to_one_hot(vec):
    arg = torch.argmax(vec, 0, keepdim=True)
    one_hot = torch.zeros_like(vec)
    one_hot.scatter_(0, arg, 1).float()
    return one_hot


def get_rand_labels(num_classes, batch_size, one_hot=False):
    label = torch.FloatTensor(batch_size, num_classes).uniform_(-1, 1)
    if one_hot:
        label = F.one_hot(label, num_classes)
    return label.to('cuda')


def get_sequential_labels(num_classes, batch_size, one_hot=False):
    rep = batch_size // num_classes + 1
    if one_hot:
        arr = xp.eye(num_classes, dtype=xp.float32)
        arr = xp.tile(arr, (rep, 1))[:batch_size]
        return torch.from_numpy(arr).float().to('cuda')
    else:
        arr = torch.arange(num_classes, dtype=torch.float32)
        arr = arr.repeat(rep)[:batch_size]
        return arr.to('cuda')


def Variable_Float(x, batch_size):
    return Variable(torch.cuda.FloatTensor(batch_size, 1).fill_(x), requires_grad=False)


def make_table_img(images, ref_images, results):
    blank = torch.zeros_like(images[0]).unsqueeze(0)
    ref_img = torch.cat([blank] + list(torch.split(ref_images, 1)), dim=3)
    in_out_img = torch.cat([images] + results, dim=2)
    res_img = torch.cat([ref_img, in_out_img], dim=0)

    return res_img


def make_dataloader(dataset, args, mode='train'):
    if mode == 'test':
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=dataset.bs,
            num_workers=args.num_workers,
            pin_memory=True)
        return loader

    if args.sampler == 'none':
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=dataset.bs,
            shuffle=True,
            drop_last=True,
            num_workers=args.num_workers,
            pin_memory=True)
    elif args.sampler == 'time':
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=dataset.bs,
            sampler=TimeImbalancedDatasetSampler(dataset),
            drop_last=True,
            num_workers=args.num_workers,
            pin_memory=True)
    elif args.sampler == 'class':
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=dataset.bs,
            sampler=ImbalancedDatasetSampler(dataset),
            drop_last=True,
            num_workers=args.num_workers,
            pin_memory=True)
    else:
        print('{} is invalid sampler name'.format(args.sampler))
        exit()

    return loader


def make_network(args, num_classes, name):
    from cunet import Conditional_UNet, Conditional_UNet_V2
    from disc import SNDisc, SNDisc_, SNResNet64ProjectionDiscriminator, SNResNetProjectionDiscriminator
    from glob import glob

    if args.generator == 'cUNet':
        inference = Conditional_UNet(num_classes=num_classes)
    elif args.generator == 'cUNetV2':
        inference = Conditional_UNet_V2(num_classes=num_classes)
    else:
        print('{} is invalid generator'.format(args.generator))
        exit()

    if args.disc == 'SNDisc':
        discriminator = SNDisc(num_classes=num_classes)
    elif args.disc == 'SNDiscV2':
        discriminator = SNDisc_(num_classes=num_classes)
    elif args.disc == 'SNRes64':
        discriminator = SNResNet64ProjectionDiscriminator(num_classes=num_classes)
    elif args.disc == 'SNRes':
        discriminator = SNResNetProjectionDiscriminator(num_classes=num_classes)
    else:
        print('{} is invalid discriminator'.format(args.disc))
        exit()

    if args.resume_cp:
        exist_cp = [args.resume_cp]
    else:
        exist_cp = sorted(glob(os.path.join(args.save_dir, name, '*')))

    if len(exist_cp) != 0:
        print('Load checkpoint:{}'.format(exist_cp[-1]))
        sd = torch.load(exist_cp[-1])
        inference.load_state_dict(sd['inference'])
        discriminator.load_state_dict(sd['discriminator'])
        epoch = sd['epoch']
        global_step = sd['global_step']
        print('Success checkpoint loading!')
    else:
        print('Initialize training status.')
        epoch = 0
        global_step = 0

    estimator = torch.load(args.estimator_path)

    return inference, discriminator, estimator, epoch, global_step

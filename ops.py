import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
# import cupy as cp


xp = np # cpu or gpu
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
    return torch.mean(torch.stack([F.l1_loss(a_, b_) for a_,b_ in zip(a, b)]))


def pred_loss(preds, labels, cls=None):
    if cls == 'CE':
        criterion = nn.CrossEntropyLoss()
        # one-hot to 0~4 label
        labels_ = torch.argmax(labels, dim=1)
        loss = criterion(preds, labels_)
    elif cls == 'mse':
        criterion = nn.MSELoss()
        loss = criterion(preds, labels)
    else:
        print('set weather loss name')
        exit()
    return loss


def dis_hinge(dis_fake, dis_real):
    loss = torch.mean(torch.relu(1. - dis_real)) +\
        torch.mean(torch.relu(1. + dis_fake))
    return loss


def gen_hinge(dis_fake):
    return torch.mean(-dis_fake)


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

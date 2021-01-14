import argparse
import os
from args import get_args
args = get_args()

# GPU Setting
os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import trange
from collections import OrderedDict
from glob import glob

import torch
import torch.nn as nn
import torchvision as tv
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image, make_grid

if args.amp:
    from apex import amp, optimizers

from ops import *
from dataset import ImageLoader, SequenceFlickrDataLoader


class WeatherTransfer(object):

    def __init__(self, args):

        self.args = args
        self.batch_size = args.batch_size
        self.global_step = 0

        self.name = 'Seq_{}_D-{}_sampler-{}_GDratio{}_adam-b1{}-b2{}_ep{}'.format(
            args.name,
            args.disc,
            args.sampler,
            '1-' + str(args.GD_train_ratio),
            args.adam_beta1,
            args.adam_beta2,
            args.epsilon
        )

        comment = '_lr-{}_bs-{}_ne-{}'.format(args.lr, args.batch_size, args.num_epoch)

        self.writer = SummaryWriter(comment=comment + '_name-' + self.name)
        self.name = self.name + comment
        os.makedirs(os.path.join(args.save_dir, self.name), exist_ok=True)

        # Consts
        self.real = Variable_Float(1., self.batch_size)
        self.fake = Variable_Float(0., self.batch_size)
        self.lmda = 0.
        self.args.lr = args.lr * (args.batch_size / 16)

        # torch >= 1.7
        train_transform = nn.Sequential(
            transforms.RandomRotation(10),
            # transforms.RandomResizedCrop(args.input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.5,
                contrast=0.3,
                saturation=0.3,
                hue=0
            ),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        )

        # torch >= 1.7
        test_transform = nn.Sequential(
            # transforms.Resize((args.input_size,) * 2),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        )

        self.cols = ['tempC', 'uvIndex', 'visibility', 'windspeedKmph', 'cloudcover', 'humidity', 'pressure', 'DewPointC']
        self.num_classes = len(self.cols)

        self.transform = {'train': train_transform, 'test': test_transform}
        self.train_set, self.test_set = self.load_data(varbose=True)

        self.build()

    def load_data(self, varbose=False):

        print('Start loading image files...')
        df = pd.read_pickle(args.pkl_path)
        # --- normalize --- #
        df_ = df.loc[:, self.cols].fillna(0)
        df_mean = df_.mean()
        df_std = df_.std()

        df.loc[:, self.cols] = (df.loc[:, self.cols].fillna(0) - df_mean) / df_std
        # ------------------ #
        # --- memory min - max value --- #
        self.sig_max = torch.from_numpy(np.array([np.max(df[col].values) for col in self.cols])).to('cuda')
        self.sig_min = torch.from_numpy(np.array([np.min(df[col].values) for col in self.cols])).to('cuda')
        # ------------------------------ #

        print('loaded {} signals data'.format(len(df)))
        df_shuffle = df.sample(frac=1)
        # df_sep = {'train': df_shuffle[df_shuffle['mode'] == 't_train'],
        #           'test': df_shuffle[df_shuffle['mode'] == 'test']}
        df_sep = {
            'train': df_shuffle[(df_shuffle['mode'] == 't_train') | (df_shuffle['mode'] == 'train')],
            'test': df_shuffle[df_shuffle['mode'] == 'test']}
        del df, df_shuffle
        loader = lambda s: SequenceFlickrDataLoader(args.image_root, args.csv_root, df_sep[s], self.cols, df_mean, df_std, transform=self.transform[s])

        train_set = loader('train')
        test_set = loader('test')
        print('train:{} test:{} sets have already loaded.'.format(len(train_set), len(test_set)))
        return train_set, test_set

    def build(self):
        args = self.args

        # Models
        print('Build Models...')
        self.inference, self.discriminator, self.estimator, self.epoch, self.global_step = make_network(args, self.num_classes, self.name)
        self.estimator.eval()

        # Models to CUDA
        [i.to('cuda') for i in [self.inference, self.discriminator, self.estimator]]

        # Optimizer
        self.g_opt = torch.optim.Adam(self.inference.parameters(), lr=args.lr, betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.lr/20)
        self.d_opt = torch.optim.Adam(self.discriminator.parameters(), lr=args.lr, betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.lr/20)

        # apex
        if args.amp:
            self.inference, self.g_opt = amp.initialize(self.inference, self.g_opt, opt_level='O1')
            self.discriminator, self.d_opt = amp.initialize(self.discriminator, self.d_opt, opt_level='O1')

        # multi gpu
        if args.multi_gpu and torch.cuda.device_count() > 1:
            self.inference = nn.DataParallel(self.inference)
            self.discriminator = nn.DataParallel(self.discriminator)
            self.estimator = nn.DataParallel(self.estimator)

        self.train_loader = make_dataloader(self.train_set, args)
        self.test_loader = make_dataloader(self.test_set, args)

        test_data_iter = iter(self.test_loader)
        # torch >= 1.7
        self.test_random_sample = []
        for i in range(1):
            img, label, seq = test_data_iter.next()
            img = self.test_set.transform(img.to('cuda'))
            self.test_random_sample.append((img, label.to('cuda'), seq.to('cuda')))
        del test_data_iter, self.test_loader

        self.scalar_dict = {}
        self.image_dict = {}
        self.shift_lmda = lambda a, b: (1. - self.lmda) * a + self.lmda * b
        print('Build has been completed.')

    def update_inference(self, images, labels, seq_labels):
        # --- UPDATE(Inference) --- #
        self.g_opt.zero_grad()
        seq_len = self.train_set.seq_len
        seq_labels_ = seq_labels.view(-1, self.num_classes)
        # for real
        pred_labels = self.estimator(images).detach()
        pred_labels = torch.minimum(self.sig_max, pred_labels)
        pred_labels = torch.maximum(self.sig_min, pred_labels)
        pred_labels = pred_labels.float()

        images = torch.cat([torch.cat([image.unsqueeze(0)] * seq_len, dim=0) for image in images], dim=0)
        fake_out = self.inference(images, seq_labels_)
        fake_c_out = self.estimator(fake_out)
        fake_d_out = self.discriminator(fake_out, seq_labels_)

        # Calc Generator Loss
        g_loss_adv = gen_hinge(fake_d_out)  # Adversarial loss
        g_loss_l1 = l1_loss(fake_out, images)  # L1 Loss
        g_loss_w = pred_loss(fake_c_out, seq_labels_,
                             l_type=self.args.wloss_type, weight=self.args.wloss_weight)   # Weather prediction

        # sequence loss
        g_loss_seq = seq_loss(fake_c_out, seq_labels, seq_len)
        # reconstruction loss
        diff = torch.mean(torch.abs(fake_out - images), [1, 2, 3])
        lmda = torch.mean(torch.abs(torch.transpose(torch.stack([pred_labels] * seq_len), 1, 0) - seq_labels), 2)
        loss_con = torch.mean(diff / (lmda.reshape(-1) + self.args.epsilon))  # Reconstraction loss

        lmda_con, lmda_w, lmda_seq = (1, 1, 1.5)

        g_loss = g_loss_adv + lmda_con * loss_con + lmda_w * g_loss_w + lmda_seq * g_loss_seq

        if self.args.amp:
            with amp.scale_loss(g_loss, self.g_opt) as scale_loss:
                scale_loss.backward()
        else:
            g_loss.backward()
        self.g_opt.step()

        self.scalar_dict.update({
            'losses/g_loss/train': g_loss.item(),
            'losses/g_loss_adv/train': g_loss_adv.item(),
            'losses/g_loss_l1/train': g_loss_l1.item(),
            'losses/g_loss_w/train': g_loss_w.item(),
            'losses/loss_con/train': loss_con.item(),
            'losses/loss_seq/train': g_loss_seq.item(),
            'variables/lmda': self.lmda
        })

        self.image_dict.update({
            'io/train': [images.to('cpu'), fake_out.detach().to('cpu')]
        })
        return g_loss_adv.item(), loss_con.item(), g_loss_w.item(), g_loss_seq.item()

    def update_discriminator(self, images, labels, seq_labels):

        # --- UPDATE(Discriminator) ---#
        self.d_opt.zero_grad()
        seq_len = self.train_set.seq_len
        seq_labels_ = seq_labels.view(-1, self.num_classes)
        # for real
        pred_labels = self.estimator(images).detach()
        pred_labels = torch.minimum(self.sig_max, pred_labels)
        pred_labels = torch.maximum(self.sig_min, pred_labels)
        pred_labels = pred_labels.float()
        real_d_out_pred = self.discriminator(images, pred_labels)[0]
        # for fake
        images = torch.cat([torch.cat([image.unsqueeze(0)] * seq_len, dim=0) for image in images], dim=0)
        fake_out_ = self.inference(images, seq_labels_)
        fake_d_out_ = self.discriminator(fake_out_, seq_labels_)

        d_loss = dis_hinge(fake_d_out_, real_d_out_pred)

        if self.args.amp:
            with amp.scale_loss(d_loss, self.d_opt) as scale_loss:
                scale_loss.backward()
        else:
            d_loss.backward()
        self.d_opt.step()

        self.scalar_dict.update({
            'losses/d_loss/train': d_loss.item()
        })

        return d_loss.item()

    def evaluation(self):
        images, labels, seq_labels = self.test_random_sample[0]
        seq_len = self.train_set.seq_len
        seq_labels_ = seq_labels.view(-1, self.num_classes)

        images = torch.cat([torch.cat([image.unsqueeze(0)] * seq_len, dim=0) for image in images], dim=0)
        with torch.no_grad():
            fake_out = self.inference(images, seq_labels_)
            fake_c_out = self.estimator(fake_out)
            fake_d_out = self.discriminator(fake_out, seq_labels_)

        g_loss_adv = gen_hinge(fake_d_out)  # Adversarial loss
        g_loss_l1 = l1_loss(fake_out, images)  # L1 Loss
        g_loss_w = pred_loss(fake_c_out, seq_labels_,
                             l_type=self.args.wloss_type, weight=self.args.wloss_weight)   # Weather prediction

        # sequence loss
        g_loss_seq = seq_loss(fake_c_out, seq_labels, seq_len)
        # reconstruction loss
        diff = torch.mean(torch.abs(fake_out - images), [1, 2, 3])
        lmda = torch.mean(torch.abs(torch.transpose(torch.stack([labels] * seq_len), 1, 0) - seq_labels), 2)
        loss_con = torch.mean(diff / (lmda.reshape(-1) + self.args.epsilon))  # Reconstraction loss
        # discriminator loss
        with torch.no_grad():
            real_d_out = self.discriminator(images[::seq_len], labels)[0]
        d_loss = dis_hinge(fake_d_out, real_d_out)

        # --- WRITING SUMMARY ---#
        self.scalar_dict.update({
            'losses/g_loss_adv/test': g_loss_adv.item(),
            'losses/g_loss_l1/test': g_loss_l1.item(),
            'losses/g_loss_w/test': g_loss_w.item(),
            'losses/loss_con/test': loss_con.item(),
            'losses/loss_seq/test': g_loss_seq.item(),
            'losses/d_loss/test': d_loss.item()
        })

        self.image_dict.update({
            'io/test': [images.to('cpu'), fake_out.detach().to('cpu')]
        })

    def update_summary(self):
        # Summarize
        seq_len = self.train_set.seq_len
        for k, v in self.scalar_dict.items():
            spk = k.rsplit('/', 1)
            self.writer.add_scalars(spk[0], {spk[1]: v}, self.global_step)
        for k, v in self.image_dict.items():
            images, fake_out = v
            fake_out_row = make_grid(fake_out, nrow=seq_len, normalize=True, scale_each=True)
            images_row = make_grid(images[::seq_len], nrow=1, normalize=True, scale_each=True)
            out = torch.cat([images_row, fake_out_row], dim=2)
            self.writer.add_image(k, out, self.global_step)

    def train(self):
        args = self.args

        # train setting
        eval_per_step = 1000 * args.GD_train_ratio
        display_per_step = 1000 * args.GD_train_ratio

        self.all_step = args.num_epoch * len(self.train_set) // self.batch_size

        tqdm_iter = trange(args.num_epoch, desc='Training', leave=True)
        for epoch in tqdm_iter:
            if epoch > 0:
                self.epoch += 1

            for i, data in enumerate(self.train_loader):
                self.global_step += 1

                if self.global_step % eval_per_step == 0:
                    out_path = os.path.join(args.save_dir, self.name, ('cUNet_est' + '_e{:04d}_s{}.pt').format(self.epoch, self.global_step))
                    if args.multi_gpu and torch.cuda.device_count() > 1:
                        state_dict = {
                            'inference': self.inference.module.state_dict(),
                            'discriminator': self.discriminator.module.state_dict(),
                            'epoch': self.epoch,
                            'global_step': self.global_step
                        }
                    else:
                        state_dict = {
                            'inference': self.inference.state_dict(),
                            'discriminator': self.discriminator.state_dict(),
                            'epoch': self.epoch,
                            'global_step': self.global_step
                        }
                    torch.save(state_dict, out_path)

                tqdm_iter.set_description('Training [ {} step ]'.format(self.global_step))
                if args.lmda:
                    self.lmda = args.lmda
                else:
                    self.lmda = self.global_step / self.all_step

                images, con, seq = (d.to('cuda') for d in data)
                # torch >= 1.7
                images = self.train_set.transform(images)

                if images.size(0) != self.batch_size:
                    continue

                # --- TRAINING --- #
                if (self.global_step - 1) % args.GD_train_ratio == 0:
                    g_loss, r_loss, w_loss, seq_loss = self.update_inference(images, con, seq)
                d_loss = self.update_discriminator(images, con, seq)
                tqdm_iter.set_postfix(OrderedDict(
                    d_loss=d_loss,
                    g_loss=g_loss,
                    r_loss=r_loss,
                    w_loss=w_loss,
                    seq_loss=seq_loss
                ))

                # --- EVALUATION ---#
                if (self.global_step % eval_per_step == 0):
                    self.evaluation()

                # --- UPDATE SUMMARY ---#
                if self.global_step % display_per_step == 0:
                    self.update_summary()
        print('Done: training')


if __name__ == '__main__':
    wt = WeatherTransfer(args)
    wt.train()

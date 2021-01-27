import argparse
import os
from args import get_args
import itertools
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
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image, make_grid
import torch.utils.data.distributed

if args.amp:
    from apex import amp, optimizers

from ops import *
from dataset import SequenceFlickrDataLoader, TimeLapseLoader


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
        self.sig_step_num = 10

        # torch >= 1.7
        train_transform = nn.Sequential(
            transforms.RandomRotation(10),
            # transforms.RandomResizedCrop(args.input_size),
            transforms.RandomHorizontalFlip(),
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

        self.transform = {'train': train_transform, 'seq': train_transform, 'test': test_transform}
        self.train_set, self.seq_set, self.test_set = self.load_data(varbose=True)

        self.build()

    def load_data(self, varbose=False):

        print('Start loading image files...')
        df = pd.read_pickle(args.pkl_path)
        df_test = pd.read_pickle(args.test_pkl_path)
        # --- normalize --- #
        df_ = df.loc[:, self.cols].fillna(0)
        df_mean = df_.mean()
        df_std = df_.std()

        df.loc[:, self.cols] = (df.loc[:, self.cols].fillna(0) - df_mean) / df_std
        df_test.loc[:, self.cols] = (df_test.loc[:, self.cols].fillna(0) - df_mean) / df_std
        # ------------------ #
        # --- memory min - max value --- #
        self.sig_max = np.array([np.max(df[col].values) for col in self.cols])
        self.sig_min = np.array([np.min(df[col].values) for col in self.cols])
        self.sig_list = np.linspace(self.sig_min, self.sig_max, self.sig_step_num).T
        self.sig_max, self.sig_min, self.sig_list = [torch.from_numpy(_).float().to('cuda') for _ in [self.sig_max, self.sig_min, self.sig_list]]
        # ------------------------------ #

        print('loaded {} signals data'.format(len(df)))
        df_shuffle = df.sample(frac=1)
        # df_sep = {'train': df_shuffle[df_shuffle['mode'] == 't_train'],
        #           'test': df_shuffle[df_shuffle['mode'] == 'test']}
        df_sep = {
            'train': df_shuffle[(df_shuffle['mode'] == 't_train') | (df_shuffle['mode'] == 'train')],
            'test': df_test}
        del df, df_shuffle, df_test

        train_set = SequenceFlickrDataLoader(args.image_root, args.csv_root, df_sep['train'], self.cols, df_mean, df_std,
                                             bs=args.batch_size, transform=self.transform['train'])
        test_set = SequenceFlickrDataLoader(args.image_root, args.csv_root, df_sep['test'], self.cols, df_mean, df_std,
                                            bs=5, transform=self.transform['test'], mode='test')
        seq_set = TimeLapseLoader(args.vid_root, bs=args.batch_size, transform=self.transform['train'])
        print('train:{} test:{} sets have already loaded.'.format(len(train_set), len(test_set)))
        return train_set, seq_set, test_set

    def build(self):
        args = self.args

        # Models
        print('Build Models...')
        self.inference, self.discriminator, self.seq_disc, self.estimator, self.epoch, self.global_step = make_seq_network(args, self.num_classes, self.name)
        self.estimator.eval()

        # Models to CUDA
        [i.to('cuda') for i in [self.inference, self.discriminator, self.seq_disc, self.estimator]]

        # Optimizer
        self.g_opt = torch.optim.Adam(self.inference.parameters(), lr=args.lr, betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.lr/20)
        self.d_opt = torch.optim.Adam(self.discriminator.parameters(), lr=args.lr, betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.lr/20)
        self.seq_d_opt = torch.optim.Adam(self.seq_disc.parameters(), lr=args.lr, betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.lr/20)

        # apex
        if args.amp:
            self.inference, self.g_opt = amp.initialize(self.inference, self.g_opt, opt_level='O1')
            self.discriminator, self.d_opt = amp.initialize(self.discriminator, self.d_opt, opt_level='O1')
            self.seq_disc, self.seq_d_opt = amp.initialize(self.seq_disc, self.seq_d_opt, opt_level='O1')

        # multi gpu
        if args.multi_gpu and torch.cuda.device_count() > 1:
            self.inference = nn.DataParallel(self.inference)
            self.discriminator = nn.DataParallel(self.discriminator)
            self.seq_disc = nn.DataParallel(self.seq_disc)
            self.estimator = nn.DataParallel(self.estimator)

        self.train_set.transform = self.train_set.transform.to('cuda')
        self.test_set.transform = self.test_set.transform.to('cuda')

        args.distributed = False
        self.train_loader = make_dataloader(self.train_set, args)
        args.sampler = 'none'
        seq_loader = make_dataloader(self.seq_set, args)
        seq_loader = iter(seq_loader)
        self.seq_test_samples = seq_loader.__next__()
        self.seq_loader = itertools.cycle(iter(seq_loader))
        self.test_loader = make_dataloader(self.test_set, args, mode='test')

        self.seq_test_samples = torch.stack([
            self.seq_set.transform(self.seq_test_samples[i].to('cuda', non_blocking=True))
            for i in range(self.batch_size)], dim=0)
        self.seq_test_samples = torch.transpose(self.seq_test_samples, 1, 2)

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
        pred_labels = self.estimator(images[::seq_len]).detach()
        pred_labels = torch.minimum(self.sig_max, pred_labels)
        pred_labels = torch.maximum(self.sig_min, pred_labels)
        pred_labels = pred_labels.float()

        fake_out = self.inference(images, seq_labels_)
        fake_c_out = self.estimator(fake_out)
        fake_d_out = self.discriminator(fake_out, seq_labels_)[0]
        # (bs, c , seq_len, w, h)
        fake_seq_out = fake_out.view(self.batch_size, -1, 3, self.args.input_size, self.args.input_size).clone()
        # (bs, c, seq_len, w, h)
        fake_seq_out = torch.transpose(fake_seq_out, 1, 2)
        fake_seq_d_out = self.seq_disc(fake_seq_out)

        ### -- Calc Generator Loss --- ###
        # Adversarial loss
        g_loss_adv = gen_hinge(fake_d_out)
        g_loss_seq_adv = gen_hinge(fake_seq_d_out)
        # L1 Loss
        g_loss_l1 = l1_loss(fake_out, images)
        # Weather prediction
        g_loss_w = pred_loss(fake_c_out, seq_labels_,
                             l_type=self.args.wloss_type, weight=self.args.wloss_weight)

        # Sequence loss
        g_loss_seq = seq_loss(fake_c_out, seq_labels, seq_len)
        # Reconstruction loss
        weight = torch.tensor([1, 1, 0.3, 1, 1, 1, 1, 1]).to('cuda')
        diff = torch.mean(torch.abs(fake_out - images), [1, 2, 3])
        pred_labels = weight * pred_labels
        lmda = torch.mean(torch.abs(torch.transpose(torch.stack([pred_labels] * seq_len), 1, 0) - seq_labels), 2)
        loss_con = torch.mean(diff / (lmda.reshape(-1) + self.args.epsilon))

        lmda_con, lmda_w, lmda_seq = (1.3, 1, 1)

        g_loss = g_loss_adv + g_loss_seq_adv + lmda_con * loss_con + lmda_w * g_loss_w + lmda_seq * g_loss_seq
        ### ------ ###

        if self.args.amp:
            with amp.scale_loss(g_loss, self.g_opt) as scale_loss:
                scale_loss.backward()
        else:
            g_loss.backward()
        self.g_opt.step()

        self.image_dict.update({
            'io/train': [images.to('cpu'), fake_out.detach().to('cpu')]
        })
        return g_loss.item(), g_loss_adv.item(), g_loss_seq_adv.item(), loss_con.item(), g_loss_w.item(), g_loss_seq.item(), g_loss_l1.item()

    def update_discriminator(self, images, labels, seq_labels, sequence):

        # --- UPDATE(Discriminator) ---#
        self.d_opt.zero_grad()
        self.seq_d_opt.zero_grad()
        seq_len = self.train_set.seq_len
        seq_labels_ = seq_labels.view(-1, self.num_classes)
        # for real
        pred_labels = self.estimator(images[::seq_len]).detach()
        pred_labels = torch.minimum(self.sig_max, pred_labels)
        pred_labels = torch.maximum(self.sig_min, pred_labels)
        pred_labels = pred_labels.float()
        real_d_out_pred = self.discriminator(images[::seq_len], pred_labels)[0]

        # for fake
        fake_seq_out = self.inference(images, seq_labels_).detach()
        fake_d_out_ = self.discriminator(fake_seq_out, seq_labels_)[0]

        # for sequence
        real_seq_d_out = self.seq_disc(sequence)
        # (bs* seq_len, c, w, h) -> (bs, seq_len, c, w, h)
        fake_seq_out = fake_seq_out.view(self.batch_size, -1, 3, self.args.input_size, self.args.input_size)
        # (bs, c, seq_len, w, h)
        fake_seq_out = torch.transpose(fake_seq_out, 1, 2)
        fake_seq_d_out = self.seq_disc(fake_seq_out)

        d_loss = dis_hinge(fake_d_out_, real_d_out_pred)
        seq_d_loss = dis_hinge(fake_seq_d_out, real_seq_d_out)

        if self.args.amp:
            with amp.scale_loss(d_loss, self.d_opt) as scale_loss:
                scale_loss.backward(retain_graph=True)
            with amp.scale_loss(seq_d_loss, self.seq_d_opt) as scale_loss_:
                scale_loss_.backward(retain_graph=True)
        else:
            d_loss.backward(retain_graph=True)
            seq_d_loss.backward(retain_graph=True)
        self.d_opt.step()
        self.seq_d_opt.step()

        return d_loss.item(), seq_d_loss.item()

    def eval_each_sig_effect(self, images, labels, photo):
        test_bs = images.size(0)
        for i in range(test_bs):
            out_l = []
            for j in range(self.num_classes):
                sig_expand = torch.cat([labels[i]] * self.sig_step_num).view(-1, self.num_classes)
                sig_expand[:, j] = self.sig_list[j]
                image_expand = torch.cat([images[i].unsqueeze(0)] * self.sig_step_num, dim=0)
                out = self.inference(image_expand, sig_expand).detach().to('cpu')
                out_l.append(out)
            outs = torch.cat(out_l, dim=2).float()
            os.makedirs(os.path.join('runs', self.name, photo[i]), exist_ok=True)
            save_image(
                outs,
                fp=os.path.join('runs', self.name, photo[i], 'step_' + str(self.global_step) + '.png'),
                normalize=True, scale_each=True, nrow=self.sig_step_num
            )

        return

    def eval_seq_transform(self, images, labels, seq_labels):
        seq_len = self.train_set.seq_len
        test_bs = images.size(0)
        seq_labels_ = seq_labels.view(-1, self.num_classes)
        bs_seq = test_bs * seq_len
        images_ = torch.cat([images[1].unsqueeze(0)] * bs_seq, dim=0)
        out = self.inference(images_, seq_labels_).detach().to('cpu').float()
        images_ = images_.to('cpu').float()
        fake_out_row = make_grid(out, nrow=seq_len, normalize=True, scale_each=True)
        images_row = make_grid(images_[::seq_len], nrow=1, normalize=True, scale_each=True)
        out = torch.cat([images_row, fake_out_row], dim=2)

        return out

    def eval_loss(self, images, labels, seq_labels, sequence):
        seq_len = self.train_set.seq_len
        seq_labels_ = seq_labels.view(-1, self.num_classes)
        test_set_bs = self.test_set.bs
        sequence = sequence[:test_set_bs]
        images = images.unsqueeze(1).repeat(1, seq_len, 1, 1, 1).view(-1, 3, self.args.input_size, self.args.input_size)
        with torch.no_grad():
            fake_out = self.inference(images, seq_labels_)
            fake_c_out = self.estimator(fake_out)
            fake_d_out = self.discriminator(fake_out, seq_labels_)[0]
            fake_seq_out = fake_out.view(test_set_bs, -1, 3, self.args.input_size, self.args.input_size)
            fake_seq_out = torch.transpose(fake_seq_out, 1, 2)
            fake_seq_d_out = self.seq_disc(fake_seq_out)

        # Adversarial loss
        g_loss_adv = gen_hinge(fake_d_out)
        # L1 Loss
        g_loss_l1 = l1_loss(fake_out, images)
        # Weather prediction
        g_loss_w = pred_loss(fake_c_out, seq_labels_,
                             l_type=self.args.wloss_type, weight=self.args.wloss_weight)
        # sequence loss
        g_loss_seq = seq_loss(fake_c_out, seq_labels, seq_len)
        # reconstruction loss
        weight = torch.tensor([1, 1, 0.3, 1, 1, 1, 1, 1]).to('cuda')
        labels_ = weight * labels
        diff = torch.mean(torch.abs(fake_out - images), [1, 2, 3])
        lmda = torch.mean(torch.abs(torch.transpose(torch.stack([labels_] * seq_len), 1, 0) - seq_labels), 2)
        loss_con = torch.mean(diff / (lmda.reshape(-1) + self.args.epsilon))
        # discriminator loss
        with torch.no_grad():
            real_d_out = self.discriminator(images[::seq_len], labels)[0]
            real_seq_d_out = self.seq_disc(sequence)

        d_loss = dis_hinge(fake_d_out, real_d_out)
        seq_d_loss = dis_hinge(fake_seq_d_out, real_seq_d_out)

        return [_.item() for _ in [g_loss_adv, g_loss_l1, g_loss_w, loss_con, g_loss_seq, d_loss, seq_d_loss]]

    def evaluation(self):
        losses_l = []
        seq_out_l = []
        for images, labels, seq_labels, photos in self.test_loader:
            images, labels, seq_labels = (d.to('cuda', non_blocking=True) for d in [images, labels, seq_labels])
            images = self.test_set.transform(images)
            losses = self.eval_loss(images, labels, seq_labels, self.seq_test_samples)
            losses_l.append(losses)
            seq_out = self.eval_seq_transform(images, labels, seq_labels)
            seq_out_l.append(seq_out)
            self.eval_each_sig_effect(images, labels, photos)
        losses = np.mean(np.array(losses_l), axis=0)
        seq_out = torch.cat(seq_out_l, dim=1)

        self.seq_test_samples.to('cpu')
        # --- WRITING SUMMARY ---#
        # g_loss_adv, g_loss_l1, g_loss_w, loss_con, g_loss_seq, d_loss
        self.scalar_dict.update({
            'losses/g_loss_adv/test': losses[0],
            'losses/g_loss_l1/test': losses[1],
            'losses/g_loss_w/test': losses[2],
            'losses/loss_con/test': losses[3],
            'losses/loss_seq/test': losses[4],
            'losses/d_loss/test': losses[5],
            'losses/d_seq_loss/test': losses[6]
        })

        self.image_dict.update({
            'io/test': seq_out
        })

    def update_summary(self):
        # Summarize
        seq_len = self.train_set.seq_len
        for k, v in self.scalar_dict.items():
            spk = k.rsplit('/', 1)
            self.writer.add_scalars(spk[0], {spk[1]: v}, self.global_step)
        for k, v in self.image_dict.items():
            if 'train' in k:
                images, fake_out = v
                fake_out_row = make_grid(fake_out.float(), nrow=seq_len, normalize=True, scale_each=True)
                images_row = make_grid(images[::seq_len].float(), nrow=1, normalize=True, scale_each=True)
                out = torch.cat([images_row, fake_out_row], dim=2)
                self.writer.add_image(k, out, self.global_step)
            else:
                self.writer.add_image(k, v, self.global_step)

    def train(self):
        args = self.args
        # train setting
        eval_per_step = 1000 * args.GD_train_ratio
        display_per_step = 1000 * args.GD_train_ratio
        self.all_step = args.num_epoch * len(self.train_set) // self.batch_size

        g_loss_l = []
        g_loss_adv_l = []
        g_loss_seq_adv_l = []
        d_loss_l = []
        d_seq_loss_l = []
        g_loss_w_l = []
        g_loss_con_l = []
        g_loss_l1_l = []
        g_loss_seq_l = []

        torch.backends.cudnn.benchmark = True

        tqdm_iter = trange(args.num_epoch, desc='Training', leave=True)
        for epoch in tqdm_iter:
            if epoch > 0:
                self.epoch += 1

            for i, data in enumerate(self.train_loader):
                tqdm_iter.set_description('Training [ {} step ]'.format(self.global_step))
                if args.lmda:
                    self.lmda = args.lmda
                else:
                    self.lmda = self.global_step / self.all_step

                seq_len = self.train_set.seq_len

                images, con, seq_sig = (d.to('cuda', non_blocking=True) for d in data)
                images = self.train_set.transform(images)

                sequence = self.seq_loader.__next__()
                sequence = torch.stack([
                    self.seq_set.transform(sequence[i].to('cuda', non_blocking=True))
                    for i in range(self.batch_size)], dim=0)
                sequence = torch.transpose(sequence, 1, 2)

                if images.size(0) != self.batch_size or sequence.size(0) != self.batch_size:
                    continue

                # --- TRAINING --- #
                # images = torch.cat([torch.cat([image.unsqueeze(0)] * seq_len, dim=0) for image in images], dim=0
                images = images.unsqueeze(1).repeat(1, seq_len, 1, 1, 1).view(-1, 3, self.args.input_size, self.args.input_size)
                if self.global_step % args.GD_train_ratio == 0:
                    g_loss, g_loss_adv, g_loss_seq_adv, r_loss, w_loss, seq_loss, l1_loss = self.update_inference(images, con, seq_sig)
                    g_loss_l.append(g_loss)
                    g_loss_adv_l.append(g_loss_adv)
                    g_loss_seq_adv_l.append(g_loss_seq_adv)
                    g_loss_w_l.append(w_loss)
                    g_loss_con_l.append(r_loss)
                    g_loss_seq_l.append(seq_loss)
                    g_loss_l1_l.append(l1_loss)
                d_loss, d_seq_loss = self.update_discriminator(images, con, seq_sig, sequence)
                d_loss_l.append(d_loss)
                d_seq_loss_l.append(d_seq_loss)
                tqdm_iter.set_postfix(OrderedDict(
                    d_loss=d_loss,
                    g_loss=g_loss,
                    s_d_loss=d_seq_loss,
                    s_g_loss=g_loss_seq_adv,
                ))

                # --- EVALUATION ---#
                if (self.global_step % eval_per_step == 0):
                    self.scalar_dict.update({
                        'losses/g_loss/train': np.mean(g_loss_l[-100:]),
                        'losses/g_loss_adv/train': np.mean(g_loss_adv_l[-100:]),
                        'losses/g_loss_seq_adv/train': np.mean(g_loss_seq_adv_l[-100:]),
                        'losses/g_loss_l1/train': np.mean(g_loss_l1_l[-100:]),
                        'losses/g_loss_w/train': np.mean(g_loss_w_l[-100:]),
                        'losses/loss_con/train': np.mean(g_loss_con_l[-100:]),
                        'losses/loss_seq/train': np.mean(g_loss_seq_l[-100:]),
                        'losses/d_loss/train': np.mean(d_loss_l[-100:]),
                        'losses/d_seq_loss/train': np.mean(d_seq_loss_l[-100:])
                    })
                    del images, sequence
                    self.seq_test_samples = self.seq_test_samples.to('cuda', non_blocking=True)
                    self.evaluation()
                    g_loss_l = []
                    g_loss_adv_l = []
                    g_loss_seq_adv_l = []
                    d_loss_l = []
                    d_seq_loss_l = []
                    g_loss_w_l = []
                    g_loss_con_l = []
                    g_loss_l1_l = []
                    g_loss_seq_l = []

                # --- UPDATE SUMMARY ---#
                if self.global_step % display_per_step == 0:
                    self.update_summary()

                if self.global_step % eval_per_step == 0:
                    out_path = os.path.join(args.save_dir, self.name, ('cUNet_est' + '_e{:04d}_s{:06d}.pt').format(self.epoch, self.global_step))
                    if args.multi_gpu and torch.cuda.device_count() > 1:
                        state_dict = {
                            'inference': self.inference.module.state_dict(),
                            'discriminator': self.discriminator.module.state_dict(),
                            'seq_disc': self.seq_disc.module.state_dict(),
                            'epoch': self.epoch,
                            'global_step': self.global_step
                        }
                    else:
                        state_dict = {
                            'inference': self.inference.state_dict(),
                            'discriminator': self.discriminator.state_dict(),
                            'seq_disc': self.seq_disc.state_dict(),
                            'epoch': self.epoch,
                            'global_step': self.global_step
                        }
                    torch.save(state_dict, out_path)

                self.global_step += 1
        print('Done: training')


if __name__ == '__main__':
    wt = WeatherTransfer(args)
    wt.train()

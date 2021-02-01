import argparse
import os
from args import get_args
import itertools
args_ = get_args()

# GPU Setting
os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = args_.gpu

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

if args_.amp:
    from apex import amp, optimizers
    from apex.parallel import DistributedDataParallel
    from apex.fp16_utils import *
    from apex.multi_tensor_apply import multi_tensor_applier

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
        self.name = self.name + comment
        if self.args.local_rank == 0:
            self.writer = SummaryWriter(comment=comment + '_name-' + self.name)
            os.makedirs(os.path.join(args.save_dir, self.name), exist_ok=True)

        # Consts
        self.real = Variable_Float(1., self.batch_size)
        self.fake = Variable_Float(0., self.batch_size)
        self.lmda = 0.
        self.args.lr = args.lr
        self.sig_step_num = 10

        self.args.distributed = False

        if 'WORLD_SIZE' in os.environ:
            self.args.distributed = int(os.environ['WORLD_SIZE']) > 1

        self.args.gpu = 0
        self.args.world_size = 1
        if args.distributed:
            # FOR DISTRIBUTED:  Set the device according to local_rank.
            self.args.gpu = self.args.local_rank
            torch.cuda.set_device(self.args.gpu)
            # FOR DISTRIBUTED:  Initialize the backend.  torch.distributed.launch will provide
            # environment variables, and requires that you use init_method=`env://`.
            torch.distributed.init_process_group(backend='nccl',
                                                 init_method='env://')
            self.args.world_size = torch.distributed.get_world_size()

        torch.backends.cudnn.benchmark = True

        # torch >= 1.7
        train_transform = nn.Sequential(
            transforms.RandomRotation(10),
            # transforms.RandomResizedCrop(args.input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        )

        seq_transform = nn.Sequential(
            transforms.RandomResizedCrop(self.args.input_size),
            transforms.RandomAffine(degrees=10, translate=(0.2, 0.2), scale=(0.8, 1.2)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomApply(torch.nn.ModuleList([
                transforms.ColorJitter(
                    brightness=0.5,
                    contrast=0.3,
                    saturation=0.3,
                    hue=0)
            ])),
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

        self.transform = {'train': train_transform, 'seq': seq_transform, 'test': test_transform}
        self.train_set, self.seq_set, self.test_set = self.load_data(varbose=True)

        self.build()

    def load_data(self, varbose=False):
        args = self.args

        if args.local_rank == 0:
            print('Start loading image files...')
        df = pd.read_pickle(args.pkl_path)
        df_test = pd.read_pickle(args.test_pkl_path)
        # --- normalize --- #
        df_ = df.loc[:, self.cols].fillna(0)
        self.df_mean = df_.mean()
        self.df_std = df_.std()

        df.loc[:, self.cols] = (df.loc[:, self.cols].fillna(0) - self.df_mean) / self.df_std
        df_test.loc[:, self.cols] = (df_test.loc[:, self.cols].fillna(0) - self.df_mean) / self.df_std
        # ------------------ #
        # --- memory min - max value --- #
        self.sig_max = np.array([np.max(df[col].values) for col in self.cols])
        self.sig_min = np.array([np.min(df[col].values) for col in self.cols])
        self.sig_list = np.linspace(self.sig_min, self.sig_max, self.sig_step_num).T
        self.sig_max, self.sig_min, self.sig_list = [torch.from_numpy(_).float().cuda() for _ in [self.sig_max, self.sig_min, self.sig_list]]
        # ------------------------------ #
        if args.local_rank == 0:
            print('loaded {} signals data'.format(len(df)))
        df_shuffle = df.sample(frac=1)
        # df_sep = {'train': df_shuffle[df_shuffle['mode'] == 't_train'],
        #           'test': df_shuffle[df_shuffle['mode'] == 'test']}
        df_sep = {
            'train': df_shuffle[(df_shuffle['mode'] == 't_train') | (df_shuffle['mode'] == 'train')],
            'test': df_test}
        del df, df_shuffle, df_test

        train_set = SequenceFlickrDataLoader(args.image_root, args.csv_root, df_sep['train'], self.cols, self.df_mean, self.df_std,
                                             bs=args.batch_size, seq_len=args.seq_len, transform=self.transform['train'])
        test_set = SequenceFlickrDataLoader(args.image_root, args.csv_root, df_sep['test'], self.cols, self.df_mean, self.df_std,
                                            bs=1, seq_len=args.seq_len, transform=self.transform['test'], mode='test')
        seq_set = TimeLapseLoader(args.vid_root, bs=args.batch_size, seq_len=args.seq_len, transform=self.transform['seq'])
        if args.local_rank == 0:
            print('train:{} test:{} sets have already loaded.'.format(len(train_set), len(test_set)))
        return train_set, seq_set, test_set

    def build(self):
        args = self.args

        # Models
        if args.local_rank == 0:
            print('Build Models...')
        self.inference, self.discriminator, self.seq_disc, self.estimator, self.epoch, self.global_step = make_seq_network(args, self.num_classes, self.name)
        self.estimator.eval()

        # Models to CUDA
        memory_format = torch.contiguous_format
        [i.cuda().to(memory_format=memory_format) for i in [self.inference, self.discriminator, self.seq_disc, self.estimator]]

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
        if args.distributed:
            self.inference = DistributedDataParallel(self.inference, delay_allreduce=True)
            self.discriminator = DistributedDataParallel(self.discriminator, delay_allreduce=True)
            self.seq_disc = DistributedDataParallel(self.seq_disc, delay_allreduce=True)
            self.estimator = DistributedDataParallel(self.estimator, delay_allreduce=True)

        self.train_set.transform = self.train_set.transform.cuda()
        self.seq_set.transform = self.seq_set.transform.cuda()
        self.test_set.transform = self.test_set.transform.cuda()

        self.train_sampler = None
        self.seq_sampler = None
        self.test_sampler = None
        if args.distributed:
            self.train_sampler = torch.utils.data.distributed.DistributedSampler(
                self.train_set, num_replicas=torch.distributed.get_world_size(),
                rank=torch.distributed.get_rank()
            )
            self.seq_sampler = torch.utils.data.distributed.DistributedSampler(
                self.seq_set, num_replicas=torch.distributed.get_world_size(),
                rank=torch.distributed.get_rank()
            )
            self.test_sampler = torch.utils.data.distributed.DistributedSampler(
                self.test_set, num_replicas=torch.distributed.get_world_size(),
                rank=torch.distributed.get_rank()
            )

        self.train_loader = make_dataloader(self.train_set, args, sampler=self.train_sampler)
        args.sampler = 'none'
        seq_loader = make_dataloader(self.seq_set, args, sampler=self.seq_sampler)
        seq_loader = iter(seq_loader)
        self.seq_test_samples = seq_loader.__next__()
        self.seq_loader = itertools.cycle(iter(seq_loader))
        self.test_loader = make_dataloader(self.test_set, args, mode='test', sampler=self.test_sampler)

        self.seq_test_samples = torch.stack([
            self.seq_set.transform(self.seq_test_samples[i].cuda())
            for i in range(self.batch_size)], dim=0)
        self.seq_test_samples = torch.transpose(self.seq_test_samples, 1, 2)

        self.scalar_dict = {}
        self.image_dict = {}
        self.shift_lmda = lambda a, b: (1. - self.lmda) * a + self.lmda * b
        if args.local_rank == 0:
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
        fake_c_out = self.estimator(fake_out).detach()
        fake_d_out = self.discriminator(fake_out, seq_labels_)[0]
        # (bs, c , seq_len, w, h)
        # fake_seq_out = fake_out.view(self.batch_size, -1, 3, self.args.input_size, self.args.input_size)
        # (bs, c, seq_len, w, h)
        # fake_seq_out = torch.transpose(fake_seq_out, 1, 2)
        # fake_seq_d_out = self.seq_disc(fake_seq_out)
        fake_seq_d_out = self.seq_disc(torch.transpose(
            fake_out.view(self.batch_size, -1, 3, self.args.input_size, self.args.input_size),
            1, 2
        ))

        # for cycle consistency loss
        pred_labels_ = pred_labels.unsqueeze(1).repeat(1, seq_len, 1).view(-1, self.num_classes)
        fake_images = self.inference(fake_out, pred_labels_)

        ### -- Calc Generator Loss --- ###
        # Adversarial loss
        g_loss_adv = gen_hinge(fake_d_out)
        g_loss_seq_adv = gen_hinge(fake_seq_d_out)
        # L1 Loss
        g_loss_l1 = l1_loss(fake_out, images)
        # cycle consistency loss
        g_loss_cyc = l1_loss(fake_images, images)
        # Weather prediction
        g_loss_w = pred_loss(fake_c_out, seq_labels_,
                             l_type=self.args.wloss_type, weight=self.args.wloss_weight)

        # Sequence loss
        g_loss_seq = seq_loss(fake_c_out, seq_labels, seq_len)
        # Reconstruction loss
        weight = torch.tensor([1, 1, 0.3, 1, 1, 1, 1, 1]).cuda()
        diff = torch.mean(torch.abs(fake_out - images), [1, 2, 3])
        pred_labels = weight * pred_labels
        lmda = torch.mean(torch.abs(torch.transpose(torch.stack([pred_labels] * seq_len), 1, 0) - seq_labels), 2)
        g_loss_con = torch.mean(diff / (lmda.reshape(-1) + self.args.epsilon))

        lmda_con, lamda_cyc, lmda_w, lmda_seq = (1, 1, 1, 1)

        g_loss = g_loss_adv + g_loss_seq_adv + (lmda_con * g_loss_con) + (lamda_cyc * g_loss_cyc) \
            + (lmda_w * g_loss_w) + (lmda_seq * g_loss_seq)

        ### ------ ###
        if self.args.amp:
            with amp.scale_loss(g_loss, self.g_opt) as scale_loss:
                scale_loss.backward()
        else:
            g_loss.backward()
        self.g_opt.step()

        fake_out = fake_out.detach()
        losses = [g_loss, g_loss_adv, g_loss_seq_adv, g_loss_con, g_loss_cyc, g_loss_w, g_loss_seq, g_loss_l1]
        if self.args.distributed:
            losses_ = [to_python_float(reduce_tensor(loss.data, self.args)) for loss in losses]
        else:
            losses_ = [loss.item() for loss in losses]

        if self.global_step % self.display_per_step == 0:
            if self.args.distributed:
                images = torch.cat(gather_tensor(images, self.args), dim=0)
                fake_out = torch.cat(gather_tensor(fake_out, self.args), dim=0)
            if self.args.local_rank == 0:
                self.image_dict.update({
                    'io/train': [images.cpu().clone(), fake_out.cpu().clone()]
                })
                del images, fake_out

        return losses_

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
        d_loss = dis_hinge(fake_d_out_, real_d_out_pred)
        if self.args.amp:
            with amp.scale_loss(d_loss, self.d_opt) as scale_loss:
                scale_loss.backward(
                    retain_graph=(self.global_step % (self.args.GD_train_ratio * 2) == 0))
        else:
            d_loss.backward(
                retain_graph=(self.global_step % (self.args.GD_train_ratio * 2) == 0))

        self.d_opt.step()
        losses = [d_loss]

        # for sequence
        if self.global_step % (self.args.GD_train_ratio * 2) == 0:
            real_seq_d_out = self.seq_disc(sequence)
            # (bs* seq_len, c, w, h) -> (bs, seq_len, c, w, h)
            # fake_seq_out = fake_seq_out.view(self.batch_size, -1, 3, self.args.input_size, self.args.input_size)
            # (bs, c, seq_len, w, h)
            # fake_seq_out = torch.transpose(fake_seq_out, 1, 2)
            # fake_seq_d_out = self.seq_disc(fake_seq_out)
            fake_seq_d_out = self.seq_disc(torch.transpose(
                fake_seq_out.view(self.batch_size, -1, 3, self.args.input_size, self.args.input_size),
                1, 2
            ))
            seq_d_loss = dis_hinge(fake_seq_d_out, real_seq_d_out)

            if self.args.amp:
                with amp.scale_loss(seq_d_loss, self.seq_d_opt) as scale_loss_:
                    scale_loss_.backward()
            else:
                seq_d_loss.backward()

            self.seq_d_opt.step()
            losses.append(seq_d_loss)

        if self.args.distributed:
            losses_ = [to_python_float(reduce_tensor(loss.data, self.args)) for loss in losses]
        else:
            losses_ = [loss.item() for loss in losses]

        return losses_

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
            if self.args.local_rank == 0:
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
        images_ = torch.cat([images, ] * bs_seq, dim=0)
        out = self.inference(images_, seq_labels_).detach().float()
        images_ = images_.float()

        fake_out_row = make_grid(out, nrow=seq_len, normalize=True, scale_each=True, padding=0)
        images_row = make_grid(images_[::seq_len], nrow=1, normalize=True, scale_each=True, padding=0)
        out = torch.cat([images_row, fake_out_row], dim=2)

        return out

    def eval_loss(self, images, labels, seq_labels, sequence):
        seq_len = self.train_set.seq_len
        seq_labels_ = seq_labels.view(-1, self.num_classes)
        test_set_bs = self.test_set.bs
        sequence = sequence[:test_set_bs]
        images = images.unsqueeze(1).repeat(1, seq_len, 1, 1, 1).view(-1, 3, self.args.input_size, self.args.input_size)
        labels_ = labels.unsqueeze(1).repeat(1, seq_len, 1).view(-1, self.num_classes)
        with torch.no_grad():
            fake_out = self.inference(images, seq_labels_)
            fake_c_out = self.estimator(fake_out)
            fake_d_out = self.discriminator(fake_out, seq_labels_)[0]
            # fake_seq_out = fake_out.view(test_set_bs, -1, 3, self.args.input_size, self.args.input_size)
            # fake_seq_out = torch.transpose(fake_seq_out, 1, 2)
            # fake_seq_d_out = self.seq_disc(fake_seq_out)
            fake_seq_d_out = self.seq_disc(torch.transpose(
                fake_out.view(test_set_bs, -1, 3, self.args.input_size, self.args.input_size),
                1, 2
            ))
            fake_images = self.inference(fake_out, labels_)

        # Adversarial loss
        g_loss_adv = gen_hinge(fake_d_out)
        # L1 Loss
        g_loss_l1 = l1_loss(fake_out, images)
        # cycle consistency loss
        g_loss_cyc = l1_loss(fake_images, images)
        # Weather prediction
        g_loss_w = pred_loss(fake_c_out, seq_labels_,
                             l_type=self.args.wloss_type, weight=self.args.wloss_weight)
        # sequence loss
        g_loss_seq = seq_loss(fake_c_out, seq_labels, seq_len)
        # reconstruction loss
        weight = torch.tensor([1, 1, 0.3, 1, 1, 1, 1, 1]).cuda()
        labels_ = weight * labels
        diff = torch.mean(torch.abs(fake_out - images), [1, 2, 3])
        lmda = torch.mean(torch.abs(torch.transpose(torch.stack([labels_] * seq_len), 1, 0) - seq_labels), 2)
        g_loss_con = torch.mean(diff / (lmda.reshape(-1) + self.args.epsilon))
        # discriminator loss
        with torch.no_grad():
            real_d_out = self.discriminator(images[::seq_len], labels)[0]
            real_seq_d_out = self.seq_disc(sequence)

        d_loss = dis_hinge(fake_d_out, real_d_out)
        seq_d_loss = dis_hinge(fake_seq_d_out, real_seq_d_out)

        losses = [g_loss_adv, g_loss_l1, g_loss_w, g_loss_con, g_loss_cyc, g_loss_seq, d_loss, seq_d_loss]
        if self.args.distributed:
            losses_ = [to_python_float(reduce_tensor(loss.data, self.args)) for loss in losses]
        else:
            losses_ = [loss.item() for loss in losses]

        return losses_

    def evaluation(self):
        losses_l = []
        seq_out_l = []
        for images, labels, seq_labels, photos in self.test_loader:
            images, labels, seq_labels = (d.cuda() for d in [images, labels, seq_labels])
            images = self.test_set.transform(images)
            losses = self.eval_loss(images, labels, seq_labels, self.seq_test_samples)
            losses_l.append(losses)
            seq_out = self.eval_seq_transform(images, labels, seq_labels)
            self.eval_each_sig_effect(images, labels, photos)
            if self.args.local_rank == 0:
                seq_out_l.append(seq_out.cpu())
        losses = np.mean(np.array(losses_l), axis=0)

        self.seq_test_samples.cpu()
        # --- WRITING SUMMARY ---#
        # g_loss_adv, g_loss_l1, g_loss_w, loss_con, g_loss_seq, d_loss
        if self.args.local_rank == 0:
            self.scalar_dict.update({
                'losses/g_loss_adv/test': losses[0],
                'losses/g_loss_l1/test': losses[1],
                'losses/g_loss_w/test': losses[2],
                'losses/loss_con/test': losses[3],
                'losses/loss_cyc/test': losses[4],
                'losses/loss_seq/test': losses[5],
                'losses/d_loss/test': losses[6],
                'losses/d_seq_loss/test': losses[7]
            })

            seq_out = torch.cat(seq_out_l, dim=1)
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
        self.image_dict = {}

    def train(self):
        args = self.args
        # train setting
        self.eval_per_step = 1000 * args.GD_train_ratio
        self.display_per_step = 1000 * args.GD_train_ratio
        self.all_step = args.num_epoch * len(self.train_set) // self.batch_size
        g_loss_l = []
        g_loss_adv_l = []
        g_loss_seq_adv_l = []
        d_loss_l = []
        d_seq_loss_l = []
        g_loss_w_l = []
        g_loss_con_l = []
        g_loss_cyc_l = []
        g_loss_l1_l = []
        g_loss_seq_l = []

        tqdm_iter = trange(args.num_epoch, desc='Training', leave=True)
        for epoch in tqdm_iter:
            if epoch > 0:
                self.epoch += 1
            if args.distributed:
                self.train_sampler.set_epoch(epoch)
                self.seq_sampler.set_epoch(epoch)

            for i, data in enumerate(self.train_loader):
                if args.local_rank == 0:
                    tqdm_iter.set_description('Training [ {} step ]'.format(self.global_step))

                if args.lmda:
                    self.lmda = args.lmda
                else:
                    self.lmda = self.global_step / self.all_step

                seq_len = self.train_set.seq_len

                images, con, seq_sig = (d.cuda() for d in data)
                images = self.train_set.transform(images)

                sequence = self.seq_loader.__next__()
                sequence = torch.stack([
                    self.seq_set.transform(sequence[i].cuda())
                    for i in range(self.batch_size)], dim=0)
                sequence = torch.transpose(sequence, 1, 2)

                if images.size(0) != self.batch_size or sequence.size(0) != self.batch_size:
                    continue

                # --- TRAINING --- #
                images = images.unsqueeze(1).repeat(1, seq_len, 1, 1, 1).view(-1, 3, self.args.input_size, self.args.input_size)
                if self.global_step % args.GD_train_ratio == 0:
                    g_loss, g_loss_adv, g_loss_seq_adv, r_loss, cyc_loss, w_loss, seq_loss, l1_loss \
                        = self.update_inference(images, con, seq_sig)
                    g_loss_l.append(g_loss)
                    g_loss_adv_l.append(g_loss_adv)
                    g_loss_seq_adv_l.append(g_loss_seq_adv)
                    g_loss_w_l.append(w_loss)
                    g_loss_con_l.append(r_loss)
                    g_loss_cyc_l.append(cyc_loss)
                    g_loss_seq_l.append(seq_loss)
                    g_loss_l1_l.append(l1_loss)
                d_losses = self.update_discriminator(images, con, seq_sig, sequence)
                d_loss = d_losses[0]
                d_loss_l.append(d_loss)
                if len(d_losses) == 2:
                    d_seq_loss = d_losses[1]
                    d_seq_loss_l.append(d_seq_loss)

                # --- EVALUATION --- #
                if self.global_step % self.eval_per_step == 0:
                    del images, sequence
                    self.seq_test_samples = self.seq_test_samples.cuda()
                    self.evaluation()

                    g_loss_l = g_loss_l[-100:]
                    g_loss_adv_l = g_loss_adv_l[-100:]
                    g_loss_seq_adv_l = g_loss_seq_adv_l[-100:]
                    d_loss_l = d_loss_l[-100:]
                    d_seq_loss_l = d_seq_loss_l[-100:]
                    g_loss_w_l = g_loss_w_l[-100:]
                    g_loss_con_l = g_loss_con_l[-100:]
                    g_loss_cyc_l = g_loss_cyc_l[-100:]
                    g_loss_l1_l = g_loss_l1_l[-100:]
                    g_loss_seq_l = g_loss_seq_l[-100:]

                if args.local_rank == 0:
                    tqdm_iter.set_postfix(OrderedDict(
                        d_loss=d_loss,
                        g_loss=g_loss,
                        s_d_loss=d_seq_loss,
                        s_g_loss=g_loss_seq_adv,
                    ))

                    # --- LOSS POST-PROCESS --- #
                    if self.global_step % self.eval_per_step == 0:
                        self.scalar_dict.update({
                            'losses/g_loss/train': np.mean(g_loss_l),
                            'losses/g_loss_adv/train': np.mean(g_loss_adv_l),
                            'losses/g_loss_seq_adv/train': np.mean(g_loss_seq_adv_l),
                            'losses/g_loss_l1/train': np.mean(g_loss_l1_l),
                            'losses/g_loss_w/train': np.mean(g_loss_w_l),
                            'losses/loss_con/train': np.mean(g_loss_con_l),
                            'losses/loss_cyc/train': np.mean(g_loss_cyc_l),
                            'losses/loss_seq/train': np.mean(g_loss_seq_l),
                            'losses/d_loss/train': np.mean(d_loss_l),
                            'losses/d_seq_loss/train': np.mean(d_seq_loss_l)
                        })

                    # --- UPDATE SUMMARY --- #
                    if self.global_step % self.display_per_step == 0:
                        self.update_summary()

                    # --- SAVE CHECKPOINT --- #
                    if self.global_step % self.eval_per_step == 0:
                        out_path = os.path.join(args.save_dir, self.name, ('cUNet_est' + '_e{:04d}_s{:06d}.pt').format(self.epoch, self.global_step))
                        if args.distributed:
                            state_dict = {
                                'inference': self.inference.module.state_dict(),
                                'discriminator': self.discriminator.module.state_dict(),
                                'seq_disc': self.seq_disc.module.state_dict(),
                                'epoch': self.epoch,
                                'global_step': self.global_step,
                                'dataset_mean': self.df_mean.values,
                                'dataset_std': self.df_std.values
                            }
                        else:
                            state_dict = {
                                'inference': self.inference.state_dict(),
                                'discriminator': self.discriminator.state_dict(),
                                'seq_disc': self.seq_disc.state_dict(),
                                'epoch': self.epoch,
                                'global_step': self.global_step,
                                'dataset_mean': self.df_mean.values,
                                'dataset_std': self.df_std.values
                            }
                        torch.save(state_dict, out_path)

                self.global_step += 1

        print('Done: training')


if __name__ == '__main__':
    wt = WeatherTransfer(args_)
    wt.train()

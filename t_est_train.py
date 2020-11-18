import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--image_root', type=str,
                    default="/mnt/HDD8T/takamuro/dataset/photos_usa_2016/"
                    # default='/mnt/fs2/2018/matsuzaki/dataset_fromnitta/Image/'
                    )
parser.add_argument('--name', type=str, default='cUNet')
# Nmaing rule : cUNet_[c(classifier) or e(estimator)]_[detail of condition]_[epoch]_[step]
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--save_dir', type=str, default='cp/transfer')
parser.add_argument('--pkl_path', type=str,
                    default='/mnt/fs2/2019/Takamuro/m2_research/flicker_data/from_nitta/param03/temp_WoGray-LowConfOutlier-TestAndTrain_for_transfer-esttrain214938_test500.pkl'
                    )
parser.add_argument('--estimator_path', type=str,
                    default='/mnt/fs2/2019/Takamuro/m2_research/weather_transfer/cp/estimator/'
                    'est_res101_flicker-p03th01-WoOutlier_sep-train_aug_pre_loss-mse-reduction-none-grad-all-1/est_resnet101_20_step22680.pt'
                    )
parser.add_argument('--input_size', type=int, default=224)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--lmda', type=float, default=None)
parser.add_argument('--num_epoch', type=int, default=150)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--GD_train_ratio', type=int, default=1)
parser.add_argument('--sampler', action='store_true')
parser.add_argument('--augmentation', action='store_true')
parser.add_argument('--image_only', action='store_true')
parser.add_argument('--resume_cp', type=str)
parser.add_argument('-b1', '--adam_beta1', type=float, default=0.0)
parser.add_argument('-b2', '--adam_beta2', type=float, default=0.999)
args = parser.parse_args()
# args = parser.parse_args(args=['--augmentation', '--name', 'debug', '--resume_cp', './cp/transfer/cUNet_w-e_res101_expt4_RandomSig-LowConf_aug-True_sampler-False_dataset-temp_WoGray_for_transfer-esttrain214938_test500/cUNet_w-e_res101_expt4_RandomSig-LowConf_aug-True_sampler-False_dataset-temp_WoGray_for_transfer-esttrain214938_test500_e0010_s138000.pt'])

# GPU Setting
os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import trange, tqdm
from glob import glob

import torch
import torch.nn as nn
import torchvision as tv
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image, make_grid

from ops import *
from dataset import ImageLoader, FlickrDataLoader, ClassImageLoader
from sampler import ImbalancedDatasetSampler
from cunet import Conditional_UNet
from disc import SNResNetProjectionDiscriminator
from utils import MakeOneHot


class WeatherTransfer(object):

    def __init__(self, args):

        self.args = args
        self.batch_size = args.batch_size
        self.global_step = 0

        self.name = '{}_aug-{}_sampler-{}_GDratio{}_adam-b1{}-b2{}_norm-{}_dataset-{}'.format(args.name, args.augmentation, args.sampler, '1-' + str(args.GD_train_ratio), 
                                                                                        args.adam_beta1, args.adam_beta2, args.norm, args.pkl_path.split('/')[-1].split('.')[0])
        os.makedirs(os.path.join(args.save_dir, self.name), exist_ok=True)
        comment = '_lr-{}_bs-{}_ne-{}_name-{}'.format(args.lr, args.batch_size, args.num_epoch, self.name)
        self.writer = SummaryWriter(comment=comment)

        # Consts
        self.real = Variable_Float(1., self.batch_size)
        self.fake = Variable_Float(0., self.batch_size)
        self.lmda = 0.

        if args.augmentation:
            train_transform = transforms.Compose([
                transforms.RandomRotation(10),
                transforms.RandomResizedCrop(args.input_size),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(
                        brightness=0.5,
                        contrast=0.3,
                        saturation=0.3,
                        hue=0
                    ),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        else:
            train_transform = transforms.Compose([
                transforms.Resize((args.input_size,)*2),
                transforms.RandomRotation(10),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])

        test_transform = transforms.Compose([
            transforms.Resize((args.input_size,) * 2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        self.cols = ['tempC', 'uvIndex', 'visibility', 'windspeedKmph', 'cloudcover', 'humidity', 'pressure', 'FeelsLikeC', 'DewPointC']
        self.num_classes = len(self.cols)

        self.transform = {'train': train_transform, 'test': test_transform}
        self.train_set, self.test_set = self.load_data(varbose=True, image_only=args.image_only)

        self.build()

    def load_data(self, varbose=False, image_only=False, train_data_rate=0.7):

        print('Start loading image files...')
        if image_only:
            paths = glob(os.path.join(args.image_root, '*'))
            print('loaded {} data'.format(len(paths)))
            pivot = int(len(paths) * train_data_rate)
            paths_sep = {'train': paths[:pivot], 'test': paths[pivot:]}
            loader = lambda s: ImageLoader(paths_sep[s], transform=self.transform[s])

        else:
            df = pd.read_pickle(args.pkl_path)
            # --- normalize --- #
            temp = pd.read_pickle('/mnt/fs2/2019/okada/from_nitta/parm_0.3/sepalated_data_wo-outlier.pkl')
            df_ = temp.loc[:, self.cols].fillna(0)
            df_mean = df_.mean()
            df_std = df_.std()

            df.loc[:, self.cols] = (df.loc[:, self.cols].fillna(0) - df_mean) / df_std
            # ------------------ #
            print('loaded {} signals data'.format(len(df)))
            df_shuffle = df.sample(frac=1)
            df_sep = {'train': df_shuffle[df_shuffle['mode'] == 'train'], 'test': df_shuffle[df_shuffle['mode'] == 'test']}
            del df, df_shuffle, temp
            loader = lambda s: FlickrDataLoader(args.image_root, df_sep[s], self.cols, transform=self.transform[s])

        train_set = loader('train')
        test_set = loader('test')
        print('train:{} test:{} sets have already loaded.'.format(len(train_set), len(test_set)))
        return train_set, test_set

    def build(self):
        args = self.args

        # Models
        print('Build Models...')

        self.inference = Conditional_UNet(num_classes=self.num_classes)
        self.discriminator = SNResNetProjectionDiscriminator(num_classes=self.num_classes)

        if args.resume_cp:
            exist_cp = [args.resume_cp]
        else:
            exist_cp = sorted(glob(os.path.join(args.save_dir, self.name, '*')))
        if len(exist_cp) != 0:
            print('Load checkpoint:{}'.format(exist_cp[-1]))
            sd = torch.load(exist_cp[-1])
            self.inference.load_state_dict(sd['inference'])
            self.discriminator.load_state_dict(sd['discriminator'])
            self.epoch = sd['epoch']
            self.global_step = sd['global_step']
            print('Success checkpoint loading!')
        else:
            print('Initialize training status.')
            self.epoch = 0
            self.global_step = 0

        self.estimator = torch.load(args.estimator_path)
        self.estimator.eval()

        # Models to CUDA
        [i.to('cuda') for i in [self.inference, self.discriminator, self.estimator]]

        # Optimizer
        self.g_opt = torch.optim.Adam(self.inference.parameters(), lr=args.lr, betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.lr/20)
        self.d_opt = torch.optim.Adam(self.discriminator.parameters(), lr=args.lr, betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.lr/20)

        # これらのloaderにsamplerは必要ないのか？
        self.train_loader = torch.utils.data.DataLoader(
                self.train_set,
                batch_size=args.batch_size,
                shuffle=True,
                drop_last=True,
                num_workers=args.num_workers)

        if args.sampler:
            self.random_loader = torch.utils.data.DataLoader(
                self.train_set,
                batch_size=args.batch_size,
                sampler=ImbalancedDatasetSampler(self.train_set),
                drop_last=True,
                num_workers=args.num_workers)
        else:
            self.random_loader = torch.utils.data.DataLoader(
                    self.train_set,
                    batch_size=args.batch_size,
                    shuffle=True,
                    drop_last=True,
                    num_workers=args.num_workers)

        if not args.image_only:
            self.test_loader = torch.utils.data.DataLoader(
                    self.test_set,
                    batch_size=args.batch_size,
                    shuffle=True,
                    drop_last=True,
                    num_workers=args.num_workers)
            test_data_iter = iter(self.test_loader)
            self.test_random_sample = [tuple(d.to('cuda') for d in test_data_iter.next()) for i in range(2)]
            del test_data_iter, self.test_loader

        self.scalar_dict = {}
        self.image_dict = {}
        self.shift_lmda = lambda a, b: (1. - self.lmda) * a + self.lmda * b
        print('Build has been completed.')

    def update_inference(self, images, r_labels, rand_images):
        # --- UPDATE(Inference) --- #
        self.g_opt.zero_grad()

        # for real
        pred_labels = self.estimator(images).detach()

        fake_out = self.inference(images, r_labels)
        fake_c_out = self.estimator(fake_out)
        fake_res = self.discriminator(fake_out, r_labels)
        fake_d_out = fake_res[0]
        # fake_feat = fake_res[3]

        # Calc Generator Loss
        g_loss_adv = gen_hinge(fake_d_out)       # Adversarial loss
        g_loss_l1 = l1_loss(fake_out, images)
        g_loss_w = pred_loss(fake_c_out, r_labels)   # Weather prediction

        # abs_loss = torch.mean(torch.abs(fake_out - images), [1, 2, 3])
        diff = torch.mean(torch.abs(fake_out - images), [1, 2, 3])
        lmda = torch.mean(torch.abs(pred_labels - r_labels), 1)
        loss_con = torch.mean(diff / (lmda + 1e-7))  # Reconstraction loss

        lmda_con, lmda_w = (1, 1)

        g_loss = g_loss_adv + lmda_con * loss_con + lmda_w * g_loss_w

        g_loss.backward()
        self.g_opt.step()

        self.scalar_dict.update({
            'losses/g_loss/train': g_loss.item(),
            'losses/g_loss_adv/train': g_loss_adv.item(),
            'losses/g_loss_l1/train': g_loss_l1.item(),
            'losses/g_loss_w/train': g_loss_w.item(),
            'losses/loss_con/train': loss_con.item(),
            'variables/lmda': self.lmda
            })

        self.image_dict.update({
            'io/train': torch.cat([images, fake_out, rand_images], dim=3),
            })

    def update_discriminator(self, images, labels):

        # --- UPDATE(Discriminator) ---#
        self.d_opt.zero_grad()

        # for real
        real_c_out = self.estimator(images)
        pred_labels = real_c_out.detach()

        real_d_out_pred = self.discriminator(images, pred_labels)[0]

        # for fake
        fake_out = self.inference(images, labels)
        fake_d_out = self.discriminator(fake_out.detach(), labels)[0]

        d_loss = dis_hinge(fake_d_out, real_d_out_pred)

        d_loss.backward()
        self.d_opt.step()

        self.scalar_dict.update({
            'losses/d_loss/train': d_loss.item()
            })

    def evaluation(self):
        g_loss_l1_ = []
        g_loss_adv_ = []
        g_loss_w_ = []
        fake_out_li = []
        d_loss_ = []
        # loss_con_ = []

        images, labels = self.test_random_sample[0]
        # if not args.supervised:
        #     labels_ = self.estimator(images).detach()
        blank = torch.zeros_like(images[0]).unsqueeze(0)
        ref_images, ref_labels = self.test_random_sample[1]

        for i in range(self.batch_size):
            with torch.no_grad():
                if ref_labels is None:
                    ref_labels = self.estimator(ref_images)
                ref_labels_expand = torch.cat([ref_labels[i]] * self.batch_size).view(-1, self.num_classes)
                fake_out_ = self.inference(images, ref_labels_expand)
                fake_c_out_ = self.estimator(fake_out_)
                # fake_d_out_ = self.discriminator(fake_out_, labels)[0]  # Dへの入力はfake_out_ と re_labels_expandではないのか？
                real_d_out_ = self.discriminator(images, labels)[0]
                fake_d_out_ = self.discriminator(fake_out_, ref_labels_expand)[0]

            # diff = torch.mean(torch.abs(fake_out_ - images), [1, 2, 3])
            # lmda = torch.mean(torch.abs(pred_labels_ - ref_labels_expand), 1)
            # loss_con_ = torch.mean(diff / (lmda + 1e-7))

            fake_out_li.append(fake_out_)
            g_loss_adv_.append(gen_hinge(fake_d_out_).item())
            g_loss_l1_.append(l1_loss(fake_out_, images).item())
            g_loss_w_.append(pred_loss(fake_c_out_, ref_labels_expand).item())
            d_loss_.append(dis_hinge(fake_d_out_, real_d_out_).item())
            # loss_con_.append(torch.mean(diff / (lmda + 1e-7).item())

        # --- WRITING SUMMARY ---#
        self.scalar_dict.update({
                'losses/g_loss_adv/test': np.mean(g_loss_adv_),
                'losses/g_loss_l1/test': np.mean(g_loss_l1_),
                'losses/g_loss_w/test': np.mean(g_loss_w_),
                'losses/d_loss/test': np.mean(d_loss_)
                })
        ref_img = torch.cat([blank] + list(torch.split(ref_images, 1)), dim=3)
        in_out_img = torch.cat([images] + fake_out_li, dim=3)
        res_img = torch.cat([ref_img, in_out_img], dim=0)

        self.image_dict.update({
            'images/test': res_img,
                })

    def update_summary(self):
        # Summarize
        for k, v in self.scalar_dict.items():
            spk = k.rsplit('/', 1)
            self.writer.add_scalars(spk[0], {spk[1]: v}, self.global_step)
        for k, v in self.image_dict.items():
            grid = make_grid(v,
                    nrow=1,
                    normalize=True, scale_each=True)
            self.writer.add_image(k, grid, self.global_step)

    def train(self):
        args = self.args

        # train setting
        eval_per_step = 1000
        display_per_step = 1000
        save_per_epoch = 5

        self.all_step = args.num_epoch * len(self.train_set) // self.batch_size

        tqdm_iter = trange(args.num_epoch, desc='Training', leave=True)
        for epoch in tqdm_iter:
            if epoch > 0:
                self.epoch += 1

            for i, (data, rand_data) in enumerate(zip(self.train_loader, self.random_loader)):
                self.global_step += 1

                if self.global_step % eval_per_step == 0:
                    out_path = os.path.join(args.save_dir, self.name, (self.name + '_e{:04d}_s{}.pt').format(self.epoch, self.global_step))
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

                images, con = (d.to('cuda') for d in data)
                rand_images, r_con = (d.to('cuda') for d in rand_data)

                # --- master --- #
                # rand_signals = self.estimator(rand_images).detach()
                # -------------- #
                # --- expt 4 --- #
                rand_signals = r_con
                # -------------- #

                if images.size(0) != self.batch_size:
                    continue

                self.update_discriminator(images, rand_signals)
                if self.global_step % args.GD_train_ratio == 0:
                    self.update_inference(images, rand_signals, rand_images)

                # --- EVALUATION ---#
                if (self.global_step % eval_per_step == 0) and not args.image_only:
                    self.evaluation()

                # --- UPDATE SUMMARY ---#
                if self.global_step % display_per_step == 0:
                    self.update_summary()
        print('Done: training')


if __name__ == '__main__':
    wt = WeatherTransfer(args)
    wt.train()

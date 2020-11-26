import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--image_root', type=str,
                    default='/mnt/HDD8T/takamuro/dataset/photos_usa_224_2016-2017'
                    )
parser.add_argument('--name', type=str, default='cUNet')
# Nmaing rule : cUNet_[c(classifier) or e(classifier)]_[detail of condition]_[epoch]_[step]
parser.add_argument('--gpus', type=str, default='1')
parser.add_argument('--save_dir', type=str, default='cp/transfer')
parser.add_argument('--pkl_path', type=str,
                    # default='/mnt/fs2/2019/Takamuro/m2_research/flicker_data/wwo/2016_17/equal_con-cnn-mlp/outdoor_all_dbdate_wwo_weather_selected_ent_owner_2016_17_delnoise_addpred_equal_con-cnn-mlp.pkl'
                    default='/mnt/fs2/2019/Takamuro/m2_research/flicker_data/wwo/2016_17/equal_con-cnn-mlp/outdoor_all_dbdate_wwo_weather_selected_ent_owner_2016_17_delnoise_addpred_equal_con-cnn-mlp-time6-18_Wo-person-animals.pkl'
                    )
parser.add_argument('--classifier_path', type=str,
                    default='/mnt/fs2/2019/Takamuro/m2_research/weather_transferV2/cp/classifier/cls_res101_1122_NotPreTrain/resnet101_epoch15_step59312.pt'
                    )
parser.add_argument('--input_size', type=int, default=224)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--lmda', type=float, default=None)
parser.add_argument('--num_epoch', type=int, default=150)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--image_only', action='store_true')
parser.add_argument('--GD_train_ratio', type=int, default=5)
parser.add_argument('--sampler', action='store_true')
parser.add_argument('--loss_lamda_cw', '-lm', type=float, nargs=2, default=[1, 1])
parser.add_argument('-b1', '--adam_beta1', type=float, default=0.5)
parser.add_argument('-b2', '--adam_beta2', type=float, default=0.9)
parser.add_argument('--amp', action='store_true')
parser.add_argument('--multi_gpu', action='store_true')
parser.add_argument('--weather_loss', type=str, default='CE')
args = parser.parse_args()
# args = parser.parse_args(args=['--gpu', '3', '--sampler', '--name', 'cUNet_w-c_res101-0317_RamCom_sampler', 
#                                  '--classifier_path', '/mnt/fs2/2019/Takamuro/m2_research/weather_transfer/cp/classifier/cls_res101_i2w_sep-val_aug_20200408/resnet101_epoch25_step96382.pt'])

# GPU Setting
os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import trange
from collections import OrderedDict
from glob import glob

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image, make_grid

if args.amp:
    from apex import amp, optimizers

from ops import *
from dataset import ImageLoader, FlickrDataLoader
from sampler import ImbalancedDatasetSampler
from cunet import Conditional_UNet
from disc import SNResNetProjectionDiscriminator
from utils import MakeOneHot


class WeatherTransfer(object):

    def __init__(self, args):

        self.args = args
        self.batch_size = args.batch_size
        if args.batch_size % 8 != 0:
            print('set batch size multiple of 8')
        self.global_step = 0

        self.name = 'Flickr_{}_sampler-{}_loss_lamda-c{}-w{}-{}_b1-{}_b2-{}_GDratio-{}_amp-{}_MGpu-{}'.format(self.args.name, self.args.sampler, self.args.loss_lamda_cw[0],
                                                                                        self.args.loss_lamda_cw[1], self.args.weather_loss, self.args.adam_beta1, self.args.adam_beta2,
                                                                                        self.args.GD_train_ratio, self.args.amp, self.args.multi_gpu)

        comment = '_lr-{}*{}_bs-{}_ne-{}'.format(str((args.batch_size / 16)), self.args.lr, self.args.batch_size, self.args.num_epoch)

        self.writer = SummaryWriter(comment=comment + '_name-' + self.name)
        self.name = self.name + comment
        os.makedirs(os.path.join(args.save_dir, self.name), exist_ok=True)

        # Consts
        self.real = Variable_Float(1., self.batch_size)
        self.fake = Variable_Float(0., self.batch_size)
        self.lmda = 0.
        self.args.lr = args.lr * (args.batch_size / 16)

        # torch >= 1.7
        # train_transform = nn.Sequential([
        train_transform = transforms.Compose([
            transforms.RandomRotation(10),
            # transforms.RandomResizedCrop(args.input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                    brightness=0.5,
                    contrast=0.3,
                    saturation=0.3,
                    hue=0
                ),
            transforms.ToTensor(),
            # torch >= 1.7
            # transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        # torch >= 1.7
        # test_transform = nn.Sequential([
        test_transform = transforms.Compose([
            # transforms.Resize((args.input_size,) * 2),
            transforms.ToTensor(),
            # torch >= 1.7
            # transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        self.cols = ['tempC', 'uvIndex', 'visibility', 'windspeedKmph', 'cloudcover', 'humidity', 'pressure', 'FeelsLikeC', 'DewPointC']
        self.num_classes = len(['Clear', 'Clouds', 'Rain', 'Snow', 'Mist'])

        self.transform = {'train': train_transform, 'test': test_transform}
        self.train_set, self.test_set = self.load_data(varbose=True, image_only=args.image_only)

        self.build()

    def load_data(self, varbose=False, image_only=False, train_data_rate=0.7):
        args = self.args
        print('Start loading image files...')

        if not image_only:
            df = pd.read_pickle(args.pkl_path)
            print('loaded {} data'.format(len(df)))
            df_shuffle = df.sample(frac=1)
            df_sep = {'train': df_shuffle[df_shuffle['mode'] == 't_train'],
                      'test': df_shuffle[df_shuffle['mode'] == 'test']}
            del df, df_shuffle
            print(df_sep['train'].condition.value_counts())
            loader = lambda s: FlickrDataLoader(args.image_root, df_sep[s], self.cols, transform=self.transform[s], class_id=True)

        else:
            print('image loader')
            paths = glob(os.path.join(args.image_root, '*'))
            print('loaded {} data'.format(len(paths)))
            pivot = int(len(paths) * train_data_rate)
            paths_sep = {'train': paths[:pivot], 'test': paths[pivot:]}
            loader = lambda s: ImageLoader(paths_sep[s], transform=self.transform[s])

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

        self.classifier = torch.load(args.classifier_path)
        self.classifier.eval()

        # Optimizer
        args.lr = args.lr * args.batch_size / 16
        self.g_opt = torch.optim.Adam(self.inference.parameters(), lr=args.lr, betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.lr/20)
        self.d_opt = torch.optim.Adam(self.discriminator.parameters(), lr=args.lr, betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.lr/20)

        # Models to CUDA
        [i.to('cuda:0') for i in [self.inference, self.discriminator, self.classifier]]

        # apex
        if args.amp:
            self.inference, self.g_opt = amp.initialize(self.inference, self.g_opt, opt_level='O1')
            self.discriminator, self.d_opt = amp.initialize(self.discriminator, self.d_opt, opt_level='O1')

        if args.multi_gpu and torch.cuda.device_count() > 1:
            self.inference = nn.DataParallel(self.inference)
            self.discriminator = nn.DataParallel(self.discriminator)
            self.classifier = nn.DataParallel(self.classifier)

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
                    batch_size=self.batch_size,
                    shuffle=True,
                    drop_last=True,
                    num_workers=args.num_workers)
            test_data_iter = iter(self.test_loader)

            self.test_random_sample = [tuple(d.to('cuda:0') for d in test_data_iter.next()) for i in range(2)]
            # torch >= 1.7
            # self.test_random_sample = []
            # for i in range(2):
            #     img, label = test_data_iter.next()
            #     img = self.test_set.transform(img.to('cuda:0'))
            #     self.test_random_sample.append((img, label.to('cuda:0')))
            del test_data_iter, self.test_loader

        self.scalar_dict = {}
        self.image_dict = {}
        self.shift_lmda = lambda a, b: (1. - self.lmda) * a + self.lmda * b
        print('Build has been completed.')

    def update_inference(self, images, r_labels, rand_images, labels=None):
        # labels, r_labels is one hot, target_label is label(0~4)
        # --- UPDATE(Inference) --- #
        self.g_opt.zero_grad()
        # for real
        pred_labels = self.classifier(images).detach()
        # --- master --- #
        pred_labels = F.softmax(pred_labels, dim=1)
        ep = 1e-7
        # -------------- #
        # --- experiment1 --- #
        # one-hot
        # pred_labels = torch.eye(self.num_classes)[torch.argmax(pred_labels, dim=1)].to('cuda:0')
        # ep = 1e-2
        # ------------------- #

        fake_out = self.inference(images, r_labels)
        fake_res = self.discriminator(fake_out, r_labels)
        fake_d_out = fake_res[0]
        # fake_feat = fake_res[3]
        fake_c_out = self.classifier(fake_out)
        if self.args.weather_loss == 'mse':
            fake_c_out = F.softmax(fake_c_out, dim=1)

        # Calc Generator Loss
        g_loss_adv = gen_hinge(fake_d_out)  # Adversarial loss
        g_loss_l1 = l1_loss(fake_out, images)
        g_loss_w = pred_loss(fake_c_out, r_labels, cls=self.args.weather_loss)   # Weather prediction

        # abs_loss = torch.mean(torch.abs(fake_out - images), [1, 2, 3])

        diff = torch.mean(torch.abs(fake_out - images), [1, 2, 3])
        lmda = torch.mean(torch.abs(pred_labels - r_labels), 1)
        loss_con = torch.mean(diff / (lmda + ep))  # Reconstraction loss

        lmda_con, lmda_w = tuple(self.args.loss_lamda_cw)

        g_loss = g_loss_adv + lmda_con * loss_con + lmda_w * g_loss_w

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
            'variables/lmda': self.lmda,
            'variables/denominator_loss_con': torch.mean(lmda).item()
            })

        self.image_dict.update({
            'io/train': torch.cat([images, fake_out.detach(), rand_images], dim=3),
            })

        return g_loss_adv.item(), loss_con.item(), g_loss_w.item()

    def update_discriminator(self, images, r_labels, labels=None):

        # --- UPDATE(Discriminator) ---#
        self.d_opt.zero_grad()

        # for real
        pred_labels = self.classifier(images).detach()
        # --- master --- #
        pred_labels = F.softmax(pred_labels, dim=1)
        # -------------- #
        # --- experiment1 --- #
        # one-hot
        # pred_labels = torch.eye(self.num_classes)[torch.argmax(pred_labels, dim=1)].to('cuda:0')
        # ------------------- #

        real_d_out_pred = self.discriminator(images, pred_labels)[0]

        # for fake
        fake_out = self.inference(images, r_labels)
        fake_d_out = self.discriminator(fake_out.detach(), r_labels)[0]

        d_loss = dis_hinge(fake_d_out, real_d_out_pred)

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
        g_loss_l1_ = []
        g_loss_adv_ = []
        g_loss_w_ = []
        fake_out_li = []
        d_loss_ = []
        # loss_con_ = []

        images, labels = self.test_random_sample[0]

        blank = torch.zeros_like(images[0]).unsqueeze(0)
        ref_images, ref_labels = self.test_random_sample[1]

        labels = F.one_hot(labels, self.num_classes).float()
        ref_labels = F.one_hot(ref_labels, self.num_classes).float()

        for i in range(self.batch_size):
            with torch.no_grad():
                if ref_labels is None:
                    ref_labels = self.classifier(ref_images)
                    ref_labels = F.softmax(ref_labels, dim=1)
                ref_labels_expand = torch.cat([ref_labels[i]] * self.batch_size).view(-1, self.num_classes)
                fake_out_ = self.inference(images, ref_labels_expand)
                fake_c_out_ = self.classifier(fake_out_)
                if self.args.weather_loss == 'mse':
                    fake_c_out_ = F.softmax(fake_c_out_, dim=1)
                # fake_d_out_ = self.discriminator(fake_out_, labels)[0]
                real_d_out_ = self.discriminator(images, labels)[0]
                fake_d_out_ = self.discriminator(fake_out_, ref_labels_expand)[0]

            # diff = torch.mean(torch.abs(fake_out_ - images), [1, 2, 3])
            # lmda = torch.mean(torch.abs(pred_labels_ - ref_labels_expand), 1)
            # loss_con_ = torch.mean(diff / (lmda + 1e-7))

            fake_out_li.append(fake_out_)
            # g_loss_adv_.append(adv_loss(fake_d_out_, self.real).item())
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
        eval_per_step = 1000 * args.GD_train_ratio
        display_per_step = 1000 * args.GD_train_ratio

        self.all_step = args.num_epoch * len(self.train_set) // self.batch_size

        tqdm_iter = trange(args.num_epoch, desc='Training', leave=True)
        for epoch in tqdm_iter:
            if epoch > 0:
                self.epoch += 1

            for i, (data, rand_data) in enumerate(zip(self.train_loader, self.random_loader)):
                self.global_step += 1

                if self.global_step % eval_per_step == 0:
                    out_path = os.path.join(args.save_dir, self.name, ('cUNet_cls' + '_e{:04d}_s{}.pt').format(self.epoch, self.global_step))
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

                images, con = (d.to('cuda:0') for d in data)
                rand_images, r_con = (d.to('cuda:0') for d in rand_data)

                # torch >= 1.7
                # images = self.train_set.transform(images)
                # rand_images = self.train_set.transform(rand_images)

                if images.size(0) != self.batch_size:
                    continue

                # --- LABEL PREPROCESS --- #
                # rand_labels = self.classifier(rand_images).detach()
                # --- master --- #
                # rand_labels = F.softmax(rand_labels, dim=1)
                # -------------- #
                # --- experiment1 --- #
                # one-hot
                # rand_labels = torch.eye(self.num_classes)[torch.argmax(rand_labels, dim=1)].to('cuda:0')
                # ------------------- #
                # --- experiment2 --- #
                rand_labels = torch.eye(self.num_classes)[r_con].to('cuda:0')  # rand_labels:one_hot, r_con:label[0~4]
                # ------------------- #
                labels = torch.eye(self.num_classes)[con].to('cuda:0')

                # --- TRAINING --- #
                if (self.global_step - 1) % args.GD_train_ratio == 0:
                    g_loss, r_loss, w_loss = self.update_inference(images, rand_labels, rand_images, labels=labels)
                d_loss = self.update_discriminator(images, rand_labels, labels)
                tqdm_iter.set_postfix(OrderedDict(d_loss=d_loss, g_loss=g_loss, r_loss=r_loss, w_loss=w_loss))

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

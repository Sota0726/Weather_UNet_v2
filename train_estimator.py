import os
from args import get_args
from predictor import Predicor

import numpy as np
from tqdm import trange
from collections import OrderedDict

args = get_args()

os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

import torch
from torch.utils.tensorboard import SummaryWriter

if args.amp:
    from apex import amp, optimizers
from ops import l1_loss, adv_loss  # , soft_transform


if __name__ == '__main__':
    estimator = Predicor(args, _type='est')

    transform = estimator.train_set.transform.to('cuda')
    train_loader = estimator.train_loader
    test_loader = estimator.test_loader
    model = estimator.model
    model.to('cuda')

    name = '{}_sampler-{}_PreTrained-{}'.format(args.name, args.sampler, args.pre_trained)

    comment = '_lr-{}_bs-{}_ne-{}_x{}_name-{}'.format(args.lr,
                                                      args.batch_size,
                                                      args.num_epoch,
                                                      args.input_size,
                                                      name)
    writer = SummaryWriter(comment=comment)

    # train setting
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    if args.amp:
        model, opt = amp.initialize(model, opt, opt_level='O1')
    num_classes = estimator.num_classes
    criterion = estimator.criterion

    eval_per_iter = 500
    save_per_epoch = 5
    global_step = 0

    tqdm_iter = trange(args.num_epoch, desc='Training', leave=True)
    for epoch in tqdm_iter:
        loss_li = []
        diff_mse_li = []
        diff_l1_li = []
        for i, data in enumerate(train_loader, start=0):
            inputs, labels = (d.to('cuda') for d in data)
            inputs = transform(inputs)
            tqdm_iter.set_description('Training [ {} step ]'.format(global_step))

            # optimize
            opt.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            if args.amp:
                with amp.scale_loss(loss, opt) as scale_loss:
                    scale_loss.backward()
            else:
                loss.backward()

            opt.step()

            diff_l1 = l1_loss(outputs.detach(), labels)
            diff_mse = adv_loss(outputs.detach(), labels)

            diff_mse_li.append(diff_mse.item())
            diff_l1_li.append(diff_l1.item())

            if global_step % eval_per_iter == 0:
                diff_mse_li_ = []
                diff_l1_li_ = []
                for j, data_ in enumerate(test_loader, start=0):
                    with torch.no_grad():
                        inputs_, labels_ = (d.to('cuda') for d in data_)
                        inputs_ = transform(inputs_)
                        outputs_ = model(inputs_).detach()
                        diff_mse_ = adv_loss(outputs_, labels_)
                        diff_l1_ = l1_loss(outputs_, labels_)
                        diff_mse_li_.append(diff_mse_.item())
                        diff_l1_li_.append(diff_l1_.item())

                # write summary
                train_mse = np.mean(diff_mse_li)
                train_diff_l1 = np.mean(diff_l1_li)
                test_mse = np.mean(diff_mse_li_)
                test_diff_l1 = np.mean(diff_l1_li_)
                tqdm_iter.set_postfix(OrderedDict(train_l1=train_diff_l1, test_l1=test_diff_l1))
                writer.add_scalars('mse_loss', {'train': train_mse,
                                                'test': test_mse}, global_step)
                writer.add_scalars('l1_loss', {'train': train_diff_l1,
                                               'test': test_diff_l1}, global_step)
                diff_mse_li = []
                diff_l1_li = []

            global_step += 1

        if epoch % save_per_epoch == 0:
            out_path = os.path.join(estimator.save_dir, '{}_epoch{}_step{}.pt'.format(args.predictor, str(epoch), str(global_step)))
            torch.save(model, out_path)

    print('Done: training')

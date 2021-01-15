import os
from args import get_args
from predictor import Predictor

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


if __name__ == '__main__':
    classifier = Predictor(args, _type='cls')

    transform = classifier.train_set.transform.to('cuda')
    train_loader = classifier.train_loader
    test_loader = classifier.test_loader
    model = classifier.model
    model.to('cuda')

    # train setting
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = classifier.criterion
    eval_metric = classifier.eval_metric
    global_step = 0
    eval_per_iter = 500
    save_per_epoch = 5
    tqdm_iter = trange(args.num_epoch, desc='Training', leave=True)

    comment = '_lr-{}_bs-{}_ne-{}_x{}_pre-train-{}_sampler-{}_name-{}'.format(
        args.lr,
        args.batch_size,
        args.num_epoch,
        args.input_size,
        args.pre_trained,
        args.sampler,
        args.name
    )
    writer = SummaryWriter(comment=comment)

    if args.amp:
        model, opt = amp.initialize(model, opt, opt_level='O1')

    loss_li = []
    performance_li = []

    for epoch in tqdm_iter:

        for i, data in enumerate(train_loader, start=0):
            tqdm_iter.set_description('Training [ {} step ]'.format(global_step))
            inputs, labels = (d.to('cuda') for d in data)
            if args.dataset != 'i2w':
                inputs = transform(inputs)
            opt.zero_grad()

            outputs = model(inputs)

            loss = criterion(outputs, labels)
            if args.dataset == 'celebA':
                performance = classifier.get_accuracy(outputs.detach(), labels)
            else:
                performance = classifier.get_precision(outputs.detach(), labels)

            loss_li.append(loss.item())
            performance_li.append(performance)

            if args.amp:
                with amp.scale_loss(loss, opt) as scale_loss:
                    scale_loss.backward()
            else:
                loss.backward()

            opt.step()

            if global_step % eval_per_iter == 0:

                loss_li_ = []
                performance_li_ = []
                for j, data_ in enumerate(test_loader):
                    with torch.no_grad():
                        inputs_, labels_ = (d.to('cuda') for d in data_)
                        if args.dataset != 'i2w':
                            inputs_ = transform(inputs_)
                        outputs_ = model(inputs_)
                        loss_ = criterion(outputs_, labels_)
                        if args.dataset == 'celebA':
                            performance_ = classifier.get_accuracy(outputs_, labels_)
                        else:
                            performance_ = classifier.get_precision(outputs_, labels_)
                        loss_li_.append(loss_.item())
                        performance_li_.append(performance_)
                if args.dataset == 'celebA':
                    tqdm_iter.set_postfix(OrderedDict(train_acc=np.mean(performance_li), test_acc=np.mean(performance_li_)))
                else:
                    tqdm_iter.set_postfix(OrderedDict(train_prec=np.mean(performance_li), test_prec=np.mean(performance_li_)))
                writer.add_scalars('loss', {
                    'train': np.mean(loss_li),
                    'test': np.mean(loss_li_)},
                    global_step)
                writer.add_scalars(eval_metric, {
                    'train': np.mean(performance_li),
                    'test': np.mean(performance_li_)},
                    global_step)
                loss_li = []
                performance_li = []

            global_step += 1

        if epoch % save_per_epoch == 0:
            out_path = os.path.join(classifier.save_dir, '{}_epoch{}_step{}.pt'.format(args.predictor, str(epoch), str(global_step)))
            torch.save(model, out_path)

    print('Done: training')

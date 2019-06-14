"""
Train CMC with AlexNet
"""
from __future__ import print_function

import os
import sys
import time
import torch
import torch.backends.cudnn as cudnn
import argparse
import socket

import tensorboard_logger as tb_logger

from torchvision import transforms
from dataset import RGB2Lab, ImageFolderInstance
from util import adjust_learning_rate, AverageMeter
from models.alexnet import alexnet
from NCE.NCEAverage import NCEAverage
from NCE.NCECriterion import NCECriterion

from spawn import spawn


def parse_option():

    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10, help='print frequency')
    parser.add_argument('--tb_freq', type=int, default=500, help='tb frequency')
    parser.add_argument('--save_freq', type=int, default=5, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=256, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=32, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=400, help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.03, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='250,300,350', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    # resume path
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')

    # model definition
    parser.add_argument('--model', type=str, default='alexnet', choices=['alexnet'])
    parser.add_argument('--nce_k', type=int, default=4096)
    parser.add_argument('--nce_t', type=float, default=0.07)
    parser.add_argument('--nce_m', type=float, default=0.5)
    parser.add_argument('--feat_dim', type=int, default=128, help='dim of feat for inner product')

    # specify folder
    parser.add_argument('--data_folder', type=str, default=None, help='path to data')
    parser.add_argument('--model_path', type=str, default=None, help='path to save model')
    parser.add_argument('--tb_path', type=str, default=None, help='path to tensorboard')

    opt = parser.parse_args()

    if (opt.data_folder is None) or (opt.save_path is None) or (opt.tb_path is None):
        raise ValueError('one or more of the folders is None: data_folder | model_path | tb_path')

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = 'memory_nce_{}_{}_lr_{}_decay_{}_bsz_{}'.format(opt.nce_k, opt.model, opt.learning_rate,
                                                                     opt.weight_decay, opt.batch_size)

    opt.model_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.model_folder):
        os.makedirs(opt.model_folder)

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    if not os.path.isdir(opt.data_folder):
        raise ValueError('data path not exist: {}'.format(opt.data_folder))

    return opt


def get_train_loader(args):
    """get the train loader"""
    data_folder = os.path.join(args.data_folder, 'train')
    normalize = transforms.Normalize(mean=[(0 + 100) / 2, (-86.183 + 98.233) / 2, (-107.857 + 94.478) / 2],
                                     std=[(100 - 0) / 2, (86.183 + 98.233) / 2, (107.857 + 94.478) / 2])
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        RGB2Lab(),
        transforms.ToTensor(),
        normalize,
    ])
    train_dataset = ImageFolderInstance(data_folder, transform=train_transform)
    train_sampler = None

    # train loader
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.num_workers, pin_memory=True, sampler=train_sampler)

    # num of samples
    n_data = len(train_dataset)
    print('number of samples: {}'.format(n_data))

    return train_loader, n_data


def set_model(args, n_data):
    # set the model
    if args.model == 'alexnet':
        model = alexnet(args.feat_dim)
    else:
        raise ValueError('model not supported yet {}'.format(args.model))
    contrast = NCEAverage(args.feat_dim, n_data, args.nce_k, args.nce_t, args.nce_m)
    criterion_l = NCECriterion(n_data)
    criterion_ab = NCECriterion(n_data)

    if torch.cuda.is_available():
        model = model.cuda()
        contrast = contrast.cuda()
        criterion_ab = criterion_ab.cuda()
        criterion_l = criterion_l.cuda()
        cudnn.benchmark = True

    return model, contrast, criterion_ab, criterion_l


def set_optimizer(args, model):
    # return optimizer
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.learning_rate,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    return optimizer


def train(epoch, train_loader, model, contrast, criterion_l, criterion_ab, optimizer, opt):
    """
    one epoch training
    """
    model.train()
    contrast.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    l_loss_meter = AverageMeter()
    ab_loss_meter = AverageMeter()
    l_prob_meter = AverageMeter()
    ab_prob_meter = AverageMeter()

    end = time.time()
    for idx, (inputs, _, index) in enumerate(train_loader):
        data_time.update(time.time() - end)

        bsz = inputs.size(0)
        inputs = inputs.float()
        if torch.cuda.is_available():
            index = index.cuda(async=True)
            inputs = inputs.cuda()

        # ===================forward=====================
        feat_l, feat_ab = model(inputs)
        out_l, out_ab = contrast(feat_l, feat_ab, index)

        l_loss = criterion_l(out_l)
        ab_loss = criterion_ab(out_ab)
        l_prob = out_l[:, 0].mean()
        ab_prob = out_ab[:, 0].mean()

        loss = l_loss + ab_loss

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================meters=====================
        losses.update(loss.item(), bsz)
        l_loss_meter.update(l_loss.item(), bsz)
        l_prob_meter.update(l_prob.item(), bsz)
        ab_loss_meter.update(ab_loss.item(), bsz)
        ab_prob_meter.update(ab_prob.item(), bsz)

        batch_time.update(time.time() - end)
        end = time.time()

        # tensorboard logger
        pass

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'l_p {lprobs.val:.3f} ({lprobs.avg:.3f})\t'
                  'ab_p {abprobs.val:.3f} ({abprobs.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, lprobs=l_prob_meter,
                   abprobs=ab_prob_meter))
            print(out_l.shape)
            sys.stdout.flush()

    return l_loss_meter.avg, l_prob_meter.avg, ab_loss_meter.avg, ab_prob_meter.avg


def main():

    # parse the args
    args = parse_option()

    # set the loader
    train_loader, n_data = get_train_loader(args)

    # set the model
    model, contrast, criterion_ab, criterion_l = set_model(args, n_data)

    # set the optimizer
    optimizer = set_optimizer(args, model)

    # optionally resume from a checkpoint
    args.start_epoch = 1
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch'] + 1
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            contrast.load_state_dict(checkpoint['contrast'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # tensorboard
    logger = tb_logger.Logger(logdir=args.tb_folder, flush_secs=2)

    # routine
    for epoch in range(args.start_epoch, args.epochs + 1):

        adjust_learning_rate(epoch, args, optimizer)
        print("==> training...")

        time1 = time.time()
        l_loss, l_prob, ab_loss, ab_prob = train(epoch, train_loader, model, contrast, criterion_l, criterion_ab,
                                                 optimizer, args)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        # tensorboard logger
        logger.log_value('l_loss', l_loss, epoch)
        logger.log_value('l_prob', l_prob, epoch)
        logger.log_value('ab_loss', ab_loss, epoch)
        logger.log_value('ab_prob', ab_prob, epoch)

        # save model
        if epoch % args.save_freq == 0:
            print('==> Saving...')
            state = {
                'opt': args,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'contrast': contrast.state_dict(),
                'epoch': epoch,
            }
            save_file = os.path.join(args.model_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            torch.save(state, save_file)

        pass


if __name__ == '__main__':
    main()

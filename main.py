import torch
import time
import shutil
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
import math
import numpy as np
from dataset.inat import INaturalist
from dataset.imagenet import ImageNetLT
import warnings
import torch.backends.cudnn as cudnn
import random
from randaugment import rand_augment_transform
import torchvision
from utils.shotacc import  shot_acc
from utils.runtime import checkpoint_prefix_mode, str2bool
import argparse
import os
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast
import logging
from expert_model.model import EMAResNet50Model
from loss_sd import MixedCEKDLoss


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='imagenet', choices=['inat', 'imagenet'])
parser.add_argument('--data', required=True, metavar='DIR',
                    help='path to the dataset root')
parser.add_argument('--arch', default='resnext50', choices=['resnet50', 'resnext50'])
parser.add_argument('--num_experts', default=3, type=int,
                    help='number of experts (M in the paper)')
parser.add_argument('--workers', default=12, type=int)
parser.add_argument('--epochs', default=200, type=int)
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--schedule', default=[160, 180], nargs='*', type=int,
                    help='epoch milestones for step LR when --cos is false')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=5e-4, type=float,
                    metavar='W', help='SGD weight decay (default: 5e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=500, type=int,
                    metavar='N', help='logging interval in iterations (default: 500)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--eta', default=0.0, type=float, help='DED loss weight (eta in the paper)')
parser.add_argument('--beta', default=0.0, type=float, help='SC loss weight (beta in the paper)')
parser.add_argument('--warmup_epochs', default=0, type=int,
                    help='number of learning-rate warmup epochs')
parser.add_argument('--cos', default=True, type=str2bool,
                    help='use cosine learning-rate decay')
parser.add_argument('--use_norm', default=True, type=str2bool,
                    help='use the normalized classifier described in the paper')
parser.add_argument('--reduce_dimension', default=True, type=str2bool,
                    help='reduce each expert branch dimension as described in the paper')
parser.add_argument('--randaug_m', default=10, type=int, help='fixed RandAugment magnitude')
parser.add_argument('--randaug_n', default=2, type=int, help='fixed number of RandAugment operations')
parser.add_argument('--seed', default=None, type=int, help='seed for initializing training')
parser.add_argument('--evaluate', action='store_true', help='evaluate a loaded checkpoint')

global best_acc1, best_many, best_med, best_few


def main():

    args = parser.parse_args()
    if args.evaluate and not args.resume.strip():
        parser.error("--evaluate requires a nonempty --resume checkpoint path")
    if args.evaluate and not os.path.isfile(args.resume):
        parser.error(f"evaluation checkpoint not found: {args.resume}")

    args.store_name = '_'.join(
        [args.dataset, args.arch, 'num_experts', str(args.num_experts),'batchsize', str(args.batch_size), 'epochs', str(args.epochs), 'wd', str(args.weight_decay),  'lr', str(args.lr), 'eta', str(args.eta), 'beta', str(args.beta),
            'evaluate', str(args.evaluate)])
    args.store_name = os.path.join('logs', args.store_name)
    
    os.makedirs(args.store_name, exist_ok=True)
    log_filename = os.path.join(args.store_name, 'training.log')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler()
        ],
        force=True
    )
    logging.info("=" * 80)
    logging.info("Arguments: %s", vars(args))
    logging.info("PyTorch version: %s", torch.__version__)
    logging.info("Visible GPUs: %d", torch.cuda.device_count())

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')
    ngpus_per_node = torch.cuda.device_count()
    main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
    
    num_classes = 1000 if args.dataset == 'imagenet' else 8142

    # Build the online encoder and its EMA teacher.
    print("=> creating model '{}'".format(args.arch))
    if args.arch in ['resnet50', 'resnext50']:
        model = EMAResNet50Model(arch=args.arch, num_classes=num_classes, reduce_dimension=args.reduce_dimension, use_norm=args.use_norm, 
                num_experts=args.num_experts)

    else:
        raise NotImplementedError('This model is not supported')
    print(model)

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        model = torch.nn.DataParallel(model).cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    best_acc1, best_many, best_med, best_few = 0.0, 0.0, 0.0, 0.0
    scaler = GradScaler()

    # Resume strictly across DataParallel and single-GPU checkpoints.
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location="cpu")
            args.start_epoch = checkpoint['epoch']
            best_acc1 = float(checkpoint.get("best_acc1", 0.0))
            best_many = float(checkpoint.get("best_many", 0.0))
            best_med = float(checkpoint.get("best_med", 0.0))
            best_few = float(checkpoint.get("best_few", 0.0))

            state_dict = checkpoint['state_dict']
            prefix_mode = checkpoint_prefix_mode(state_dict.keys(), model.state_dict().keys())
            if prefix_mode == "strip":
                torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(
                    state_dict, "module."
                )
                model.load_state_dict(state_dict, strict=True)
            elif prefix_mode == "inner":
                model.module.load_state_dict(state_dict, strict=True)
            else:
                model.load_state_dict(state_dict, strict=True)

            optimizer.load_state_dict(checkpoint['optimizer'])
            if "scaler" in checkpoint:
                scaler.load_state_dict(checkpoint["scaler"])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = args.seed is None

    txt_train = f'dataset/ImageNet_LT/ImageNet_LT_train.txt' if args.dataset == 'imagenet' \
        else f'dataset/iNaturalist18/iNaturalist18_train.txt'
    txt_val = f'dataset/ImageNet_LT/ImageNet_LT_test.txt' if args.dataset == 'imagenet' \
        else f'dataset/iNaturalist18/iNaturalist18_val.txt'

    normalize = transforms.Normalize((0.466, 0.471, 0.380), (0.195, 0.194, 0.192)) if args.dataset == 'inat' \
        else transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    rgb_mean = (0.485, 0.456, 0.406)
    ra_params = dict(translate_const=int(224 * 0.45), img_mean=tuple([min(255, round(255 * x)) for x in rgb_mean]), )
    ra_params2 = dict(translate_const=int(96 * 0.45), img_mean=tuple([min(255, round(255 * x)) for x in rgb_mean]), )


    train_transform_large = [
        transforms.RandomResizedCrop(224, scale=(0.08, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.0)
        ], p=1.0),
        rand_augment_transform('rand-n{}-m{}-mstd0.5'.format(args.randaug_n, args.randaug_m), ra_params),
        transforms.ToTensor(),
        normalize,
    ]
    
    train_transform_small = [
        transforms.RandomResizedCrop(96, scale=(0.08, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.0)
        ], p=1.0),
        rand_augment_transform('rand-n{}-m{}-mstd0.5'.format(args.randaug_n, args.randaug_m), ra_params2),
        transforms.ToTensor(),
        normalize,
    ]

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])

    train_transform = [transforms.Compose(train_transform_large), transforms.Compose(train_transform_small)]

    val_dataset = INaturalist(
        root=args.data,
        txt=txt_val,
        transform=val_transform, train=False,
    ) if args.dataset == 'inat' else ImageNetLT(
        root=args.data,
        txt=txt_val,
        transform=val_transform, train=False)

    train_dataset = INaturalist(
        root=args.data,
        txt=txt_train,
        transform=train_transform
    ) if args.dataset == 'inat' else ImageNetLT(
        root=args.data,
        txt=txt_train,
        transform=train_transform)

    cls_num_list = train_dataset.cls_num_list
    args.cls_num = len(cls_num_list)

    train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    criterion = MixedCEKDLoss(cls_num_list).cuda(args.gpu)


    if args.evaluate:
        txt_test = f'dataset/ImageNet_LT/ImageNet_LT_test.txt' if args.dataset == 'imagenet' \
            else f'dataset/iNaturalist18/iNaturalist18_val.txt'
        test_dataset = INaturalist(
            root=args.data,
            txt=txt_test,
            transform=val_transform, train=False
        ) if args.dataset == 'inat' else ImageNetLT(
            root=args.data,
            txt=txt_test,
            transform=val_transform, train=False)

        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
        acc1, many, med, few = validate(train_loader, test_loader, model, criterion, 1, args)
        return
    
    for epoch in range(args.start_epoch, args.epochs):
        adjust_lr(optimizer, epoch, args)

        train(train_loader, model, criterion,  optimizer, epoch, scaler, args)

        acc1, many, med, few  = validate(train_loader, val_loader, model, criterion, epoch, args)

        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        

        if is_best:
            best_many = many
            best_med = med
            best_few = few
        
        
        logging.info(f'Best Prec@1: {best_acc1:.3f} | '
                f'Many Prec@1: {best_many:.3f} | '
                f'Med Prec@1: {best_med:.3f} | '
                f'Few Prec@1: {best_few:.3f} | '
                )

        save_checkpoint(args, {
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'best_many': best_many,
            'best_med': best_med,
            'best_few': best_few,
            'optimizer': optimizer.state_dict(),
            'scaler': scaler.state_dict(),
        }, is_best)


def train(train_loader, model, criterion, optimizer, epoch, scaler, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Time', ':6.3f')
    loss_all = AverageMeter('Loss', ':.4e')

    model.train()
    end = time.time()
    for i, data in enumerate(train_loader):
        data_time.update(time.time() - end)

        inputs, targets = data
        inputs_s = inputs[2].cuda(non_blocking=True)

        inputs_v1, inputs_v2 = inputs[0].cuda(non_blocking=True), inputs[1].cuda(non_blocking=True) 
        inputs_q = torch.cat([inputs_v1, inputs_v2], dim=0)
        targets = targets.cuda(non_blocking=True)
        batch_size = targets.shape[0]
        
        # SC uses a convex mixture of two full-resolution views.
        mixer = np.random.beta(1, 1)
        
        inputs_mix = inputs_v1 * mixer + inputs_v2 * (1 - mixer)

        inputs_all = torch.cat([inputs_q, inputs_mix], dim=0)

        with autocast():
            output = model(inputs_all)
            output_s = model(inputs_s)
            
            with torch.no_grad():
                output_k = model(inputs_q, teacher=True)

        
        logits = output["logits"].transpose(0, 1)[:, :2*batch_size, :]
        logits_k = output_k['logits'].transpose(0, 1)
        logits_s = output_s["logits"].transpose(0, 1)
        logits_m = output["logits"].transpose(0, 1)[:, 2*batch_size:, :]

        with autocast():
            ce_loss, sc_loss, ded_loss = criterion(logits, logits_k, logits_m, logits_s, targets, args)

        # Linearly ramp SC and DED during the first 20 epochs.
        loss = ce_loss + min(epoch / 20, 1.) * (args.beta * sc_loss + args.eta * ded_loss)

        loss_all.update(loss.item(), batch_size)
        
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            output = ('Epoch: [{0}][{1}/{2}] \t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time, data_time=data_time,
                loss=loss_all))
            logging.info(output)

def validate(train_loader, val_loader, model, criterion_ce, epoch, args):
    model.eval()
    batch_time = AverageMeter('Time', ':6.3f')
    ce_loss_all = AverageMeter('CE_Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    prediction_chunks = []
    label_chunks = []
    
    with torch.no_grad():
        end = time.time()
        for i, data in enumerate(val_loader):
            inputs, targets = data
            inputs, targets = inputs.cuda(non_blocking=True), targets.cuda(non_blocking=True)
            batch_size = targets.size(0)
            
            # Final inference uses the EMA teacher.
            output = model(inputs, teacher=True)
            logits = output['output']

            ce_loss = F.cross_entropy(logits, targets)
            _, preds = F.softmax(logits.detach(), dim=1).max(dim=1)
            prediction_chunks.append(preds.cpu())
            label_chunks.append(targets.detach().cpu())

            acc1 = accuracy(logits, targets, topk=(1,))
            ce_loss_all.update(ce_loss.item(), batch_size)
            top1.update(acc1[0].item(), batch_size)


            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0 or i == len(val_loader) - 1:
                output_info = ('Test: [{0}/{1}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'CE_Loss {ce_loss.val:.4f} ({ce_loss.avg:.4f})\t'
                          'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, ce_loss=ce_loss_all, top1=top1, ))
                logging.info(output_info)

        total_preds = torch.cat(prediction_chunks, dim=0)
        total_labels = torch.cat(label_chunks, dim=0)
        many_acc_top1, median_acc_top1, low_acc_top1 = shot_acc(
            total_preds, total_labels, train_loader.dataset.cls_num_list, acc_per_cls=False)

        return top1.avg, many_acc_top1, median_acc_top1, low_acc_top1

def save_checkpoint(args, state, is_best):
    filename = os.path.join(args.store_name, 'ckpt.pth.tar')
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename.replace('pth.tar', 'best.pth.tar'))

class TwoCropTransform:
    def __init__(self, transform1):
        self.transform1 = transform1

    def __call__(self, x):
        return [self.transform1(x), self.transform1(x)]


def adjust_lr(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if epoch < args.warmup_epochs:
        lr = lr / args.warmup_epochs * (epoch + 1)
    elif args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * (epoch - args.warmup_epochs + 1) / (args.epochs - args.warmup_epochs + 1)))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred)).contiguous()

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()

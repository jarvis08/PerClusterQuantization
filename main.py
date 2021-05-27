import argparse
import os
import time
import shutil
from datetime import datetime
from tqdm import tqdm
import logging
import io

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

# from tensorboardX import SummaryWriter
from torchsummary import summary

import torchvision
import torchvision.transforms as transforms
from models import *
import json

from models import *
from utils import *
from models.quantization import *

parser = argparse.ArgumentParser(description='PyTorch Cifar10 Training')
parser.add_argument('--epoch', default=100, type=int, help='Number of epochs to fine-tune')
parser.add_argument('--batch', default=128, type=int, help='Mini-batch size')
parser.add_argument('--lr', default=0.1, type=float, help='Initial Learning Rate')
parser.add_argument('--weight_decay', default=1e-4, type=float, help='Weight-decay value')
parser.add_argument('--arch', default='resnet18', type=str, help='Architecture to quantize')
parser.add_argument('--dataset', default='cifar10', type=str, help='Dataset to use')
parser.add_argument('--bit', default=32, type=int, help='Target bit-width to be quantized (value 32 means pretraining)')
parser.add_argument('--cluster', default=1, type=int, help='Number of clusters')
parser.add_argument('--pre_path', default='', type=str, help="Pretrained model's path")
parser.add_argument('--mode', default='eval', type=str, help="pre or tune or eval")
parser.add_argument('--fused', default=False, type=bool, help="Path of Model, fused or not")
parser.add_argument('--darknet_data', default=False, type=bool, help="Evaluate model with dataset preprocessed with darknet")

args = parser.parse_args()
max_epoch = args.epoch
arch = args.arch
batch_size = args.batch
dataset = args.dataset
target_bit = args.bit
initial_lr = args.lr
weight_decay = args.weight_decay
pretrained_model = args.pre_path
mode = args.mode
if mode == 'fine':
    max_epoch = 1
fused = args.fused
use_darknet_preprocessed_dataset = args.darknet_data
print(vars(args))

n_class = 0
if dataset == 'cifar10' or dataset == 'svhn':
    n_class = 10
elif dataset == 'imagenet':
    n_class = 1000


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
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


def train_epoch(train_loader, model, criterion, optimizer, epoch):
    losses = AverageMeter()
    top1 = AverageMeter()

    model.train()
    with tqdm(train_loader, unit="batch", ncols=90) as t:
        for i, (input, target) in enumerate(t):
            t.set_description("Epoch {}".format(epoch))

            input, target = input.cuda(), target.cuda()
            output = model(input)
            loss = criterion(output, target)
            prec = accuracy(output, target)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec.item(), input.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            t.set_postfix(loss=losses.avg, acc=top1.avg)


def validate(test_loader, model, criterion, is_darknet):
    losses = AverageMeter()
    top1 = AverageMeter()

    model.eval()
    with torch.no_grad():
        if is_darknet:
            for i in range(1000):
                _in = test_loader[0][i]
                _targ = test_loader[1][i]
                input, target = _in.cuda(), _targ.cuda()
                output = model(input)
                loss = criterion(output, target)
                prec = accuracy(output, target)[0]
                losses.update(loss.item(), input.size(0))
                top1.update(prec.item(), input.size(0))
            print("Acc : {}".format(top1.avg))
        else:
            with tqdm(test_loader, unit="batch", ncols=90) as t:
                for i, (input, target) in enumerate(t):
                    t.set_description("Validate")

                    input, target = input.cuda(), target.cuda()
                    output = model(input)
                    loss = criterion(output, target)
                    prec = accuracy(output, target)[0]
                    losses.update(loss.item(), input.size(0))
                    top1.update(prec.item(), input.size(0))
                    # exit()
                    t.set_postfix(loss=losses.avg, acc=top1.avg)
    return top1.avg


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def save_checkpoint(state, is_best, path, mode):
    filepath = os.path.join(path, 'checkpoint_{}.pth'.format(mode))
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(path, 'model_best_{}.pth.tar'.format(mode)))


if __name__=='__main__':
    save_dir = "result"
    if mode != "eval":
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_dir = os.path.join(save_dir, mode)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_dir = os.path.join(save_dir, arch + '_' + str(target_bit) + 'bit')
        now = datetime.now().strftime("%m-%d-%H%M")
        save_dir = os.path.join(save_dir, now)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        with open(os.path.join(save_dir, "params.json"), 'w') as f:
            json.dump(vars(args), f, indent=4)

    best_prec = 0
    use_gpu = torch.cuda.is_available()
    assert use_gpu, "Code works on GPU"

    model = None
    pre_model = None
    arch_sup = ['resnet18', 'alexnet']
    assert arch in arch_sup, 'Architecture not support!'
    if mode == 'eval':
        if fused:
            model = create_fused_resnet18(model, bit=target_bit, num_classes=n_class)
            checkpoint = torch.load(pretrained_model)
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            model = fuse_resnet18(model)
        else:
            model = resnet18(num_classes=n_class)
            checkpoint = torch.load(pretrained_model)
            model.load_state_dict(checkpoint['state_dict'], strict=False)
    elif mode == 'fine':
        model = resnet18(num_classes=n_class)
        checkpoint = torch.load(pretrained_model)
        model.load_state_dict(checkpoint['state_dict'], strict=False)

        fused_model = create_fused_resnet18(model, bit=target_bit, num_classes=n_class)
        model = set_fused_resnet18_params(fused_model, model)
    else:
        model = resnet18(num_classes=n_class)
    # elif arch == 'alexnet':
    #     model = alexnet(dataset=dataset, pretrained=False, num_classes=n_class)

    summary(model, (3, 32, 32))

    model.cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=initial_lr, momentum=0.9, weight_decay=weight_decay)

    opt_scheduler = None
    if arch == 'resnet18':
        opt_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)
    elif arch == 'alexnet':
        opt_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 80], gamma=0.1)
    cudnn.benchmark = True

    print('Load CIFAR-10 dataset..')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_loader = None
    if mode != 'eval':
        train_dataset = torchvision.datasets.CIFAR10(
            root='./data',
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    test_dataset = torchvision.datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]))
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)

    if use_darknet_preprocessed_dataset:
        test_loader = load_preprocessed_cifar10_from_darknet()


    if mode == "eval":
        validate(test_loader, model, criterion, use_darknet_preprocessed_dataset)
        exit()

    for e in range(1, max_epoch + 1):
        train_epoch(train_loader, model, criterion, optimizer, e)
        opt_scheduler.step()

        prec = validate(test_loader, model, criterion, use_darknet_preprocessed_dataset)

        is_best = prec > best_prec
        best_prec = max(prec, best_prec)
        print('best acc: {:1f}'.format(best_prec))
        save_checkpoint({
            'epoch': e,
            'state_dict': model.state_dict(),
            'best_prec': best_prec,
            'optimizer': optimizer.state_dict(),
        }, is_best, save_dir, mode)

    #save_fused_network_in_darknet_form(model, arch)
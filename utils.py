import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os
import shutil
from datetime import datetime
import json
from tqdm import tqdm


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


def save_checkpoint(state, is_best, path):
    filepath = os.path.join(path, 'checkpoint.pth')
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(path, 'model_best.pth.tar'))


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


def validate(test_loader, model, criterion):
    losses = AverageMeter()
    top1 = AverageMeter()

    model.eval()
    with torch.no_grad():
        with tqdm(test_loader, unit="batch", ncols=90) as t:
            for i, (input, target) in enumerate(t):
                t.set_description("Validate")

                input, target = input.cuda(), target.cuda()
                output = model(input)
                loss = criterion(output, target)
                prec = accuracy(output, target)[0]
                losses.update(loss.item(), input.size(0))
                top1.update(prec.item(), input.size(0))

                t.set_postfix(loss=losses.avg, acc=top1.avg)
    return top1.avg


def validate_darknet_dataset(test_loader, model, criterion):
    losses = AverageMeter()
    top1 = AverageMeter()

    model.eval()
    with torch.no_grad():
        #for i in range(1000):
        for i in range(1):
            _in = test_loader[0][i]
            _targ = test_loader[1][i]
            input, target = _in.cuda(), _targ.cuda()
            output = model(input)
            loss = criterion(output, target)
            prec = accuracy(output, target)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec.item(), input.size(0))
        print("Acc : {}".format(top1.avg))
    return top1.avg


def get_normalizer(dataset):
    if dataset == 'imagenet':
       return transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    else:
        return transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))


def get_train_loader(dataset, normalizer, batch_size):
    train_loader = None
    if dataset == 'imagenet':
        print("ImageNet pretraining not developed yet")
        exit()
    else:
        train_dataset = torchvision.datasets.CIFAR10(
            root='./data',
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalizer,
            ]))
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    return train_loader


def get_test_loader(dataset, normalizer, batch_size):
    test_loader = None
    if dataset == 'imagenet':
        print("ImageNet pretraining not developed yet")
        exit()
    else:
        test_dataset = torchvision.datasets.CIFAR10(
            root='./data',
            train=False,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                normalizer,
            ]))
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    return test_loader


def add_path(prev_path, to_add):
    path = os.path.join(prev_path, to_add)
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def set_save_dir(args, quantize=False):
    save_dir = 'result'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_dir = os.path.join(save_dir, args.mode)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_dir = os.path.join(save_dir, args.arch + '_' + str(args.bit) + 'bit')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    now = datetime.now().strftime("%m-%d-%H%M")
    save_dir = os.path.join(save_dir, now)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(os.path.join(save_dir, "params.json"), 'w') as f:
        json.dump(vars(args), f, indent=4)
    return save_dir


def load_preprocessed_cifar10_from_darknet():
    input = torch.tensor(np.fromfile("result/darknet/cifar_test_dataset.bin", dtype='float32').reshape((10000, 1, 3, 32, 32)))
    target = torch.tensor(np.fromfile("result/darknet/cifar_test_target.bin", dtype='int32').reshape((10000, 1)), dtype=torch.long)
    return input, target
import torch
import torchvision
import torchvision.models as vision_models
import torchvision.transforms as transforms

import numpy as np
from tqdm import tqdm

import os
import shutil
from datetime import datetime
import json
import logging

from models.kmeans import get_pcq_batch


def set_logger(path):
    logging.basicConfig(filename=os.path.join(path, "train.log"), level=logging.DEBUG)
    logger = logging.getLogger()
    return logger


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
        shutil.copyfile(filepath, os.path.join(path, 'best.pth'))


def train_epoch(model, train_loader, criterion, optimizer, epoch, logger):
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

            logger.debug("[Epoch] {}, step {}/{} [Loss] {:.5f} (avg: {:.5f}) [Score] {:.3f} (avg: {:.3f})"
                         .format(epoch, i + 1, len(t), loss.item(), losses.avg, prec.item(), top1.avg))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            t.set_postfix(loss=losses.avg, acc=top1.avg)


def validate(model, test_loader, criterion, logger=None):
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

    if logger:
        logger.debug("[Validation] Loss: {:.5f}, Score: {:.3f}".format(losses.avg, top1.avg))
    return top1.avg


def pcq_validate(model, test_loader, criterion, kmeans, num_partitions, logger=None):
    losses = AverageMeter()
    top1 = AverageMeter()

    model.eval()
    with torch.no_grad():
        with tqdm(test_loader, unit="batch", ncols=90) as t:
            for i, (input, target) in enumerate(t):
                t.set_description("Validate")

                input, target, cluster = get_pcq_batch(kmeans, input, target, num_partitions)
                model.set_cluster_information_of_batch(cluster.cuda())

                input, target = input.cuda(), target.cuda()
                output = model(input)
                loss = criterion(output, target)
                prec = accuracy(output, target)[0]
                losses.update(loss.item(), input.size(0))
                top1.update(prec.item(), input.size(0))

                t.set_postfix(loss=losses.avg, acc=top1.avg)

    if logger:
        logger.debug("[Validation] Loss: {:.5f}, Score: {:.3f}".format(losses.avg, top1.avg))
    return top1.avg


def validate_darknet_dataset(model, test_loader, criterion):
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


def load_dnn_model(args, tools):
    model = None
    if args.quantized:
        model = tools.quantized_model_initializer(bit=args.bit, num_clusters=args.cluster)
    elif args.fused:
        model = tools.fused_model_initializer(bit=args.bit, smooth=args.smooth, quant_noise=args.quant_noise, q_prob=args.q_prob)
    else:
        if args.dataset == 'imagenet':
            if args.arch == 'MobileNetV3':
                return vision_models.mobilenet_v3_small(pretrained=True)
            elif args.arch == 'ResNet18':
                exit()
        else:
            model = tools.pretrained_model_initializer()

    checkpoint = torch.load(args.dnn_path)
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    return model


def get_normalizer(dataset):
    if dataset == 'imagenet':
       return transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    else:
        return transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))


def get_train_loader(args, normalizer):
    if args.dataset == 'imagenet':
        train_dataset = torchvision.datasets.ImageFolder(root=os.path.join(args.imagenet, 'train'),
                                                        transform=transforms.Compose([
                                                            transforms.Resize(256),
                                                            transforms.CenterCrop(224),
                                                            transforms.ToTensor(),
                                                            normalizer,
                                                        ]))
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch, shuffle=True, num_workers=10)
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
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch, shuffle=True, num_workers=2)
    return train_loader


def get_test_loader(args, normalizer):
    if args.dataset == 'imagenet':
        test_dataset = torchvision.datasets.ImageFolder(root=os.path.join(args.imagenet, 'val'),
                                                        transform=transforms.Compose([
                                                            transforms.Resize(256),
                                                            transforms.CenterCrop(224),
                                                            transforms.ToTensor(),
                                                            normalizer,
                                                        ]))
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.val_batch, shuffle=False, num_workers=2)
    else:
        test_dataset = torchvision.datasets.CIFAR10(
            root='./data',
            train=False,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                normalizer,
            ]))
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.val_batch, shuffle=False, num_workers=2)
    return test_loader


def add_path(prev_path, to_add):
    path = os.path.join(prev_path, to_add)
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def set_kmeans_dir(args):
    path = add_path('', 'result')
    path = add_path(path, 'kmeans')
    path = add_path(path, args.dataset)
    path = add_path(path, datetime.now().strftime("%m-%d-%H%M"))
    with open(os.path.join(path, "params.json"), 'w') as f:
        kmeans_args = {'k': args.cluster, 'num_partitions': args.partition,
                       'epoch': args.kmeans_epoch, 'batch': args.batch}
        json.dump(kmeans_args, f, indent=4)
    return path


def set_save_dir(args):
    path = add_path('', 'result')
    path = add_path(path, args.mode)
    path = add_path(path, args.arch + '_' + str(args.bit) + 'bit')
    path = add_path(path, datetime.now().strftime("%m-%d-%H%M"))
    with open(os.path.join(path, "params.json"), 'w') as f:
        json.dump(vars(args), f, indent=4)
    return path


def load_preprocessed_cifar10_from_darknet():
    input = torch.tensor(np.fromfile("result/darknet/cifar_test_dataset.bin", dtype='float32').reshape((10000, 1, 3, 32, 32)))
    target = torch.tensor(np.fromfile("result/darknet/cifar_test_target.bin", dtype='int32').reshape((10000, 1)), dtype=torch.long)
    return input, target

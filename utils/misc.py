import torch
import torchvision.models as vision_models
# import horovod.torch as hvd

import numpy as np
from tqdm import tqdm

import os
import shutil
from datetime import datetime
import json
import logging
import random
from time import time


class RuntimeHelper(object):
    """
        apply_fake_quantization : Flag used in layers
        batch_cluster           : Cluster information of current batch
        kmeans                  : Trained K-Means model's object
        pcq_initialized         : Initialize mean and variance of BatchNorm
    """

    def __init__(self):
        self.apply_fake_quantization = False
        self.batch_cluster = None
        self.qn_prob = 0.0

        self.range_update_phase = False
        self.pcq_initialized = False

        self.num_clusters = None
        self.data_per_cluster = None

    def set_pcq_arguments(self, args):
        self.num_clusters = args.cluster
        self.data_per_cluster = args.data_per_cluster


class Phase2DataLoader(object):
    def __init__(self, loader, num_clusters, num_data_per_cluster):
        self.data_loader = loader
        self.len_loader = len(loader)
        self.batch_size = num_clusters * num_data_per_cluster

        bc = []
        for c in range(num_clusters):
            per_cluster = [c for _ in range(num_data_per_cluster)]
            bc += per_cluster
        self.batch_cluster = torch.cuda.LongTensor(bc).cuda()

        self.generator = iter(self.data_loader)
        self.iterated = 0

    def initialize_phase2_generator(self):
        self.generator = iter(self.data_loader)
        self.iterated = 0

    def get_next_data(self):
        # if self.iterated == self.len_loader:
        #     self.initialize_phase2_generator()
        # input, _ = next(self.generator)
        # self.iterated += 1

        data = next(self.generator)
        if isinstance(data[0], dict):
            input = data[0]['data'].cuda(non_blocking=True)
        else:
            input = data[0].cuda(non_blocking=True)

        return input


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


def save_pretraining_model_checkpoint(state, is_best, path, epoch=None):
    if epoch is not None:
        filepath = os.path.join(path, 'checkpoint_{}.pth'.format(epoch))
    else:
        filepath = os.path.join(path, 'checkpoint.pth')
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(path, 'best.pth'))


def save_checkpoint(state, is_best, path):
    filepath = os.path.join(path, 'checkpoint.pth')
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(path, 'best.pth'))


def train_epoch(model, train_loader, criterion, optimizer, epoch, logger, hvd=None):
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

            if hvd:
                if hvd.rank() == 0:
                    logger.debug("[Epoch] {}, step {}/{} [Loss] {:.5f} (avg: {:.5f}) [Score] {:.3f} (avg: {:.3f})"
                                 .format(epoch, i + 1, len(t), loss.item(), losses.avg, prec.item(), top1.avg))
            else:
                logger.debug("[Epoch] {}, step {}/{} [Loss] {:.5f} (avg: {:.5f}) [Score] {:.3f} (avg: {:.3f})"
                             .format(epoch, i + 1, len(t), loss.item(), losses.avg, prec.item(), top1.avg))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            t.set_postfix(loss=losses.avg, acc=top1.avg)


def validate(model, test_loader, criterion, logger=None, hvd=None):
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
        if hvd:
            if hvd.rank() == 0:
                logger.debug("[Validation] Loss: {:.5f}, Score: {:.3f}".format(losses.avg, top1.avg))
        else:
            logger.debug("[Validation] Loss: {:.5f}, Score: {:.3f}".format(losses.avg, top1.avg))
    return top1.avg


def pcq_validate(model, clustering_model, test_loader, criterion, runtime_helper, logger=None, hvd=None):
    losses = AverageMeter()
    top1 = AverageMeter()

    model.eval()
    with torch.no_grad():
        with tqdm(test_loader, unit="batch", ncols=90) as t:
            for i, (input, target) in enumerate(t):
                t.set_description("Validate")
                input, target = input.cuda(), target.cuda()
                runtime_helper.batch_cluster = clustering_model.predict_cluster_of_batch(input)
                output = model(input)
                loss = criterion(output, target)
                prec = accuracy(output, target)[0]
                losses.update(loss.item(), input.size(0))
                top1.update(prec.item(), input.size(0))

                t.set_postfix(loss=losses.avg, acc=top1.avg)

    if logger:
        if hvd:
            if hvd.rank() == 0:
                logger.debug("[Validation] Loss: {:.5f}, Score: {:.3f}".format(losses.avg, top1.avg))
        else:
            logger.debug("[Validation] Loss: {:.5f}, Score: {:.3f}".format(losses.avg, top1.avg))
    return top1.avg


def validate_darknet_dataset(model, test_loader, criterion):
    losses = AverageMeter()
    top1 = AverageMeter()

    model.eval()
    with torch.no_grad():
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


def load_dnn_model(arg_dict, tools, path=None):
    model = None
    if arg_dict['quantized']:
        if arg_dict['dataset'] == 'cifar100':
            model = tools.quantized_model_initializer(arg_dict, num_classes=100)
        else:
            model = tools.quantized_model_initializer(arg_dict)
    elif arg_dict['fused']:
        if arg_dict['dataset'] == 'cifar100':
            model = tools.fused_model_initializer(arg_dict, num_classes=100)
        else:
            model = tools.fused_model_initializer(arg_dict)
    else:
        if arg_dict['dataset'] == 'imagenet':
            if arg_dict['arch'] == 'MobileNetV3':
                return vision_models.mobilenet_v3_small(pretrained=True)
            elif arg_dict['arch'] == 'ResNet18':
                return vision_models.resnet18(pretrained=True)
            elif arg_dict['arch'] == 'AlexNet':
                return vision_models.alexnet(pretrained=True)
            elif arg_dict['arch'] == 'ResNet50':
                return vision_models.resnet50(pretrained=True)
            elif arg_dict['arch'] == 'DenseNet121':
                return vision_models.densenet121(pretrained=True)
        elif arg_dict['dataset'] == 'cifar100':
            model = tools.pretrained_model_initializer(num_classes=100)
        else:
            model = tools.pretrained_model_initializer()
    if path is not None:
        checkpoint = torch.load(path)
    else:
        checkpoint = torch.load(arg_dict['dnn_path'])
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    return model


def load_optimizer(optim, path):
    checkpoint = torch.load(path)
    optim.load_state_dict(checkpoint['optimizer'])
    epoch_to_start = checkpoint['epoch'] + 1
    return optim, epoch_to_start


def load_tuning_info(path):
    dir_path = path.replace('/checkpoint.pth', '')
    int_params_path = os.path.join(dir_path, 'quantized/params.json')
    with open(int_params_path, 'r') as f:
        saved_args = json.load(f)
        best_epoch = saved_args['best_epoch']
        best_int_val_score = saved_args['best_int_val_score']
    return dir_path, best_epoch, best_int_val_score


def check_file_exist(path):
    return os.path.isfile(path) 


def add_path(prev_path, to_add):
    path = os.path.join(prev_path, to_add)
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def set_clustering_dir(args):
    path = add_path('', 'result')
    path = add_path(path, args.clustering_method)
    path = add_path(path, args.dataset)
    path = add_path(path, datetime.now().strftime("%m-%d-%H%M"))
    with open(os.path.join(path, "params.json"), 'w') as f:
        if args.clustering_method == 'kmeans':
            args_to_save = {'k': args.cluster, 'num_partitions': args.partition, 'tol': args.kmeans_tol,
                            'n_inits': args.kmeans_init, 'epoch': args.kmeans_epoch, 'batch': args.batch}
        else:
            args_to_save = {'k': args.cluster, 'num_partitions': args.partition}
        json.dump(args_to_save, f, indent=4)
    return path


def set_save_dir(args):
    path = add_path('', 'result')
    path = add_path(path, args.mode)
    path = add_path(path, args.dataset)
    path = add_path(path, args.arch + '_' + str(args.bit) + 'bit')
    path = add_path(path, datetime.now().strftime("%m-%d-%H%M"))
    with open(os.path.join(path, "params.json"), 'w') as f:
        json.dump(vars(args), f, indent=4)
    return path


def set_logger(path):
    logging.basicConfig(filename=os.path.join(path, "train.log"), level=logging.DEBUG)
    logger = logging.getLogger()
    return logger


def load_preprocessed_cifar10_from_darknet():
    input = torch.tensor(
        np.fromfile("result/darknet/cifar_test_dataset.bin", dtype='float32').reshape((10000, 1, 3, 32, 32)))
    target = torch.tensor(np.fromfile("result/darknet/cifar_test_target.bin", dtype='int32').reshape((10000, 1)),
                          dtype=torch.long)
    return input, target


# def metric_average(val, name):
#     tensor = torch.tensor(val)
#     avg_tensor = hvd.allreduce(tensor, name=name)
#     return avg_tensor.item()


def get_time_cost_in_string(t):
    if t > 3600:
        return '{:.1f}h'.format(t / 3600)
    elif t > 60:
        return '{:.1f}m'.format(t / 60)
    else:
        return '{:.1f}s'.format(t)

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
        if self.iterated == self.len_loader:
            self.initialize_phase2_generator()
        input, _ = next(self.generator)
        self.iterated += 1
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
            if i==4: break


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
                if i==4: break

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
                if i==4: break

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

def make_indices_list(clustering_model, train_loader, args, runtime_helper):
    total_list = [[] for _ in range(args.cluster)]

    idx = 0
    with torch.no_grad():
        with tqdm(train_loader, unit="batch", ncols=90) as t:
            for i, (input, target) in enumerate(t):
                t.set_description("Indices per Cluster")
                runtime_helper.batch_cluster = clustering_model.predict_cluster_of_batch(input)
                for c in runtime_helper.batch_cluster:
                    total_list[c].append(idx)
                    idx += 1
                t.set_postfix()
    # Cluster length list
    len_per_cluster = []
    for c in range(args.cluster):
        len_per_cluster.append(len(total_list[c]))
    return total_list, len_per_cluster


def make_phase2_list(args, indices_per_cluster, len_per_cluster):
    for c in range(args.cluster):
        random.shuffle(indices_per_cluster[c])

    n = args.data_per_cluster
    if args.phase2_loader_strategy == 'mean':
        counted = sum(len_per_cluster) // args.cluster
    elif args.phase2_loader_strategy == 'min':
        counted = min(len_per_cluster)
    else:
        counted = max(len_per_cluster)
    len_loader = counted // n

    cluster_cross_sorted = []
    cur_idx = [0 for _ in range(args.cluster)]
    for loops in range(len_loader):
        for c in range(args.cluster):
            end = cur_idx[c] + n
            share = end // len_per_cluster[c]
            remainder = end % len_per_cluster[c]
            if share < 1:
                cluster_cross_sorted += indices_per_cluster[c][cur_idx[c]:remainder]
                cur_idx[c] += n
            else:
                cluster_cross_sorted += indices_per_cluster[c][cur_idx[c]:len_per_cluster[c]]
                random.shuffle(indices_per_cluster[c])
                cluster_cross_sorted += indices_per_cluster[c][:remainder]
                cur_idx[c] = remainder
    return cluster_cross_sorted


def save_indices_list(args, indices_list_per_cluster, len_per_cluster):
    path = add_path('', 'result')
    path = add_path(path, 'indices')
    path = add_path(path, args.dataset)
    path = add_path(path, "Partition{}".format(args.partition))
    path = add_path(path, "{}data_per_cluster".format(args.data_per_cluster))
    path = add_path(path, datetime.now().strftime("%m-%d-%H%M"))
    with open(os.path.join(path, "params.json"), 'w') as f:
        indices_args = {'indices_list': indices_list_per_cluster, 'len_per_cluster': len_per_cluster,
                        'data_per_cluster': args.data_per_cluster, 'dataset': args.dataset,
                        'partition': args.partition}
        json.dump(indices_args, f, indent=4)


def load_indices_list(args):
    with open(os.path.join(args.indices_path, 'params.json'), 'r') as f:
        saved_args = json.load(f)
    assert args.dataset == saved_args['dataset'], \
        "Dataset should be same. \n" \
        "Loaded dataset: {}, Current dataset: {}".format(saved_args['dataset'], args.dataset)
    assert args.partition == saved_args['partition'], \
        "partition should be same. \n" \
        "Loaded partition: {}, Current partition: {}".format(saved_args['partition'], args.partition)
    assert args.data_per_cluster == saved_args['data_per_cluster'], \
        "Data per cluster should be same. \n" \
        "Loaded data per cluster: {}, current data per cluster: {}".format(saved_args['data_per_cluster'], args.data_per_cluster)
    return saved_args['indices_list'], saved_args['len_per_cluster']


def visualize_clustering_res(visual_loader, indices_list, len_indices_list, model, num_ctr):
    import sklearn
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    pca = sklearn.decomposition.PCA(n_components=2)
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:brown', 'tab:cyan', 'tab:olive', 'tab:purple', 'tab:gray', 'tab:red', 'tab:pink']

    for image, _ in visual_loader:
        whole_data = model.get_partitioned_batch(image)
        pca.fit(whole_data)
        centroids = model.model.cluster_centers_
        pca_whole_data = pca.transform(whole_data)
        pca_centroids = pca.transform(centroids)
        # plot
        plt.figure(figsize=(8, 8))
        for i in range(num_ctr):
            plt.scatter(pca_whole_data[indices_list[i], 0], pca_whole_data[indices_list[i], 1], c=colors[i], s=10, label='cluster {} - {}'.format(i, len_indices_list[i]), alpha=0.7, edgecolors='none')
        plt.legend()
        for i in range(num_ctr):
            plt.scatter(pca_centroids[i, 0], pca_centroids[i, 1], c=colors[i], s=30, label="centroid", edgecolors='black', alpha=0.7, linewidth=2)
        plt.suptitle('Train Dataset')
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.savefig("k-means_partition_{}_cluster_{}.png".format(model.args.partition, num_ctr))
        plt.show()


def initialize_pcq_model(model, loader, criterion):
    losses = AverageMeter()
    top1 = AverageMeter()

    model.train()
    with torch.no_grad():
        with tqdm(loader, unit="batch", ncols=90) as t:
            for i, (input, target) in enumerate(t):
                t.set_description("Initialize PCQ")
                input, target = input.cuda(), target.cuda()
                output = model(input)
                loss = criterion(output, target)
                prec = accuracy(output, target)[0]
                losses.update(loss.item(), input.size(0))
                top1.update(prec.item(), input.size(0))

                t.set_postfix(loss=losses.avg, acc=top1.avg)
    return top1.avg


def get_finetuning_model(arg_dict, tools):
    if arg_dict['dataset'] == 'cifar100':
        fused_model = tools.fused_model_initializer(arg_dict, num_classes=100)
    else:
        fused_model = tools.fused_model_initializer(arg_dict)

    if arg_dict['fused']:
        checkpoint = torch.load(arg_dict['dnn_path'])
        fused_model.load_state_dict(checkpoint['state_dict'], strict=False)
    else:
        pretrained_model = load_dnn_model(arg_dict, tools)
        fused_model = tools.fuser(fused_model, pretrained_model)
    return fused_model


def visualize_clustering_res(data_loader, clustering_model, indices_per_cluster, len_per_cluster, num_clusters):
    import sklearn
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt

    pca = sklearn.decomposition.PCA(n_components=2)

    images = []
    for image, _ in data_loader:
        images.append(image)

    data = clustering_model.get_partitioned_batch(torch.cat(images))
    pca.fit(data)
    centroids = clustering_model.model.cluster_centers_
    pca_data = pca.transform(data)
    pca_centroids = pca.transform(centroids)

    plt.figure(figsize=(8, 8))
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:brown', 'tab:cyan', 'tab:olive', 'tab:purple', 'tab:gray', 'tab:red', 'tab:pink']
    for i in range(num_clusters):
        plt.scatter(pca_data[indices_per_cluster[i], 0], pca_data[indices_per_cluster[i], 1], c=colors[i], s=10,
                    label='cluster {} - {}'.format(i, len_per_cluster[i]), alpha=0.7, edgecolors='none')
    plt.legend()
    for i in range(num_clusters):
        plt.scatter(pca_centroids[i, 0], pca_centroids[i, 1], c=colors[i], s=30, label="centroid", edgecolors='black', alpha=0.7, linewidth=2)
    plt.suptitle('Train Dataset')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.show()
    plt.savefig("k-means clustering trial 1.png")


def test_augmented_clustering(model, non_augmented_loader, augmented_loader):
    non_aug_indices = []
    for i, (input, target) in enumerate(non_augmented_loader):
        batch_cluster = model.predict_cluster_of_batch(input)
        non_aug_indices.extend(batch_cluster.tolist())

    aug_indices = []
    for i, (input, target) in enumerate(augmented_loader):
        batch_cluster = model.predict_cluster_of_batch(input)
        aug_indices.extend(batch_cluster.tolist())

    cnt_data_assigned_to_different_cluster = 0
    for i in range(len(non_aug_indices)):
        if non_aug_indices[i] != aug_indices[i]:
            cnt_data_assigned_to_different_cluster += 1
    print("Datum assigned to different cluster = {}".format(cnt_data_assigned_to_different_cluster))
    exit()

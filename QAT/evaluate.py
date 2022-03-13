from torch import nn
import torch.backends.cudnn as cudnn
from torchsummary import summary

from models import *
from utils import *
from tqdm import tqdm
import csv

import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy

def get_dict(module):
    data = module.act_range.data.transpose(0, 1)
    info = dict()
    info["channels"] = str(module.out_channels)
    info["mean"] = str(data.mean(dim=0))
    info["min"] = str(data.min(dim=0).values.data)
    info["max"] = str(data.max(dim=0).values.data)
    info["std"] = str(data.std(dim=0).data)
    return info


def traverse(target, total_dict, name_stack=None):
    for name, m in target.named_children():
        if isinstance(m, nn.Conv2d):
            if name_stack:
                total_dict[name_stack + name] = get_dict(m)
            else:
                total_dict[name] = get_dict(m)
        elif isinstance(m, nn.Sequential):
            tmp = dict()
            traverse(m, tmp, name + ' ')
            total_dict[name] = tmp
        elif isinstance(m, BasicBlock):
            traverse(m, total_dict, 'ResBlock ' + name + ' ')
    return total_dict


def save_graph(data, identifier, sort=False):
    mean_ = data.mean(axis=1)
    data = data.transpose(1, 0)
    if sort:
        sorted_list = sorted(list(data), key=lambda x: x[1]-x[0])
        data = np.array(sorted_list)

    fig = plt.figure(figsize=(20, 20))
    plt.axis([data.min() - 1, data.max() + 1, 0, 0.1 * (data.shape[0] + 1)])
    plt.axvline(mean_[0], 0, 1, color='red', linestyle='--', linewidth=2)
    plt.axvline(mean_[1], 0, 1, color='red', linestyle='--', linewidth=2)

    cnt = 0
    for i in range(data.shape[0]):
        cnt += 0.1
        plt.hlines(cnt, data[i][0], data[i][1])
        plt.text(data[i][0] - 0.2, cnt, '{:.4f}'.format(data[i][0]), ha='right', va='center')
        plt.text(data[i][1] + 0.2, cnt, '{:.4f}'.format(data[i][1]), ha='left', va='center')
    #plt.show()
    plt.xlabel('range')
    plt.savefig(identifier)


def visualize(args, model):
    path = os.path.join('', 'skt_range')
    if not os.path.exists(path):
        os.makedirs(path)
    path = os.path.join(path, args.arch + args.dataset + '_smooth_' + str(args.smooth))
    if not os.path.exists(path):
        os.makedirs(path)

    print("Save range output")
    print("Save dir: ", path)

    conv_idx = 0
    for m in model.features:
        if isinstance(m, nn.Conv2d):
            conv_idx += 1
            act_data = m.act_range.data.cpu().numpy()
            weight_ = m.weight.view(m.weight.size(0), -1).detach().cpu().numpy()
            weight_data = np.zeros((2, m.out_channels))
            weight_data[0] = weight_.min(axis=1)
            weight_data[1] = weight_.max(axis=1)
            save_graph(act_data, path + f'/conv{conv_idx}_output.png')
            save_graph(act_data, path + f'/conv{conv_idx}_output_sorted.png', sort=True)
            save_graph(weight_data, path + f'/conv{conv_idx}_weight.png')
            save_graph(weight_data, path + f'/conv{conv_idx}_weight_sorted.png', sort=True)


def save_range_out_dict(args, model):
    path = os.path.join('', 'range')
    if not os.path.exists(path):
        os.makedirs(path)
    path = os.path.join(path, args.dataset)
    if not os.path.exists(path):
        os.makedirs(path)
    path = os.path.join(path, args.arch + '_smooth_' + str(args.smooth))
    if not os.path.exists(path):
        os.makedirs(path)

    print("Save dir: ", path)
    group = dict()
    group = traverse(model, group, 'conv')


def _evaluate(args, tools):
    runtime_helper = RuntimeHelper()
    runtime_helper.set_pcq_arguments(args)
    arg_dict = deepcopy(vars(args))
    if runtime_helper:
        arg_dict['runtime_helper'] = runtime_helper
    model = load_dnn_model(arg_dict, tools)
    model.cuda()
    # if not args.quantized:
    #    if args.dataset == 'imagenet':
    #        summary(model, (3, 224, 224))
    #    else:
    #        summary(model, (3, 32, 32))

    criterion = nn.CrossEntropyLoss().cuda()
    cudnn.benchmark = True

    if args.darknet:
        darknet_loader = load_preprocessed_cifar10_from_darknet()
        validate_darknet_dataset(model, darknet_loader, criterion)

    else:
        normalizer = get_normalizer(args.dataset)
        test_dataset = get_test_dataset(args, normalizer)
        test_loader = get_data_loader(test_dataset, batch_size=args.val_batch, shuffle=False, workers=args.worker)
        if args.cluster > 1:
            clustering_model = tools.clustering_method(args)
            clustering_model.load_clustering_model()
            pcq_validate(model, clustering_model, test_loader, criterion, runtime_helper)
        else:
            validate(model, test_loader, criterion)
            # save_range_out(args, model)
            visualize(args, model)
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
import torchvision.models as models

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
    mean_ = data.transpose(1, 0).mean(axis=1)

    scale = data.shape[0] // 96
    fig = plt.figure(figsize=(20, 20*scale))
    plt.axis([data.min() - 1, data.max() + 1, 0, 0.1 * (data.shape[0] + 1)])
    plt.axhline(mean_[0], 0, 1, color='red', linestyle='--', linewidth=2)
    plt.axhline(mean_[1], 0, 1, color='red', linestyle='--', linewidth=2)

    cnt = 0
    for i in range(data.shape[0]):
        cnt += 0.1
        plt.vlines(cnt, data[i][0], data[i][1])
        plt.text(data[i][0] - 0.2, cnt, '{:.4f}'.format(data[i][0]), ha='right', va='center')
        plt.text(data[i][1] + 0.2, cnt, '{:.4f}'.format(data[i][1]), ha='left', va='center')
    #plt.show()
    plt.xlabel('range')
    plt.savefig(identifier)


def save_results_to_csv(output, weight, path, output_range_ratio, weight_range_ratio, overlapped=False):
    output_mean = output.transpose(1, 0).mean(axis=1)
    weight_mean = weight.transpose(1, 0).mean(axis=1)
    out_channels = output.shape[0]
    flag = 'weight' if 'weight' in path else 'output'
    with open(path, 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['out_channels', out_channels, f'based on {flag}', 'output_ratio', output_range_ratio, 'weight_ratio', weight_range_ratio, 'overlap', overlapped])
        writer.writerow(['output'])
        writer.writerow(['', 'min', 'max'])
        for i in range(output.shape[0]):
            writer.writerow([i, output[i][0], output[i][1]])
        writer.writerow(['mean', output_mean[0], output_mean[1]])

        writer.writerow(['weight'])
        writer.writerow(['', 'min', 'max'])
        for i in range(weight.shape[0]):
            writer.writerow([i, weight[i][0], weight[i][1]])
        writer.writerow(['mean', weight_mean[0], weight_mean[1]])

def save_std_results_to_csv(output, weight, path, overlapped=False):
    output_mean = output.transpose(1, 0).mean(axis=1)
    weight_mean = weight.transpose(1, 0).mean(axis=1)
    out_channels = output.shape[0]
    flag = 'weight' if 'weight' in path else 'output'
    with open(path, 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['out_channels', out_channels, f'based on {flag}', 'overlap', overlapped])
        writer.writerow(['output'])
        writer.writerow(['', 'min', 'max'])
        for i in range(output.shape[0]):
            writer.writerow([i, output[i][0], output[i][1]])
        writer.writerow(['mean', output_mean[0], output_mean[1]])

        writer.writerow(['weight'])
        writer.writerow(['', 'min', 'max'])
        for i in range(weight.shape[0]):
            writer.writerow([i, weight[i][0], weight[i][1]])
        writer.writerow(['mean', weight_mean[0], weight_mean[1]])

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
    #for m in model.features:
    conv_output = []
    conv_weight = []
    output_range = []
    weight_range = []
    output_std = []
    weight_std = []
    model.cpu()

    conv_cnt = 0
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            conv_cnt += 1
            act_data = m.act_range.data.numpy()
            weight_ = m.weight.view(m.weight.size(0), -1).detach().numpy()
            weight_data = np.zeros((2, m.out_channels))
            weight_data[0] = weight_.min(axis=1)
            weight_data[1] = weight_.max(axis=1)
            act_data_ = act_data.transpose(1, 0)
            weight_data_ = weight_data.transpose(1, 0)

            output_range_sorted = sorted(list(act_data_), key=lambda x: x[1] - x[0], reverse=True)
            output_range.append((output_range_sorted[0][1] - output_range_sorted[0][0]) / (output_range_sorted[-1][1] - output_range_sorted[-1][0]))

            weight_range_sorted = sorted(list(weight_data_), key=lambda x: x[1]-x[0], reverse=True)
            weight_range.append((weight_range_sorted[0][1] - weight_range_sorted[0][0]) / (weight_range_sorted[-1][1] - weight_range_sorted[-1][0]))

            conv_output.append(output_range_sorted)
            conv_weight.append(weight_range_sorted)

            output_std.append(np.array([output_range_sorted[i][1] - output_range_sorted[i][0] for i in range(act_data_.shape[0])]).std())
            weight_std.append(np.array([weight_range_sorted[i][1] - weight_range_sorted[i][0] for i in range(act_data_.shape[0])]).std())

    # output_indices = np.argsort(np.array(output_range))[-3:]
    # weight_indices = np.argsort(np.array(weight_range))[-3:]

    output_indices = np.argsort(np.array(output_std))[-3:]
    weight_indices = np.argsort(np.array(weight_std))[-3:]

    overlap = set(output_indices) & set(weight_indices)

    with open(path + '/master.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        np_output_range = np.array(output_range)
        np_weight_range = np.array(weight_range)
        np_output_std = np.array(output_std)
        np_weight_std = np.array(weight_std)
        writer.writerow(['num layers', conv_cnt])
        writer.writerow(['', 'min', 'max', 'mean', 'std'])
        writer.writerow(['output_ratio', np_output_range.min(), np_output_range.max(), np_output_range.mean(), np_output_range.std()])
        writer.writerow(['weight_ratio', np_weight_range.min(), np_weight_range.max(), np_weight_range.mean(), np_weight_range.std()])
        writer.writerow(['output_std', np_output_std.min(), np_output_std.max(), np_output_std.mean(),
                         np_output_std.std()])
        writer.writerow(['weight_std', np_weight_std.min(), np_weight_std.max(), np_weight_std.mean(),
                         np_weight_std.std()])

    # for idx in output_indices:
    #     if idx in overlap:
    #         save_results_to_csv(np.array(conv_output[idx]), np.array(conv_weight[idx]), path + f'/conv{idx}_output_sorted.csv', output_range[idx], weight_range[idx], overlapped=True)
    #     else:
    #         save_results_to_csv(np.array(conv_output[idx]), np.array(conv_weight[idx]), path + f'/conv{idx}_output_sorted.csv', output_range[idx], weight_range[idx])
    # for idx in weight_indices:
    #     if idx in overlap:
    #         continue
    #     save_results_to_csv(np.array(conv_output[idx]), np.array(conv_weight[idx]), path + f'/conv{idx}_weight_sorted.csv', output_range[idx], weight_range[idx])
    for idx in output_indices:
        if idx in overlap:
            save_std_results_to_csv(np.array(conv_output[idx]), np.array(conv_weight[idx]), path + f'/conv{idx}_output_sorted_std.csv', overlapped=True)
        else:
            save_std_results_to_csv(np.array(conv_output[idx]), np.array(conv_weight[idx]), path + f'/conv{idx}_output_sorted_std.csv')
    for idx in weight_indices:
        if idx in overlap:
            continue
        save_std_results_to_csv(np.array(conv_output[idx]), np.array(conv_weight[idx]), path + f'/conv{idx}_weight_sorted_std.csv')



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

    if args.skt:
        model = tools.pretrained_model_initializer(pretrained=True, smooth=args.smooth)
    else:
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
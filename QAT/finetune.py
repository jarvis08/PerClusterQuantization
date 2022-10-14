from copy import deepcopy

import torch
from torchsummary import summary

from utils import *
from .models import *
from tqdm import tqdm
from time import time
# import matplotlib.pyplot as plt
import math
import csv

import warnings
warnings.filterwarnings("ignore")

def draw_weight_out_violin_graph(input_data, weight_data, identifier, arch):
    plt.rcParams['font.size'] = 12
    in_channels = len(input_data)
    pos = [i for i in range(in_channels)]

    fig, ax = plt.subplots(nrows=2, ncols=1)
    # fig.tight_layout()

    violin_1 = ax[0].violinplot(input_data, positions=pos, showmedians=True, quantiles=[[0.25] for _ in range(in_channels)])
    violin_2 = ax[1].violinplot(weight_data, positions=pos, showmedians=True, quantiles=[[0.25] for _ in range(in_channels)])

    violin_1['cquantiles'].set_edgecolor('#ff2222')
    violin_1['cmedians'].set_edgecolor('black')
    violin_1['cquantiles'].set_linewidth(2)
    violin_1['cmedians'].set_linewidth(2)

    # ax[0].set_xlabel('Convolution Layers', size=12, alpha=0.8)
    ax[0].set_ylabel('Input Range', size=12, alpha=0.8)
    ax[0].set_xlim(-1, in_channels)
    ax[0].set_xticks(pos)
    ax[0].set_xticklabels([i for i in pos], fontsize=8)
    # ax[0].set_title(f'{arch} Input Range per In Channels')

    violin_2['cquantiles'].set_edgecolor('#ff2222')
    violin_2['cmedians'].set_edgecolor('black')
    violin_2['cquantiles'].set_linewidth(2)
    violin_2['cmedians'].set_linewidth(2)

    ax[1].set_xlabel('Convolution Layers', size=12, alpha=0.8)
    ax[1].set_ylabel('Weight Range', size=12, alpha=0.8)
    ax[1].set_xlim(-1, in_channels)
    ax[1].set_xticks(pos)
    ax[1].set_xticklabels([i for i in pos], fontsize=8)
    # ax[1].set_title(f'{arch} Weight Range per In Channels')

    # plt.show()
    plt.savefig(identifier)
    plt.close(fig)


def draw_violin_graph(data, max_, identifier, arch, epoch):
    plt.style.use('default')
    if '20' in arch:
        plt.rcParams['figure.figsize'] = (36, 12)
    elif '50' in arch:
        plt.rcParams['figure.figsize'] = (60, 12)
    elif '121' in arch:
        plt.rcParams['figure.figsize'] = (80, 12)
    else:
        plt.rcParams['figure.figsize'] = (6, 6)
    plt.rcParams['font.size'] = 12
    pos = [i for i in range(len(data))]

    fig, ax = plt.subplots()

    violin = ax.violinplot(data, positions=pos, showmedians=True, quantiles=[[0.25] for _ in range(len(data))])

    violin['cquantiles'].set_edgecolor('#ff2222')
    violin['cmedians'].set_edgecolor('black')
    violin['cquantiles'].set_linewidth(2)
    violin['cmedians'].set_linewidth(2)

    ax.set_xlabel('Convolution Layers', size=12, alpha=0.8)
    ax.set_ylabel('Output Range', size=12, alpha=0.8)
    ax.set_xlim(-1, len(data))
    ax.set_ylim(0, max_ + 5)
    ax.set_xticks(pos)
    ax.set_xticklabels([i for i in pos], fontsize=8)
    ax.set_title(f'{arch} epoch {epoch} Output Range per Output Channels')

    # plt.show()
    plt.savefig(identifier)
    plt.close(fig)


# def visualize(args, model, epoch):
#     path = os.path.join('', 'skt_range_yongjoo')
#     if not os.path.exists(path):
#         os.makedirs(path)
#     # path = os.path.join(path, args.arch + args.dataset)
#     # if not os.path.exists(path):
#     #     os.makedirs(path)
#     violin_path  = os.path.join(path, 'violin_output')
#     if not os.path.exists(violin_path):
#         os.makedirs(violin_path)
#     # naming = args.arch + '_' + args.dataset + '_' + str(epoch)
#     naming = args.arch + '_' + args.dataset
#
#     print("Save range output")
#     print("Save dir: ", violin_path)
#
#     conv_cnt = 0
#     model.cpu()
#     range_per_group = []
#     # weight, layer input group
#     input_group = []
#     weight_group = []
#     max_ = 0
#
#     # get input, weight distribution per input channel
#     for m in model.modules():
#     #         # out_channel = m.out_channels
#     #         # input_per_out_filter_group = m.input_range.transpose(1, 0).numpy()
#     #         # weight_per_out_filter_group = m.weight.transpose(1, 0).reshape(m.weight.size(1), -1).numpy()
#     #         # output_per_out_filter_group = m.act_range.transpose(1,0).numpy()
#     #
#     #         # input_min_max_per_group = input_per_out_filter_group.max(axis=1) - input_per_out_filter_group.min(axis=1)
#     #         # range_per_group.append(min_max_per_group)
#     #         # if min_max_per_group.max() > max_:
#     #         #     max_ = min_max_per_group.max()
#
#     # violin
#     # range_per_group_sorted = sorted(list(range_per_group))
#     # draw_violin_graph(range_per_group, max_, violin_path + f'/{naming}.png', naming, epoch)


def validate_setting_bits(model, loader, criterion):
    print('\n>>> Setting bits..')
    # from .utils.misc import accuracy

    model.eval()
    with torch.no_grad():
        with tqdm(loader, unit="batch", ncols=90) as t:
            for i, (input, target) in enumerate(t):
                t.set_description("Validate")
                input, target = input.cuda(), target.cuda()
                model.set_mixed_bits(input)


def set_lower_weights(model, pre_fixed_channel_ratio):
    total_channel = 0
    for fused in model.modules():
        if isinstance(fused, FusedConv2d):
            in_channel = fused.in_channels
            total_channel += in_channel
            num_lower_channels = math.ceil(in_channel * pre_fixed_channel_ratio)
            weight_per_filter_group = fused.conv.weight.transpose(1, 0)

            weight_group = weight_per_filter_group.reshape(in_channel, -1)
            weight_range = torch.max(weight_group.max(dim=1).values.abs(), weight_group.min(dim=1).values.abs())

            fused.fixed_indices = torch.topk(weight_range, num_lower_channels, largest=False, sorted=False)[1].cuda()
    model.total_ch_sum = total_channel


def _finetune(args, tools, data_loaders, clustering_model):
    tuning_start_time = time()

    runtime_helper = RuntimeHelper()
    runtime_helper.set_pcq_arguments(args)

    arg_dict = deepcopy(vars(args))
    arg_dict['runtime_helper'] = runtime_helper

    pretrained_model = load_dnn_model(arg_dict, tools)
    pretrained_model.cuda()

    train_loader = data_loaders['aug_train']
    val_loader = data_loaders['val']
    test_loader = data_loaders['test']

    criterion = torch.nn.CrossEntropyLoss().cuda()
    if args.nnac and clustering_model.final_cluster is None:
        clustering_model.nn_aware_clustering(pretrained_model, train_loader, args.arch)

    model = get_finetuning_model(arg_dict, tools, pretrained_model)
    if pretrained_model:
        del pretrained_model
    model.cuda()

    # if args.mixed_precision:
    #     runtime_helper.set_yj_arguments(args)
    #     # if not args.grad_method:
    #     #     runtime_helper.grad_method = ~runtime_helper.grad_method
    #     model.percentile_tensor = torch.tensor(args.percentile, dtype=torch.float, device='cuda')
    #     # # try inference once to record input precisions
    #     # identifier = f'[TRAIN_Ratio]percentile_{args.percentile}_ema_{args.smooth}_weight_scailing_{args.weight_scailing}_weight_only_{args.weight_only}_'
    #     identifier = f'GRAD_{args.input_grad}_{args.arch[:4]}_DATA_{args.dataset[5:]}_METH_{args.grad_method}_CON_{args.const_portion}'
    #     set_lower_weights(model, args.pre_fixed_channel)
    #     validate_setting_bits(model, val_loader, criterion)
    #     # pretrained_model.cpu()
    #     initial_set_mixed_bits_per_input_channels(model, args.percentile, identifier=identifier)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    opt_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    save_path_fp = ''
    epoch_to_start = 1
    best_epoch = 0
    best_int_val_score = 0
    if args.fused:
        optimizer, epoch_to_start = load_optimizer(optimizer, args.dnn_path)
        save_path_fp, best_epoch, best_int_val_score = load_tuning_info(args.dnn_path)

    if not save_path_fp:
        save_path_fp = set_save_dir(args, allow_existence=False)
        args.dnn_path = save_path_fp
        print("Save dir: " + args.dnn_path)
    save_path_int = add_path(save_path_fp, 'quantized')
    logger = set_logger(save_path_fp)

    quantized_model = None
    trial = 0
    ratio = 99
    ## channel searching epochs
    if args.run_mode == 'paper':
        if args.compression_ratio == 0:
            model.set_channel_bits_to_eight()
        else:
            runtime_helper.set_channel_arguments(args, True)
            model.quantile_tensor = torch.tensor(args.quantile, dtype=torch.float, device='cuda')
            print("<<<<<<<<<<<< Channel Searching START >>>>>>>>>>>>>>>>")
            for e in range(1, args.channel_epoch + 1):
                if ratio < args.compression_ratio:
                    break
                ratio, trial = channel_searching_train_epoch(model, train_loader, criterion, optimizer, e, logger, trial, args.compression_ratio)
                ## do not update learning rates
                # opt_scheduler.step()
                # with open(identifier + '.csv', 'a') as csvfile:
                #     writer = csv.writer(csvfile)
                #     writer.writerow(([e, '{:2f}'.format(losses), '{:2f}%'.format(ratio)]))
        print("<<<<<<<<<<<< Channel Searching END >>>>>>>>>>>>>>>>")
        print("[Final Ratio] ----------> {:2f}%".format(ratio))
    runtime_helper.set_channel_arguments(args, False)

    layer_idx = 1
    for fused in model.modules():
        with open(f"{args.arch}_{args.dataset}_{args.compression_ratio}.csv", 'a') as csvfile:
            writer = csv.writer(csvfile)
            if isinstance(fused, FusedConv2d):
                total_ch = fused.out_channels
                four = len(fused.low_group) / total_ch * 100
                eight = len(fused.high_group) / total_ch * 100
                writer.writerow([layer_idx, '{:2f}%'.format(four), '{:2f}%'.format(eight)])
                print("".format(layer_idx, four, eight))

    ## normal training mode
    print("<<<<<<<<<<<< Training START >>>>>>>>>>>>>>>>")
    for e in range(epoch_to_start, args.epoch + 1):
        if e > args.fq:
            runtime_helper.apply_fake_quantization = True

        train_epoch(model, train_loader, criterion, optimizer, e, logger)
        opt_scheduler.step()

        if args.fold_convbn:
            tools.folder(model)

        fp_score = 0

        state = {
            'epoch': e,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        save_checkpoint(state, False, save_path_fp)

        # Test quantized model, and save if performs the best
        # if e > args.fq:
        if e >= args.epoch:
            model.set_quantization_params()
            # if args.mixed_precision:
            #     model.set_mixed_quantization_params(args.method)
            # else:
            #     model.set_quantization_params()
            if quantized_model is None:
                if args.dataset == 'cifar100':
                    quantized_model = tools.quantized_model_initializer(arg_dict, num_classes=100)
                else:
                    quantized_model = tools.quantized_model_initializer(arg_dict)
            quantized_model = tools.quantizer(model, quantized_model)
            quantized_model.cuda()

            #int_score = validate(quantized_model, val_loader, criterion, logger)
            int_score = validate(quantized_model, test_loader, criterion, logger)

            if int_score > best_int_val_score:
                best_epoch = e
                # Save best model's FP model
                with open(os.path.join(save_path_fp, "params.json"), 'w') as f:
                    tmp = vars(args)
                    tmp['best_epoch'] = e
                    tmp['best_score'] = fp_score
                    json.dump(tmp, f, indent=4)
                shutil.copyfile(os.path.join(save_path_fp, 'checkpoint.pth'), os.path.join(save_path_fp, 'best.pth'))

                # Save best model's INT model
                best_int_val_score = int_score
                with open(os.path.join(save_path_int, "params.json"), 'w') as f:
                    tmp = vars(args)
                    tmp['best_epoch'] = e
                    tmp['best_int_val_score'] = best_int_val_score
                    json.dump(tmp, f, indent=4)
                filepath = os.path.join(save_path_int, 'checkpoint.pth')
                torch.save({'state_dict': quantized_model.state_dict()}, filepath)
            print('Best INT-val Score: {:.2f} (Epoch: {})'.format(best_int_val_score, best_epoch))


    test_score = best_int_val_score

    # if args.record_val:
    #     with open('SATUR_' + identifier + '.csv', 'a') as csvfile:
    #         writer = csv.writer(csvfile)
    #         for m in quantized_model.modules():
    #             i = 0
    #             if isinstance(m, QuantizedConv2d):
    #                 ch_ratio = m.low_size / m.total_size * 100
    #                 writer.writerow(([i, '{:2f}%'.format(ch_ratio)]))
    #                 i += 1

    '''
    # Test quantized model which scored the best with validation dataset
    if test_loader is None:
        test_score = best_int_val_score
    else:
        arg_dict['quantized'] = True
        quantized_model = load_dnn_model(arg_dict, tools, os.path.join(save_path_int, 'checkpoint.pth')).cuda()

        if args.cluster > 1:
            test_score = pcq_validate(quantized_model, clustering_model, test_loader, criterion, runtime_helper, logger)
        else:
            test_score = validate(quantized_model, test_loader, criterion, logger)
    '''

    with open(os.path.join(save_path_int, "params.json"), 'w') as f:
        tmp = vars(args)
        tmp['best_epoch'] = best_epoch
        tmp['best_int_val_score'] = best_int_val_score
        tmp['int_test_score'] = test_score
        json.dump(tmp, f, indent=4)

    tuning_time_cost = get_time_cost_in_string(time() - tuning_start_time)
    if args.cluster > 1:
        if args.nnac:
            method = f'DAQ+{args.quant_base}({args.clustering_method}, K{args.cluster}S{args.sub_cluster}P{args.partition}TOP-K{args.cluster}Sim{args.sim_threshold}-{args.repr_method})'
        else:
            method = f'DAQ+{args.quant_base}({args.clustering_method}, K{args.cluster}S{args.sub_cluster}P{args.partition}-{args.repr_method})'
    else:
        method = args.quant_base

    pc = ''
    # if args.per_channel:
    #     pc = 'PerChannel, '
    # if args.symmetric:
    #     pc += 'Symmetric, '

    if args.run_mode == 'paper':
        with open(f'./[{args.run_mode.upper()}]{args.arch}_{args.dataset}_C{int(args.compression_ratio)}%.txt','a') as f:
            f.write(
                'quantile {} / channel {:.2f}% / ACC {:.2f} # {}, {}, {}, LR: {}, W-decay: {}, Epoch: {}, Batch: {}, {}Bit(First/Last/AddCat): {}({}/{}/{}), Smooth: {}, Best-epoch: {}, Time: {}, GPU: {}, Path: {}\n'
                .format(args.quantile, ratio, test_score, args.arch,
                        args.dataset, method, args.lr, args.weight_decay, args.epoch, args.batch, args.percentile,
                        pc, args.bit, args.bit_first, args.bit_classifier, args.bit_addcat, args.smooth, best_epoch,
                        tuning_time_cost, args.gpu, save_path_fp))
    else:
        with open(f'./[{args.run_mode.upper()}]{args.arch}_{args.dataset}_{args.bit}_F{args.bit_first}L{args.bit_classifier}_{args.gpu}.txt', 'a') as f:
            f.write('{:.2f} # {}, {}, {}, LR: {}, W-decay: {}, Epoch: {}, Batch: {}, {}Bit(First/Last/AddCat): {}({}/{}/{}), Smooth: {}, Best-epoch: {}, Time: {}, GPU: {}, Path: {}\n'
                    .format(test_score, args.arch, args.dataset, method, args.lr, args.weight_decay, args.epoch, args.batch, args.percentile,
                            pc, args.bit, args.bit_first, args.bit_classifier, args.bit_addcat, args.smooth, best_epoch,
                            tuning_time_cost, args.gpu, save_path_fp))

    # range_fname = None
    # for i in range(9999999):
    #     range_fname = './range-{}-{}-Batch{}-FQ{}-K{}-{}.txt'.format(args.arch, method, args.batch, args.fq, args.cluster, i)
    #     if not check_file_exist(range_fname):
    #         break
    # with open(range_fname, 'a') as f:
    #     for name, param in model.named_parameters():
    #         if 'act_range' in name:
    #             f.write('{}\n'.format(name))
    #             if 'norm' in name:
    #                 f.write('{:.4f}, {:.4f}\n'.format(param[0].item(), param[1].item()))
    #             else:
    #                 for c in range(args.cluster):
    #                     f.write('{:.4f}, {:.4f}\n'.format(param[c][0].item(), param[c][1].item()))
    # save_fused_network_in_darknet_form(model, args)

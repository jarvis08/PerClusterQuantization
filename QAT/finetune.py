from copy import deepcopy

import torch
# from torchsummary import summary

from utils import *
from .models import *
from tqdm import tqdm
from time import time

import matplotlib.pyplot as plt


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


def visualize(args, model, epoch):
    path = os.path.join('', 'skt_range_yongjoo')
    if not os.path.exists(path):
        os.makedirs(path)
    # path = os.path.join(path, args.arch + args.dataset)
    # if not os.path.exists(path):
    #     os.makedirs(path)
    violin_path  = os.path.join(path, 'violin_output')
    if not os.path.exists(violin_path):
        os.makedirs(violin_path)
    naming = args.arch + '_' + args.dataset + '_' + str(epoch)

    print("Save range output")
    print("Save dir: ", violin_path)

    conv_cnt = 0
    model.cpu()
    range_per_group = []
    # weight, layer input group
    input_group = []
    weight_group = []
    max_ = 0

    for m in model.modules():
        if isinstance(m, FusedConv2d):
            conv_cnt += 1
            # out_channel = m.out_channels
            # (2, 96) -> (96, 2)
            output_per_out_filter_group = m.act_violin_range.transpose(1,0).numpy()
            # output_per_out_filter_group = m.act_range.transpose(1,0).numpy()

            min_max_per_group = output_per_out_filter_group.max(axis=1) - output_per_out_filter_group.min(axis=1)
            range_per_group.append(min_max_per_group)
            if min_max_per_group.max() > max_:
                max_ = min_max_per_group.max()

            # out_channel = m.out_channels
            # input_per_out_filter_group = m.input_range.transpose(1, 0).numpy()
            # weight_per_out_filter_group = m.weight.transpose(1, 0).reshape(m.weight.size(1), -1).numpy()
            # output_per_out_filter_group = m.act_range.transpose(1,0).numpy()

            # input_min_max_per_group = input_per_out_filter_group.max(axis=1) - input_per_out_filter_group.min(axis=1)
            # range_per_group.append(min_max_per_group)
            # if min_max_per_group.max() > max_:
            #     max_ = min_max_per_group.max()

    # violin
    # range_per_group_sorted = sorted(list(range_per_group))
    draw_violin_graph(range_per_group, max_, violin_path + f'/{naming}.png', naming, epoch)


def set_mixed_bits_per_input_channels(model, percentile):
    quantile_tensor = torch.tensor(percentile, device='cuda')
    for m in model.modules():
        if isinstance(m, FusedConv2d):
            in_channel = m.in_channels
            # [16. 3. 4.4] -> [3, 16, 4, 4]
            weight_per_filter_group = m.conv.weight.transpose(1, 0)

            weight_group = weight_per_filter_group.reshape(in_channel, -1)
            min_max_per_group = torch.zeros((2, in_channel), device='cuda')
            min_max_per_group[0], min_max_per_group[1] = weight_group.min(dim=1).values, weight_group.max(
                dim=1).values
            cur_range = min_max_per_group[1] - min_max_per_group[0]

            cur_quantile = torch.quantile(cur_range, quantile_tensor)
            m.w_bit.data = torch.where(cur_range <= cur_quantile, 4, 8).type(torch.int8)
            m.low_group = (m.w_bit.data == 4).nonzero(as_tuple=True)[0]
            m.high_group = (m.w_bit.data == 8).nonzero(as_tuple=True)[0]


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
    if args.nnac and clustering_model.final_cluster is None:
        clustering_model.nn_aware_clutering(pretrained_model, train_loader)

    model = get_finetuning_model(arg_dict, tools, pretrained_model)
    if pretrained_model:
        del pretrained_model
    model.cuda()

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    opt_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    criterion = torch.nn.CrossEntropyLoss().cuda()

    if args.mixed_precision:
        set_mixed_bits_per_input_channels(model, args.percentile)

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
    for e in range(epoch_to_start, args.epoch + 1):
        if e > args.fq:
            runtime_helper.apply_fake_quantization = True

        if args.cluster > 1:
            pcq_epoch(model, clustering_model, train_loader, criterion, optimizer, runtime_helper, e, logger)
        else:
            train_epoch(model, train_loader, criterion, optimizer, e, logger)

        if e == 1 or e % 10 == 0:
            visualize(args, model, e)
            model.cuda()
        opt_scheduler.step()

        # fp_score = 0
        # if args.dataset != 'imagenet':
        #     if args.cluster > 1:
        #         fp_score = pcq_validate(model, clustering_model, val_loader, criterion, runtime_helper, logger)
        #     else:
        #         fp_score = validate(model, val_loader, criterion, logger)
        #
        # state = {
        #     'epoch': e,
        #     'state_dict': model.state_dict(),
        #     'optimizer': optimizer.state_dict(),
        # }
        # save_checkpoint(state, False, save_path_fp)
        #
        # # Test quantized model, and save if performs the best
        # if e > args.fq:
        #     model.set_quantization_params()
        #     if quantized_model is None:
        #         if args.dataset == 'cifar100':
        #             quantized_model = tools.quantized_model_initializer(arg_dict, num_classes=100)
        #         else:
        #             quantized_model = tools.quantized_model_initializer(arg_dict)
        #     quantized_model = tools.quantizer(model, quantized_model)
        #     quantized_model.cuda()
        #
        #     if args.cluster > 1:
        #         int_score = pcq_validate(quantized_model, clustering_model, val_loader, criterion, runtime_helper,
        #                                  logger)
        #     else:
        #         int_score = validate(quantized_model, val_loader, criterion, logger)
        #
        #     if int_score > best_int_val_score:
        #         best_epoch = e
        #         # Save best model's FP model
        #         with open(os.path.join(save_path_fp, "params.json"), 'w') as f:
        #             tmp = vars(args)
        #             tmp['best_epoch'] = e
        #             tmp['best_score'] = fp_score
        #             json.dump(tmp, f, indent=4)
        #         shutil.copyfile(os.path.join(save_path_fp, 'checkpoint.pth'), os.path.join(save_path_fp, 'best.pth'))
        #
        #         # Save best model's INT model
        #         best_int_val_score = int_score
        #         with open(os.path.join(save_path_int, "params.json"), 'w') as f:
        #             tmp = vars(args)
        #             tmp['best_epoch'] = e
        #             tmp['best_int_val_score'] = best_int_val_score
        #             json.dump(tmp, f, indent=4)
        #         filepath = os.path.join(save_path_int, 'checkpoint.pth')
        #         torch.save({'state_dict': quantized_model.state_dict()}, filepath)
        #     print('Best INT-val Score: {:.2f} (Epoch: {})'.format(best_int_val_score, best_epoch))

    # # Test quantized model which scored the best with validation dataset
    # if test_loader is None:
    #     test_score = best_int_val_score
    # else:
    #     arg_dict['quantized'] = True
    #     quantized_model = load_dnn_model(arg_dict, tools, os.path.join(save_path_int, 'checkpoint.pth')).cuda()
    #
    #     if args.cluster > 1:
    #         test_score = pcq_validate(quantized_model, clustering_model, test_loader, criterion, runtime_helper, logger)
    #     else:
    #         test_score = validate(quantized_model, test_loader, criterion, logger)
    #
    # with open(os.path.join(save_path_int, "params.json"), 'w') as f:
    #     tmp = vars(args)
    #     tmp['best_epoch'] = best_epoch
    #     tmp['best_int_val_score'] = best_int_val_score
    #     tmp['int_test_score'] = test_score
    #     json.dump(tmp, f, indent=4)
    #
    # tuning_time_cost = get_time_cost_in_string(time() - tuning_start_time)
    # if args.cluster > 1:
    #     method = f'DAQ({args.clustering_method}, K{args.cluster}S{args.sub_cluster}P{args.partition}-{args.repr_method})+{args.quant_base}'
    # else:
    #     method = args.quant_base
    #
    # pc = ''
    # if args.per_channel:
    #     pc = 'PerChannel, '
    # if args.symmetric:
    #     pc += 'Symmetric, '
    #
    # with open(f'./qat_{args.arch}_{args.dataset}_{args.bit}_F{args.bit_first}L{args.bit_classifier}_{args.gpu}.txt', 'a') as f:
    #     f.write('{:.2f} # {}, {}, {}, LR: {}, W-decay: {}, Epoch: {}, Batch: {}, {}Bit(First/Last/AddCat): {}({}/{}/{}), Smooth: {}, Best-epoch: {}, Time: {}, GPU: {}, Path: {}\n'
    #             .format(test_score, args.arch, args.dataset, method, args.lr, args.weight_decay, args.epoch, args.batch,
    #                     pc, args.bit, args.bit_first, args.bit_classifier, args.bit_addcat, args.smooth, best_epoch,
    #                     tuning_time_cost, args.gpu, save_path_fp))

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

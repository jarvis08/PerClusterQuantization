from copy import deepcopy

import torch
from torchsummary import summary

from utils import *
from .models import *
from tqdm import tqdm
from time import time

import warnings
warnings.filterwarnings("ignore")

def _finetune(args, tools, data_loaders, clustering_model):
    tuning_start_time = time()

    runtime_helper = RuntimeHelper()
    runtime_helper.set_pcq_arguments(args)

    arg_dict = deepcopy(vars(args))
    arg_dict['runtime_helper'] = runtime_helper

    pretrained_model = load_dnn_model(arg_dict, tools)
    pretrained_model.cuda()

    train_loader = data_loaders['aug_train']
    #val_loader = data_loaders['val']
    test_loader = data_loaders['test']
    if args.nnac and clustering_model.final_cluster is None:
        clustering_model.nn_aware_clustering(pretrained_model, train_loader, args.arch)

    model = get_finetuning_model(arg_dict, tools, pretrained_model)
    if pretrained_model:
        del pretrained_model
    model.cuda()

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    opt_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    criterion = torch.nn.CrossEntropyLoss().cuda()

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

    model.change_classifier_input_bit(8)
    for e in range(epoch_to_start, args.epoch + 1):
        if e > args.fq:
            runtime_helper.apply_fake_quantization = True

        if args.cluster > 1:
            pcq_epoch(model, clustering_model, train_loader, criterion, optimizer, runtime_helper, e, logger)
        else:
            train_epoch(model, train_loader, criterion, optimizer, e, logger)
        opt_scheduler.step()

        if args.fold_convbn:
            tools.folder(model)

        fp_score = 0
        if args.dataset != 'imagenet':
            if args.cluster > 1:
                #fp_score = pcq_validate(model, clustering_model, val_loader, criterion, runtime_helper, logger)
                fp_score = pcq_validate(model, clustering_model, test_loader, criterion, runtime_helper, logger)
            else:
                #fp_score = validate(model, val_loader, criterion, logger)
                fp_score = validate(model, test_loader, criterion, logger)

        state = {
            'epoch': e,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        save_checkpoint(state, False, save_path_fp)

        # Test quantized model, and save if performs the best
        if e > args.fq:
            model.set_quantization_params()
            if quantized_model is None:
                if args.dataset == 'cifar100':
                    quantized_model = tools.quantized_model_initializer(arg_dict, num_classes=100)
                else:
                    quantized_model = tools.quantized_model_initializer(arg_dict)
            quantized_model = tools.quantizer(model, quantized_model)
            quantized_model.cuda()

            if args.cluster > 1:
                #int_score = pcq_validate(quantized_model, clustering_model, val_loader, criterion, runtime_helper,
                #                         logger)
                int_score = pcq_validate(quantized_model, clustering_model, test_loader, criterion, runtime_helper,
                                         logger)
            else:
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
    if args.per_channel:
        pc = 'PerChannel, '
    if args.symmetric:
        pc += 'Symmetric, '

    with open(f'./qat_{args.arch}_{args.dataset}_{args.bit}_F{args.bit_first}L{args.bit_classifier}_{args.gpu}.txt', 'a') as f:
        f.write('{:.2f} # {}, {}, {}, LR: {}, W-decay: {}, Epoch: {}, Batch: {}, {}Bit(First/Last/AddCat): {}({}/{}/{}), Smooth: {}, Best-epoch: {}, Time: {}, GPU: {}, Path: {}\n'
                .format(test_score, args.arch, args.dataset, method, args.lr, args.weight_decay, args.epoch, args.batch,
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

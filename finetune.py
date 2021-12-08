from copy import deepcopy

import torch
from torchsummary import summary

from utils import *
from models import *
from tqdm import tqdm
from time import time


def pcq_epoch(model, clustering_model, train_loader, criterion, optimizer, runtime_helper, epoch, logger):
    losses = AverageMeter()
    top1 = AverageMeter()
    model.train()
    container = InputContainer(train_loader, clustering_model, runtime_helper.num_clusters,
                               clustering_model.args.dataset, clustering_model.args.batch)
    container.initialize_generator()
    container.set_next_batch()
    with tqdm(range(len(train_loader)), desc="Epoch {}".format(epoch), ncols=90) as t:
        for i, _ in enumerate(t):
            input, target, runtime_helper.batch_cluster = container.get_batch()
            input, target = input.cuda(), target.cuda()
            output = model(input)

            loss = criterion(output, target)

            prec = accuracy(output, target)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec.item(), input.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            container.set_next_batch()

            logger.debug("[Epoch] {}, step {}/{} [Loss] {:.5f} (avg: {:.5f}) [Score] {:.3f} (avg: {:.3f})"
                         .format(epoch, i + 1, len(train_loader), loss.item(), losses.avg, prec.item(), top1.avg))
            t.set_postfix(loss=losses.avg, acc=top1.avg)

            if container.ready_cluster is None:
                break


def _finetune(args, tools):
    tuning_start_time = time()
    normalizer = get_normalizer(args.dataset)

    clustering_train_loader = None
    test_loader = None
    augmented_train_dataset = get_augmented_train_dataset(args, normalizer)
    if args.dataset != 'imagenet':
        augmented_train_dataset, _ = split_dataset_into_train_and_val(augmented_train_dataset, args.dataset)
        train_loader = get_data_loader(augmented_train_dataset, batch_size=args.batch, shuffle=True, workers=args.worker)

        non_augmented_train_dataset = get_non_augmented_train_dataset(args, normalizer)
        if args.cluster > 1 and not args.clustering_path:
            non_augmented_train_dataset, val_dataset = \
                split_dataset_into_train_and_val(non_augmented_train_dataset, args.dataset)
            clustering_train_loader = get_data_loader(non_augmented_train_dataset,
                                                      batch_size=256, shuffle=True, workers=args.worker)
        else:
            _, val_dataset = split_dataset_into_train_and_val(non_augmented_train_dataset, args.dataset)

        test_dataset = get_test_dataset(args, normalizer)
        test_loader = get_data_loader(test_dataset, batch_size=args.val_batch, shuffle=False, workers=args.worker)
    else:
        train_loader = get_data_loader(augmented_train_dataset, batch_size=args.batch, shuffle=True, workers=args.worker)
        val_dataset = get_test_dataset(args, normalizer)
    val_loader = get_data_loader(val_dataset, batch_size=args.val_batch, shuffle=False, workers=args.worker)

    runtime_helper = RuntimeHelper()
    runtime_helper.set_pcq_arguments(args)

    arg_dict = deepcopy(vars(args))
    arg_dict['runtime_helper'] = runtime_helper
    model = get_finetuning_model(arg_dict, tools)
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

    clustering_model = None
    if args.cluster > 1:
        clustering_model = tools.clustering_method(args)
        if not args.clustering_path:
            args.clustering_path = set_clustering_dir(args)
            if args.clustering_method == 'dist':
                clustering_model.train_clustering_model(clustering_train_loader, train_loader)
            else:
                clustering_model.train_clustering_model(clustering_train_loader)
        else:
            clustering_model.load_clustering_model()

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
        opt_scheduler.step()

        fp_score = 0
        if args.dataset != 'imagenet':
            if args.cluster > 1:
                fp_score = pcq_validate(model, clustering_model, val_loader, criterion, runtime_helper, logger)
            else:
                fp_score = validate(model, val_loader, criterion, logger)

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
                int_score = pcq_validate(quantized_model, clustering_model, val_loader, criterion, runtime_helper,
                                         logger)
            else:
                int_score = validate(quantized_model, val_loader, criterion, logger)

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

    with open(os.path.join(save_path_int, "params.json"), 'w') as f:
        tmp = vars(args)
        tmp['best_epoch'] = best_epoch
        tmp['best_int_val_score'] = best_int_val_score
        tmp['int_test_score'] = test_score
        json.dump(tmp, f, indent=4)

    tuning_time_cost = get_time_cost_in_string(time() - tuning_start_time)
    if args.cluster > 1:
        method = f'DAQ({args.clustering_method}, K{args.cluster}P{args.partition}-{args.repr_method})+{args.quant_base}'
    else:
        method = args.quant_base

    pc = ''
    if args.per_channel:
        pc = 'PerChannel, '
    if args.symmetric:
        pc += 'Symmetric, '

    with open('./exp_results.txt', 'a') as f:
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

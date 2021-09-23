from copy import deepcopy

import torch
from torchsummary import summary

from utils import *
from models import *
from tqdm import tqdm
from time import time
import torch.cuda.nvtx as nvtx


def pcq_epoch(model, clustering_model, phase1_loader, phase2_loader, criterion, optimizer, runtime_helper, epoch, logger):
    losses = AverageMeter()
    top1 = AverageMeter()
    phase1_iter = iter(phase1_loader)

    model.train()
    with tqdm(phase1_loader, unit="batch", ncols=90) as t:
        for i in range(len(phase1_loader)):
            nvtx.range_push("train data Loading")
            input, target = next(phase1_iter)
            nvtx.range_pop()
            t.set_description("Epoch {}".format(epoch))

            # Phase-1
            nvtx.range_push("copy data to cuda")
            input, target = input.cuda(), target.cuda()
            nvtx.range_pop()

            nvtx.range_push("get ctr info of cur batch")
            runtime_helper.batch_cluster = clustering_model.predict_cluster_of_batch(input)
            nvtx.range_pop()

            nvtx.range_push("Forward")
            output = model(input)
            nvtx.range_pop()
            loss = criterion(output, target)
            prec = accuracy(output, target)[0]

            losses.update(loss.item(), input.size(0))
            top1.update(prec.item(), input.size(0))

            optimizer.zero_grad()

            nvtx.range_push("Backward")
            loss.backward()
            nvtx.range_pop()
            optimizer.step()

            # Phase-2
            runtime_helper.range_update_phase = True

            nvtx.range_push("phase2 data loading")
            phase2_input = phase2_loader.get_next_data()
            nvtx.range_pop()

            runtime_helper.batch_cluster = phase2_loader.batch_cluster

            nvtx.range_push("phase2 forward")
            with torch.no_grad():
                model(phase2_input.cuda())
            nvtx.range_pop()

            runtime_helper.range_update_phase = False

            logger.debug("[Epoch] {}, step {}/{} [Loss] {:.5f} (avg: {:.5f}) [Score] {:.3f} (avg: {:.3f})"
                         .format(epoch, i + 1, len(t), loss.item(), losses.avg, prec.item(), top1.avg))
            t.set_postfix(loss=losses.avg, acc=top1.avg)
            if i == 4: break


def _finetune(args, tools):
    tuning_start_time = time()
    normalizer = get_normalizer(args.dataset)

    augmented_train_dataset = get_augmented_train_dataset(args, normalizer)
    non_augmented_train_dataset = get_non_augmented_train_dataset(args, normalizer)

    test_loader = None
    if args.dataset != 'imagenet':
        train_dataset, _ = split_dataset_into_train_and_val(augmented_train_dataset, args.dataset)
        non_augmented_train_dataset, val_dataset = split_dataset_into_train_and_val(non_augmented_train_dataset, args.dataset)
        test_dataset = get_test_dataset(args, normalizer)
        test_loader = get_sequential_loader(args, test_dataset)
        train_loader = get_shuffled_loader(args, train_dataset)
    else:
        val_dataset = get_test_dataset(args, normalizer)
        train_loader = get_shuffled_loader(args, augmented_train_dataset)
    val_loader = get_sequential_loader(args, val_dataset)

    runtime_helper = RuntimeHelper()
    runtime_helper.set_pcq_arguments(args)

    arg_dict = deepcopy(vars(args))
    arg_dict['runtime_helper'] = runtime_helper
    model = get_finetuning_model(arg_dict, tools)
    model.cuda()

    # if args.dataset == 'imagenet':
    #    summary(model, (3, 224, 224))
    # else:
    #    summary(model, (3, 32, 32))

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

    if args.quant_noise:
        runtime_helper.qn_prob = args.qn_prob - 0.1
        tools.shift_qn_prob(model)

    phase2_loader = None
    clustering_model = None
    if args.cluster > 1:
        clustering_model = tools.clustering_method(args)
        if not args.clustering_path:
            args.clustering_path = set_clustering_dir(args)
            clustering_model.train_clustering_model(train_loader)
        else:
            clustering_model.load_clustering_model()

        # Make non-augmented dataset/loader for Phase-2 training
        if args.indices_path:
            indices_per_cluster, len_per_cluster = load_indices_list(args)
        else:
            sequential_non_aug_loader = get_sequential_loader(args, non_augmented_train_dataset)
            # test_augmented_clustering(clustering_model, non_augmented_loader, train_loader)
            indices_per_cluster, len_per_cluster = make_indices_list(clustering_model, sequential_non_aug_loader, args, runtime_helper)
            save_indices_list(args, indices_per_cluster, len_per_cluster)
            # check_cluster_distribution(clustering_model, non_augmented_loader)
            if args.visualize_clustering:
                visualize_clustering_res(sequential_non_aug_loader, clustering_model, indices_per_cluster, len_per_cluster, args.cluster)

        list_for_phase2 = make_phase2_list(args, indices_per_cluster, len_per_cluster)
        phase2_dataset = torch.utils.data.Subset(non_augmented_train_dataset, list_for_phase2)
        loader = torch.utils.data.DataLoader(phase2_dataset, batch_size=args.data_per_cluster * args.cluster,
                                             num_workers=args.worker, shuffle=False)
        phase2_loader = Phase2DataLoader(loader, args.cluster, args.data_per_cluster)

        if args.pcq_initialization:
            nvtx.range_push("pcq initialize")
            runtime_helper.batch_cluster = phase2_loader.batch_cluster
            initialize_pcq_model(model, phase2_loader.data_loader, criterion)
            nvtx.range_pop()

    if not save_path_fp:
        save_path_fp = set_save_dir(args)
    save_path_int = add_path(save_path_fp, 'quantized')
    logger = set_logger(save_path_fp)

    quantized_model = None
    runtime_helper.pcq_initialized = True
    for e in range(epoch_to_start, args.epoch + 1):
        if e > args.fq:
            runtime_helper.apply_fake_quantization = True

        # TODO: Quantnoise prob-increasing method
        if args.quant_noise and e % args.qn_increment_epoch == 1:
            model.runtime_helper.qn_prob += 0.1
            tools.shift_qn_prob(model)

        if args.cluster > 1:
            pcq_epoch(model, clustering_model, train_loader, phase2_loader, criterion, optimizer, runtime_helper, e,
                      logger)
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
            nvtx.range_push("quantize model")
            quantized_model = tools.quantizer(model, quantized_model)
            nvtx.range_pop()
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
    method = ''
    if args.quant_noise:
        method += 'QN{:.1f}+'.format(args.qn_prob)
    if args.cluster > 1:
        method += 'PCQ({}), K: {}'.format(args.clustering_method, args.cluster)
    elif not args.quant_noise:
        method += 'QAT'

    bn = ''
    if args.bn_momentum < 0.1:
        bn += 'BN-momentum: {:.3f}, '.format(args.bn_momentum)

    with open('./exp_results.txt', 'a') as f:
        f.write('{:.2f} # {}, {}, LR: {}, {}Epoch: {}, Batch: {}, FQ: {}, Best-epoch: {}, Time: {}, GPU: {}, Path: {}\n'
                .format(test_score, args.arch, method, args.lr, bn, args.epoch, args.batch, args.fq, best_epoch, tuning_time_cost, args.gpu, save_path_fp))

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

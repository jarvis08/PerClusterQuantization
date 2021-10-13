from copy import deepcopy

import torch
from torchsummary import summary

from utils import *
from models import check2mix_alexnet, check2mix_resnet
from models import quantized_alexnet, quantized_resnet
from tqdm import tqdm
from time import time


def get_finetuning_model(arg_dict, tools):
    if arg_dict['arch'] == 'AlexNetSmall':
        fused_model_initializer = check2mix_alexnet.fused_alexnet_small
    else:
        fused_model_initializer = check2mix_resnet.fused_resnet20

    if arg_dict['dataset'] == 'cifar100':
        fused_model = fused_model_initializer(arg_dict, num_classes=100)
    else:
        fused_model = fused_model_initializer(arg_dict)

    if arg_dict['fused']:
        checkpoint = torch.load(arg_dict['dnn_path'])
        fused_model.load_state_dict(checkpoint['state_dict'], strict=False)
    else:
        pretrained_model = load_dnn_model(arg_dict, tools)
        if arg_dict['arch'] == 'ResNet20':
            fused_model = check2mix_resnet.set_fused_resnet_with_fold_method(fused_model, pretrained_model)
        else:
            fused_model = tools.fuser(fused_model, pretrained_model)
    return fused_model


def find_layer_of_longest_range(model, loader, to_search=2):
    model.eval()
    with torch.no_grad():
        with tqdm(loader, unit="batch", ncols=90) as t:
            for i, (input, _) in enumerate(t):
                t.set_description("Set-range")
                input = input.cuda()
                _ = model(input)

        print("Find top-{} layers of longest range..".format(to_search))
        found_layers = []
        for n in range(to_search):
            found_layers.append(model.find_longest_range(found_layers))
        model.reset_ranges()
    return found_layers


def find_layer_of_largest_number_of_params(model, to_search=2):
    print("Find top-{} layers of largest number of parameters..".format(to_search))
    found_layers = []
    model.eval()
    with torch.no_grad():
        for _ in range(to_search):
            found_layers.append(model.find_biggest_layer(found_layers))
    return found_layers


def raise_bit_level(model, target_layer):
    with torch.no_grad():
        target_layer.w_bit.data = torch.tensor(8, dtype=torch.int8)
        target_layer.a_bit.data = torch.tensor(8, dtype=torch.int8)


def _check_and_finetune(args, tools):
    tuning_start_time = time()
    normalizer = get_normalizer(args.dataset)

    test_loader = None
    augmented_train_dataset = get_augmented_train_dataset(args, normalizer)
    if args.dataset != 'imagenet':
        augmented_train_dataset, _ = split_dataset_into_train_and_val(augmented_train_dataset, args.dataset)
        train_loader = get_data_loader(augmented_train_dataset, batch_size=args.batch, shuffle=True, workers=args.worker)

        non_augmented_train_dataset = get_non_augmented_train_dataset(args, normalizer)
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

    if not save_path_fp:
        save_path_fp = set_save_dir(args, allow_existence=False)
        args.dnn_path = save_path_fp
        print("Save dir: " + args.dnn_path)
    save_path_int = add_path(save_path_fp, 'quantized')
    logger = set_logger(save_path_fp)

    if args.mode == 'check2mix':
        if args.check_method == 'both':
            range_layers = find_layer_of_longest_range(model, train_loader, to_search=1)
            n_params_layers = find_layer_of_largest_number_of_params(model)
            raise_bit_level(model, range_layers[0])
            if id(range_layers[0]) != id(n_params_layers[0]):
                raise_bit_level(model, n_params_layers[0])
            else:
                raise_bit_level(model, n_params_layers[1])
        elif args.check_method == 'range':
            layers = find_layer_of_longest_range(model, train_loader, to_search=args.n_mix)
            raise_bit_level(model, layers[0])
            if args.n_mix == 2:
                raise_bit_level(model, layers[1])
        elif args.check_method == 'n_params':
            layers = find_layer_of_largest_number_of_params(model, to_search=args.n_mix)
            raise_bit_level(model, layers[0])
            if args.n_mix == 2:
                raise_bit_level(model, layers[1])
    runtime_helper.check2mix = False

    quantized_model = None
    for e in range(epoch_to_start, args.epoch + 1):
        if e > args.fq:
            runtime_helper.apply_fake_quantization = True

        train_epoch(model, train_loader, criterion, optimizer, e, logger)
        opt_scheduler.step()

        fp_score = 0
        if args.dataset != 'imagenet':
            fp_score = validate(model, val_loader, criterion, logger)

        state = {
            'epoch': e,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        save_checkpoint(state, False, save_path_fp)

        # Test quantized model, and save if performs the best
        if e > args.fq:
            if arg_dict['arch'] == 'ResNet20':
                model = check2mix_resnet.fold_resnet(model)
            model.set_quantization_params()
            if quantized_model is None:
                if args.dataset == 'cifar100':
                    quantized_model = tools.quantized_model_initializer(arg_dict, num_classes=100)
                else:
                    quantized_model = tools.quantized_model_initializer(arg_dict)

            if arg_dict['arch'] == 'ResNet20':
                quantized_model = quantized_resnet.quantize_folded_resnet(model, quantized_model)
            else:
                quantized_model = tools.quantizer(model, quantized_model)
            quantized_model.cuda()

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
        f.write('{:.2f} # {}, {}, LR: {}, {}Epoch: {}, Batch: {}, Bit(First/Last): {}({}/{}), FQ: {}, Best-epoch: {}, Time: {}, GPU: {}, Path: {}\n'
                .format(test_score, args.arch, method, args.lr, bn, args.epoch, args.batch, args.bit,
                    args.first_bit, args.classifier_bit, args.fq, best_epoch, tuning_time_cost, args.gpu, save_path_fp))

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

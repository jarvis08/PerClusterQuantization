from copy import deepcopy

import torch
from torchsummary import summary

from utils import *
from models import *
from tqdm import tqdm
from time import time


def make_indices_list(train_loader, args, runtime_helper):
    total_list = [[] for _ in range(args.cluster)]

    idx = 0
    with torch.no_grad():
        with tqdm(train_loader, unit="batch", ncols=90) as t:
            for i, (input, target) in enumerate(t):
                t.set_description("Indices per Cluster")
                runtime_helper.set_cluster_information_of_batch(input)
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
    min_count = min(len_per_cluster)
    max_count = max(len_per_cluster)

    cluster_cross_sorted = []
    n = args.data_per_cluster
    if args.use_max_cnt:
        cur_idx = [0 for _ in range(args.cluster)]
        for loops in range(max_count // n):
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
    else:
        done = 0
        for loops in range(min_count // n):
            for c in range(args.cluster):
                cluster_cross_sorted += indices_per_cluster[c][done:done + n]
            done += n

    return cluster_cross_sorted


def save_indices_list(args, indices_list_per_cluster, len_per_cluster):
    path = add_path('', 'result')
    path = add_path(path, 'indices')
    path = add_path(path, args.dataset)
    path = add_path(path, "{}data_per_cluster".format(args.data_per_cluster))
    path = add_path(path, datetime.now().strftime("%m-%d-%H%M"))
    with open(os.path.join(path, "params.json"), 'w') as f:
        indices_args = {'indices_list': indices_list_per_cluster, 'len_per_cluster': len_per_cluster,
                        'data_per_cluster': args.data_per_cluster, 'dataset': args.dataset}
        json.dump(indices_args, f, indent=4)


def load_indices_list(args):
    with open(os.path.join(args.indices_path, 'params.json'), 'r') as f:
        saved_args = json.load(f)
    assert args.dataset == saved_args['dataset'], \
        "Dataset should be same. \n" \
        "Model's dataset: {}, Loaded dataset: {}".format(saved_args['dataset'], args.dataset)
    assert args.data_per_cluster == saved_args['data_per_cluster'], \
        "Data per cluster should be same. \n" \
        "Model's arg: {}, Loaded arg: {}".format(saved_args['data_per_cluster'], args.data_per_cluster)
    return saved_args['indices_list'], saved_args['len_per_cluster']


def initialize_pcq_model(model, loader, criterion):
    losses = AverageMeter()
    top1 = AverageMeter()

    model.train()
    with torch.no_grad():
        if isinstance(loader, list):
            for c in range(len(loader)):
                with tqdm(loader[c], unit="batch", ncols=90) as t:
                    for i, (input, target) in enumerate(t):
                        t.set_description("Validate")
                        input, target = input.cuda(), target.cuda()
                        output = model(input)
                        loss = criterion(output, target)
                        prec = accuracy(output, target)[0]
                        losses.update(loss.item(), input.size(0))
                        top1.update(prec.item(), input.size(0))

                        t.set_postfix(loss=losses.avg, acc=top1.avg)
        else:
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


def pcq_epoch(model, phase1_loader, phase2_loader, criterion, optimizer, runtime_helper, epoch, logger):
    losses = AverageMeter()
    top1 = AverageMeter()

    model.train()
    with tqdm(phase1_loader, unit="batch", ncols=90) as t:
        for i, (input, target) in enumerate(t):
            t.set_description("Epoch {}".format(epoch))

            # Phase-1
            runtime_helper.set_cluster_information_of_batch(input)
            input, target = input.cuda(), target.cuda()
            output = model(input)

            loss = criterion(output, target)
            prec = accuracy(output, target)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec.item(), input.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Phase-2
            runtime_helper.range_update_phase = True
            phase2_input, _ = phase2_loader.get_next_data()
            runtime_helper.batch_cluster = phase2_loader.batch_cluster
            with torch.no_grad():
                model(phase2_input.cuda())
            runtime_helper.range_update_phase = False

            logger.debug("[Epoch] {}, step {}/{} [Loss] {:.5f} (avg: {:.5f}) [Score] {:.3f} (avg: {:.3f})"
                         .format(epoch, i + 1, len(t), loss.item(), losses.avg, prec.item(), top1.avg))
            t.set_postfix(loss=losses.avg, acc=top1.avg)


def get_finetuning_model(arg_dict, tools):
    pretrained_model = load_dnn_model(arg_dict, tools)
    fused_model = tools.fused_model_initializer(arg_dict)
    fused_model = tools.fuser(fused_model, pretrained_model)
    return fused_model


def _finetune(args, tools):
    tuning_start_time = time()
    normalizer = get_normalizer(args.dataset)
    train_dataset = get_train_dataset(args, normalizer)
    train_loader = get_data_loader(args, train_dataset)
    test_loader = get_test_loader(args, normalizer)

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

    if args.quant_noise:
        runtime_helper.qn_prob = args.qn_prob - 0.1
        tools.shift_qn_prob(model)

    epoch_to_start = 1
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    if args.fused:
        optimizer, epoch_to_start = load_optimizer(optimizer, args.dnn_path)
    opt_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    criterion = torch.nn.CrossEntropyLoss().cuda()

    phase2_loader = None
    if args.cluster > 1:
        # Set K-means model
        kmeans = KMeans(args)
        if not args.kmeans_path:
            kmeans_train_loader = get_data_loader(args, train_dataset, usage='kmeans')
            args.kmeans_path = set_kmeans_dir(args)
            kmeans.train_kmeans_model(kmeans_train_loader)
        else:
            kmeans.load_kmeans_model()
        runtime_helper.kmeans = kmeans
        # check_cluster_distribution(runtime_helper.kmeans, train_loader)

        # Make non-augmented dataset/loader for Phase-2 training
        non_augmented_dataset = get_train_dataset_without_augmentation(args, normalizer)
        if args.indices_path:
            indices_per_cluster, len_per_cluster = load_indices_list(args)
        else:
            non_augmented_loader = get_data_loader(args, non_augmented_dataset, usage='initializer')
            indices_per_cluster, len_per_cluster = make_indices_list(non_augmented_loader, args, runtime_helper)
            save_indices_list(args, indices_per_cluster, len_per_cluster)

        list_for_phase2 = make_phase2_list(args, indices_per_cluster, len_per_cluster)
        phase2_dataset = torch.utils.data.Subset(non_augmented_dataset, list_for_phase2)

        loader = torch.utils.data.DataLoader(phase2_dataset, batch_size=args.data_per_cluster * args.cluster,
                                             num_workers=args.worker, shuffle=False)
        phase2_loader = Phase2DataLoader(loader, args.cluster, args.data_per_cluster)

        if args.pcq_initialization:
            runtime_helper.batch_cluster = phase2_loader.batch_cluster
            initialize_pcq_model(model, phase2_loader.data_loader, criterion)

    runtime_helper.pcq_initialized = True
    save_path_fp = set_save_dir(args)
    save_path_int = add_path(save_path_fp, 'quantized')
    logger = set_logger(save_path_fp)
    best_score_int = 0
    best_epoch = 0
    for e in range(epoch_to_start, args.epoch + 1):
        if e > args.fq:
            runtime_helper.apply_fake_quantization = True

        # TODO: Quantnoise prob-increasing method
        if args.quant_noise and e % args.qn_increment_epoch == 1:
            model.runtime_helper.qn_prob += 0.1
            tools.shift_qn_prob(model)

        if args.quant_noise and e % 3 == 1:
            runtime_helper.qn_prob += 0.1
            tools.shift_qn_prob(model)

        if args.cluster > 1:
            pcq_epoch(model, train_loader, phase2_loader, criterion, optimizer, runtime_helper, e, logger)
        else:
            train_epoch(model, train_loader, criterion, optimizer, e, logger)
        opt_scheduler.step()

        if args.cluster > 1:
            fp_score = pcq_validate(model, test_loader, criterion, runtime_helper, logger)
        else:
            fp_score = validate(model, test_loader, criterion, logger)

        state = {
            'epoch': e,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        save_checkpoint(state, False, save_path_fp)

        # Test quantized model, and save if performs the best
        if e > args.fq:
            if tools.folder:
                folded_model = tools.folder(deepcopy(model))
            else:
                folded_model = deepcopy(model)
            folded_model.set_quantization_params()
            quantized_model = tools.quantized_model_initializer(arg_dict)
            quantized_model = tools.quantizer(folded_model, quantized_model)
            quantized_model.cuda()
            del folded_model

            if args.cluster > 1:
                val_score = pcq_validate(quantized_model, test_loader, criterion, runtime_helper, logger)
            else:
                val_score = validate(quantized_model, test_loader, criterion, logger)

            if val_score > best_score_int:
                best_epoch = e
                # Save best model's FP model
                with open(os.path.join(save_path_fp, "params.json"), 'w') as f:
                    tmp = vars(args)
                    tmp['best_epoch'] = e
                    tmp['best_score'] = fp_score
                    json.dump(tmp, f, indent=4)
                shutil.copyfile(os.path.join(save_path_fp, 'checkpoint.pth'), os.path.join(save_path_fp, 'best.pth'))

                # Save best model's INT model
                best_score_int = val_score
                with open(os.path.join(save_path_int, "params.json"), 'w') as f:
                    tmp = vars(args)
                    tmp['best_epoch'] = e
                    tmp['best_score'] = best_score_int
                    json.dump(tmp, f, indent=4)
                filepath = os.path.join(save_path_int, 'checkpoint.pth')
                torch.save({'state_dict': quantized_model.state_dict()}, filepath)
            del quantized_model

    tuning_time_cost = get_time_cost_in_string(time() - tuning_start_time)
    method = ''
    if args.quant_noise:
        method += 'QN{:.1f}+'.format(args.qn_prob)
    if args.cluster > 1:
        method += 'PCQ'
    elif not args.quant_noise:
        method += 'QAT'

    bn = ''
    if args.bn_momentum < 0.1:
        bn += 'BN{:.3f}, '.format(args.bn_momentum)

    with open('./exp_results.txt', 'a') as f:
        f.write('{:.2f} # {}, {}, Batch {}, Best-epoch {}, {}Time {}, Path {}\n'
                .format(best_score_int, args.arch, method, args.batch, best_epoch, bn, tuning_time_cost, save_path_fp))

    # with open('./test.txt', 'a') as f:
    #     for name, param in model.named_parameters():
    #         if 'act_range' in name:
    #             f.write('{}\n'.format(name))
    #             if 'norm' in name:
    #                 for c in range(args.cluster):
    #                     f.write('{:.4f}, {:.4f}\n'.format(param[0].item(), param[1].item()))
    #             else:
    #                 for c in range(args.cluster):
    #                     f.write('{:.4f}, {:.4f}\n'.format(param[c][0].item(), param[c][1].item()))
    # save_fused_network_in_darknet_form(model, args)

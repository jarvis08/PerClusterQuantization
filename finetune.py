from copy import deepcopy

import torch
from torchsummary import summary

from utils import *
from models import *
from tqdm import tqdm
from time import time


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


def pcq_epoch(model, clustering_model, phase1_loader, phase2_loader, criterion, optimizer, runtime_helper, epoch, logger):
    losses = AverageMeter()
    top1 = AverageMeter()

    model.train()
    with tqdm(phase1_loader, unit="batch", ncols=90) as t:
        for i, (input, target) in enumerate(t):
            t.set_description("Epoch {}".format(epoch))

            # Phase-1
            input, target = input.cuda(), target.cuda()
            runtime_helper.batch_cluster = clustering_model.predict_cluster_of_batch(input)
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
    if arg_dict['dataset'] == 'cifar100':
        fused_model = tools.fused_model_initializer(arg_dict, num_classes=100)
    else:
        fused_model = tools.fused_model_initializer(arg_dict)
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


def _finetune(args, tools):
    tuning_start_time = time()
    normalizer = get_normalizer(args.dataset)
    train_dataset, val_dataset = get_train_dataset(args, normalizer)
    train_loader = get_data_loader(args, train_dataset)
    val_loader = get_data_loader(args, val_dataset)
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

    best_score = 0
    best_epoch = 0
    if args.fused:
        optimizer, epoch_to_start = load_optimizer(optimizer, args.dnn_path)
        params_path = arg_dict['dnn_path']
        with open(os.path.join(params_path, "params.json"), 'r') as f:
            saved_args = json.load(f)
            best_score = saved_args['best_score']
            best_epoch = saved_args['best_epoch']

    opt_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    criterion = torch.nn.CrossEntropyLoss().cuda()

    phase2_loader = None
    clustering_model = None
    if args.cluster > 1:
        clustering_model = tools.clustering_method(args)
        if not args.clustering_path:
            clustering_train_loader = get_data_loader(args, train_dataset, usage='clustering')
            args.clustering_path = set_clustering_dir(args)
            clustering_model.train_clustering_model(clustering_train_loader)
        else:
            clustering_model.load_clustering_model()

        # Make non-augmented dataset/loader for Phase-2 training
        non_augmented_dataset = get_train_dataset_without_augmentation(args, normalizer)
        if args.indices_path:
            indices_per_cluster, len_per_cluster = load_indices_list(args)
        else:
            non_augmented_loader = get_data_loader(args, non_augmented_dataset, usage='initializer')
            indices_per_cluster, len_per_cluster = make_indices_list(clustering_model, non_augmented_loader, args, runtime_helper)
            save_indices_list(args, indices_per_cluster, len_per_cluster)
            #check_cluster_distribution(clustering_model, non_augmented_loader)
            if args.visualize_clustering:
                visualize_clustering_res(non_augmented_loader, clustering_model, indices_per_cluster, len_per_cluster, args.cluster)

        list_for_phase2 = make_phase2_list(args, indices_per_cluster, len_per_cluster)
        phase2_dataset = torch.utils.data.Subset(non_augmented_dataset, list_for_phase2)

        loader = torch.utils.data.DataLoader(phase2_dataset, batch_size=args.data_per_cluster * args.cluster,
                                             num_workers=args.worker, shuffle=False)
        phase2_loader = Phase2DataLoader(loader, args.cluster, args.data_per_cluster)

        if args.pcq_initialization:
            runtime_helper.batch_cluster = phase2_loader.batch_cluster
            initialize_pcq_model(model, phase2_loader.data_loader, criterion)

    quantized_model = None
    runtime_helper.pcq_initialized = True
    save_path_fp = set_save_dir(args)
    save_path_int = add_path(save_path_fp, 'quantized')
    logger = set_logger(save_path_fp)
    for e in range(epoch_to_start, args.epoch + 1):
        if e > args.fq:
            runtime_helper.apply_fake_quantization = True

        # TODO: Quantnoise prob-increasing method
        if args.quant_noise and e % args.qn_increment_epoch == 1:
            model.runtime_helper.qn_prob += 0.1
            tools.shift_qn_prob(model)

        if args.cluster > 1:
            pcq_epoch(model, clustering_model, train_loader, phase2_loader, criterion, optimizer, runtime_helper, e, logger)
        else:
            train_epoch(model, train_loader, criterion, optimizer, e, logger)
        opt_scheduler.step()

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

        if e > args.fq and fp_score > best_score:
            best_epoch = e
            best_score = fp_score
            # Save best model's FP model
            with open(os.path.join(save_path_fp, "params.json"), 'w') as f:
                tmp = vars(args)
                tmp['best_epoch'] = e
                tmp['best_score'] = fp_score
                json.dump(tmp, f, indent=4)
            shutil.copyfile(os.path.join(save_path_fp, 'checkpoint.pth'), os.path.join(save_path_fp, 'best.pth'))

    # Test quantized model, and save if performs the best
    # if last_epoch is not the best epoch, load the best model
    if fp_score < best_score:
        del model
        model = load_dnn_model(arg_dict, tools, os.path.join(save_path_fp, 'best.pth'))

    model.set_quantization_params()
    if quantized_model is None:
        if args.dataset == 'cifar100':
            quantized_model = tools.quantized_model_initializer(arg_dict, num_classes=100)
        else:
            quantized_model = tools.quantized_model_initializer(arg_dict)
    quantized_model = tools.quantizer(model, quantized_model)
    quantized_model.cuda()

    if args.cluster > 1:
        int_score = pcq_validate(quantized_model, clustering_model, test_loader, criterion, runtime_helper, logger)
    else:
        int_score = validate(quantized_model, test_loader, criterion, logger)

    # Save best model's INT model
    with open(os.path.join(save_path_int, "params.json"), 'w') as f:
        tmp = vars(args)
        tmp['fp_epoch'] = best_epoch
        tmp['fp_score'] = best_score
        tmp['int_score'] = int_score
        json.dump(tmp, f, indent=4)
    filepath = os.path.join(save_path_int, 'checkpoint.pth')
    torch.save({'state_dict': quantized_model.state_dict()}, filepath)

    tuning_time_cost = get_time_cost_in_string(time() - tuning_start_time)
    method = ''
    if args.quant_noise:
        method += 'QN{:.1f}+'.format(args.qn_prob)
    if args.cluster > 1:
        method += 'PCQ({})'.format(args.clustering_method)
    elif not args.quant_noise:
        method += 'QAT'

    bn = ''
    if args.bn_momentum < 0.1:
        bn += 'BN-momentum: {:.3f}, '.format(args.bn_momentum)

    n_cluster = ''
    if 'PCQ' in method:
        n_cluster += 'K: {}, '.format(args.cluster)

    with open('./exp_results.txt', 'a') as f:
        f.write('{:.2f} # {}, {}, LR: {}, Epoch: {}, Batch: {}, FQ: {}, {}Best-epoch: {}, {}Time: {}, GPU: {}, Path: {}\n'
                .format(int_score, args.arch, method, args.lr, args.epoch, args.batch, args.fq, n_cluster, best_epoch, bn, tuning_time_cost, args.gpu, save_path_fp))

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
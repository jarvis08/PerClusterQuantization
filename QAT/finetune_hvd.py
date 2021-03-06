from copy import deepcopy

import torch
import torch.backends.cudnn as cudnn
import horovod.torch as hvd
from torchsummary import summary

from utils import *
from models import *
from tqdm import tqdm


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
    # shuffle list
    min_count = 99999999999999999
    for c in range(args.cluster):
        if min_count > len(total_list[c]):
            min_count = len(total_list[c])
        random.shuffle(total_list[c])
    return total_list, min_count


def initialize_pcq_model(model, loader, criterion):
    losses = AverageMeter()
    top1 = AverageMeter()

    model.eval()
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


def get_data_loader_hvd(args, dataset, usage=None):
    if usage == 'kmeans':
        if args.dataset == 'imagenet':
            loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True, num_workers=10)
        else:
            loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True, num_workers=2)
    elif usage == 'initializer':
        if args.dataset == 'imagenet':
            loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False, num_workers=10)
        else:
            loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=False, num_workers=2)
    else:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, hvd.size(), hvd.rank())
        if args.dataset == 'imagenet':
            loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch, sampler=sampler, num_workers=10)
        else:
            loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch, sampler=sampler, num_workers=2)
    return loader


def pcq_epoch(model, train_loader, criterion, optimizer, runtime_helper, epoch, logger):
    losses = AverageMeter()
    top1 = AverageMeter()

    model.train()
    with tqdm(train_loader, unit="batch", ncols=90) as t:
        for i, (input, target) in enumerate(t):
            t.set_description("Epoch {}".format(epoch))
            runtime_helper.set_cluster_information_of_batch(input)
            input, target = runtime_helper.sort_by_cluster_info(input, target)
            input, target = input.cuda(), target.cuda()
            output = model(input)

            loss = criterion(output, target)
            prec = accuracy(output, target[0])
            losses.update(loss.item(), input.size(0))
            top1.update(prec.item(), input.size(0))

            if hvd.rank() == 0:
                logger.debug("[Epoch] {}, step {}/{} [Loss] {:.5f} (avg: {:.5f}) [Score] {:.3f} (avg: {:.3f})"
                             .format(epoch, i + 1, len(t), loss.item(), losses.avg, prec.item(), top1.avg))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            t.set_postfix(loss=losses.avg, acc=top1.avg)


# def metric_average(val, name):
#     tensor = torch.tensor(val)
#     avg_tensor = hvd.allreduce(tensor, name=name)
#     return avg_tensor.item()


def get_finetuning_model(arg_dict, tools):
    pretrained_model = load_dnn_model(arg_dict, tools)
    fused_model = tools.fused_model_initializer(arg_dict)
    fused_model = tools.fuser(fused_model, pretrained_model)
    return fused_model, arg_dict


def hvd_finetune(args, tools):
    save_path_fp = set_save_dir(args)
    save_path_int = add_path(save_path_fp, 'quantized')
    logger = set_logger(save_path_fp)
    # init hvd
    hvd.init()
    torch.set_num_threads(1)
    torch.cuda.set_device(hvd.local_rank())

    normalizer = get_normalizer(args.dataset)
    criterion = torch.nn.CrossEntropyLoss().cuda()
    test_loader = get_test_loader(args, normalizer)
    train_dataset = get_train_dataset(args, normalizer)
    train_loader = get_data_loader_hvd(args, train_dataset)
    runtime_helper = RuntimeHelper()
    arg_dict = deepcopy(vars(args))

    if runtime_helper:
        arg_dict['runtime_helper'] = runtime_helper

    cudnn.benchmark = True

    model, arg_dict = get_finetuning_model(arg_dict, tools)
    model.cuda()
    model.eval()

    if args.quant_noise:
        runtime_helper.qn_prob = args.qn_prob - 0.1
        tools.qn_forward_pre_hooker(model)

    epoch_to_start = 1

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)

    if args.fused:
        optimizer, epoch_to_start = load_optimizer(optimizer, args.dnn_path)

    optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())
    opt_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    if hvd.rank() == 0:
        #        if args.dataset == 'imagenet':
        #           summary(model, (3, 224, 224))
        #        else:
        #           summary(model, (3, 32, 32))

        if args.cluster > 1:
            kmeans = KMeans(args)
            kmeans_train_loader = get_data_loader_hvd(args, train_dataset, usage='kmeans')
            if not args.kmeans_path:
                args.kmeans_path = set_kmeans_dir(args)
                kmeans.train_kmeans_model(kmeans_train_loader)
            else:
                kmeans.load_kmeans_model()
            runtime_helper.kmeans = kmeans
            # check_cluster_distribution(runtime_helper.kmeans, train_loader)

            loaders = []
            non_augmented_dataset = get_train_dataset_without_augmentation(args, normalizer)
            non_augmented_loader = get_data_loader(args, non_augmented_dataset, usage='initializer')
            indices_per_cluster, min_count = make_indices_list(non_augmented_loader, args, runtime_helper)
            # Load Minimum number of images
            list_with_minimum = []
            done = 0
            for loops in range(min_count // 8):
                for c in range(args.cluster):
                    list_with_minimum += indices_per_cluster[c][done:done + 8]
                done += 8
            sorted_dataset = torch.utils.data.Subset(non_augmented_dataset, list_with_minimum)
            sorted_loader = torch.utils.data.DataLoader(sorted_dataset, batch_size=64, num_workers=2, shuffle=False)
            bc = []
            for c in range(args.cluster):
                tmp = [c, 8]
                bc.append(tmp)
            runtime_helper.batch_cluster = torch.cuda.LongTensor(bc)
            initialize_pcq_model(model, sorted_loader, criterion)

            # # Load images per cluster
            # for c in range(args.cluster):
            #     cur_dataset = torch.utils.data.Subset(indices_train_loader.dataset, indices_per_cluster[c])
            #     loaders.append(torch.utils.data.DataLoader(cur_dataset, batch_size=8, num_workers=2, shuffle=False))
            # initialize_pcq_model(model, loaders, criterion, runtime_helper)

    runtime_helper.pcq_initialized = True

    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    best_score_int = 0
    for e in range(epoch_to_start, args.epoch + 1):
        if e > args.fq:
            runtime_helper.apply_fake_quantization = True

        # TODO: Quantnoise prob-increasing method
        if args.quant_noise and e % 3 == 1:
            runtime_helper.qn_prob += 0.1
            tools.qn_forward_pre_hooker(model)
        # TODO: In Fused/PCQ-Conv/Linear, use runtimehelper.qn_prob

        if args.cluster > 1:
            pcq_epoch(model, train_loader, criterion, optimizer, runtime_helper, e, logger)
        else:
            train_epoch(model, train_loader, criterion, optimizer, e, logger)

        opt_scheduler.step()

        if args.cluster > 1:
            fp_score = pcq_validate(model, test_loader, criterion, runtime_helper, logger, sort_input=True)
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
                val_score = pcq_validate(quantized_model, test_loader, criterion, runtime_helper, logger, True)
            else:
                val_score = validate(quantized_model, test_loader, criterion, logger, True)

            if val_score > best_score_int:
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

    with open('./exp_results.txt', 'a') as f:
        f.write('{:.2f}\n'.format(best_score_int))

    with open('./test.txt', 'a') as f:
        for name, param in model.named_parameters():
            if 'act_range' in name:
                f.write('{}\n'.format(name))
                if 'norm' in name:
                    for c in range(args.cluster):
                        f.write('{:.4f}, {:.4f}\n'.format(param[0].item(), param[1].item()))
                else:
                    for c in range(args.cluster):
                        f.write('{:.4f}, {:.4f}\n'.format(param[c][0].item(), param[c][1].item()))
    # save_fused_network_in_darknet_form(model, args)

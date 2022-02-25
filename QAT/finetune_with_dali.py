import torch

from utils import *
from models import *
from tqdm import tqdm
from time import time


def dali_train_epoch(model, train_loader, criterion, optimizer, epoch, logger):
    losses = AverageMeter()
    top1 = AverageMeter()

    model.train()
    with tqdm(train_loader, unit="batch", ncols=90) as t:
        for i, data in enumerate(t):
            t.set_description("Epoch {}".format(epoch))
            input = data[0]["data"]
            target = data[0]["label"].squeeze(-1).long()
            output = model(input)
            loss = criterion(output, target)
            prec = accuracy(output, target)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec.item(), input.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            logger.debug("[Epoch] {}, step {}/{} [Loss] {:.5f} (avg: {:.5f}) [Score] {:.3f} (avg: {:.3f})"
                         .format(epoch, i + 1, len(t), loss.item(), losses.avg, prec.item(), top1.avg))
            t.set_postfix(loss=losses.avg, acc=top1.avg)


def dali_validate(model, test_loader, criterion, logger=None):
    losses = AverageMeter()
    top1 = AverageMeter()

    model.eval()
    with torch.no_grad():
        with tqdm(test_loader, unit="batch", ncols=90) as t:
            for i, data in enumerate(t):
                t.set_description("Validate")
                input = data[0]["data"]
                target = data[0]["label"].squeeze(-1).long()
                output = model(input)
                loss = criterion(output, target)
                prec = accuracy(output, target)[0]
                losses.update(loss.item(), input.size(0))
                top1.update(prec.item(), input.size(0))

                t.set_postfix(loss=losses.avg, acc=top1.avg)

    if logger is not None:
        logger.debug("[Validation] Loss: {:.5f}, Score: {:.3f}".format(losses.avg, top1.avg))
    return top1.avg


def dali_pcq_epoch(model, clustering_model, train_loader, criterion, optimizer, runtime_helper, epoch, logger):
    losses = AverageMeter()
    top1 = AverageMeter()
    model.train()
    container = InputContainer(runtime_helper.num_clusters, clustering_model.args.batch)
    with tqdm(train_loader, unit="batch", ncols=90) as t:
        for i, data in enumerate(t):
            t.set_description("Epoch {}".format(epoch))

            images, targets = data[0]["data"], data[0]["label"].squeeze(-1).long()
            cluster_info = clustering_model.predict_cluster_of_batch(images)
            images, targets = images.cpu(), targets.cpu()

            input, target, runtime_helper.batch_cluster = container.gather_and_get_data(images, targets, cluster_info)
            if input is None:
                continue

            input, target = input.cuda(), target.cuda()
            output = model(input)
            loss = criterion(output, target)

            prec = accuracy(output, target)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec.item(), input.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            logger.debug("[Epoch] {}, step {}/{} [Loss] {:.5f} (avg: {:.5f}) [Score] {:.3f} (avg: {:.3f})"
                         .format(epoch, i + 1, len(t), loss.item(), losses.avg, prec.item(), top1.avg))
            t.set_postfix(loss=losses.avg, acc=top1.avg)

    #leftover = container.check_leftover()
    #if leftover:
    #    with tqdm(range(leftover), unit="batch", ncols=90) as t:
    #        for _ in t:
    #            t.set_description("Leftover")
    #            input, target, runtime_helper.batch_cluster = container.get_leftover()
    #            input = input.cuda()
    #            target = target.cuda()
    #            output = model(input)

    #            loss = criterion(output, target)
    #            prec = accuracy(output, target)[0]
    #            losses.update(loss.item(), input.size(0))
    #            top1.update(prec.item(), input.size(0))

    #            optimizer.zero_grad()
    #            loss.backward()
    #            optimizer.step()

    #            logger.debug("[Epoch] {}, step {}/{} [Loss] {:.5f} (avg: {:.5f}) [Score] {:.3f} (avg: {:.3f})"
    #                         .format(epoch, i + 1, len(t), loss.item(), losses.avg, prec.item(), top1.avg))
    #            t.set_postfix(loss=losses.avg, acc=top1.avg)
    #return top1.avg



def dali_pcq_validate(model, clustering_model, test_loader, criterion, runtime_helper, logger=None):
    losses = AverageMeter()
    top1 = AverageMeter()

    model.eval()
    with torch.no_grad():
        with tqdm(test_loader, unit="batch", ncols=90) as t:
            for i, data in enumerate(t):
                t.set_description("Validate")
                input = data[0]["data"]
                target = data[0]["label"].squeeze(-1).long()
                runtime_helper.batch_cluster = clustering_model.predict_cluster_of_batch(input)
                output = model(input)
                loss = criterion(output, target)
                prec = accuracy(output, target)[0]
                losses.update(loss.item(), input.size(0))
                top1.update(prec.item(), input.size(0))

                t.set_postfix(loss=losses.avg, acc=top1.avg)

    if logger is not None:
        logger.debug("[Validation] Loss: {:.5f}, Score: {:.3f}".format(losses.avg, top1.avg))
    return top1.avg


def _finetune_with_dali(args, tools):
    tuning_start_time = time()
    train_loader = get_dali_loader(args, 'train')
    val_loader = get_dali_loader(args, 'val')
    test_loader = None

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

    if args.quant_noise:
        runtime_helper.qn_prob = args.qn_prob - 0.1
        tools.shift_qn_prob(model)

    clustering_model = None
    if args.cluster > 1:
        clustering_model = tools.clustering_method(args)
        if not args.clustering_path:
            args.clustering_path = set_clustering_dir(args)
            clustering_model.train_clustering_model(train_loader)
        else:
            clustering_model.load_clustering_model()

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
            dali_pcq_epoch(model, clustering_model, train_loader, criterion, optimizer, runtime_helper, e, logger)
        else:
            dali_train_epoch(model, train_loader, criterion, optimizer, e, logger)

        opt_scheduler.step()
        train_loader.reset()

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
                int_score = dali_pcq_validate(quantized_model, clustering_model, val_loader, criterion, runtime_helper,
                                         logger)
            else:
                int_score = dali_validate(quantized_model, val_loader, criterion, logger)

            if int_score > best_int_val_score:
                best_epoch = e
                shutil.copyfile(os.path.join(save_path_fp, 'checkpoint.pth'), os.path.join(save_path_fp, 'best.pth'))

                # Save best model's INT model
                best_int_val_score = int_score
                with open(os.path.join(save_path_int, "params.json"), 'w') as f:
                    tmp = vars(args)
                    tmp['best_epoch'] = best_epoch
                    tmp['best_int_val_score'] = best_int_val_score
                    json.dump(tmp, f, indent=4)
                filepath = os.path.join(save_path_int, 'checkpoint.pth')
                torch.save({'state_dict': quantized_model.state_dict()}, filepath)
            print('Best INT-val Score: {:.2f} (Epoch: {})'.format(best_int_val_score, best_epoch))
            val_loader.reset()

    # Test quantized model which scored the best with validation dataset
    arg_dict['quantized'] = True
    quantized_model = load_dnn_model(arg_dict, tools, os.path.join(save_path_int, 'checkpoint.pth')).cuda()

    if args.dataset == 'imagenet':
        test_loader = val_loader

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

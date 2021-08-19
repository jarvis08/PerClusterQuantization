from copy import deepcopy

import torch
import torch.backends.cudnn as cudnn
from torchsummary import summary

from utils import *
from models import *
from tqdm import tqdm


def get_train_loader(args, normalizer, hvd=None):
    if args.dataset == 'imagenet':
        train_dataset = torchvision.datasets.ImageFolder(root=os.path.join(args.imagenet, 'train'),
                                                        transform=transforms.Compose([
                                                            transforms.Resize(256),
                                                            transforms.CenterCrop(224),
                                                            transforms.ToTensor(),
                                                            normalizer]))
        if args.horovod:
            sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, hvd.size(), hvd.rank())
        else:
            sampler = torch.utils.data.RandomSampler(train_dataset)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch, num_workers=10, sampler=sampler)
    else:
        train_dataset = torchvision.datasets.CIFAR10(
            root='./data',
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalizer]))
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch, shuffle=True, num_workers=2)
    return train_loader


def pcq_epoch(model, train_loader, criterion, optimizer, runtime_helper, epoch, logger):
    losses = AverageMeter()
    top1 = AverageMeter()

    model.train()
    with tqdm(train_loader, unit="batch", ncols=90) as t:
        for i, (input, target) in enumerate(t):
            t.set_description("Epoch {}".format(epoch))

            runtime_helper.get_pcq_batch(input)
            input, target = runtime_helper.sort_by_cluster_info(input, target)
            input, target = input.cuda(), target.cuda()
            output = model(input)

            loss = criterion(output, target)
            prec = accuracy(output, target)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec.item(), input.size(0))

            logger.debug("[Epoch] {}, step {}/{} [Loss] {:.5f} (avg: {:.5f}) [Score] {:.3f} (avg: {:.3f})"
                         .format(epoch, i + 1, len(t), loss.item(), losses.avg, prec.item(), top1.avg))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            t.set_postfix(loss=losses.avg, acc=top1.avg)


def get_finetuning_model(arg_dict, tools):
    pretrained_model = load_dnn_model(arg_dict, tools)
    fused_model = tools.fused_model_initializer(arg_dict)
    fused_model = tools.fuser(fused_model, pretrained_model)
    return pretrained_model, fused_model, arg_dict


def _finetune(args, tools):
    normalizer = get_normalizer(args.dataset)


    if args.horovod:
        import horovod.torch as hvd
        hvd.init()
        torch.set_num_threads(1)
        torch.cuda.set_device(hvd.local_rank())
        train_loader = get_train_loader(args, normalizer, hvd)
    else:
        train_loader = get_train_loader(args, normalizer)
    test_loader = get_test_loader(args, normalizer)

    runtime_helper = RuntimeHelper()
    arg_dict = deepcopy(vars(args))
    if runtime_helper:
        arg_dict['runtime_helper'] = runtime_helper
    pretrained_model, model, arg_dict = get_finetuning_model(arg_dict, tools)

    model.cuda()
    model.eval()
    #if args.dataset == 'imagenet':
    #    summary(model, (3, 224, 224))
    #else:
    #    summary(model, (3, 32, 32))

    if args.quant_noise:
        runtime_helper.qn_prob = args.qn_prob - 0.1
        tools.qn_forward_pre_hooker(model)

    epoch_to_start = 1
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    if args.fused:
        optimizer, epoch_to_start = load_optimizer(optimizer, args.dnn_path)
    if args.horovod:
        optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())
    opt_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    criterion = torch.nn.CrossEntropyLoss().cuda()

    cudnn.benchmark = True

    if args.cluster > 1:
        kmeans = KMeans(args)
        if not args.kmeans_path:
            args.kmeans_path = set_kmeans_dir(args)
            kmeans.train_kmeans_model(train_loader)
        else:
            kmeans.load_kmeans_model()
        runtime_helper.kmeans = kmeans

    # check_cluster_distribution(runtime_helper.kmeans, train_loader)

    if args.horovod:
        hvd.broadcast_parameters(model.state_dict(), root_rank=0)

    save_path_fp = set_save_dir(args)
    save_path_int = add_path(save_path_fp, 'quantized')
    logger = set_logger(save_path_fp)
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

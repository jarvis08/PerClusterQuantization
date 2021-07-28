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
                                                            normalizer,]))
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


# def pcq_epoch(model, train_loader, criterion, optimizer, kmeans, epoch, logger):
def pcq_epoch(model, train_loader, criterion, optimizer, kmeans, num_partitions, epoch, logger):
    losses = AverageMeter()
    top1 = AverageMeter()

    model.train()
    with tqdm(train_loader, unit="batch", ncols=90) as t:
        for i, (input, target) in enumerate(t):
            t.set_description("Epoch {}".format(epoch))

            input, target, cluster = get_pcq_batch(kmeans, input, target, num_partitions)
            model.set_cluster_information_of_batch(cluster.cuda())
            # input, target = kmeans.get_pcq_batch(input, target)

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


def get_finetuning_model(args, tools):
    pretrained_model = load_dnn_model(args, tools)
    fused_model = tools.fused_model_initializer(vars(args))
    fused_model = tools.fuser(fused_model, pretrained_model)
    return fused_model


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

    model = get_finetuning_model(args, tools)
    model.cuda()
    model.eval()
    if args.dataset == 'imagenet':
        summary(model, (3, 224, 224))
    else:
        summary(model, (3, 32, 32))

    criterion = torch.nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    if args.horovod:
        optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())
    opt_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    cudnn.benchmark = True

    kmeans_model = None
    if args.cluster > 1:
        if not args.kmeans_path:
            args.kmeans_path = set_kmeans_dir(args)
            kmeans_model = train_kmeans(args, train_loader)
        else:
            kmeans_model = load_kmeans_model(args.kmeans_path)
    # pcq_tools = None
    # if args.cluster > 1:
    #     pcq_tools = PCQtools(args)
    #     if not args.kmeans_path:
    #         args.kmeans_path = set_kmeans_dir(args)
    #         pcq_tools.train_kmeans(train_loader)

    if args.horovod:
        hvd.broadcast_parameters(model.state_dict(), root_rank=0)

    save_path_fp = set_save_dir(args)
    save_path_int = add_path(save_path_fp, 'quantized')
    logger = set_logger(save_path_fp)
    best_score_int = 0
    for e in range(1, args.epoch + 1):
        if e > args.fq:
           model.start_fake_quantization()

        if kmeans_model:
            pcq_epoch(model, train_loader, criterion, optimizer, kmeans_model, args.partition, e, logger)
        else:
            train_epoch(model, train_loader, criterion, optimizer, e, logger)
        opt_scheduler.step()

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
            quantized_model = tools.quantized_model_initializer(bit=args.bit, num_clusters=args.cluster)
            quantized_model = tools.quantizer(folded_model, quantized_model)
            quantized_model.cuda()
            del folded_model

            if kmeans_model:
                val_score = pcq_validate(quantized_model, test_loader, criterion, kmeans_model, args.partition, logger)
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

    # save_fused_network_in_darknet_form(model, args)

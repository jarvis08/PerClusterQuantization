from torch import nn
import torch.backends.cudnn as cudnn
from torchsummary import summary

from models import *
from utils import *
from tqdm import tqdm


def pcq_validate(model, test_loader, criterion, kmeans, num_partitions):
    losses = AverageMeter()
    top1 = AverageMeter()

    model.eval()
    with torch.no_grad():
        with tqdm(test_loader, unit="batch", ncols=90) as t:
            for i, (input, target) in enumerate(t):
                t.set_description("Validate")

                input, target, cluster = get_pcq_batch(kmeans, input, target, num_partitions)
                input, target, cluster = input.cuda(), target.cuda(), cluster.cuda()
                output = model(input, cluster_info=cluster)
                loss = criterion(output, target)
                prec = accuracy(output, target)[0]
                losses.update(loss.item(), input.size(0))
                top1.update(prec.item(), input.size(0))

                t.set_postfix(loss=losses.avg, acc=top1.avg)
    return top1.avg


def _evaluate(args, tools):
    model = load_dnn_model(args, tools)
    if not args.quantized:
        if args.dataset == 'imagenet':
            summary(model, (3, 224, 224))
        else:
            summary(model, (3, 32, 32))

    model.cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    cudnn.benchmark = True

    if args.darknet:
        darknet_loader = load_preprocessed_cifar10_from_darknet()
        validate_darknet_dataset(model, darknet_loader, criterion)
    elif args.cluster > 1:
        kmeans_model = load_kmeans_model(args.kmeans_path)
        normalizer = get_normalizer(args.dataset)
        test_loader = get_test_loader(args, normalizer)
        pcq_validate(model, test_loader, criterion, kmeans_model, args.partition)
    else:
        normalizer = get_normalizer(args.dataset)
        test_loader = get_test_loader(args, normalizer)
        validate(model, test_loader, criterion)

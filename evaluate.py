from torch import nn
import torch.backends.cudnn as cudnn
from torchsummary import summary

from models import *
from utils import *
from tqdm import tqdm


def _evaluate(args, tools):
    model = load_dnn_model(args, tools)
    model.cuda()
    if not args.quantized:
        if args.dataset == 'imagenet':
            summary(model, (3, 224, 224))
        else:
            summary(model, (3, 32, 32))

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

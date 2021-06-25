import torch.backends.cudnn as cudnn
from torchsummary import summary

from models import *
from quantization import *
from utils import *


def get_model(args):
    initializer = None
    if args.quantized:
        if args.arch == 'alexnet':
            if args.dataset == 'imagenet':
                initializer = quantized_alexnet
            else:
                initializer = quantized_alexnet_small
        elif args.arch == 'resnet':
            if args.dataset == 'imagenet':
                initializer = quantized_resnet18
            else:
                initializer = quantized_resnet20
    elif args.fused:
        if args.arch == 'alexnet':
            if args.dataset == 'imagenet':
                initializer = fused_alexnet
            else:
                initializer = fused_alexnet_small
        elif args.arch == 'resnet':
            if args.dataset == 'imagenet':
                initializer = fused_resnet18
            else:
                initializer = fused_resnet20
    else:
        if args.arch == 'alexnet':
            if args.dataset == 'imagenet':
                initializer = alexnet
            else:
                initializer = alexnet_small
        elif args.arch == 'resnet':
            if args.dataset == 'imagenet':
                initializer = resnet18
            else:
                initializer = resnet20

    if args.fused:
        model = initializer(bit=args.bit, smooth=args.smooth)
    elif args.quantized:
        model = initializer(bit=args.bit)
    else:
        model = initializer()
    checkpoint = torch.load(args.path)
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    return model


def _evaluate(args):
    model = get_model(args)
    if args.dataset == 'imagenet':
        summary(model, (3, 224, 224))
    else:
        if not args.quantized:
            summary(model, (3, 32, 32))
    model.cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    cudnn.benchmark = True

    if args.darknet:
        darknet_loader = load_preprocessed_cifar10_from_darknet()
        validate_darknet_dataset(darknet_loader, model, criterion)
    else:
        normalizer = get_normalizer(args.dataset)
        test_loader = get_test_loader(args.dataset, normalizer, args.batch)
        validate(test_loader, model, criterion)
import torch.backends.cudnn as cudnn
from torchsummary import summary

from models import *
from utils import *


def get_model(args, tools):
    if args.quantized:
        model = tools.quantized_model_initializer(bit=args.bit)
    elif args.fused:
        model = tools.fused_model_initializer(bit=args.bit, smooth=args.smooth)
    else:
        model = tools.pretrained_model_initializer()
    checkpoint = torch.load(args.path)
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    return model


def _evaluate(args, tools):
    model = get_model(args, tools)
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
        validate_darknet_dataset(darknet_loader, model, criterion)
    else:
        normalizer = get_normalizer(args.dataset)
        test_loader = get_test_loader(args.dataset, normalizer, args.batch)
        validate(test_loader, model, criterion)

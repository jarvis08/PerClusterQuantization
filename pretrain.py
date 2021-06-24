import torch.backends.cudnn as cudnn
from torchsummary import summary

from models import *
from quantization import *
from utils import *


def get_pretraining_model(arch, dataset):
    if arch == 'alexnet':
        if dataset == 'imagenet':
            return alexnet()
        else:
            return alexnet_small()
    elif arch == 'resnet':
        if dataset == 'imagenet':
            return resnet18()
        if dataset == 'imagenet':
            return resnet20()


def _pretrain(args):
    save_dir = set_save_dir(args)
    model = get_pretraining_model(args.arch, args.dataset)
    if args.dataset == 'imagenet':
        summary(model, (3, 224, 224))
    else:
        summary(model, (3, 32, 32))
    model.cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    opt_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    cudnn.benchmark = True

    normalizer = get_normalizer(args.dataset)
    train_loader = get_train_loader(args.dataset, normalizer, args.batch)
    test_loader = get_test_loader(args.dataset, normalizer, args.batch)

    best_prec = 0
    for e in range(1, args.epoch + 1):
        train_epoch(train_loader, model, criterion, optimizer, e)
        opt_scheduler.step()

        prec = validate(test_loader, model, criterion, False)

        is_best = prec > best_prec
        best_prec = max(prec, best_prec)
        print('best acc: {:1f}'.format(best_prec))
        save_checkpoint({
            'epoch': e,
            'state_dict': model.state_dict(),
            'best_prec': best_prec,
            'optimizer': optimizer.state_dict(),
        }, is_best, save_dir)

import torch
import torch.backends.cudnn as cudnn
from torchsummary import summary


from models import *
from utils import *


def get_train_loader(args, normalizer):
    if args.dataset == 'imagenet':
        train_dataset = torchvision.datasets.ImageFolder(root=os.path.join(args.imagenet, 'train'),
                                                        transform=transforms.Compose([
                                                            transforms.Resize(256),
                                                            transforms.CenterCrop(224),
                                                            transforms.ToTensor(),
                                                            normalizer,
                                                        ]))
    elif args.dataset == 'cifar':
        train_dataset = torchvision.datasets.CIFAR10(
            root='./data',
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalizer,
            ]))
    elif args.dataset == 'cifar100':
        train_dataset = torchvision.datasets.CIFAR100(
            root='./data',
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalizer,
            ]))
    else:
        train_dataset = torchvision.datasets.SVHN(
            root='./data',
            split='train',
            download=True,
            transform=transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalizer,
            ]))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch, shuffle=True, num_workers=args.worker)
    return train_loader


def _pretrain(args, tools):
    if args.dataset == 'cifar100':
        model = tools.pretrained_model_initializer(num_classes=100)
    else:
        model = tools.pretrained_model_initializer()

    if args.dnn_path != '':
        checkpoint = torch.load(args.dnn_path)
        model.load_state_dict(checkpoint['state_dict'], strict=False)

    model.cuda()
    # if args.dataset == 'imagenet':
    #     summary(model, (3, 224, 224))
    # else:
    #     summary(model, (3, 32, 32))

    criterion = torch.nn.CrossEntropyLoss().cuda()
    ## original
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)

    ## mobilenet
    # optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr, momentum=0.9, alpha=0.9, weight_decay=args.weight_decay)

    # cifar-100
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay, nesterov=True)
    epoch_to_start = 1
    best_prec = 0
    if args.dnn_path != '':
        optimizer, epoch_to_start = load_optimizer(optimizer, args.dnn_path)
        best_prec = checkpoint['best_prec']

    ## original scheduler
    # opt_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    # cifar-100
    opt_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=0)
    # opt_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epoch + 1, eta_min=0, last_epoch=-1, verbose=False)
    cudnn.benchmark = True

    normalizer = get_normalizer(args.dataset)
    train_loader = get_train_loader(args, normalizer)
    test_loader = get_test_loader(args, normalizer)

    save_path = set_save_dir(args)
    logger = set_logger(save_path)

    for e in range(epoch_to_start, args.epoch + 1):
        train_epoch(model, train_loader, criterion, optimizer, e, logger)
        opt_scheduler.step()
        prec = validate(model, test_loader, criterion, logger)
        is_best = prec > best_prec
        if is_best:
            with open(os.path.join(save_path, "params.json"), 'w') as f:
                tmp = vars(args)
                tmp['best_epoch'] = e
                tmp['best_score'] = prec
                json.dump(tmp, f, indent=4)
        best_prec = max(prec, best_prec)
        print('best acc: {:1f}'.format(best_prec))
        if e % 10 == 0:
            periodic = e
        else:
            periodic = None
        save_pretraining_model_checkpoint({
            'epoch': e,
            'state_dict': model.state_dict(),
            'best_prec': best_prec,
            'optimizer': optimizer.state_dict(),
        }, is_best, save_path, periodic)
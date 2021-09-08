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
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch, shuffle=True, num_workers=args.worker)
    else:
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
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch, shuffle=True, num_workers=args.worker)
    return train_loader


def _pretrain(args, tools):
    # model = tools.pretrained_model_initializer()
    model = vision_models.densenet121(pretrained=True)
    model.cuda()
    if args.dataset == 'imagenet':
        summary(model, (3, 224, 224))
    else:
        summary(model, (3, 32, 32))
    criterion = torch.nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    # optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr, momentum=0.9, alpha=0.9, weight_decay=args.weight_decay)
    opt_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    # opt_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epoch + 1, eta_min=0, last_epoch=-1, verbose=False)
    cudnn.benchmark = True

    normalizer = get_normalizer(args.dataset)
    train_loader = get_train_loader(args, normalizer)
    test_loader = get_test_loader(args, normalizer)

    save_path = set_save_dir(args)
    logger = set_logger(save_path)
    best_prec = 0
    for e in range(1, args.epoch + 1):
        train_epoch(model, train_loader, criterion, optimizer, e, logger)
        opt_scheduler.step()
        prec = validate(model, test_loader, criterion, logger)
        is_best = prec > best_prec
        best_prec = max(prec, best_prec)
        print('best acc: {:1f}'.format(best_prec))
        save_checkpoint({
            'epoch': e,
            'state_dict': model.state_dict(),
            'best_prec': best_prec,
            'optimizer': optimizer.state_dict(),
        }, is_best, save_path)

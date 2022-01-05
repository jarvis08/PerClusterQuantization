import torch
import torch.backends.cudnn as cudnn
from torchsummary import summary


from QAT.models import *
from utils import *


def get_milestones(max_epoch):
    return [int(max_epoch / 2), int(max_epoch / 4 * 3)]


def _pretrain(args, tools):
    if args.dataset == 'cifar100':
        model = tools.pretrained_model_initializer(num_classes=100)
    else:
        model = tools.pretrained_model_initializer()

    if args.dnn_path:
        checkpoint = torch.load(args.dnn_path)
        model.load_state_dict(checkpoint['state_dict'], strict=False)

    model.cuda()
    if args.dataset == 'imagenet':
        summary(model, (3, 224, 224))
    else:
        summary(model, (3, 32, 32))

    criterion = torch.nn.CrossEntropyLoss().cuda()

    ## mobilenet
    # optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr, momentum=0.9, alpha=0.9, weight_decay=args.weight_decay)
    # opt_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epoch + 1, eta_min=0, last_epoch=-1, verbose=False)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    opt_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=get_milestones(args.epoch), gamma=0.1)
    # opt_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay, gamma=0.1)
    # opt_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=0)
    epoch_to_start = 1
    best_prec = 0
    if args.dnn_path:
        optimizer, epoch_to_start = load_optimizer(optimizer, args.dnn_path)
        best_prec = checkpoint['best_prec']

    normalizer = get_normalizer(args.dataset)
    augmented_train_dataset = get_augmented_train_dataset(args, normalizer)
    non_augmented_train_dataset = get_non_augmented_train_dataset(args, normalizer)
    train_dataset, _ = split_dataset_into_train_and_val(augmented_train_dataset, args.dataset)
    non_augmented_train_dataset, val_dataset = split_dataset_into_train_and_val(non_augmented_train_dataset, args.dataset)

    train_loader = get_data_loader(train_dataset, batch_size=args.batch, shuffle=True, workers=args.worker)
    val_loader = get_data_loader(val_dataset, batch_size=args.val_batch, shuffle=False, workers=args.worker)

    save_path = set_save_dir(args)
    logger = set_logger(save_path)

    best_epoch = 0
    for e in range(epoch_to_start, args.epoch + 1):
        train_epoch(model, train_loader, criterion, optimizer, e, logger)
        opt_scheduler.step()
        prec = validate(model, val_loader, criterion, logger)
        is_best = prec > best_prec
        if is_best:
            best_epoch = e
            with open(os.path.join(save_path, "params.json"), 'w') as f:
                tmp = vars(args)
                tmp['best_epoch'] = e
                tmp['best_score'] = prec
                json.dump(tmp, f, indent=4)
        best_prec = max(prec, best_prec)
        print('Best-acc.: {:2f}, Best-epoch: {}'.format(best_prec, best_epoch))

        save_pretraining_model_checkpoint({
            'epoch': e,
            'state_dict': model.state_dict(),
            'best_prec': best_prec,
            'optimizer': optimizer.state_dict(),
            #}, is_best, save_path, periodic)
        }, is_best, save_path, None)

    with open('./pretraining_results.txt', 'a') as f:
        f.write('{:.2f} # {}, {}, LR: {}, Epoch: {}, Batch: {}, Best-epoch: {}, Path: {}\n'
                .format(best_prec, args.arch, args.dataset, args.lr, args.epoch, args.batch, best_epoch, save_path))

import torch.backends.cudnn as cudnn
from torchsummary import summary

from models import *
from quantization import *
from utils import *


def quantize_model(fp_model, args):
    int_model = None
    if args.arch == 'alexnet':
        if args.dataset == 'imagenet':
            int_model = quantized_alexnet(bit=args.bit)
            int_model = quantize_alexnet(fp_model, int_model)
        else:
            int_model = quantized_alexnet_small(bit=args.bit)
            int_model = quantize_alexnet(fp_model, int_model)
    elif args.arch == 'resnet':
        if args.dataset == 'imagenet':
            int_model = quantized_resnet18(bit=args.bit)
            int_model = quantize_resnet(fp_model, int_model, 'resnet18')
        else:
            int_model = quantized_resnet20(bit=args.bit)
            int_model = quantize_resnet(fp_model, int_model, 'resnet20')
    else:
        print("Arch. not supported")
        exit()
    return int_model


def get_finetuning_model(args):
    pre_initializer = None
    fused_initializer = None
    if args.arch == 'alexnet':
        if args.dataset == 'imagenet':
            pre_initializer = alexnet
            fused_initializer = fused_alexnet
        else:
            pre_initializer = alexnet_small
            fused_initializer = fused_alexnet_small
    elif args.arch == 'resnet':
        if args.dataset == 'imagenet':
            pre_initializer = resnet18
            fused_initializer = fused_resnet18
        else:
            pre_initializer = resnet20
            fused_initializer = fused_resnet20
    else:
        print("Arch. not supported")
        exit()

    pre_model = pre_initializer()
    checkpoint = torch.load(args.path)
    pre_model.load_state_dict(checkpoint['state_dict'], strict=False)
    fused_model = fused_initializer(bit=args.bit, smooth=args.smooth)
    if args.arch == 'alexnet':
        fused_model = set_fused_alexnet(fused_model, pre_model)
    elif args.arch == 'resnet':
        if args.dataset == 'imagenet':
            fused_model = set_fused_resnet(fused_model, pre_model, 'resnet18')
        else:
            fused_model = set_fused_resnet(fused_model, pre_model, 'resnet20')
    else:
        print("Arch. not supported")
        exit()
    return fused_model


def _finetune(args):
    save_dir = set_save_dir(args)
    model = get_finetuning_model(args)
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

        prec = validate(test_loader, model, criterion)

        is_best = prec > best_prec
        best_prec = max(prec, best_prec)
        print('best acc: {:1f}'.format(best_prec))
        save_checkpoint({
            'epoch': e,
            'state_dict': model.state_dict(),
            'best_prec': best_prec,
            'optimizer': optimizer.state_dict(),
        }, is_best, save_dir)

    # Quantize and save quantized model
    if args.arch == 'resnet':
        if args.dataset == 'imagenet':
            model = fuse_resnet(model, 'resnet18')
        else:
            model = fuse_resnet(model, 'resnet20')
    model.set_quantization_params()
    save_fused_network_in_darknet_form(model, args.arch)

    quantized_model = quantize_model(model, args)
    path = set_save_dir(args, quantize=True)
    f_path = os.path.join(path, 'checkpoint.pth')
    torch.save({'state_dict': quantized_model.state_dict()}, f_path)

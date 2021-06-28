import torch
import torch.backends.cudnn as cudnn
from torchsummary import summary

from models import *
from utils import *


def get_finetuning_model(args, tools):
    pre_model = tools.pretrained_model_initializer()
    checkpoint = torch.load(args.path)
    pre_model.load_state_dict(checkpoint['state_dict'], strict=False)

    fused_model = tools.fused_model_initializer(bit=args.bit, smooth=args.smooth)
    fused_model = tools.fuser(fused_model, pre_model)
    return fused_model


def _finetune(args, tools):
    save_path = set_save_dir(args)
    model = get_finetuning_model(args, tools)
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
        }, is_best, save_path)

    if 'ResNet' in args.arch:
        model = fold_resnet(model)
    model.set_quantization_params()
    save_fused_network_in_darknet_form(model, args)

    quantized_model = tools.quantizer(model, args)
    path = add_path(save_dir, 'quantized')
    f_path = os.path.join(path, 'checkpoint.pth')
    torch.save({'state_dict': quantized_model.state_dict()}, f_path)



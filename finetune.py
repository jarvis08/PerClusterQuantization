import torch
import torch.backends.cudnn as cudnn
from torchsummary import summary
# from torch.utils.tensorboard import SummaryWriter

from utils import *
from models import *
from tqdm import tqdm


def pcq_epoch(model, train_loader, criterion, optimizer, kmeans, num_partitions, epoch):
    losses = AverageMeter()
    top1 = AverageMeter()

    model.train()
    with tqdm(train_loader, unit="batch", ncols=90) as t:
        for i, (input, target) in enumerate(t):
            t.set_description("Epoch {}".format(epoch))

            input, target, cluster = get_pcq_batch(kmeans, input, target, num_partitions)
            input, target, cluster = input.cuda(), target.cuda(), cluster.cuda()
            output = model(input, cluster_info=cluster)
            loss = criterion(output, target)
            prec = accuracy(output, target)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec.item(), input.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            t.set_postfix(loss=losses.avg, acc=top1.avg)


def get_finetuning_model(args, tools):
    pretrained_model = load_dnn_model(args, tools)
    fused_model = tools.fused_model_initializer(bit=args.bit, smooth=args.smooth)
    fused_model = tools.fuser(fused_model, pretrained_model)
    return fused_model


def _finetune(args, tools):
    # save_path = set_save_dir(args)
    model = get_finetuning_model(args, tools)
    if args.dataset == 'imagenet':
        summary(model, (3, 224, 224))
    else:
        summary(model, (3, 32, 32))
    model.cuda()
    criterion = torch.nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    opt_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    cudnn.benchmark = True

    normalizer = get_normalizer(args.dataset)
    train_loader = get_train_loader(args, normalizer)
    test_loader = get_test_loader(args, normalizer)

    kmeans_model = None
    if args.cluster > 1:
        if not args.kmeans_path:
            args.kmeans_path = set_kmeans_dir(args)
            kmeans_model = train_kmeans(args, train_loader)
        else:
            kmeans_model = load_kmeans_model(args.kmeans_path)

    best_prec = 0
    for e in range(1, args.epoch + 1):
        if e > args.fq:
            model.start_fake_quantization()

        if kmeans_model:
            pcq_epoch(model, train_loader, criterion, optimizer, kmeans_model, args.partition, e)
        else:
            train_epoch(model, train_loader, criterion, optimizer, e)
        opt_scheduler.step()

        prec = validate(model, test_loader, criterion)

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
    if 'mobilenet' in args.arch:
        model = fold_mobilenet(model)
    model.set_quantization_params()
    # save_fused_network_in_darknet_form(model, args)

    quantized_model = tools.quantized_model_initializer(bit=args.bit, num_clusters=args.cluster)
    quantized_model = tools.quantizer(model, quantized_model)
    path = add_path(save_path, 'quantized')
    f_path = os.path.join(path, 'checkpoint.pth')
    torch.save({'state_dict': quantized_model.state_dict()}, f_path)

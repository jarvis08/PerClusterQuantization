from torch import nn
import torch.backends.cudnn as cudnn
# from torchsummary import summary

from models import *
from utils import *
from tqdm import tqdm


def pcq_rt_validate(model, clustering_model, test_loader, criterion, runtime_helper, logger=None, hvd=None):
    losses = AverageMeter()
    top1 = AverageMeter()

    model.eval()
    with torch.no_grad():
        with tqdm(test_loader, unit="batch", ncols=90) as t:
            for i, (input, target) in enumerate(t):
                t.set_description("Validate")
                input, target = input.cuda(), target.cuda()
                runtime_helper.batch_cluster = clustering_model.predict_cluster_of_batch(input)
                output = model(input)
                loss = criterion(output, target)
                prec = accuracy(output, target)[0]
                losses.update(loss.item(), input.size(0))
                top1.update(prec.item(), input.size(0))

                t.set_postfix(loss=losses.avg, acc=top1.avg)

    if logger:
        if hvd:
            if hvd.rank() == 0:
                logger.debug("[Validation] Loss: {:.5f}, Score: {:.3f}".format(losses.avg, top1.avg))
        else:
            logger.debug("[Validation] Loss: {:.5f}, Score: {:.3f}".format(losses.avg, top1.avg))
    return top1.avg


def _evaluate(args, tools):
    runtime_helper = RuntimeHelper()
    arg_dict = deepcopy(vars(args))
    if runtime_helper:
        arg_dict['runtime_helper'] = runtime_helper
    model = load_dnn_model(arg_dict, tools)
    model.cuda()
    pcq_alexnet_trained_activation_ranges(model)
    # qat_alexnet_trained_activation_ranges(model)
    # qat_resnet_trained_activation_ranges(model)
    # qat_resnet50_trained_activation_ranges(model)
    exit()
    # if not args.quantized:
    #     if args.dataset == 'imagenet':
    #         summary(model, (3, 224, 224))
    #     else:
    #         summary(model, (3, 32, 32))

    criterion = nn.CrossEntropyLoss().cuda()
    cudnn.benchmark = True

    if args.darknet:
        darknet_loader = load_preprocessed_cifar10_from_darknet()
        validate_darknet_dataset(model, darknet_loader, criterion)

    else:
        normalizer = get_normalizer(args.dataset)
        test_dataset = get_test_dataset(args, normalizer)
        test_loader = get_data_loader(test_dataset, batch_size=args.val_batch, shuffle=False, workers=args.worker)
        if args.cluster > 1:
            clustering_model = tools.clustering_method(args)
            clustering_model.load_clustering_model()
            pcq_validate(model, clustering_model, test_loader, criterion, runtime_helper)
        else:
            validate(model, test_loader, criterion)

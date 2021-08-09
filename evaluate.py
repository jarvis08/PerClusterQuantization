from torch import nn
import torch.backends.cudnn as cudnn
from torchsummary import summary

from models import *
from utils import *
from tqdm import tqdm


def _evaluate(args, tools):
    runtime_helper = RuntimeHelper()
    arg_dict = deepcopy(vars(args))
    if runtime_helper:
        arg_dict['runtime_helper'] = runtime_helper
    model = load_dnn_model(arg_dict, tools)
    model.cuda()
    if not args.quantized:
        if args.dataset == 'imagenet':
            summary(model, (3, 224, 224))
        else:
            summary(model, (3, 32, 32))

    criterion = nn.CrossEntropyLoss().cuda()
    cudnn.benchmark = True

    if args.darknet:
        darknet_loader = load_preprocessed_cifar10_from_darknet()
        validate_darknet_dataset(model, darknet_loader, criterion)

    else:
        normalizer = get_normalizer(args.dataset)
        test_loader = get_test_loader(args, normalizer)
        if args.cluster > 1:
            kmeans = KMeans(args)
            kmeans.load_kmeans_model()
            runtime_helper.kmeans = kmeans
            pcq_validate(model, test_loader, criterion, runtime_helper)
        else:
            normalizer = get_normalizer(args.dataset)
            test_loader = get_test_loader(args, normalizer)
            validate(model, test_loader, criterion)

from pytorchcv.model_provider import get_model as ptcv_get_model
from .utils import *
from .bit_config import *
from utils.misc import SKT_Helper, get_time_cost_in_string, load_dnn_model, set_save_dir, set_kt_save_dir, set_log_dir, set_kt_log_dir
from HAWQ.utils.models.q_densenet import q_densenet
from HAWQ.utils.models.q_alexnet import q_alexnet
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.utils.data.distributed
import torch.utils.data
import argparse
import json
import os
import random
import shutil
import time
import logging
import warnings

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp

import pandas as pd

mp.set_sharing_strategy('file_system')

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    help='model architecture')
parser.add_argument('--teacher-arch',
                    type=str,
                    default='resnet101',
                    help='teacher network used to do distillation')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=1, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--act-range-momentum',
                    type=float,
                    default=-1,
                    help='momentum of the activation range moving average, '
                         '-1 stands for using minimum of min and maximum of max')
parser.add_argument('--quant-mode',
                    type=str,
                    default='symmetric',
                    choices=['asymmetric', 'symmetric'],
                    help='quantization mode')
parser.add_argument('--save-path',
                    type=str,
                    default='checkpoints/imagenet/test/',
                    help='path to save the quantized model')
parser.add_argument('--data-percentage',
                    type=float,
                    default=1,
                    help='data percentage of training data')
parser.add_argument('--fix-BN', action='store_true',
                    help='whether to fix BN statistics and fold BN during training')
parser.add_argument('--fix-BN-threshold',
                    type=int,
                    default=None,
                    help='when to start training with fixed and folded BN,'
                         'after the threshold iteration, the original fix-BN will be overwritten to be True')
parser.add_argument('--checkpoint-iter',
                    type=int,
                    default=-1,
                    help='the iteration that we save all the featuremap for analysis')
parser.add_argument('--evaluate-times',
                    type=int,
                    default=-1,
                    help='The number of evaluations during one epoch')
parser.add_argument('--quant-scheme',
                    type=str,
                    default='uniform4',
                    help='quantization bit configuration')
parser.add_argument('--resume-quantize',
                    action='store_true',
                    help='if True map the checkpoint to a quantized model,'
                         'otherwise map the checkpoint to an ordinary model and then quantize')
parser.add_argument('--act-percentile',
                    type=float,
                    default=0,
                    help='the percentage used for activation percentile'
                         '(0 means no percentile, 99.9 means cut off 0.1%)')
parser.add_argument('--weight-percentile',
                    type=float,
                    default=0,
                    help='the percentage used for weight percentile'
                         '(0 means no percentile, 99.9 means cut off 0.1%)')

parser.add_argument('--channel-wise',
                    action='store_false',
                    help='whether to use channel-wise quantizaiton or not')

# parser.add_argument('--channel-wise',
#                     type=bool,
#                     default=False,
#                     help='whether to use channel-wise quantizaiton or not')

parser.add_argument('--bias-bit',
                    type=int,
                    default=32,
                    help='quantizaiton bit-width for bias')
parser.add_argument('--distill-method',
                    type=str,
                    default='None',
                    help='you can choose None or KD_naive')
parser.add_argument('--distill-alpha',
                    type=float,
                    default=0.95,
                    help='how large is the ratio of normal loss and teacher loss')
parser.add_argument('--temperature',
                    type=float,
                    default=6,
                    help='how large is the temperature factor for distillation')
parser.add_argument('--fixed-point-quantization',
                    action='store_true',
                    help='whether to skip deployment-oriented operations and '
                         'use fixed-point rather than integer-only quantization')

parser.add_argument('--transfer_param', action='store_true',
                    help='copy params of torchcv pretrained models')
parser.add_argument('--dnn_path', default='', type=str,
                    help="Pretrained model's path")

# skt arguments
parser.add_argument('--mixed_precision',
                    action='store_true',
                    help='For SKT')
parser.add_argument('--replace_grad',
                    type=float,
                    default=1e-5,
                    help='grad value to replace input gradient for manipulation')
parser.add_argument('--range_ratio',
                    type=float,
                    default=0.5,
                    help="ratio of setting a channel with a range less than the max range as a low bit channel")
parser.add_argument('--quantile',
                    type=float,
                    default=0.5,
                    help="criteria for determining whether the abs of input gradient is small or large ")
parser.add_argument('--schedule_unit',
                    type=str,
                    default='epoch',
                    help="whether to handle gradients per iteration or epoch ")
parser.add_argument('--schedule_count',
                    default=1,
                    type=int,
                    help="schedule counts")


best_acc1 = 0
quantize_arch_dict = {'resnet18': q_resnet18,
                      'resnet20': q_resnet20,
                      'resnet50': q_resnet50,
                      'alexnet': q_alexnet,
                      'densenet121': q_densenet,
                      'inceptionv3': q_inceptionv3,
                      'mobilenetv2_w1': q_mobilenetv2_w1}

args_hawq, _ = parser.parse_known_args()
args_hawq.save_path = os.path.join("/home/work/JK-Data/checkpoint/{}/{}/{}/".format(
    args_hawq.arch, args_hawq.data, os.getpid()))
if not os.path.exists(args_hawq.save_path):
    os.makedirs(args_hawq.save_path)

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S', filename=args_hawq.save_path + 'log{}.log'.format(os.getpid()))
logging.getLogger().setLevel(logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler())

logging.info(args_hawq)


def main(args_daq, data_loaders, clustering_model=None):
    args = argparse.Namespace(**vars(args_hawq), **vars(args_daq))
    print(vars(args))
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()

    # Simply call main_worker function
    main_worker(args.gpu, ngpus_per_node, args, data_loaders, clustering_model)


def main_worker(gpu, ngpus_per_node, args, data_loaders, clustering_model):
    def set_args_arch(args):
        arch = args.arch.lower()
        if arch == 'resnet20':
            return arch + "_" + args.data.lower()
        elif arch == 'densenet':
            return "densenet121"
        else:
            return arch

    def reset_args_arch(args):
        arch = args.arch.lower()
        if 'resnet20' in arch:
            return "resnet20"
        else:
            return arch

    def create_model(args):
        pretrained = args.pretrained and not args.resume
        logging.info(
            "=> using pre-trained PyTorchCV model '{}'".format(args.arch))
        model = ptcv_get_model(args.arch, pretrained=True)
        if args.distill_method != 'None':
            logging.info(
                "=> using pre-trained PyTorchCV teacher '{}'".format(args.teacher_arch))
            teacher = ptcv_get_model(args.teacher_arch, pretrained=pretrained)
            return model, teacher
        return model, None

    def transfer_param(args, model):
        if args.arch.lower() == 'resnet50':
            import torchvision.models as vision_models
            vision = vision_models.resnet50(pretrained=True)
            vision_dict = vision.state_dict()
            model_dict = model.state_dict()
            for cv, our in zip(model_dict.items(), vision_dict.items()):
                model_dict[cv[0]].copy_(vision_dict[our[0]])
        elif args.arch.lower() == 'densenet121':
            import torchvision.models as vision_models
            vision = vision_models.densenet121(pretrained=True)
            vision_dict = vision.state_dict()
            model_dict = model.state_dict()
            for cv, our in zip(model_dict.items(), vision_dict.items()):
                model_dict[cv[0]].copy_(vision_dict[our[0]])
        elif args.arch.lower() == 'alexnet':
            checkpoint = torch.load(args.dnn_path)
            loaded_dict = checkpoint['state_dict']
            model_dict = model.state_dict()
            for cur, from_ in zip(model_dict.items(), loaded_dict.items()):
                model_dict[cur[0]] = loaded_dict[from_[0]]
        else:
            checkpoint = torch.load(args.dnn_path)
            loaded_dict = checkpoint['state_dict']
            model_dict = model.state_dict()
            for cur, from_ in zip(model_dict.items(), loaded_dict.items()):
                model_dict[cur[0]].copy_(loaded_dict[from_[0]])
        return model_dict

    def eval_resume(args, model):
        if args.resume and not args.resume_quantize:
            if os.path.isfile(args.resume):
                logging.info("=> loading checkpoint '{}'".format(args.resume))

                checkpoint = torch.load(args.resume)['state_dict']
                model_key_list = list(model.state_dict().keys())
                for key in model_key_list:
                    if 'num_batches_tracked' in key:
                        model_key_list.remove(key)
                i = 0
                modified_dict = {}
                for key, value in checkpoint.items():
                    if 'scaling_factor' in key:
                        continue
                    if 'num_batches_tracked' in key:
                        continue
                    if 'weight_integer' in key:
                        continue
                    if 'min' in key or 'max' in key:
                        continue
                    modified_key = model_key_list[i]
                    modified_dict[modified_key] = value
                    i += 1
                logging.info(model.load_state_dict(
                    modified_dict, strict=False))
            else:
                logging.info(
                    "=> no checkpoint found at '{}'".format(args.resume))
        return model

    def quant_resume(args, model):
        if args.resume and args.resume_quantize:
            if os.path.isfile(args.resume):
                logging.info(
                    "=> loading quantized checkpoint '{}'".format(args.resume))
                checkpoint = torch.load(args.resume)['state_dict']
                modified_dict = {}
                for key, value in checkpoint.items():
                    if 'num_batches_tracked' in key:
                        continue
                    if 'weight_integer' in key:
                        continue
                    if 'bias_integer' in key:
                        continue

                    modified_key = key.replace("module.", "")
                    modified_dict[modified_key] = value
                model.load_state_dict(modified_dict, strict=False)
            else:
                logging.info(
                    "=> no quantized checkpoint found at '{}'".format(args.resume))
        return model

    def resume_optimizer(args, optimizer):
        # optionally resume optimizer and meta information from a checkpoint
        if args.resume:
            if os.path.isfile(args.resume):
                if args.gpu is None:
                    checkpoint = torch.load(args.resume)
                else:
                    # Map model to be loaded to specified single gpu.
                    loc = 'cuda:{}'.format(args.gpu)
                    checkpoint = torch.load(args.resume, map_location=loc)
                args.start_epoch = checkpoint['epoch']
                best_acc1 = checkpoint['best_acc1']
                if args.gpu is not None:
                    # best_acc1 may be from a checkpoint from a different GPU
                    best_acc1 = best_acc1.to(args.gpu)
                optimizer.load_state_dict(checkpoint['optimizer'])
                logging.info("=> loaded optimizer and meta information from checkpoint '{}' (epoch {})".
                             format(args.resume, checkpoint['epoch']))
            else:
                logging.info(
                    "=> no checkpoint found at '{}'".format(args.resume))
        return optimizer

    def get_quantize_model(args, model, model_dict, quantize_arch, num_clusters, skt_helper=None):
        if args.arch.lower() == 'alexnet':
            return quantize_arch(model, model_dict, num_clusters, skt_helper)
        return quantize_arch(model, num_clusters, skt_helper)

    def set_quantize_param(args, model, bit_config):
        name_counter = 0

        fix_BN = True if args.arch == 'densenet121' else args.fix_BN
        for name, m in model.named_modules():
            if name in bit_config.keys():
                name_counter += 1
                setattr(m, 'quant_mode', 'symmetric')
                setattr(m, 'bias_bit', args.bias_bit)
                setattr(m, 'quantize_bias', (args.bias_bit != 0))
                setattr(m, 'per_channel', args.channel_wise)
                setattr(m, 'act_percentile', args.act_percentile)
                setattr(m, 'act_range_momentum', args.act_range_momentum)
                setattr(m, 'weight_percentile', args.weight_percentile)
                setattr(m, 'fix_flag', False)
                setattr(m, 'fix_BN', fix_BN)
                setattr(m, 'fix_BN_threshold', args.fix_BN_threshold)
                setattr(m, 'training_BN_mode', fix_BN)
                setattr(m, 'checkpoint_iter_threshold', args.checkpoint_iter)
                setattr(m, 'save_path', args.save_path)
                setattr(m, 'fixed_point_quantization',
                        args.fixed_point_quantization)

                if type(bit_config[name]) is tuple:
                    bitwidth, symmetric = bit_config[name]
                else:
                    bitwidth, symmetric = bit_config[name], False

                if hasattr(m, 'activation_bit'):
                    setattr(m, 'activation_bit', bitwidth)
                    if bitwidth == 4 and not symmetric:
                        setattr(m, 'quant_mode', 'asymmetric')
                else:
                    setattr(m, 'weight_bit', bitwidth)

        logging.info("match all modules defined in bit_config: {}".format(
            len(bit_config.keys()) == name_counter))
        return model

    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        logging.info("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    skt_helper = None
    if args.mixed_precision:
        assert args.schedule_unit in ['iter', 'epoch'], f'Not supported schedule unit : {args.schedule_unit}'
        skt_helper = SKT_Helper()
        skt_helper.set_skt_arguments(args)

    prev_arch = args.arch
    args.arch = set_args_arch(args)
    fp_model, teacher = create_model(args)  # Create Model
    args.arch = reset_args_arch(args)
    model_dict = transfer_param(args, fp_model) if args.transfer_param else None
    fp_model = eval_resume(args, fp_model)

    quantize_arch = quantize_arch_dict[args.arch]
    model = get_quantize_model(args, fp_model, model_dict, quantize_arch, args.cluster, skt_helper)
    bit_config = bit_config_dict["bit_config_" + args.arch + "_" + args.quant_scheme]
    model = set_quantize_param(args, model, bit_config)
    # logging.info(model)
    model = quant_resume(args, model)

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        if args.distill_method != 'None':
            teacher = teacher.cuda(args.gpu)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    optimizer = resume_optimizer(args, optimizer)

    cudnn.benchmark = True

    train_loader = data_loaders['aug_train']
    val_loader = data_loaders['val']
    test_loader = data_loaders['test']
    cluster_train_loader = data_loaders['non_aug_train']

    if clustering_model is not None and clustering_model.model is None:
    #     clustering_model.feature_index = clustering_model.get_high_corr_features(model, cluster_train_loader)
        clustering_model.train_clustering_model(cluster_train_loader)

    if args.nnac and clustering_model.final_cluster is None:
        ###
        # model.toggle_full_precision()
        # freeze_model(model)
        # clustering_model.zero_max_nn_aware_clustering(
        # clustering_model.max_nn_aware_clustering(
        #     model, cluster_train_loader, args.arch)
        ###
        sub_model = get_quantize_model(args, fp_model, model_dict, quantize_arch, args.sub_cluster)
        sub_model = set_quantize_param(args, sub_model, bit_config)
        sub_model = sub_model.cuda(args.gpu)
        
        print("EMA training epochs...")
        ema_epoch = 2 if args.data == 'imagenet' else 10
        for epoch in range(args.start_epoch, ema_epoch):
            train_ema(train_loader, sub_model, clustering_model, criterion, epoch, args)
            
        sub_model.toggle_full_precision()
        freeze_model(sub_model)
        clustering_model.ema_nn_aware_clustering(sub_model, cluster_train_loader, args.arch)
        del sub_model
    del fp_model
    
    if args.evaluate:
        validate(test_loader, model, criterion, args)
        return

    best_epoch = 0
    best_acc1 = 0
    tuning_start_time = time.time()
    tuning_fin_time = None
    one_epoch_time = None

    ### LOG DIRECTORY ###
    # finetune_path = set_save_dir(args)
    # log_path = set_log_dir(args)

    finetune_path = set_kt_save_dir(args)
    log_path = set_kt_log_dir(args)

    if not os.path.exists(finetune_path):
        os.mkdir(finetune_path)
    if not os.path.exists(log_path):
        os.mkdir(log_path)
        
    
    def confusion_matrix(data_loader, model, clustering_model, args):
        def prediction(output, target):
            with torch.no_grad():
                if type(output) is tuple:
                    output = output[0]
                _, pred = output.topk(1, 1, True, True)
                pred = pred.t()
                correct = pred.eq(target.view(1, -1).expand_as(pred))
                return torch.squeeze(correct)
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        saturated = 0
        
        cluster_confusion_matrix = torch.zeros([7, args.cluster], dtype=torch.long).cuda()

        freeze_model(model)
        model.eval()

        with torch.no_grad():
            with tqdm(data_loader, desc="experiment ", ncols=95) as t:
                for i, (images, target) in enumerate(t):
                    images = images.cuda(args.gpu, non_blocking=True)
                    target = target.cuda(args.gpu, non_blocking=True)
                    cluster = clustering_model.predict_cluster_of_batch(images).cuda(args.gpu, non_blocking=True)
                    
                    unique, count = torch.unique(cluster, return_counts=True)
                    cluster_confusion_matrix[0].put_(unique, count, accumulate=True)                # Total Cluster Counts
                    
                    batch_size = target.size(0)
                    
                    output = model(images, cluster)
                    INT_pred = prediction(output, target)
                    
                    model.toggle_full_precision()
                    output = model(images, cluster)
                    model.toggle_full_precision()
                    
                    FP_pred = prediction(output, target)

                    #############
                    cluster_confusion_matrix[1].scatter_reduce_(0, cluster, FP_pred.type(torch.long), reduce="sum")  # Full Precision Correct
                    cluster_confusion_matrix[2].scatter_reduce_(0, cluster, INT_pred.type(torch.long), reduce="sum") # Quantized Accuracy
                    
                    TP = torch.logical_and(FP_pred, INT_pred).type(torch.long)
                    cluster_confusion_matrix[3].scatter_reduce_(0, cluster, TP, reduce="sum")  # Both Positive
                    TN = torch.logical_and(~FP_pred, ~INT_pred).type(torch.long)
                    cluster_confusion_matrix[4].scatter_reduce_(0, cluster, TN, reduce="sum")  # Both Negative
                    FP = torch.logical_and(~FP_pred, INT_pred).type(torch.long)
                    cluster_confusion_matrix[5].scatter_reduce_(0, cluster, FP, reduce="sum")  # FP False, QT True
                    FN = torch.logical_and(FP_pred, ~INT_pred).type(torch.long)
                    cluster_confusion_matrix[6].scatter_reduce_(0, cluster, FN, reduce="sum")  # FP True, QT False

                    #############

        print("=======================")
        columns = ['Total','Full Precision', 'Quantized', 'TP', 'TN', 'FP', 'FN']
        cluster_confusion_matrix_np = cluster_confusion_matrix.T.cpu().numpy()
        cluster_confusion_matrix_df = pd.DataFrame(cluster_confusion_matrix_np, columns=columns)
        
        cluster = args.sub_cluster if args.nnac else args.cluster
        cluster_confusion_matrix_df.to_csv(f"/workspace/PerClusterQuantization/{args.arch}/{args.data}/{cluster}/{args.cluster}.csv", index=False)
        return


    def cluster_score(train_loader, cluster_train_loader, test_loader, model, clustering_model, args):
        freeze_model(model)
        aug_score, nonaug_score, test_score = clustering_model.measure_cluster_score(model, train_loader, cluster_train_loader, test_loader, args.arch)
        # score = clustering_model.measure_cluster_distance(model, train_loader, args.arch)

        cluster = args.sub_cluster if args.nnac else args.cluster
        try:
            df_data = pd.read_csv(f"aug_{args.arch}_{args.data}_{cluster}.csv")
            
            score_np = aug_score.cpu().numpy()
            score_df = pd.DataFrame(score_np, columns=[str(args.cluster)])
            
            score_df = pd.concat([df_data, score_df], axis=1)
            score_df.to_csv(f"aug_{args.arch}_{args.data}_{cluster}.csv", index=False)
        except :
            score_np = aug_score.cpu().numpy()
            score_df = pd.DataFrame(score_np, columns=[str(args.cluster)])
            score_df.to_csv(f"aug_{args.arch}_{args.data}_{cluster}.csv", index=False)
            
        try:
            df_data = pd.read_csv(f"nonaug_{args.arch}_{args.data}_{cluster}.csv")
            
            score_np = nonaug_score.cpu().numpy()
            score_df = pd.DataFrame(score_np, columns=[str(args.cluster)])
            
            score_df = pd.concat([df_data, score_df], axis=1)
            score_df.to_csv(f"nonaug_{args.arch}_{args.data}_{cluster}.csv", index=False)
        except :
            score_np = nonaug_score.cpu().numpy()
            score_df = pd.DataFrame(score_np, columns=[str(args.cluster)])
            score_df.to_csv(f"nonaug_{args.arch}_{args.data}_{cluster}.csv", index=False)
            
        try:
            df_data = pd.read_csv(f"test_{args.arch}_{args.data}_{cluster}.csv")
            
            score_np = test_score.cpu().numpy()
            score_df = pd.DataFrame(score_np, columns=[str(args.cluster)])
            
            score_df = pd.concat([df_data, score_df], axis=1)
            score_df.to_csv(f"test_{args.arch}_{args.data}_{cluster}.csv", index=False)
        except :
            score_np = test_score.cpu().numpy()
            score_df = pd.DataFrame(score_np, columns=[str(args.cluster)])
            score_df.to_csv(f"test_{args.arch}_{args.data}_{cluster}.csv", index=False)


    ###

    # for epoch in range(args.start_epoch, 10):
    #     train_ema(train_loader, model, clustering_model, criterion, epoch, args)
    #     acc1 = validate(test_loader, model, clustering_model, criterion, args)

    # confusion_matrix(test_loader, model, clustering_model, args)
    # cluster_score(train_loader, cluster_train_loader, test_loader, model, clustering_model, args)

    # # Train EMA for couple epochs before training parameters
    # ema_epoch = 2 if args.data == 'imagenet' else 10
    # for epoch in range(args.start_epoch, ema_epoch):
    #     print("EMA training epochs...")
    #     train_ema(train_loader, model, clustering_model, criterion, epoch, args)
    #     acc1 = validate(test_loader, model, clustering_model, criterion, args)


    def initial_incremental_channel_selection(model):
        neuron_four_counter, neuron_eight_counter, ch_four_counter, ch_eight_counter = 0, 0, 0, 0
        ch_counter, element_counter = 0, 0
        range_ratio = model.skt_helper.range_ratio
        iterator = iter(model.modules())
        for cur in iterator:
            if isinstance(cur, (QuantConv2d, QuantBnConv2d)):
                in_channel = cur.in_channels
                # candidate channel selection
                if isinstance(cur, QuantConv2d):
                    weight_group = cur.weight.transpose(1, 0).reshape(in_channel, -1)
                else:
                    weight_group = cur.conv.weight.transpose(1, 0).reshape(in_channel, -1)
                weight_range = torch.max(weight_group.max(dim=1).values.abs(), weight_group.min(dim=1).values.abs())

                if cur.quant_mode == 'asymmetric':
                    input_range = cur.input_range[1] - cur.input_range[0]
                else:
                    input_range = torch.max(cur.input_range[0].abs(), cur.input_range[1].abs())

                # int4 channel selection
                weight_bits = torch.where(weight_range <= weight_range.max() * range_ratio, 1, 0)
                input_bits = torch.where(input_range <= input_range.max() * range_ratio, 1, 0)
                mask = torch.logical_and(input_bits, weight_bits)

                cur.prev_mask[mask] = True
                cur.low_group = mask.nonzero(as_tuple=True)[0]
                cur.high_group = (~mask).nonzero(as_tuple=True)[0]

                ch_four_counter += len(cur.low_group)
                ch_eight_counter += len(cur.high_group)
                neuron_four_counter += len(cur.low_group) * cur.element_size
                neuron_eight_counter += len(cur.high_group) * cur.element_size
                ch_counter += cur.in_channels
                element_counter += cur.in_channels * cur.element_size

                # initialize params for training
                cur.init_records()

        assert ch_four_counter + ch_eight_counter == ch_counter, 'total num of in-channels mismatch'
        assert neuron_four_counter + neuron_eight_counter == element_counter, 'total num of element size mismatch'

        model.total_ch = ch_counter
        model.total_element_size = element_counter

        ch_ratio = ch_four_counter / model.total_ch * 100
        neuron_ratio = neuron_four_counter / model.total_element_size * 100
        print("Initial Int-4 Neuron ratio : {:.2f}%".format(neuron_ratio))
        return ch_ratio, neuron_ratio


    if args.mixed_precision:
        validate(val_loader, model, clustering_model, criterion, args)
        initial_incremental_channel_selection(model)
        global iter_cnt, res_iters
        res_iters = 0
        if args.schedule_unit == 'epoch':
            iter_cnt = len(train_loader)
        else:
            iter_cnt = args.schedule_count

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args)

        if args.mixed_precision:
            ch_ratio, neuron_ratio = skt_train(train_loader, model, clustering_model, criterion, optimizer, epoch, args)
        else:
            train(train_loader, model, clustering_model, criterion, optimizer, epoch, args)
        tuning_fin_time = time.time()
        one_epoch_time = get_time_cost_in_string(
            tuning_fin_time - tuning_start_time)
        acc1 = validate(test_loader, model, clustering_model, criterion, args)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        if (acc1 > best_acc1):
            best_acc1 = max(acc1, best_acc1)
            best_epoch = epoch

        logging.info(f'Best acc at epoch {best_epoch}: {best_acc1}')

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'cluster': args.cluster,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer': optimizer.state_dict(),
            }, is_best, finetune_path)

    test_score = best_acc1

    time_cost = get_time_cost_in_string(tuning_fin_time - tuning_start_time)

    if args.mixed_precision:
        with open(f'{log_path}/LR_{args.lr}_range_{args.range_ratio}_({args.schedule_unit}-{args.schedule_count}).txt', 'a') as f:
            f.write(
                'Channel:{:.2f}, Neuron:{:.2f} Acc:{:.2f}, REPL:{} QUANTILE:{} LR:{}, Batch:{}, Weight decay: {}, Cluster:{} Best Epoch:{}, Time:{}, Data:{}, 1 epoch time: {}\n'.format(
                    ch_ratio, neuron_ratio, test_score, args.replace_grad, args.quantile, args.lr, args.batch_size, args.weight_decay, args.cluster,
                    best_epoch, time_cost, args.data, one_epoch_time))
    else:
        if not args.nnac:
            with open(f'{log_path}/cluster_{args.cluster}.txt', 'a') as f:
                f.write('Bit:{}, Acc:{:.2f}, LR:{}, Batch:{}, Weight decay: {}, Cluster:{} Best Epoch:{}, Time:{}, Data:{}, 1 epoch time: {}\n'.format(
                    args.quant_scheme, test_score, args.lr, args.batch_size, args.weight_decay, args.cluster, best_epoch, time_cost, args.data, one_epoch_time))
        else:
            with open(f'{log_path}/cluster_{args.sub_cluster}->{args.cluster}.txt', 'a') as f:
                f.write('Bit:{}, Acc:{:.2f}, LR:{}, Batch:{}, Weight decay: {}, Cluster:{} Best Epoch:{}, Time:{}, Data:{}, 1 epoch time: {}\n'.format(
                    args.quant_scheme, test_score, args.lr, args.batch_size, args.weight_decay, args.cluster, best_epoch, time_cost, args.data, one_epoch_time))


def train_ema(train_loader, model, clustering_model, criterion, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    # switch to train mode
    if args.fix_BN == True:
        model.eval()
    else:
        model.train()

    end = time.time()
    with torch.no_grad():
        with tqdm(train_loader, desc="Epoch {} ".format(epoch), ncols=95) as t:
            # for i, (images, target) in enumerate(t):
            for i, data in enumerate(t):
                if args.dataset == 'imagenet':
                    images, target = data[0]['data'], torch.flatten(data[0]['label']).type(torch.long)
                else:
                    images, target = data

                # measure data loading time
                data_time.update(time.time() - end)

                if args.gpu is not None:
                    images = images.cuda(args.gpu)
                    target = target.cuda(args.gpu)

                if clustering_model is None:
                    cluster = torch.zeros(images.size(0), dtype=torch.long).cuda(args.gpu)
                else:
                    cluster = clustering_model.predict_cluster_of_batch(images).cuda(args.gpu)

                # compute output
                output = model(images, cluster)
                loss = criterion(output, target)

                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                losses.update(loss.item(), images.size(0))
                top1.update(acc1[0].item(), images.size(0))
                top5.update(acc5[0].item(), images.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                t.set_postfix(acc1=top1.avg, acc5=top5.avg, loss=losses.avg)


def incremental_channel_selection(model, epoch):
    neuron_four_counter, neuron_eight_counter, ch_four_counter, ch_eight_counter = 0, 0, 0, 0
    range_ratio = model.skt_helper.range_ratio
    iterator = iter(model.modules())

    for cur in iterator:
        if isinstance(cur, (QuantConv2d, QuantBnConv2d)):
            in_channel = cur.in_channels

            # candidate channel selction
            if isinstance(cur, QuantConv2d):
                weight_group = cur.weight.transpose(1, 0).reshape(in_channel, -1)
            else:
                weight_group = cur.conv.weight.transpose(1, 0).reshape(in_channel, -1)
            weight_range = torch.max(weight_group.max(dim=1).values.abs(), weight_group.min(dim=1).values.abs())

            if cur.quant_mode == 'asymmetric':
                input_range = cur.input_range[1] - cur.input_range[0]
            else:
                input_range = torch.max(cur.input_range[0].abs(), cur.input_range[1].abs())

            # int4 channel selection
            weight_bits = torch.where(weight_range <= weight_range.max() * range_ratio, 1, 0)
            input_bits = torch.where(input_range <= input_range.max() * range_ratio, 1, 0)
            mask = torch.logical_and(input_bits, weight_bits)

            cur.prev_mask = torch.logical_or(cur.prev_mask, mask)
            cur.low_group = cur.prev_mask.nonzero(as_tuple=True)[0]
            cur.high_group = (~cur.prev_mask).nonzero(as_tuple=True)[0]

            ch_four_counter += len(cur.low_group)
            ch_eight_counter += len(cur.high_group)
            neuron_four_counter += len(cur.low_group) * cur.element_size
            neuron_eight_counter += len(cur.high_group) * cur.element_size

    ch_ratio = ch_four_counter / model.total_ch * 100
    neuron_ratio = neuron_four_counter / model.total_element_size * 100
    print("Epoch {} Int-4 Neuron ratio : {:.2f}%".format(epoch, neuron_ratio))
    return ch_ratio, neuron_ratio


def train(train_loader, model, clustering_model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    # switch to train mode
    if args.fix_BN == True:
        model.eval()
    else:
        model.train()

    end = time.time()
    with tqdm(train_loader, desc="Epoch {} ".format(epoch), ncols=95) as t:
        for i, data in enumerate(t):
            if args.dataset == 'imagenet':
                images, target = data[0]['data'], torch.flatten(data[0]['label']).type(torch.long)
            else:
                images, target = data

            # measure data loading time
            data_time.update(time.time() - end)

            if args.gpu is not None:
                images = images.cuda(args.gpu)
                target = target.cuda(args.gpu)

            if clustering_model is None:
                cluster = torch.zeros(images.size(0), dtype=torch.long).cuda(args.gpu)
            else:
                cluster = clustering_model.predict_cluster_of_batch(images).cuda(args.gpu)

            # compute output
            output = model(images, cluster)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0].item(), images.size(0))
            top5.update(acc5[0].item(), images.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            t.set_postfix(acc1=top1.avg, acc5=top5.avg, loss=losses.avg)


def skt_train(train_loader, model, clustering_model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    global iter_cnt, res_iters

    # switch to train mode
    if args.fix_BN == True:
        model.eval()
    else:
        model.train()
    
    end = time.time()
    with tqdm(train_loader, desc="Epoch {} ".format(epoch), ncols=95) as t:
        for i, data in enumerate(t):
            if args.dataset == 'imagenet':
                images, target = data[0]['data'], torch.flatten(data[0]['label']).type(torch.long)
            else:
                images, target = data

            # measure data loading time
            data_time.update(time.time() - end)

            if args.gpu is not None:
                images = images.cuda(args.gpu)
                target = target.cuda(args.gpu)

            if clustering_model is None:
                cluster = torch.zeros(images.size(0), dtype=torch.long).cuda(args.gpu)
            else:
                cluster = clustering_model.predict_cluster_of_batch(images).cuda(args.gpu)

            # compute output
            output = model(images, cluster)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0].item(), images.size(0))
            top5.update(acc5[0].item(), images.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # candidate channel selection
            if (i + 1 + res_iters) % iter_cnt == 0:
                ch_ratio, neuron_ratio = incremental_channel_selection(model, epoch)
                model.reset_input_range()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            t.set_postfix(acc1=top1.avg, acc5=top5.avg, loss=losses.avg)
    res_iters = (len(train_loader) + res_iters) % iter_cnt
    return ch_ratio, neuron_ratio


def train_kd(train_loader, model, teacher, criterion, optimizer, epoch, val_loader, args, ngpus_per_node,
             dataset_length):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    if args.fix_BN == True:
        model.eval()
    else:
        model.train()
    teacher.eval()

    end = time.time()

    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(images)
        if args.distill_method != 'None':
            with torch.no_grad():
                teacher_output = teacher(images)

        if args.distill_method == 'None':
            loss = criterion(output, target)
        elif args.distill_method == 'KD_naive':
            loss = loss_kd(output, target, teacher_output, args)
        else:
            raise NotImplementedError

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
        if i % args.print_freq == 0 and args.rank == 0:
            print('Epoch {epoch_} [{iters}]  Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(epoch_=epoch, iters=i,
                                                                                               top1=top1, top5=top5))

        if i % ((dataset_length // (
                args.batch_size * args.evaluate_times)) + 2) == 0 and i > 0 and args.evaluate_times > 0:
            acc1 = validate(val_loader, model, criterion, args)

            # switch to train mode
            if args.fix_BN == True:
                model.eval()
            else:
                model.train()

            # remember best acc@1 and save checkpoint
            global best_acc1
            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)

            if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                        and args.rank % ngpus_per_node == 0):
                if not os.path.exists(args.save_path):
                    os.makedirs(args.save_path)

                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'best_acc1': best_acc1,
                    'optimizer': optimizer.state_dict(),
                }, is_best, args.save_path)


def validate(val_loader, model, clustering_model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    # switch to evaluate mode
    freeze_model(model)
    model.eval()

    with torch.no_grad():
        end = time.time()
        with tqdm(val_loader, desc="Validate", ncols=95) as t:
            for i, data in enumerate(t):
                if args.dataset == 'imagenet':
                    images, target = data[0]['data'], torch.flatten(data[0]['label']).type(torch.long)
                else:
                    images, target = data
                    
                if args.gpu is not None:
                    images = images.cuda(args.gpu, non_blocking=True)
                    target = target.cuda(args.gpu, non_blocking=True)
                    
                if clustering_model is None:
                    cluster = torch.zeros(images.size(0), dtype=torch.long).cuda(args.gpu, non_blocking=True)
                else:
                    cluster = clustering_model.predict_cluster_of_batch(images).cuda(args.gpu, non_blocking=True)

                # compute output
                output = model(images, cluster)
                loss = criterion(output, target)

                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                losses.update(loss.item(), images.size(0))
                top1.update(acc1[0].item(), images.size(0))
                top5.update(acc5[0].item(), images.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                t.set_postfix(acc1=top1.avg, acc5=top5.avg, loss=losses.avg)

    torch.save({'convbn_scaling_factor': {k: v for k, v in model.state_dict().items() if 'convbn_scaling_factor' in k},
                'fc_scaling_factor': {k: v for k, v in model.state_dict().items() if 'fc_scaling_factor' in k},
                'weight_integer': {k: v for k, v in model.state_dict().items() if 'weight_integer' in k},
                'bias_integer': {k: v for k, v in model.state_dict().items() if 'bias_integer' in k},
                'act_scaling_factor': {k: v for k, v in model.state_dict().items() if 'act_scaling_factor' in k},
                }, args.save_path + 'quantized_checkpoint.pth.tar')

    unfreeze_model(model)

    return top1.avg


def save_checkpoint(state, is_best, filename=None):
    torch.save(state, filename + 'checkpoint.pth.tar')
    if is_best:
        shutil.copyfile(filename + 'checkpoint.pth.tar',
                        filename + 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        logging.info('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    print('lr = ', lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def loss_kd(output, target, teacher_output, args):
    """
    Compute the knowledge-distillation (KD) loss given outputs and labels.
    "Hyperparameters": temperature and alpha
    The KL Divergence for PyTorch comparing the softmaxs of teacher and student.
    The KL Divergence expects the input tensor to be log probabilities.
    """
    alpha = args.distill_alpha
    T = args.temperature
    KD_loss = F.kl_div(F.log_softmax(output / T, dim=1), F.softmax(teacher_output / T, dim=1)) * (alpha * T * T) + \
        F.cross_entropy(output, target) * (1. - alpha)

    return KD_loss


if __name__ == '__main__':
    main()

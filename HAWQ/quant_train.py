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
                    type=bool,
                    default=False,
                    help='whether to use channel-wise quantizaiton or not')

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
    main_worker(args.gpu, ngpus_per_node, args, data_loaders)


def main_worker(gpu, ngpus_per_node, args, data_loaders):
    def get_args_arch(args):
        arch = args.arch.lower()
        if arch == 'resnet20':
            return arch + "_" + args.data.lower()
        elif arch == 'densenet':
            return "densenet121"
        elif arch == 'mobilenet':
            return "mobilenetv2_w1"
        else:
            return arch

    def create_model(args):
        pretrained = args.pretrained and not args.resume
        logging.info(
            "=> using pre-trained PyTorchCV model '{}'".format(args.arch))
        model = ptcv_get_model(get_args_arch(args), pretrained=True)
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
                optimizer.load_state_dict(checkpoint['optimizer'])
                logging.info("=> loaded optimizer and meta information from checkpoint '{}' (epoch {})".
                             format(args.resume, checkpoint['epoch']))
            else:
                logging.info(
                    "=> no checkpoint found at '{}'".format(args.resume))
        return optimizer

    def get_quantize_model(args, model, model_dict, quantize_arch):
        if args.arch.lower() == 'alexnet':
            return quantize_arch(model, model_dict)
        return quantize_arch(model)

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
                    # if (bitwidth == 4) and not symmetric:
                    if 'quant_input' not in name and not symmetric:
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

    ############ PREPARE MODEL ##############

    fp_model, teacher = create_model(args)  # Create Model
    model_dict = transfer_param(args, fp_model) if args.transfer_param else None
    fp_model = eval_resume(args, fp_model)

    quantize_arch = quantize_arch_dict[args.arch]
    model = get_quantize_model(args, fp_model, model_dict, quantize_arch)
    bit_config = bit_config_dict["bit_config_" + args.arch + "_" + args.quant_scheme]
    model = set_quantize_param(args, model, bit_config)

    model = quant_resume(args, model)

    if args.gpu is not None:
        model = model.cuda(args.gpu)
        if args.distill_method != 'None':
            teacher = teacher.cuda(args.gpu)

    ######### PREPARE MODEL DONE #########
    
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    optimizer = resume_optimizer(args, optimizer)

    cudnn.benchmark = True

    train_loader = data_loaders['train']
    test_loader = data_loaders['test']
    
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

    # # Train EMA for couple epochs before training parameters
    for epoch in range(args.start_epoch, 1):
        print("EMA training epochs...")
        if args.mixed_precision:
            global update_interval, global_iter
            update_interval, global_iter = (args.schedule_count, 0) if args.schedule_unit == 'iter' else (len(train_loader), 0)
            
            skt_helper = SKT_Helper(args)
            set_mixed_precision(model, skt_helper)
            
            train_ema(train_loader, model, criterion, epoch, args)
            acc1 = validate(test_loader, model, criterion, args)

            incremental_channel_selection(model, args.range_ratio)
        else:
            train_ema(train_loader, model, criterion, epoch, args)
            acc1 = validate(test_loader, model, criterion, args)
    
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args)

        if args.mixed_precision:
            ch_ratio, neuron_ratio = skt_train(train_loader, model, criterion, optimizer, epoch, args)
        else:
            train(train_loader, model, criterion, optimizer, epoch, args)
        tuning_fin_time = time.time()
        one_epoch_time = get_time_cost_in_string(
            tuning_fin_time - tuning_start_time)
        acc1 = validate(test_loader, model, criterion, args)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        if (acc1 > best_acc1):
            best_acc1 = max(acc1, best_acc1)
            best_epoch = epoch

        logging.info(f'Best acc at epoch {best_epoch}: {best_acc1}')

        # if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
        #     save_checkpoint({
        #         'epoch': epoch + 1,
        #         'arch': args.arch,
        #         'state_dict': model.state_dict(),
        #         'best_acc1': best_acc1,
        #         'optimizer': optimizer.state_dict(),
        #     }, is_best, finetune_path)

    test_score = best_acc1

    time_cost = get_time_cost_in_string(tuning_fin_time - tuning_start_time)

    if args.mixed_precision:
        with open(f'{log_path}/LR_{args.lr}_range_{args.range_ratio}.txt', 'a') as f:
            f.write(
                'Schedule:{} {}, Channel:{:.2f}, Neuron:{:.2f} Acc:{:.2f}, REPL:{} QUANTILE:{} LR:{}, Batch:{}, Weight decay: {} Best Epoch:{}, Time:{}, Data:{}, 1 epoch time: {}\n'.format(
                    args.schedule_unit, args.schedule_count, ch_ratio, neuron_ratio, test_score, args.replace_grad, args.quantile, args.lr, args.batch_size, args.weight_decay,
                    best_epoch, time_cost, args.data, one_epoch_time))
    else:
        with open(f'{log_path}.txt', 'a') as f:
            f.write('Bit:{}, Acc:{:.2f}, LR:{}, Batch:{}, Weight decay: {}, Best Epoch:{}, Time:{}, Data:{}, 1 epoch time: {}\n'.format(
                args.quant_scheme, test_score, args.lr, args.batch_size, args.weight_decay, best_epoch, time_cost, args.data, one_epoch_time))
        

def train_ema(train_loader, model, criterion, epoch, args):
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    # switch to train mode
    if args.fix_BN == True:
        model.eval()
    else:
        model.train()

    with torch.no_grad():
        with tqdm(train_loader, desc="Epoch {} ".format(epoch), ncols=95) as t:
            # for i, (images, target) in enumerate(t):
            for i, data in enumerate(t):
                if args.dataset == 'imagenet':
                    images, target = data[0]['data'], torch.flatten(data[0]['label']).type(torch.long)
                else:
                    images, target = data

                if args.gpu is not None:
                    images = images.cuda(args.gpu)
                    target = target.cuda(args.gpu)

                # compute output
                output = model(images)
                loss = criterion(output, target)

                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                losses.update(loss.item(), images.size(0))
                top1.update(acc1[0].item(), images.size(0))
                top5.update(acc5[0].item(), images.size(0))

                t.set_postfix(acc1=top1.avg, acc5=top5.avg, loss=losses.avg)


def incremental_channel_selection(model, range_ratio):
    def get_weight_range(weight):
        w_transform = weight.transpose(1, 0).contiguous().view(weight.size(1), -1)
        return torch.max(w_transform.amin(dim=1).abs(), w_transform.amax(dim=1).abs())
    
    def get_activation_range(activation_range):
        if activation_range[0].min() >= 0.:
            return activation_range[1] - activation_range[0]
        return torch.max(activation_range[1].abs(), activation_range[0].abs())
    
    def get_weight_candidate_channel(range, ratio):
        return torch.where(range <= range.max() * ratio, 1, 0)
    
    def get_activation_max_range(x_min, x_max):
        if x_min >= 0.:
            return x_max - x_min
        return torch.max(x_max.abs(), x_min.abs())
    
    def get_activation_candidate_channel(range, max_range, ratio):
        return torch.where(range <= max_range * ratio, 1, 0)
    
    total_low_bit_channel, total_orig_bit_channel, total_low_bit_weight, total_orig_bit_weight = 0, 0, 0, 0
    for module in model.modules():
        if isinstance(module, QuantAct):
            prev_module = module
        elif isinstance(module, (QuantConv2d, QuantBnConv2d)):
            weight = module.weight.detach() if isinstance(module, QuantConv2d) else module.conv.weight.detach()
            weight_range = get_weight_range(weight)
            input_range = get_activation_range(module.input_range)
            input_max_range = get_activation_max_range(prev_module.x_min, prev_module.x_max)

            # candidate channel selection
            weight_bits = get_weight_candidate_channel(weight_range, range_ratio)
            input_bits = get_activation_candidate_channel(input_range, input_max_range, range_ratio)
            
            # low bit channel selection
            new_selected_channel = torch.logical_and(input_bits, weight_bits)
            accumulated_selected_channel = torch.logical_or(module.selected_channel, new_selected_channel)
            
            # accumulate new selected channel to selected channel pool
            module.selected_channel = accumulated_selected_channel
            module.selected_channel_index = accumulated_selected_channel.nonzero().view(-1)

            # statistic
            low_bit_channel = module.selected_channel_index.size(0)
            orig_bit_channel = module.selected_channel.size(0)
            weight_parameter_count = int(torch.numel(weight) / weight.size(1))
            
            total_low_bit_channel += low_bit_channel
            total_orig_bit_channel += orig_bit_channel
            total_low_bit_weight += low_bit_channel * weight_parameter_count
            total_orig_bit_weight += orig_bit_channel * weight_parameter_count

    channel_ratio = total_low_bit_channel / total_orig_bit_channel * 100
    parameter_ratio = total_low_bit_weight / total_orig_bit_weight * 100
    print("Int-4 channel ratio : {:.2f}%, parameter ratio : {:.2f}%".format(channel_ratio, parameter_ratio))
    return channel_ratio, parameter_ratio


def train(train_loader, model, criterion, optimizer, epoch, args):
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

            if args.gpu is not None:
                images = images.cuda(args.gpu)
                target = target.cuda(args.gpu)

            # compute output
            output = model(images)
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
            
            t.set_postfix(acc1=top1.avg, acc5=top5.avg, loss=losses.avg)


def skt_train(train_loader, model, criterion, optimizer, epoch, args):
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    global update_interval, global_iter

    # switch to train mode
    if args.fix_BN == True:
        model.eval()
    else:
        model.train()
    
    with tqdm(train_loader, desc="Epoch {} ".format(epoch), ncols=95) as t:
        for i, data in enumerate(t):
            if args.dataset == 'imagenet':
                images, target = data[0]['data'], torch.flatten(data[0]['label']).type(torch.long)
            else:
                images, target = data

            if args.gpu is not None:
                images = images.cuda(args.gpu)
                target = target.cuda(args.gpu)

            # compute output
            output = model(images)
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

            global_iter += 1
            # candidate channel selection
            if global_iter % update_interval == 0:
                channel_ratio, parameter_ratio = incremental_channel_selection(model, args.range_ratio)
                reset_model_record(model)

            t.set_postfix(acc1=top1.avg, acc5=top5.avg, loss=losses.avg)
    return channel_ratio, parameter_ratio


def validate(val_loader, model, criterion, args):
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    # switch to evaluate mode
    freeze_model(model)
    model.eval()

    with torch.no_grad():
        with tqdm(val_loader, desc="Validate", ncols=95) as t:
            for i, data in enumerate(t):
                if args.dataset == 'imagenet':
                    images, target = data[0]['data'], torch.flatten(data[0]['label']).type(torch.long)
                else:
                    images, target = data
                    
                if args.gpu is not None:
                    images = images.cuda(args.gpu, non_blocking=True)
                    target = target.cuda(args.gpu, non_blocking=True)

                # compute output
                output = model(images)
                loss = criterion(output, target)

                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                losses.update(loss.item(), images.size(0))
                top1.update(acc1[0].item(), images.size(0))
                top5.update(acc5[0].item(), images.size(0))

                t.set_postfix(acc1=top1.avg, acc5=top5.avg, loss=losses.avg)

    # torch.save({'convbn_scaling_factor': {k: v for k, v in model.state_dict().items() if 'convbn_scaling_factor' in k},
    #             'fc_scaling_factor': {k: v for k, v in model.state_dict().items() if 'fc_scaling_factor' in k},
    #             'weight_integer': {k: v for k, v in model.state_dict().items() if 'weight_integer' in k},
    #             'bias_integer': {k: v for k, v in model.state_dict().items() if 'bias_integer' in k},
    #             'act_scaling_factor': {k: v for k, v in model.state_dict().items() if 'act_scaling_factor' in k},
    #             }, args.save_path + 'quantized_checkpoint.pth.tar')

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

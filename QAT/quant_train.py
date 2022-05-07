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
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

torch.set_num_threads(8)

import warnings
warnings.filterwarnings("ignore")

from QAT.utils.models.q_alexnet import q_alexnet
from QAT.utils.models.q_densenet import q_densenet
from utils.misc import RuntimeHelper, pcq_epoch, pcq_validate, get_time_cost_in_string, load_dnn_model, set_save_dir
from .bit_config import *
from .utils import *
from pytorchcv.model_provider import get_model as ptcv_get_model


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
parser.add_argument('--fix-BN',
                    action='store_true',
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
"""
parser.add_argument('--channel-wise',
                    action='store_false',
                    help='whether to use channel-wise quantizaiton or not')
"""
parser.add_argument('--channel-wise',
                    type=bool,
                    default=False,
                    help='whether to use channel-wise quantizaiton or not')
parser.add_argument('--resize-qbit',
                    type=str,
                    default="False",
                    help='for residual tensor, true when high bit')

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

parser.add_argument('--transfer_param', action='store_true', help='copy params of torchcv pretrained models')
parser.add_argument('--dnn_path', default='', type=str, help="Pretrained model's path")

best_acc1 = 0
quantize_arch_dict = {'resnet50': q_resnet50, 'resnet50b': q_resnet50,
                      'resnet18': q_resnet18, 'resnet101': q_resnet101,
                      'resnet20': q_resnet20,
                      'alexnet': q_alexnet,
                      'densenet121': q_densenet,
                      'inceptionv3': q_inceptionv3,
                      'mobilenetv2_w1': q_mobilenetv2_w1}

args_qat, _ = parser.parse_known_args()
args_qat.save_path = os.path.join("checkpoints/{}/{}_{}_{}/".format(args_qat.arch, args_qat.data, args_qat.batch_size, os.getpid()))
if not os.path.exists(args_qat.save_path):
    os.makedirs(args_qat.save_path)

hook_counter = args_qat.checkpoint_iter
hook_keys = []
hook_keys_counter = 0

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S', filename=args_qat.save_path + 'log{}.log'.format(os.getpid()))
logging.getLogger().setLevel(logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler())

logging.info(args_qat)


def main(args_daq, data_loaders, clustering_model):
    args = argparse.Namespace(**vars(args_qat), **vars(args_daq))
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
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
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
        # pretrained = args.pretrained and not args.resume
        logging.info("=> using pre-trained PyTorchCV model '{}'".format(args.arch))
        model = ptcv_get_model(args.arch, pretrained=True)
        if args.distill_method != 'None':
            logging.info("=> using pre-trained PyTorchCV teacher '{}'".format(args.teacher_arch))
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
                    if 'num_batches_tracked' in key: model_key_list.remove(key)
                i = 0
                modified_dict = {}
                for key, value in checkpoint.items():
                    if 'scaling_factor' in key: continue
                    if 'num_batches_tracked' in key: continue
                    if 'weight_integer' in key: continue
                    if 'min' in key or 'max' in key: continue
                    modified_key = model_key_list[i]
                    modified_dict[modified_key] = value
                    i += 1
                logging.info(model.load_state_dict(modified_dict, strict=False))
            else:
                logging.info("=> no checkpoint found at '{}'".format(args.resume))
        return model

    def quant_resume(args, model):
        if args.resume and args.resume_quantize:
            if os.path.isfile(args.resume):
                logging.info("=> loading quantized checkpoint '{}'".format(args.resume))
                checkpoint = torch.load(args.resume)['state_dict']
                modified_dict = {}
                for key, value in checkpoint.items():
                    if 'num_batches_tracked' in key: continue
                    if 'weight_integer' in key: continue
                    if 'bias_integer' in key: continue

                    modified_key = key.replace("module.", "")
                    modified_dict[modified_key] = value
                model.load_state_dict(modified_dict, strict=False)
            else:
                logging.info("=> no quantized checkpoint found at '{}'".format(args.resume))
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
                logging.info("=> no checkpoint found at '{}'".format(args.resume))
        return optimizer

    def set_runtime_helper(args):
        if args.cluster > 1:
            runtime_helper = RuntimeHelper()
            runtime_helper.set_pcq_arguments(args)
            return runtime_helper
        return None

    def get_quantize_model(args, model, model_dict, quantize_arch, runtime_helper):
        if args.arch.lower() == 'alexnet':
            return quantize_arch(model, model_dict, runtime_helper)
        if args.arch.lower() == 'resnet20':
            return quantize_arch(model, args.resize_qbit, runtime_helper)
        return quantize_arch(model, runtime_helper)

    def set_quantize_param(args, model, bit_config):
        name_counter = 0

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
                setattr(m, 'fix_BN', args.fix_BN)
                setattr(m, 'fix_BN_threshold', args.fix_BN_threshold)
                setattr(m, 'training_BN_mode', args.fix_BN)
                setattr(m, 'checkpoint_iter_threshold', args.checkpoint_iter)
                setattr(m, 'save_path', args.save_path)
                setattr(m, 'fixed_point_quantization', args.fixed_point_quantization)

                if type(bit_config[name]) is tuple:
                    bitwidth = bit_config[name][0]
                    if bit_config[name][1] == 'hook':
                        m.register_forward_hook(hook_fn_forward)
                        global hook_keys
                        hook_keys.append(name)
                else:
                    bitwidth = bit_config[name]

                if hasattr(m, 'activation_bit'):
                    setattr(m, 'activation_bit', bitwidth)
                    if bitwidth == 4:
                        setattr(m, 'quant_mode', 'asymmetric')
                else:
                    setattr(m, 'weight_bit', bitwidth)

        logging.info("match all modules defined in bit_config: {}".format(len(bit_config.keys()) == name_counter))
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

    prev_arch = args.arch
    args.arch = set_args_arch(args)
    model, teacher = create_model(args)  # Create Model
    args.arch = reset_args_arch(args)
    if "true" not in args.resize_qbit.lower():
        args.resize_qbit = False
    else:
        args.resize_qbit = True
    model_dict = transfer_param(args, model) if args.transfer_param else None
    model = eval_resume(args, model)
    runtime_helper = set_runtime_helper(args)

    quantize_arch = quantize_arch_dict[args.arch]
    model = get_quantize_model(args, model, model_dict, quantize_arch, runtime_helper)

    if "resnet20" in args.arch and args.resize_qbit:
        bit_config = bit_config_dict["bit_config_" + args.arch + "_" + args.quant_scheme + "_resize"]
    else:
        bit_config = bit_config_dict["bit_config_" + args.arch + "_" + args.quant_scheme]
    model = set_quantize_param(args, model, bit_config)

    logging.info(model)

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
    test_loader = data_loaders['test']

    if args.nnac and clustering_model.final_cluster is None:
        model.toggle_full_precision()
        clustering_model.nn_aware_clustering(model, train_loader, prev_arch)
        model.toggle_full_precision()

    if args.evaluate:
        validate(test_loader, model, criterion, args)
        return

    best_epoch = 0
    register_acc = 0
    tuning_start_time = time.time()
    tuning_fin_time = None
    one_epoch_time = None

    finetune_path = set_save_dir(args)
    if not os.path.exists(finetune_path):
        os.mkdir(finetune_path)

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args)

        if args.cluster > 1:
            pcq_epoch(model, clustering_model, train_loader, criterion, optimizer, runtime_helper, epoch, logging,
                      fix_BN=args.fix_BN)
            tuning_fin_time = time.time()
            one_epoch_time = get_time_cost_in_string(tuning_fin_time - tuning_start_time)
            if epoch > 0:
                acc1 = pcq_validate(model, clustering_model, test_loader, criterion, runtime_helper, logging)

        else:
            train(train_loader, model, criterion, optimizer, epoch, logging, args)
            tuning_fin_time = time.time()
            one_epoch_time = get_time_cost_in_string(tuning_fin_time - tuning_start_time)
            if epoch > 0:
                acc1 = validate(test_loader, model, criterion, args)

        if epoch > 0:
            # remember best acc@1 and save checkpoint
            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)

            logging.info(f'Best acc at epoch {epoch}: {best_acc1}')
            if is_best:
                # record the best epoch
                best_epoch = epoch
                register_acc = best_acc1

            if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'best_acc1': best_acc1,
                    'optimizer': optimizer.state_dict(),
                }, is_best, finetune_path)

    test_score = register_acc

    time_cost = get_time_cost_in_string(tuning_fin_time - tuning_start_time)
    if not args.nnac:
        with open(f'qat_{args.arch}_{args.data}_cluster_{args.cluster}.txt', 'a') as f:
            f.write('Bit:{}, Acc:{:.2f}, LR:{}, Batch:{}, Weight decay: {}, Cluster:{} Best Epoch:{}, Time:{}, Data:{}, 1 epoch time: {}\n'.format(
                args.quant_scheme, test_score, args.lr, args.batch_size, args.weight_decay, args.cluster, best_epoch, time_cost, args.data, one_epoch_time))
    else:
        with open(f'qat_{args.arch}_{args.data}_cluster_{args.sub_cluster}->{args.cluster}.txt', 'a') as f:
            f.write('Bit:{}, Acc:{:.2f}, LR:{}, Batch:{}, Weight decay: {}, Cluster:{} Best Epoch:{}, Time:{}, Data:{}, 1 epoch time: {}\n'.format(
                args.quant_scheme, test_score, args.lr, args.batch_size, args.weight_decay, args.cluster, best_epoch, time_cost, args.data, one_epoch_time))


def train(train_loader, model, criterion, optimizer, epoch, logger, args):
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
    
    if epoch == 1:
        first_epoch_done(model)

    end = time.time()
    with tqdm(train_loader, desc="Epoch {}".format(epoch), ncols=105) as t:
        for i, (images, target) in enumerate(t):
            # measure data loading time
            data_time.update(time.time() - end)

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

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            logger.debug("[Epoch] {}, step {}/{} [Loss] {:.5f} ({:.5f}) [Acc@1] {:.3f} ({:.3f}) [Acc@5] {:3f} ({:.3f})"
                         .format(epoch, i + 1, len(train_loader), loss.item(), losses.avg, acc1.item(), top1.avg, acc5.item(), top5.avg))
            t.set_postfix(acc1=top1.avg, acc5=top5.avg, loss=losses.avg)

            #if i % args.print_freq == 0:
            #    progress.display(i)


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


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    freeze_model(model)
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        logging.info(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))

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
        shutil.copyfile(filename + 'checkpoint.pth.tar', filename + 'model_best.pth.tar')


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

import argparse
import os

# from QAT.models import *
from Clustering import *
from pretrain import _pretrain
# from evaluate import _evaluate
# from utils.lipschitz import check_lipschitz

# parser = argparse.ArgumentParser(description='[PyTorch] Per Cluster Quantization')
# parser.add_argument('--mode', default='fine', type=str, help="pre/fine/eval/lip")
# parser.add_argument('--arch', default='resnet', type=str, help='Architecture to train/eval')
# parser.add_argument('--dnn_path', default='', type=str, help="Pretrained model's path")
# parser.add_argument('--worker', default=4, type=int, help='Number of workers for input data loader')
#
# parser.add_argument('--imagenet', default='', type=str, help="ImageNet dataset path")
# parser.add_argument('--dataset', default='cifar10', type=str, help='Dataset to use')
# parser.add_argument('--dali', action='store_true', help='Use GPU data augmentation DALI')
#
# parser.add_argument('--epoch', default=100, type=int, help='Number of epochs to train')
# parser.add_argument('--batch', default=128, type=int, help='Mini-batch size')
# parser.add_argument('--val_batch', default=0, type=int, help='Validation batch size')
# parser.add_argument('--lr', default=0.01, type=float, help='Initial Learning Rate')
# parser.add_argument('--weight_decay', default=1e-4, type=float, help='Weight-decay value')
# parser.add_argument('--bn_momentum', default=0.1, type=float, help="BatchNorm2d's momentum factor")
#
# parser.add_argument('--fused', action='store_true', help='Evaluate or fine-tune fused model')
# parser.add_argument('--quantized', action='store_true', help='Evaluate quantized model')
#
# parser.add_argument('--quant_base', default='qat', type=str,
#                     help='Among qat/qn/hawq, choose fine-tuning method to apply DAQ')
# parser.add_argument('--per_channel', action='store_true',
#                     help='Use per output-channel quantization, or per tensor quantization')
# parser.add_argument('--symmetric', action='store_true',
#                     help="Use symmetric quantization for layers' weights")
# parser.add_argument('--fold_convbn', action='store_true',
#                     help="Fake Quantize CONV's weight after folding BatchNormalization")
#
# parser.add_argument('--ste', default=True, type=bool, help="Use Straight-through Estimator in Fake Quantization")
# parser.add_argument('--fq', default=1, type=int,
#                     help='Epoch to wait for fake-quantize activations. PCQ requires at least one epoch.')
# parser.add_argument('--bit', default=32, type=int, help='Target bit-width to be quantized (value 32 means pretraining)')
# parser.add_argument('--bit_conv_act', default=16, type=int,
#                     help="CONV's activation bit size when not using Conv&BN folding")
# parser.add_argument('--bit_bn_w', default=16, type=int, help="BN's weight bit size when not using CONV & BN folding")
# parser.add_argument('--bit_addcat', default=0, type=int, help="Bit size used in Skip-connection")
# parser.add_argument('--bit_first', default=0, type=int, help="First layer's bit size")
# parser.add_argument('--bit_classifier', default=0, type=int, help="Last classifier layer's bit size")
# parser.add_argument('--smooth', default=0.999, type=float, help='Smoothing parameter of EMA')
#
# parser.add_argument('--nnac', action='store_true', help="Use Neural Network Aware Clustering")
# parser.add_argument('--sim_threshold', default=0.7, type=float,
#                     help='Similarity threshold of ratio for considering similar clusters in nnac')
# parser.add_argument('--clustering_method', default='kmeans', type=str, help="Clustering method(K-means or BIRCH)")
# parser.add_argument('--mixrate', default=2, type=int,
#                     help='Number of epochs to mix augmented dataset to non-augmented dataset in training of clustering')
# parser.add_argument('--cluster', default=1, type=int, help='Number of clusters')
# parser.add_argument('--sub_cluster', default=0, type=int, help='Number of sub-clusters in NN-aware Clustering')
# parser.add_argument('--partition', default=2, type=int,
#                     help="Number of partitions to divide per channel for clustering's input")
# parser.add_argument('--partition_method', default='square', type=str, help="How to divide image into partitions")
# parser.add_argument('--repr_method', default='minmax', type=str, help="How to get representation per partition")
# parser.add_argument('--clustering_path', default='', type=str, help="Trained K-means clustering model's path")
#
# parser.add_argument('--kmeans_epoch', default=300, type=int, help='Max epoch of K-means model to train')
# parser.add_argument('--kmeans_tol', default=0.0001, type=float, help="K-means model's tolerance to detect convergence")
# parser.add_argument('--kmeans_init', default=10, type=int, help="Train K-means model n-times, and use the best model")
# parser.add_argument('--visualize_clustering', action='store_true',
#                     help="Visualize clustering result with PCA-ed training dataset")
#
# parser.add_argument('--quant_noise', default=False, type=bool, help='Apply quant noise')
# parser.add_argument('--qn_prob', default=0.2, type=float, help='quant noise probaility 0.05~0.2')
# parser.add_argument('--qn_increment_epoch', default=9999, type=int, help='quant noise qn_prob increment gap')
# parser.add_argument('--qn_each_channel', default=True, type=bool, help='qn apply conv each channel')
#
# parser.add_argument('--darknet', default=False, type=bool, help="Evaluate with dataset preprocessed in darknet")
# parser.add_argument('--horovod', default=False, type=bool, help="Use distributed training with horovod")
# parser.add_argument('--gpu', default='0', type=str, help='GPU to use')

#
#
# parser = argparse.ArgumentParser(description='[PyTorch] Per Cluster Quantization')
# parser.add_argument('--data', metavar='DIR',
#                     help='path to dataset')
# parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
#                     help='model architecture')
# parser.add_argument('--teacher-arch',
#                     type=str,
#                     default='resnet101',
#                     help='teacher network used to do distillation')
# parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
#                     help='number of data loading workers (default: 4)')
# parser.add_argument('--epochs', default=90, type=int, metavar='N',
#                     help='number of total epochs to run')
# parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
#                     help='manual epoch number (useful on restarts)')
# parser.add_argument('-b', '--batch-size', default=1, type=int,
#                     metavar='N',
#                     help='mini-batch size (default: 256), this is the total '
#                          'batch size of all GPUs on the current node when '
#                          'using Data Parallel or Distributed Data Parallel')
# parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
#                     metavar='LR', help='initial learning rate', dest='lr')
# parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
#                     help='momentum')
# parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
#                     metavar='W', help='weight decay (default: 1e-4)',
#                     dest='weight_decay')
# parser.add_argument('-p', '--print-freq', default=10, type=int,
#                     metavar='N', help='print frequency (default: 10)')
# parser.add_argument('--resume', default='', type=str, metavar='PATH',
#                     help='path to latest checkpoint (default: none)')
# parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
#                     help='evaluate model on validation set')
# parser.add_argument('--pretrained', dest='pretrained', action='store_true',
#                     help='use pre-trained model')
# parser.add_argument('--world-size', default=-1, type=int,
#                     help='number of nodes for distributed training')
# parser.add_argument('--rank', default=-1, type=int,
#                     help='node rank for distributed training')
# parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
#                     help='url used to set up distributed training')
# parser.add_argument('--dist-backend', default='nccl', type=str,
#                     help='distributed backend')
# parser.add_argument('--seed', default=None, type=int,
#                     help='seed for initializing training. ')
# parser.add_argument('--gpu', default='0', type=str,
#                     help='GPU id to use.')
# parser.add_argument('--multiprocessing-distributed', action='store_true',
#                     help='Use multi-processing distributed training to launch '
#                          'N processes per node, which has N GPUs. This is the '
#                          'fastest way to use PyTorch for either single node or '
#                          'multi node data parallel training')
# parser.add_argument('--act-range-momentum',
#                     type=float,
#                     default=-1,
#                     help='momentum of the activation range moving average, '
#                          '-1 stands for using minimum of min and maximum of max')
# parser.add_argument('--quant-mode',
#                     type=str,
#                     default='symmetric',
#                     choices=['asymmetric', 'symmetric'],
#                     help='quantization mode')
# parser.add_argument('--save-path',
#                     type=str,
#                     default='./checkpoints/imagenet/test/',
#                     help='path to save the quantized model')
# parser.add_argument('--data-percentage',
#                     type=float,
#                     default=1,
#                     help='data percentage of training data')
# parser.add_argument('--fix-BN',
#                     action='store_true',
#                     help='whether to fix BN statistics and fold BN during training')
# parser.add_argument('--fix-BN-threshold',
#                     type=int,
#                     default=None,
#                     help='when to start training with fixed and folded BN,'
#                          'after the threshold iteration, the original fix-BN will be overwritten to be True')
# parser.add_argument('--checkpoint-iter',
#                     type=int,
#                     default=-1,
#                     help='the iteration that we save all the featuremap for analysis')
# parser.add_argument('--evaluate-times',
#                     type=int,
#                     default=-1,
#                     help='The number of evaluations during one epoch')
# parser.add_argument('--quant-scheme',
#                     type=str,
#                     default='uniform4',
#                     help='quantization bit configuration')
# parser.add_argument('--resume-quantize',
#                     action='store_true',
#                     help='if True map the checkpoint to a quantized model,'
#                          'otherwise map the checkpoint to an ordinary model and then quantize')
# parser.add_argument('--act-percentile',
#                     type=float,
#                     default=0,
#                     help='the percentage used for activation percentile'
#                          '(0 means no percentile, 99.9 means cut off 0.1%)')
# parser.add_argument('--weight-percentile',
#                     type=float,
#                     default=0,
#                     help='the percentage used for weight percentile'
#                          '(0 means no percentile, 99.9 means cut off 0.1%)')
# parser.add_argument('--channel-wise',
#                     action='store_false',
#                     help='whether to use channel-wise quantizaiton or not')
# parser.add_argument('--bias-bit',
#                     type=int,
#                     default=32,
#                     help='quantizaiton bit-width for bias')
# parser.add_argument('--distill-method',
#                     type=str,
#                     default='None',
#                     help='you can choose None or KD_naive')
# parser.add_argument('--distill-alpha',
#                     type=float,
#                     default=0.95,
#                     help='how large is the ratio of normal loss and teacher loss')
# parser.add_argument('--temperature',
#                     type=float,
#                     default=6,
#                     help='how large is the temperature factor for distillation')
# parser.add_argument('--fixed-point-quantization',
#                     action='store_true',
#                     help='whether to skip deployment-oriented operations and '
#                          'use fixed-point rather than integer-only quantization')
#
#
# args = parser.parse_args()
# os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
#
# # General
# if args.imagenet:
#     args.dataset = 'imagenet'
# if args.dataset == 'cifar':
#     args.dataset = 'cifar10'
# if not args.val_batch:
#     args.val_batch = 256 if args.dataset != 'imagenet' else 128
#
# if args.quant_base == 'qn':
#     args.quant_noise = True
#
# # Default symmetric/asymmetric quantization setting
# if args.quant_base == 'hawq':
#     args.per_channel = True
#     if args.cluster == 1:
#         args.symmetric = True
#
# # First/Last layers' bit level
# if args.quant_base == 'hawq':
#     args.bit_first, args.bit_classifier = 8, 8
# if not args.bit_first:
#     args.bit_first = args.bit
# if not args.bit_classifier:
#     args.bit_classifier = args.bit
#
# # Skip-connections' bit level
# if not args.bit_addcat:
#     if args.quant_base == 'hawq':
#         args.bit_addcat = 16
#     else:
#         args.bit_addcat = args.bit
#
# # NN-aware Clustering
# if args.cluster > 1 and args.sub_cluster:
#     args.nnac = True
# print(vars(args))


# def set_func_for_target_arch(arch, clustering_method, is_pcq):
#     tools = QuantizationTool()
#
#     if is_pcq:
#         if clustering_method == 'kmeans':
#             setattr(tools, 'clustering_method', KMeansClustering)
#         elif clustering_method == 'dist':
#             setattr(tools, 'clustering_method', MinMaxDistClustering)
#         else:
#             setattr(tools, 'clustering_method', BIRCH)
#
#     if arch == 'MLP':
#         setattr(tools, 'pretrained_model_initializer', mlp)
#         setattr(tools, 'fused_model_initializer', pcq_mlp if is_pcq else fused_mlp)
#         setattr(tools, 'quantized_model_initializer', quantized_mlp)
#         setattr(tools, 'fuser', set_fused_mlp)
#         setattr(tools, 'quantizer', quantize_mlp)
#         setattr(tools, 'shift_qn_prob', modify_fused_mlp_qn_pre_hook)
#
#     if 'AlexNet' in arch:
#         setattr(tools, 'fuser', set_fused_alexnet)
#         setattr(tools, 'quantizer', quantize_alexnet)
#         if 'Small' in arch:
#             setattr(tools, 'pretrained_model_initializer', alexnet_small)
#             if is_pcq:
#                 setattr(tools, 'fused_model_initializer', pcq_alexnet_small)
#                 # setattr(tools, 'shift_qn_prob', modify_pcq_alexnet_qn_pre_hook)
#             else:
#                 setattr(tools, 'fused_model_initializer', fused_alexnet_small)
#                 # setattr(tools, 'shift_qn_prob', modify_fused_alexnet_qn_pre_hook)
#             setattr(tools, 'quantized_model_initializer', quantized_alexnet_small)
#         else:
#             setattr(tools, 'pretrained_model_initializer', alexnet)
#             if is_pcq:
#                 setattr(tools, 'fused_model_initializer', pcq_alexnet)
#                 # setattr(tools, 'shift_qn_prob', modify_pcq_alexnet_qn_pre_hook)
#             else:
#                 setattr(tools, 'fused_model_initializer', fused_alexnet)
#                 # setattr(tools, 'shift_qn_prob', modify_fused_alexnet_qn_pre_hook)
#             setattr(tools, 'quantized_model_initializer', quantized_alexnet)
#
#     elif 'ResNet' in arch:
#         setattr(tools, 'quantizer', quantize_pcq_resnet)
#         if is_pcq:
#             setattr(tools, 'fuser', set_pcq_resnet)
#         else:
#             setattr(tools, 'fuser', set_fused_resnet)
#
#         if '50' in arch:
#             setattr(tools, 'pretrained_model_initializer', resnet50)
#             if is_pcq:
#                 setattr(tools, 'fused_model_initializer', pcq_resnet50)
#                 # setattr(tools, 'shift_qn_prob', modify_pcq_resnet_qn_pre_hook)
#             else:
#                 setattr(tools, 'fused_model_initializer', fused_resnet50)
#                 # setattr(tools, 'shift_qn_prob', modify_fused_resnet_qn_pre_hook)
#             setattr(tools, 'quantized_model_initializer', quantized_resnet50)
#         else:
#             setattr(tools, 'pretrained_model_initializer', resnet20)
#             if is_pcq:
#                 setattr(tools, 'fused_model_initializer', pcq_resnet20)
#                 # setattr(tools, 'shift_qn_prob', modify_pcq_resnet_qn_pre_hook)
#             else:
#                 setattr(tools, 'fused_model_initializer', fused_resnet20)
#                 # setattr(tools, 'shift_qn_prob', modify_fused_resnet_qn_pre_hook)
#             setattr(tools, 'quantized_model_initializer', quantized_resnet20)
#
#     elif arch == 'MobileNetV3':
#         setattr(tools, 'fuser', set_fused_mobilenet)
#         setattr(tools, 'quantizer', quantize_mobilenet)
#         setattr(tools, 'pretrained_model_initializer', mobilenet)
#         setattr(tools, 'fused_model_initializer', fused_mobilenet)
#         setattr(tools, 'quantized_model_initializer', quantized_mobilenet)
#
#     elif arch == 'DenseNet121':
#         setattr(tools, 'quantized_model_initializer', quantized_densenet)
#         if is_pcq:
#             setattr(tools, 'fused_model_initializer', pcq_densenet)
#             setattr(tools, 'fuser', set_pcq_densenet)
#             setattr(tools, 'quantizer', quantize_pcq_densenet)
#             # setattr(tools, 'shift_qn_prob', modify_fused_densenet_qn_pre_hook)
#         else:
#             setattr(tools, 'fused_model_initializer', fused_densenet)
#             setattr(tools, 'fuser', set_fused_densenet)
#             setattr(tools, 'quantizer', quantize_densenet)
#             # setattr(tools, 'shift_qn_prob', modify_fused_densenet_qn_pre_hook)
#     return tools
#
#
# def specify_target_arch(arch, dataset, num_clusters):
#     arch = 'MLP' if arch == 'mlp' else arch
#     if arch == 'alexnet':
#         if dataset == 'imagenet':
#             arch = 'AlexNet'
#         else:
#             arch = 'AlexNetSmall'
#     elif arch == 'resnet':
#         if dataset == 'imagenet':
#             arch = 'ResNet50'
#         else:
#             arch = 'ResNet20'
#     elif arch == 'mobilenet':
#         arch = 'MobileNetV3'
#     elif arch =='bert':
#         arch = 'Bert'
#     elif arch == 'densenet':
#         arch = 'DenseNet121'
#
#     is_pcq = True if num_clusters > 1 else False
#     model_initializers = set_func_for_target_arch(arch, args.clustering_method, is_pcq)
#     return arch, model_initializers


if __name__ == '__main__':
    # assert args.arch in ['mlp', 'alexnet', 'resnet', 'bert', 'densenet', 'mobilenet'], 'Not supported architecture'
    # assert args.bit in [4, 8, 16, 32], 'Not supported target bit'
    # if args.mode == 'fine':
    #     assert args.bit in [4, 8], 'Please set target bit between 4 & 8'
    #     if args.dataset != 'imagenet':
    #         assert args.dnn_path, "Need pretrained model with the path('dnn_path' argument) for finetuning"

    # if args.mode == 'pre':
    #     args.arch, tools = specify_target_arch(args.arch, args.dataset, args.cluster)
    #     _pretrain(args, tools)
    # elif args.mode == 'fine':
    #     if args.quant_base == 'qat':
    #         from QAT.finetune import _finetune
    #
    #         args.arch, tools = specify_target_arch(args.arch, args.dataset, args.cluster)
    #         _finetune(args, tools)
    #     else:
    #         from HAWQ.quant_train import main
    #         main(args)
    # elif args.mode == 'eval':
    #     _evaluate(args, tools)
    # elif args.mode == 'lip':
    #     check_lipschitz(args, tools)

    from HAWQ.quant_train import main
    main()

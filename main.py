import argparse

from models import *
from pretrain import _pretrain
from finetune import _finetune
from finetune_with_dali import _finetune_with_dali
from evaluate import _evaluate
from utils.lipschitz import check_lipschitz

parser = argparse.ArgumentParser(description='[PyTorch] Per Cluster Quantization')
parser.add_argument('--mode', default='fine', type=str, help="pre/fine/eval/lip")
parser.add_argument('--arch', default='resnet', type=str, help='Architecture to train/eval')
parser.add_argument('--dnn_path', default='', type=str, help="Pretrained model's path")
parser.add_argument('--worker', default=4, type=int, help='Number of workers for input data loader')

parser.add_argument('--imagenet', default='', type=str, help="ImageNet dataset path")
parser.add_argument('--dataset', default='cifar10', type=str, help='Dataset to use')
parser.add_argument('--dali', action='store_true', help='Use GPU data augmentation DALI')

parser.add_argument('--epoch', default=100, type=int, help='Number of epochs to train')
parser.add_argument('--batch', default=128, type=int, help='Mini-batch size')
parser.add_argument('--val_batch', default=0, type=int, help='Validation batch size')
parser.add_argument('--lr', default=0.01, type=float, help='Initial Learning Rate')
parser.add_argument('--weight_decay', default=1e-4, type=float, help='Weight-decay value')
parser.add_argument('--bn_momentum', default=0.1, type=float, help="BatchNorm2d's momentum factor")

parser.add_argument('--fused', action='store_true', help='Evaluate or fine-tune fused model')
parser.add_argument('--quantized', action='store_true', help='Evaluate quantized model')


parser.add_argument('--quant_base', default='qat', type=str,
                    help='Among qat/qn/hawq, choose fine-tuning method to apply DAQ')
parser.add_argument('--per_channel', action='store_true',
                    help='Use per output-channel quantization, or per tensor quantization')
parser.add_argument('--fold_convbn', action='store_true',
                    help="Fake Quantize CONV's weight after folding BatchNormalization")

parser.add_argument('--ste', default=True, type=bool, help="Use Straight-through Estimator in Fake Quantization")
parser.add_argument('--fq', default=1, type=int, help='Epoch to wait for fake-quantize activations.'
                                                      ' PCQ requires at least one epoch.')
parser.add_argument('--bit', default=32, type=int, help='Target bit-width to be quantized (value 32 means pretraining)')
parser.add_argument('--bit_conv_act', default=16, type=int,
                    help="CONV's activation bit size when not using Conv&BN folding")
parser.add_argument('--bit_bn_w', default=16, type=int, help="BN's weight bit size when not using CONV & BN folding")
parser.add_argument('--bit_addcat', default=0, type=int, help="Bit size used in Skip-connection")
parser.add_argument('--bit_first', default=0, type=int, help="First layer's bit size")
parser.add_argument('--bit_classifier', default=0, type=int, help="Last classifier layer's bit size")
parser.add_argument('--smooth', default=0.999, type=float, help='Smoothing parameter of EMA')

parser.add_argument('--clustering_method', default='kmeans', type=str, help="Clustering method(K-means or BIRCH)")
parser.add_argument('--cluster', default=1, type=int, help='Number of clusters')
parser.add_argument('--partition', default=2, type=int,
                    help="Number of partitions to divide per channel for clustering's input")
parser.add_argument('--partition_method', default='square', type=str, help="How to divide image into partitions")
parser.add_argument('--repr_method', default='minmax', type=str, help="How to get representation per partition")
parser.add_argument('--clustering_path', default='', type=str, help="Trained K-means clustering model's path")

parser.add_argument('--kmeans_epoch', default=300, type=int, help='Max epoch of K-means model to train')
parser.add_argument('--kmeans_tol', default=0.0001, type=float, help="K-means model's tolerance to detect convergence")
parser.add_argument('--kmeans_init', default=10, type=int, help="Train K-means model n-times, and use the best model")
parser.add_argument('--visualize_clustering', action='store_true',
                    help="Visualize clustering result with PCA-ed training dataset")

parser.add_argument('--quant_noise', default=False, type=bool, help='Apply quant noise')
parser.add_argument('--qn_prob', default=0.2, type=float, help='quant noise probaility 0.05~0.2')
parser.add_argument('--qn_increment_epoch', default=9999, type=int, help='quant noise qn_prob increment gap')
parser.add_argument('--qn_each_channel', default=True, type=bool, help='qn apply conv each channel')

parser.add_argument('--darknet', default=False, type=bool, help="Evaluate with dataset preprocessed in darknet")
parser.add_argument('--horovod', default=False, type=bool, help="Use distributed training with horovod")
parser.add_argument('--gpu', default='0', type=str, help='GPU to use')
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

# General
if args.imagenet:
    args.dataset = 'imagenet'
if args.dataset == 'cifar':
    args.dataset = 'cifar10'
if not args.val_batch:
    args.val_batch = 256 if args.dataset != 'imagenet' else 128

if args.quant_base == 'qn':
    args.quant_noise = True

# First/Last layers' bit level
if args.quant_base == 'hawq':
    args.bit_first, args.bit_classifier = 8, 8
if not args.bit_first:
    args.bit_first = args.bit
if not args.bit_classifier:
    args.bit_classifier = args.bit

# Skip-connections' bit level
if not args.bit_addcat:
    if args.quant_base == 'hawq':
        args.bit_addcat = 16
    else:
        args.bit_addcat = args.bit
print(vars(args))


def set_func_for_target_arch(arch, clustering_method, is_pcq):
    tools = QuantizationTool()

    if is_pcq:
        if clustering_method == 'kmeans':
            setattr(tools, 'clustering_method', KMeans)
        elif clustering_method == 'dist':
            setattr(tools, 'clustering_method', MinMaxDistClustering)
        else:
            setattr(tools, 'clustering_method', BIRCH)

    if arch == 'MLP':
        setattr(tools, 'pretrained_model_initializer', mlp)
        setattr(tools, 'fused_model_initializer', pcq_mlp if is_pcq else fused_mlp)
        setattr(tools, 'quantized_model_initializer', quantized_mlp)
        setattr(tools, 'fuser', set_fused_mlp)
        setattr(tools, 'quantizer', quantize_mlp)
        setattr(tools, 'shift_qn_prob', modify_fused_mlp_qn_pre_hook)

    if 'AlexNet' in arch:
        setattr(tools, 'fuser', set_fused_alexnet)
        setattr(tools, 'quantizer', quantize_alexnet)
        if 'Small' in arch:
            setattr(tools, 'pretrained_model_initializer', alexnet_small)
            if is_pcq:
                setattr(tools, 'fused_model_initializer', pcq_alexnet_small)
                setattr(tools, 'shift_qn_prob', modify_pcq_alexnet_qn_pre_hook)
            else:
                setattr(tools, 'fused_model_initializer', fused_alexnet_small)
                setattr(tools, 'shift_qn_prob', modify_fused_alexnet_qn_pre_hook)
            setattr(tools, 'quantized_model_initializer', quantized_alexnet_small)
        else:
            setattr(tools, 'pretrained_model_initializer', alexnet)
            if is_pcq:
                setattr(tools, 'fused_model_initializer', pcq_alexnet)
                setattr(tools, 'shift_qn_prob', modify_pcq_alexnet_qn_pre_hook)
            else:
                setattr(tools, 'fused_model_initializer', fused_alexnet)
                setattr(tools, 'shift_qn_prob', modify_fused_alexnet_qn_pre_hook)
            setattr(tools, 'quantized_model_initializer', quantized_alexnet)

    elif 'ResNet' in arch:
        setattr(tools, 'quantizer', quantize_pcq_resnet)
        if is_pcq:
            setattr(tools, 'fuser', set_pcq_resnet)
        else:
            setattr(tools, 'fuser', set_fused_resnet)

        if '50' in arch:
            setattr(tools, 'pretrained_model_initializer', resnet50)
            if is_pcq:
                setattr(tools, 'fused_model_initializer', pcq_resnet50)
                setattr(tools, 'shift_qn_prob', modify_pcq_resnet_qn_pre_hook)
            else:
                setattr(tools, 'fused_model_initializer', fused_resnet50)
                setattr(tools, 'shift_qn_prob', modify_fused_resnet_qn_pre_hook)
            setattr(tools, 'quantized_model_initializer', quantized_resnet50)
        else:
            setattr(tools, 'pretrained_model_initializer', resnet20)
            if is_pcq:
                setattr(tools, 'fused_model_initializer', pcq_resnet20)
                setattr(tools, 'shift_qn_prob', modify_pcq_resnet_qn_pre_hook)
            else:
                setattr(tools, 'fused_model_initializer', fused_resnet20)
                setattr(tools, 'shift_qn_prob', modify_fused_resnet_qn_pre_hook)
            setattr(tools, 'quantized_model_initializer', quantized_resnet20)

    elif arch == 'MobileNetV3':
        setattr(tools, 'fuser', set_fused_mobilenet)
        setattr(tools, 'quantizer', quantize_mobilenet)
        setattr(tools, 'pretrained_model_initializer', mobilenet)
        setattr(tools, 'fused_model_initializer', fused_mobilenet)
        setattr(tools, 'quantized_model_initializer', quantized_mobilenet)

    elif arch == 'DenseNet121':
        setattr(tools, 'quantized_model_initializer', quantized_densenet)
        if is_pcq:
            setattr(tools, 'fused_model_initializer', pcq_densenet)
            setattr(tools, 'fuser', set_pcq_densenet)
            setattr(tools, 'quantizer', quantize_pcq_densenet)
            setattr(tools, 'shift_qn_prob', modify_fused_densenet_qn_pre_hook)
        else:
            setattr(tools, 'fused_model_initializer', fused_densenet)
            setattr(tools, 'fuser', set_fused_densenet)
            setattr(tools, 'quantizer', quantize_densenet)
            setattr(tools, 'shift_qn_prob', modify_fused_densenet_qn_pre_hook)
    return tools


def specify_target_arch(arch, dataset, num_clusters):
    arch = 'MLP' if arch == 'mlp' else arch
    if arch == 'alexnet':
        if dataset == 'imagenet':
            arch = 'AlexNet'
        else:
            arch = 'AlexNetSmall'
    elif arch == 'resnet':
        if dataset == 'imagenet':
            arch = 'ResNet50'
        else:
            arch = 'ResNet20'
    elif arch == 'mobilenet':
        arch = 'MobileNetV3'
    elif arch =='bert':
        arch = 'Bert'
    elif arch == 'densenet':
        arch = 'DenseNet121'

    is_pcq = True if num_clusters > 1 else False
    model_initializers = set_func_for_target_arch(arch, args.clustering_method, is_pcq)
    return arch, model_initializers


if __name__ == '__main__':
    assert args.arch in ['mlp', 'alexnet', 'resnet', 'bert', 'densenet', 'mobilenet'], 'Not supported architecture'
    assert args.bit in [4, 8, 16, 32], 'Not supported target bit'
    if args.mode == 'fine':
        assert args.bit in [4, 8], 'Please set target bit between 4 & 8'
        if args.dataset != 'imagenet':
            assert args.dnn_path, "Need pretrained model with the path('dnn_path' argument) for finetuning"

    args.arch, tools = specify_target_arch(args.arch, args.dataset, args.cluster)
    if args.mode == 'pre':
        _pretrain(args, tools)
    elif args.mode == 'fine':
        if args.dali and args.dataset == 'imagenet':
            _finetune_with_dali(args, tools)
        else:
            _finetune(args, tools)
    elif args.mode == 'eval':
        _evaluate(args, tools)
    elif args.mode == 'lip':
        check_lipschitz(args, tools)

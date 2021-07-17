import argparse

from models import *
from pretrain import _pretrain
from finetune import _finetune
from evaluate import _evaluate

parser = argparse.ArgumentParser(description='[PyTorch] Per Cluster Quantization')
parser.add_argument('--mode', default='eval', type=str, help="pre or fine or eval")
parser.add_argument('--arch', default='alexnet', type=str, help='Architecture to train/eval')
parser.add_argument('--dnn_path', default='', type=str, help="Pretrained model's path")

parser.add_argument('--imagenet', default='', type=str, help="ImageNet dataset path")
parser.add_argument('--dataset', default='cifar', type=str, help='Dataset to use')

parser.add_argument('--epoch', default=100, type=int, help='Number of epochs to train')
parser.add_argument('--batch', default=128, type=int, help='Mini-batch size')
parser.add_argument('--lr', default=0.01, type=float, help='Initial Learning Rate')
parser.add_argument('--weight_decay', default=1e-4, type=float, help='Weight-decay value')

parser.add_argument('--fused', default=False, type=bool, help="Evaluate fine-tuned, fused model")
parser.add_argument('--quantized', default=False, type=bool, help="Evaluate quantized model")

parser.add_argument('--bit', default=32, type=int, help='Target bit-width to be quantized (value 32 means pretraining)')
parser.add_argument('--smooth', default=0.999, type=float, help='Smoothing parameter of EMA')
parser.add_argument('--fq', default=0, type=int, help='Epoch to wait for Fake Quantization')

parser.add_argument('--kmeans_path', default='', type=str, help="Trained K-means clustering model's path")
parser.add_argument('--cluster', default=1, type=int, help='Number of clusters')
parser.add_argument('--partition', default=4, type=int, help="Number of partitions to divide a channel in kmeans clustering's input")
parser.add_argument('--kmeans_epoch', default=100, type=int, help='Max epoch of K-means model to train')
parser.add_argument('--kmeans_tol', default=0.0001, type=float, help="K-means model's tolerance to detect convergence")

parser.add_argument('--darknet', default=False, type=bool, help="Evaluate with dataset preprocessed in darknet")

args = parser.parse_args()
print(vars(args))


def set_func_for_target_arch(arch, is_pcq):
    tools = QuantizationTool()
    if 'AlexNet' in arch:
        setattr(tools, 'fuser', set_fused_alexnet)
        setattr(tools, 'quantizer', quantize_alexnet)
        if 'Small' in arch:
            setattr(tools, 'pretrained_model_initializer', alexnet_small)
            if is_pcq:
                setattr(tools, 'fused_model_initializer', pcq_alexnet_small)
            else:
                setattr(tools, 'fused_model_initializer', fused_alexnet_small)
            setattr(tools, 'quantized_model_initializer', quantized_alexnet_small)
        else:
            setattr(tools, 'pretrained_model_initializer', alexnet)
            if is_pcq:
                setattr(tools, 'fused_model_initializer', pcq_alexnet)
            else:
                setattr(tools, 'fused_model_initializer', fused_alexnet)
            setattr(tools, 'quantized_model_initializer', quantized_alexnet)

    elif 'ResNet' in arch:
        setattr(tools, 'fuser', set_fused_resnet)
        setattr(tools, 'quantizer', quantize_resnet)
        if '18' in arch:
            setattr(tools, 'pretrained_model_initializer', resnet18)
            if is_pcq:
                setattr(tools, 'fused_model_initializer', pcq_resnet18)
            else:
                setattr(tools, 'fused_model_initializer', fused_resnet18)
            setattr(tools, 'quantized_model_initializer', quantized_resnet18)
        else:
            setattr(tools, 'pretrained_model_initializer', resnet20)
            if is_pcq:
                setattr(tools, 'fused_model_initializer', pcq_resnet20)
            else:
                setattr(tools, 'fused_model_initializer', fused_resnet20)
            setattr(tools, 'quantized_model_initializer', quantized_resnet20)

    elif arch == 'MobileNetV3':
        setattr(tools, 'fuser', set_fused_mobilenet)
        setattr(tools, 'quantizer', quantize_mobilenet)
        setattr(tools, 'pretrained_model_initializer', mobilenet)
        setattr(tools, 'fused_model_initializer', fused_mobilenet)
        setattr(tools, 'quantized_model_initializer', quantized_mobilenet)
    return tools


def specify_target_arch(arch, dataset, num_clusters):
    if arch == 'alexnet':
        if dataset == 'imagenet':
            arch = 'AlexNet'
        else:
            arch = 'AlexNetSmall'

    elif arch == 'resnet':
        if dataset == 'imagenet':
            arch = 'ResNet18'
        else:
            arch = 'ResNet20'

    elif arch == 'mobilenet':
        arch = 'MobileNetV3'

    is_pcq = True if num_clusters > 1 else False
    model_initializers = set_func_for_target_arch(arch, is_pcq)
    return arch, model_initializers


if __name__=='__main__':
    assert args.arch in ['alexnet', 'resnet', 'densenet', 'mobilenet'], 'Not supported architecture'
    assert args.bit in [4, 8, 32], 'Not supported target bit'

    args.arch, tools = specify_target_arch(args.arch, args.dataset, args.cluster)
    if args.mode == 'pre':
        _pretrain(args, tools)
    elif args.mode == 'fine':
        _finetune(args, tools)
    else:
        _evaluate(args, tools)

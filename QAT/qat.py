import argparse
import os

from .models import *
from QAT.finetune import _finetune
from Clustering import *
from pretrain import _pretrain
from .evaluate import _evaluate
from utils.lipschitz import check_lipschitz

parser = argparse.ArgumentParser(description='[PyTorch] Per Cluster Quantization')
parser.add_argument('--arch', default='resnet', type=str, help='Architecture to train/eval')
parser.add_argument('--dnn_path', default='', type=str, help="Pretrained model's path")

parser.add_argument('--dali', action='store_true', help='Use GPU data augmentation DALI')
parser.add_argument('--torchcv', action='store_true', help='Load torchcv model')

parser.add_argument('--epoch', default=100, type=int, help='Number of epochs to train')
parser.add_argument('--lr', default=0.01, type=float, help='Initial Learning Rate')
parser.add_argument('--weight_decay', default=1e-4, type=float, help='Weight-decay value')
parser.add_argument('--bn_momentum', default=0.1, type=float, help="BatchNorm2d's momentum factor")
parser.add_argument('--multi_norm', action='store_true', help='whether use single batchnorm layer or multiple during layer fusion ')

parser.add_argument('--fused', action='store_true', help='Evaluate or fine-tune fused model')
parser.add_argument('--quantized', action='store_true', help='Evaluate quantized model')

parser.add_argument('--per_channel', action='store_true',
                    help='Use per output-channel quantization, or per tensor quantization')
parser.add_argument('--symmetric', action='store_true',
                    help="Use symmetric quantization for layers' weights")
parser.add_argument('--fold_convbn', action='store_true',
                    help="Fake Quantize CONV's weight after folding BatchNormalization")

parser.add_argument('--ste', default=True, type=bool, help="Use Straight-through Estimator in Fake Quantization")
parser.add_argument('--fq', default=1, type=int,
                    help='Epoch to wait for fake-quantize activations. PCQ requires at least one epoch.')
parser.add_argument('--bit', default=32, type=int, help='Target bit-width to be quantized (value 32 means pretraining)')
parser.add_argument('--bit_conv_act', default=16, type=int,
                    help="CONV's activation bit size when not using Conv&BN folding")
parser.add_argument('--bit_bn_w', default=16, type=int, help="BN's weight bit size when not using CONV & BN folding")
parser.add_argument('--bit_addcat', default=16, type=int, help="Bit size used in Skip-connection")
parser.add_argument('--bit_first', default=0, type=int, help="First layer's bit size")
parser.add_argument('--bit_classifier', default=0, type=int, help="Last classifier layer's bit size")
parser.add_argument('--smooth', default=0.999, type=float, help='Smoothing parameter of EMA')

parser.add_argument('--quant_noise', default=False, type=bool, help='Apply quant noise')
parser.add_argument('--qn_prob', default=0.2, type=float, help='quant noise probaility 0.05~0.2')
parser.add_argument('--qn_increment_epoch', default=9999, type=int, help='quant noise qn_prob increment gap')
parser.add_argument('--qn_each_channel', default=True, type=bool, help='qn apply conv each channel')

parser.add_argument('--gpu', default='0', type=str, help='GPU to use')
args_qat, _ = parser.parse_known_args()
#os.environ["CUDA_VISIBLE_DEVICES"] = args_qat.gpu

# General
if not args_qat.bit_first:
    args_qat.bit_first = args_qat.bit
if not args_qat.bit_classifier:
    args_qat.bit_classifier = args_qat.bit
# if not args_qat.bit_addcat:
#     args_qat.bit_addcat = args_qat.bit


def set_func_for_target_arch(arch, is_pcq, is_folded):
    tools = QuantizationTool()
    if arch == 'MLP':
        setattr(tools, 'pretrained_model_initializer', mlp)
        setattr(tools, 'fused_model_initializer', pcq_mlp if is_pcq else fused_mlp)
        setattr(tools, 'quantized_model_initializer', quantized_mlp)
        setattr(tools, 'fuser', set_fused_mlp)
        setattr(tools, 'quantizer', quantize_mlp)
        # setattr(tools, 'shift_qn_prob', modify_fused_mlp_qn_pre_hook)

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
        if is_folded:
            if is_pcq:
                setattr(tools, 'fuser', set_folded_pcq_resnet)
                setattr(tools, 'folder', fold_pcq_resnet)
            else:
                setattr(tools, 'fuser', set_folded_fused_resnet)
                setattr(tools, 'folder', fold_fused_resnet)
            setattr(tools, 'quantizer', folded_quantize_pcq_resnet)
        else:
            if is_pcq:
                setattr(tools, 'fuser', set_pcq_resnet)
            else:
                setattr(tools, 'fuser', set_fused_resnet)

            setattr(tools, 'quantizer', quantize_pcq_resnet)

        if '50' in arch:
            setattr(tools, 'pretrained_model_initializer', resnet50)
            if is_pcq:
                if is_folded:
                    setattr(tools, 'fused_model_initializer', pcq_resnet50_folded)
                else:
                    setattr(tools, 'fused_model_initializer', pcq_resnet50)
            else:
                if is_folded:
                    setattr(tools, 'fused_model_initializer', fused_resnet50_folded)
                else:
                    setattr(tools, 'fused_model_initializer', fused_resnet50)

            if is_folded:
                setattr(tools, 'quantized_model_initializer', quantized_resnet50_folded)
            else:
                setattr(tools, 'quantized_model_initializer', quantized_resnet50)
        else:
            setattr(tools, 'pretrained_model_initializer', resnet20)
            if is_pcq:
                if is_folded:
                    setattr(tools, 'fused_model_initializer', pcq_resnet20_folded)
                else:
                    setattr(tools, 'fused_model_initializer', pcq_resnet20)
            else:
                if is_folded:
                    setattr(tools, 'fused_model_initializer', fused_resnet20_folded)
                else:
                    setattr(tools, 'fused_model_initializer', fused_resnet20)

            if is_folded:
                setattr(tools, 'quantized_model_initializer', quantized_resnet20_folded)
            else:
                setattr(tools, 'quantized_model_initializer', quantized_resnet20)

    elif arch == 'MobileNetV3':
        setattr(tools, 'fuser', set_fused_mobilenet)
        setattr(tools, 'quantizer', quantize_mobilenet)
        setattr(tools, 'pretrained_model_initializer', mobilenet)
        setattr(tools, 'fused_model_initializer', fused_mobilenet)
        setattr(tools, 'quantized_model_initializer', quantized_mobilenet)

    elif arch == 'DenseNet121':
        # if is_pcq:
        #     setattr(tools, 'fuser', set_pcq_densenet)
        # elif is_folded:
        #     setattr(tools, 'fuser', set_folded_fused_densenet)
        #     setattr(tools, 'folder', fold_densenet)
        # else:
        #     setattr(tools, 'fuser', set_fused_densenet)

        setattr(tools, 'quantized_model_initializer', quantized_densenet)
        if is_pcq:
            setattr(tools, 'fused_model_initializer', pcq_densenet)
            setattr(tools, 'fuser', set_pcq_densenet)
            setattr(tools, 'quantizer', quantize_pcq_densenet)
        else:
            setattr(tools, 'fused_model_initializer', fused_densenet)
            setattr(tools, 'fuser', set_fused_densenet)
            setattr(tools, 'quantizer', quantize_densenet)
    return tools


def main(args_daq, data_loaders, clustering_model):
    args = argparse.Namespace(**vars(args_qat), **vars(args_daq))
    print(vars(args))
    assert args.arch in ['mlp', 'alexnet', 'resnet', 'resnet20', 'resnet50','bert', 'densenet', 'mobilenet'], 'Not supported architecture'
    assert args.bit in [4, 8, 16, 32], 'Not supported target bit'
    if args.mode == 'fine':
        assert args.bit in [4, 8], 'Please set target bit between 4 & 8'
        # if args.dataset != 'imagenet':
            # assert args.dnn_path, "Need pretrained model with the path('dnn_path' argument) for finetuning"

    def specify_target_arch(arch, dataset, num_clusters):
        arch = 'MLP' if arch == 'mlp' else arch
        if arch == 'alexnet':
            if dataset == 'imagenet':
                arch = 'AlexNet'
            else:
                arch = 'AlexNetSmall'
        elif arch == 'resnet20':
            arch = 'ResNet20'
        elif arch == 'resnet50':
            arch = 'ResNet50'
        elif arch == 'resnet':
            if dataset == 'imagenet':
                arch = 'ResNet50'
            else:
                arch = 'ResNet20'
        elif arch == 'mobilenet':
            arch = 'MobileNetV3'
        elif arch == 'bert':
            arch = 'Bert'
        elif arch == 'densenet':
            arch = 'DenseNet121'

        is_pcq = True if num_clusters > 1 else False
        model_initializers = set_func_for_target_arch(arch, is_pcq, args.fold_convbn)
        return arch, model_initializers
    args.arch, tools = specify_target_arch(args.arch, args.dataset, args.cluster)

    if args.mode == 'pre':
        _pretrain(args, tools)
    elif args.mode == 'fine':
        _finetune(args, tools, data_loaders, clustering_model)
    elif args.mode == 'eval':
        _evaluate(args, tools)
    elif args.mode == 'lip':
        check_lipschitz(args, tools)

import argparse

#from PerClusterQuantization.models.bert.bert import bert_small
#from PerClusterQuantization.models.bert.fused_bert import fused_bert_small, set_fused_bert_small
#from PerClusterQuantization.models.bert.quantized_bert import quantized_bert_small, quantize_bert
#from PerClusterQuantization.run_classifier import _run_classifier
from run_classifier import _run_classifier
from models import *
from pretrain import _pretrain
from finetune import _finetune
#from hvd_finetune import hvd_finetune
from evaluate import _evaluate

parser = argparse.ArgumentParser(description='[PyTorch] Per Cluster Quantization')
parser.add_argument('--mode', default='eval', type=str, help="pre or fine or eval")
parser.add_argument('--arch', default='alexnet', type=str, help='Architecture to train/eval')
parser.add_argument('--dnn_path', default='', type=str, help="Pretrained model's path")
parser.add_argument('--worker', default=0, type=int, help='Number of workers for input data loader')

parser.add_argument('--imagenet', default='', type=str, help="ImageNet dataset path")
parser.add_argument('--dataset', default='cifar', type=str, help='Dataset to use')
parser.add_argument('--num_classes', default=10, type=int, help='Cifar-10 or 100')

parser.add_argument('--epoch', default=100, type=int, help='Number of epochs to train')
parser.add_argument('--batch', default=128, type=int, help='Mini-batch size')
parser.add_argument('--val_batch', default=256, type=int, help='Validation batch size')
parser.add_argument('--lr', default=0.01, type=float, help='Initial Learning Rate')
parser.add_argument('--weight_decay', default=1e-4, type=float, help='Weight-decay value')
parser.add_argument('--bn_momentum', default=0.1, type=float, help="BatchNorm2d's momentum factor")

parser.add_argument('--fused', default=False, type=bool, help="Evaluate fine-tuned, fused model")
parser.add_argument('--quantized', default=False, type=bool, help="Evaluate quantized model")

parser.add_argument('--ste', default=True, type=bool, help="Use Straight-through Estimator in Fake Quantization")
parser.add_argument('--fq', default=1, type=int, help='Epoch to wait for fake-quantize activations.'
                                                      ' PCQ requires at least one epoch.')
parser.add_argument('--bit', default=32, type=int, help='Target bit-width to be quantized (value 32 means pretraining)')
parser.add_argument('--smooth', default=0.999, type=float, help='Smoothing parameter of EMA')
parser.add_argument('--folded_fq', default=False, type=bool, help="Fake Quantize CONV's weight after folding BatchNormalization")

parser.add_argument('--clustering_method', default='kmeans', type=str, help="Clustering method(K-means/BIRCH) to use in PCQ")
parser.add_argument('--cluster', default=1, type=int, help='Number of clusters')
parser.add_argument('--partition', default=4, type=int, help="Number of partitions to divide a channel in kmeans clustering's input")
parser.add_argument('--clustering_path', default='', type=str, help="Trained K-means clustering model's path")
parser.add_argument('--data_per_cluster', default=8, type=int, help="In Phase-2 of PCQ, number of data per cluster in a mini-batch")
parser.add_argument('--pcq_initialization', default=False, type=bool, help="Initialize PCQ model's BN & qparams before finetuning")
parser.add_argument('--phase2_loader_strategy', default='mean', type=str, help="Making data loader of Phase-2, choose length of data loader per cluster by strategy of mean/min/max length")
parser.add_argument('--indices_path', default='', type=str, help="Path to load indices_list for BN initialization and phase2 training")

parser.add_argument('--kmeans_epoch', default=300, type=int, help='Max epoch of K-means model to train')
parser.add_argument('--kmeans_tol', default=0.0001, type=float, help="K-means model's tolerance to detect convergence")
parser.add_argument('--kmeans_init', default=10, type=int, help="Train K-means model multiple times, and use the best model")
parser.add_argument('--visualize_clustering', default=False, type=bool, help="Visualize clustering result with PCA-ed training dataset")

parser.add_argument('--quant_noise', default=False, type=bool, help='Apply quant noise')
parser.add_argument('--qn_prob', default=0.1, type=float, help='quant noise probaility 0.05~0.2')
parser.add_argument('--qn_increment_epoch', default=9999, type=int, help='quant noise qn_prob increment gap')

parser.add_argument('--darknet', default=False, type=bool, help="Evaluate with dataset preprocessed in darknet")
parser.add_argument('--horovod', default=False, type=bool, help="Use distributed training with horovod")
parser.add_argument('--gpu', default='0', type=str, help='GPU to use')


args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
if args.imagenet:
    args.dataset = 'imagenet'
if not args.worker:
    if args.dataset == 'imagenet':
        args.worker = 32
    else:
        args.worker = 4
print(vars(args))

def set_func_for_target_arch(arch, clustering_method, is_pcq):
    tools = QuantizationTool()

    if is_pcq:
        if clustering_method == 'kmeans':
            setattr(tools, 'clustering_method', KMeans)
        else:
            setattr(tools, 'clustering_method', BIRCH)

    if 'AlexNet' in arch:
        setattr(tools, 'fuser', set_fused_alexnet)
        setattr(tools, 'folder', None)
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
        #setattr(tools, 'folder', fold_resnet_bn)
        setattr(tools, 'folder', None)
        setattr(tools, 'quantizer', quantize_pcq_resnet)
        if is_pcq:
            setattr(tools, 'fuser', set_pcq_resnet)
            # setattr(tools, 'folder', fold_pcq_resnet)
            # setattr(tools, 'quantizer', quantize_pcq_resnet)
        else:
            setattr(tools, 'fuser', set_fused_resnet)
            # setattr(tools, 'folder', fold_resnet)
            # setattr(tools, 'quantizer', quantize_resnet)

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
        setattr(tools, 'folder', fold_mobilenet)
        setattr(tools, 'fuser', set_fused_mobilenet)
        setattr(tools, 'quantizer', quantize_mobilenet)
        setattr(tools, 'pretrained_model_initializer', mobilenet)
        setattr(tools, 'fused_model_initializer', fused_mobilenet)
        setattr(tools, 'quantized_model_initializer', quantized_mobilenet)

    # elif arch == 'Bert':
    #     # setattr(tools, 'fuser', set_fused_bert)
    #     setattr(tools, 'fuser', set_fused_bert_small)
    #     setattr(tools, 'pretrained_model_initializer', bert_small)
    #     setattr(tools, 'fused_model_initializer', fused_bert_small)
    #     setattr(tools, 'quantized_model_initializer', quantized_bert_small)
    #     setattr(tools, 'quantizer', quantize_bert)

    elif arch == 'DenseNet121':
        # setattr(tools, 'pretrained_model_initializer', densenet121)
        setattr(tools, 'quantized_model_initializer', quantized_densenet)
        setattr(tools, 'folder', None)
        if is_pcq:
            setattr(tools, 'fused_model_initializer', pcq_densenet)
            setattr(tools, 'fuser', set_pcq_densenet)
            # setattr(tools, 'folder', fold_pcq_densenet)
            setattr(tools, 'quantizer', quantize_pcq_densenet)
            setattr(tools, 'shift_qn_prob', modify_fused_densenet_qn_pre_hook)
        else:
            setattr(tools, 'fused_model_initializer', fused_densenet)
            setattr(tools, 'fuser', set_fused_densenet)
            # setattr(tools, 'folder', fold_densenet)
            setattr(tools, 'quantizer', quantize_densenet)
            setattr(tools, 'shift_qn_prob', modify_fused_densenet_qn_pre_hook)
    return tools


def specify_target_arch(arch, dataset, num_clusters):
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


if __name__=='__main__':
    assert args.arch in ['alexnet', 'resnet', 'bert', 'densenet', 'mobilenet'], 'Not supported architecture'
    assert args.bit in [4, 8, 32], 'Not supported target bit'

    args.arch, tools = specify_target_arch(args.arch, args.dataset, args.cluster)
    if args.mode == 'pre':
        _pretrain(args, tools)
    elif args.mode == 'fine':
        #if args.horovod:
        #    hvd_finetune(args,tools)
        #else:
        _finetune(args, tools)
    elif args.mode == 'test':
        _run_classifier(args, tools)
    else:
        _evaluate(args, tools)

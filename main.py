import argparse
from pretrain import _pretrain
from finetune import _finetune
from evaluate import _evaluate


parser = argparse.ArgumentParser(description='[PyTorch] Per Cluster Quantization')
parser.add_argument('--mode', default='eval', type=str, help="pre or fine or eval")
parser.add_argument('--arch', default='alexnet', type=str, help='Architecture to train/eval')
parser.add_argument('--path', default='', type=str, help="Pretrained model's path")
parser.add_argument('--dataset', default='cifar10', type=str, help='Dataset to use')
parser.add_argument('--epoch', default=100, type=int, help='Number of epochs to train')
parser.add_argument('--batch', default=128, type=int, help='Mini-batch size')
parser.add_argument('--lr', default=0.1, type=float, help='Initial Learning Rate')
parser.add_argument('--weight_decay', default=1e-4, type=float, help='Weight-decay value')
parser.add_argument('--bit', default=32, type=int, help='Target bit-width to be quantized (value 32 means pretraining)')
parser.add_argument('--smooth', default=0.999, type=float, help='Smoothing parameter of EMA')
parser.add_argument('--cluster', default=1, type=int, help='Number of clusters')
parser.add_argument('--fused', default=False, type=bool, help="Evaluate fine-tuned model")
parser.add_argument('--quantized', default=False, type=bool, help="Evaluate quantized model")
parser.add_argument('--darknet', default=False, type=bool, help="Evaluate with dataset preprocessed in darknet")

args = parser.parse_args()
print(vars(args))

if __name__=='__main__':
    # use_gpu = torch.cuda.is_available()
    # assert use_gpu, "Code works on GPU"
    assert args.arch in ['alexnet', 'resnet', 'densenet', 'mobilenet'], 'Not supported architecture'
    assert args.bit in [4, 8, 32], 'Not supported target bit'

    if args.mode == 'pre':
        _pretrain(args)
    elif args.mode == 'fine':
        _finetune(args)
    else:
        _evaluate(args)

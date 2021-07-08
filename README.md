# Per Cluster Quantization

## Environment & Requirements

- OS: Ubuntu 18.04
- CUDA version: 10.2 / 11.1
- PyTorch version: 1.8.1

```
$ pip install scikit-learn
$ pip install tqdm
$ pip install torch-summary
```

## 1. Pre-training

```
# Pre-traininig model with CIFAR-10
$ python main.py --mode pre --arch resnet
$ python main.py --mode pre --arch alexnet

# Evaluation of pre-trained model
$ python main.py --mode eval --arch resnet --path result/pre/ResNet20_32bit/[Date]/model_best.pth.tar
$ python main.py --mode eval --arch alexnet --path result/pre/AlexNetSmall_32bit/[Date]/model_best.pth.tar
```

## 2. Fine-tuning (QAT)

```
# Fine-tuning model with CIFAR-10
$ python main.py --mode fine --arch resnet  --bit 4 --batch 8 --lr 0.0001 --weight_decay 0.0 --path result/pre/ResNet20_32bit/[Date]/model_best.pth.tar
$ python main.py --mode fine --arch alexnet --bit 4 --batch 8 --lr 0.0001 --weight_decay 0.0 --path result/pre/AlexNetSmall_32bit/[Date]/model_best.pth.tar

# Evaluation of fine-tuned model (NOT integer model, just a FP model after QAT)
$ python main.py --mode eval --arch resnet  --fused True --path result/fine/ResNet20_4bit/[Date]/model_best.pth.tar
$ python main.py --mode eval --arch alexnet --fused True --path result/fine/AlexNetSmall_4bit/[Date]/model_best.pth.tar
```

## 3. Integer Inference

```
# Evaluation of quantized model
$ python main.py --mode eval --arch resnet  --quantized True --path result/fine/ResNet20_4bit/[Date]/quantized/checkpoint.pth
$ python main.py --mode eval --arch alexnet --quantized True --path result/fine/AlexNetSmall_4bit/[Date]/quantized/checkpoint.pth
```


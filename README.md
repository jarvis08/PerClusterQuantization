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
$ python main.py --mode eval --arch resnet --dnn_path result/pre/ResNet20_32bit/[Date]/model_best.pth.tar
$ python main.py --mode eval --arch alexnet --dnn_path result/pre/AlexNetSmall_32bit/[Date]/model_best.pth.tar
```

## 2. Per Cluster Quantization

### 2-1. Fine-tuning

If K-means clustering model's path wasn't given to `--kmeans_path`, it will train a clustering model and use it automatically.

Trained clustering model will be saved into path `result/kmeans/[Dataset]/[Date]/`

```
# K-means model's path not given
python main.py --mode fine --arch alexnet --lr 0.0001 --weight_decay 0.0 --epoch 1 --batch 32 --bit 4 --cluster 10\
           --dnn_path ./result/pre/AlexNetSmall_32bit/[Date]/checkpoint.pth

# With kmeans model
python main.py --mode fine --arch alexnet --lr 0.0001 --weight_decay 0.0 --epoch 1 --batch 32 --bit 4 --cluster 10\
           --kmeans_path result/kmeans/[Dataset]/[Date]/checkpoint.pkl\
           --dnn_path ./result/pre/AlexNetSmall_32bit/07-05-2234/checkpoint.pth

# Evaluation for FP fine-tuned model
python main.py --mode eval --arch alexnet --fused True --bit 4  --cluster 10\
           --kmeans_path result/kmeans/cifar/[Date]/checkpoint.pkl\
           --dnn_path ./result/fine/AlexNetSmall_4bit/[Date]/quantized/checkpoint.pth
```

### 2-2. Integer Inference

```
python main.py --mode eval --arch alexnet --quantized True --bit 4  --cluster 10\
           --kmeans_path result/kmeans/cifar/[Date]/checkpoint.pkl\
           --dnn_path ./result/fine/AlexNetSmall_4bit/[Date]/quantized/checkpoint.pth

```

## 3. Goolge's Quantization Aware Training

### 3-1. Fine-tuning

```
# Fine-tuning model with CIFAR-10
$ python main.py --mode fine --arch resnet  --bit 4 --batch 8 --lr 0.0001 --weight_decay 0.0\
        --dnn_path result/pre/ResNet20_32bit/[Date]/model_best.pth.tar
$ python main.py --mode fine --arch alexnet --bit 4 --batch 8 --lr 0.0001 --weight_decay 0.0\
        --dnn_path result/pre/AlexNetSmall_32bit/[Date]/model_best.pth.tar

# Evaluation of fine-tuned model (NOT integer model, just a FP model after QAT)
$ python main.py --mode eval --arch resnet  --fused True\
        --dnn_path result/fine/ResNet20_4bit/[Date]/model_best.pth.tar
$ python main.py --mode eval --arch alexnet --fused True\
        --dnn_path result/fine/AlexNetSmall_4bit/[Date]/model_best.pth.tar
```

### 3-2. Integer Inference

```
$ python main.py --mode eval --arch resnet  --quantized True\
        --dnn_path result/fine/ResNet20_4bit/[Date]/quantized/checkpoint.pth
$ python main.py --mode eval --arch alexnet --quantized True\
        --dnn_path result/fine/AlexNetSmall_4bit/[Date]/quantized/checkpoint.pth
```


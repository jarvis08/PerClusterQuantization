#! /bin/bash

export TORCH_WARN_ONCE

PRETRAINED_MODEL_PATH="/workspace/pretrained_models"
CLUSTERING_MODEL_PATH="result/kmeans/cifar100/nnac_resnet20.k4.sub8.part2.mean.topk_3.sim_0.7/"

MODEL="alexnet"                # alexnet / resnet20 / resnet50 / densenet121
DATASET="cifar10"              # svhn / cifar10 / cifar100 / imagenet
PRETRAINED_MODEL=$MODEL

BATCH=128
LEARNING_RATE=0.001

CUDA_VISIBLE_DEVICES=0 python main.py \
    --mode fine \
    --epochs 100 \
    --batch $BATCH \
    --quant_base hawq \
    --arch $MODEL \
    --dataset $DATASET \
    --lr $LEARNING_RATE \
    --act-range-momentum 0.99 \
    --wd 1e-4 \
    --fix-BN \
    --pretrained \
    --channel-wise true \
    --quant-scheme uniform4 \
    --gpu 0 \
    --data $DATASET \
    --batch-size $BATCH \
    --transfer_param \
    --dnn_path $PRETRAINED_MODEL_PATH/$DATASET/$PRETRAINED_MODEL/checkpoint.pth \

#    --cluster 4 \
#    --repr_method mean \

#    --clustering_path $CLUSTERING_MODEL_PATH \

#    --sub_cluster 8 \
#    --nnac true \
#    --similarity_method and \

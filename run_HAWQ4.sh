#! /bin/bash

PRETRAINED_MODEL_PATH="/workspace/pretrained_models"

MODEL="resnet20_unfold"
DATASET="cifar100"
PRETRAINED_MODEL="resnet20"

BATCH=64

CUDA_VISIBLE_DEVICES=1 python main.py \
    --mode fine \
    --epochs 100 \
    --batch $BATCH \
    --quant_base hawq \
    --arch $MODEL \
    --dataset $DATASET \
    --lr 0.001 \
    --act-range-momentum 0.99 \
    --wd 1e-4 \
    --fix-BN \
    --pretrained \
    --channel-wise true \
    --quant-scheme uniform4 \
    --gpu 0 \
    --data $DATASET \
    --transfer_param \
    --batch-size $BATCH \
    --cluster 4 \
    --dnn_path $PRETRAINED_MODEL_PATH/$DATASET/$PRETRAINED_MODEL/checkpoint.pth \
#    --imagenet /workspace/dataset/

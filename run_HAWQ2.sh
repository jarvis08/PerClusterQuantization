#! /bin/bash

PRETRAINED_MODEL_PATH="/workspace/pretrained_models"

MODEL="resnet20_unfold"
DATASET="cifar10"
PRETRAINED_MODEL="resnet20"

CUDA_VISIBLE_DEVICES=1 python main.py \
    --mode fine \
    --quant_base hawq \
    --arch $MODEL \
    --dnn_path $PRETRAINED_MODEL_PATH/$DATASET/$PRETRAINED_MODEL/checkpoint.pth \
    --dataset $DATASET \
    --data $DATASET \
    --wd 1e-4 \
    --act-range-momentum 0.99 \
    --lr 0.001 \
    --epochs 100 
    --batch-size 32
    --gpu 0 \
    --fix-BN \
    --quant-scheme uniform4 \
    --transfer_param \
    --channel-wise \
    --pretrained \
    --cluster 4


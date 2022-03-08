#! /bin/bash

PRETRAINED_MODEL_PATH="/workspace/pretrained_models"

MODEL="resnet20_unfold"
DATASET="cifar10"
PRETRAINED_MODEL="resnet20"

CUDA_VISIBLE_DEVICES=0 python main.py \
    --mode fine \
    --epochs 100 \
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
    --dnn_path $PRETRAINED_MODEL_PATH/$DATASET/$PRETRAINED_MODEL/checkpoint.pth \
    --data $DATASET \
    --transfer_param \
    --batch-size 32 \


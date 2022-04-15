#! /bin/bash

#PRETRAINED_MODEL_PATH="/workspace/pretrained_models"
PRETRAINED_MODEL_PATH="pretrained_models"

MODEL="alexnet"
DATASET="cifar100"
PRETRAINED_MODEL="alexnet"

BATCH=128

CUDA_VISIBLE_DEVICES=3 python main.py \
    --mode fine \
    --epochs 70 \
    --batch $BATCH \
    --quant_base hawq \
    --arch $MODEL \
    --dataset $DATASET \
    --lr 0.0001 \
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
    --sub_cluster 8 \
    --cluster 4 \
    --repr_method mean \
    --nnac true \
    --dnn_path $PRETRAINED_MODEL_PATH/$DATASET/$PRETRAINED_MODEL/checkpoint.pth \
#    --imagenet /workspace/dataset/

#! /bin/bash

PRETRAINED_MODEL_PATH="/workspace/pretrained_models"

MODEL="densenet121"
DATASET="imagenet"
PRETRAINED_MODEL="densenet121"

BATCH=128

CUDA_VISIBLE_DEVICES=3 python main.py \
    --mode fine \
    --epochs 70 \
    --batch $BATCH \
    --quant_base hawq \
    --arch $MODEL \
    --dataset $DATASET \
    --lr 0.000001 \
    --act-range-momentum 0.99 \
    --wd 1e-4 \
    --fix-BN \
    --pretrained \
    --channel-wise true \
    --quant-scheme uniform4 \
    --gpu 0 \
    --data $DATASET \
    --batch-size $BATCH \
    --imagenet /workspace/dataset/
#    --transfer_param \
#    --cluster 4 \
#    --dnn_path $PRETRAINED_MODEL_PATH/$DATASET/$PRETRAINED_MODEL/checkpoint.pth \

#! /bin/bash

PRETRAINED_MODEL_PATH="/workspace/pretrained_models"
CLUSTERING_MODEL_PATH=""

MODEL="resnet20"
DATASET="cifar100"
PRETRAINED_MODEL="resnet20"

BATCH=128

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
    --batch-size $BATCH \
    --transfer_param \
    --dnn_path $PRETRAINED_MODEL_PATH/$DATASET/$PRETRAINED_MODEL/checkpoint.pth \
    # --cluster 4 \
    # --repr_method mean \
    # --sub_cluster 8 \
    # --clustering_path $CLUSTERING_MODEL_PATH \

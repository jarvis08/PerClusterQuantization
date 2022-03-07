#! /bin/bash

PRETRAINED_MODEL_PATH="/workspace/pretrained_models"

MODEL="resnet"
DATASET="cifar100"

if [MODEL == "resnet"]
then
    PRETRAINED_MODEL="resnet20"
else
    PRETRAINED_MODEL=$MODEL
fi

python main.py \
    --mode fine \
    --quant_base qat \
    --arch $MODEL \
    --dnn_path $PRETRAINED_MODEL_PATH/$DATASET/$PRETRAINED_MODEL/checkpoint.pth \
    --dataset $DATASET \
    --weight_decay 0.0001 \
    --smooth 0.99 \
    --lr 0.001 \
    --epoch 100 \
    --batch 128 \
    --gpu 0 \
    --bit_classifier 8 \
    --bit_addcat 16 \
    --bit_first 8 \
    --bit 4 \
    --per_channel \
    --symmetric \
    -fq 1 \

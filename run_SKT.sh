#! /bin/bash

PRETRAINED_MODEL_PATH="/workspace/pretrained_models"
# IMAGENET_PATH="/home/work/.Datasets/Imagenet2012"
IMAGENET_PATH="/home/work/.Datasets/Imagenet2012_rec"
# IMAGENET_PATH="/workspace/dataset"

###################################################
GPU_NUM=${1}
MODEL=${2}              # alexnet / resnet20
DATASET=${3}            # cifar10 / cifar100 / svhn
LEARNING_RATE=${4}      # 0.001 / 0.0001
MIXED_PRECISION=${5}    # true / false
RANGE_RATIO=${6}        # 0.5 ~ 0.7
SCHEDULE_UNIT=${7}      # epoch / iter
SCHEDULE_COUNT=${8}     # 1 / 10 / 100
#REPLACE_GRAD=${9} # 1e-4 / 1e-5 / 1e-6 / 1e-7 / 1e-8
#QUANTILE=${10}      # 0 / 0.25 / 0.5 / 0.75 / 1
#####################################################

if [ "$MIXED_PRECISION" = true ]; then 
    if [ "$MODEL" = resnet20 ]; then
        CUDA_VISIBLE_DEVICES=${GPU_NUM} python main.py \
            --mode fine \
            --epochs 100 \
            --batch 128 \
            --quant_base hawq \
            --arch $MODEL \
            --dataset $DATASET \
            --lr $LEARNING_RATE \
            --act-range-momentum 0.99 \
            --wd 1e-4 \
            --pretrained \
            --channel-wise false \
            --quant-scheme uniform8 \
            --gpu 0 \
            --data $DATASET \
            --batch-size 128 \
            --range_ratio $RANGE_RATIO \
            --schedule_unit $SCHEDULE_UNIT \
            --schedule_count $SCHEDULE_COUNT \
            --mixed_precision
    else
        CUDA_VISIBLE_DEVICES=${GPU_NUM} python main.py \
            --mode fine \
            --epochs 100 \
            --batch 128 \
            --quant_base hawq \
            --arch $MODEL \
            --dataset $DATASET \
            --lr $LEARNING_RATE \
            --act-range-momentum 0.99 \
            --wd 1e-4 \
            --pretrained \
            --channel-wise false \
            --quant-scheme uniform8 \
            --gpu 0 \
            --data $DATASET \
            --batch-size 128 \
            --transfer_param \
            --dnn_path $PRETRAINED_MODEL_PATH/$DATASET/$MODEL/checkpoint.pth \
            --range_ratio $RANGE_RATIO \
            --schedule_unit $SCHEDULE_UNIT \
            --schedule_count $SCHEDULE_COUNT \
            --mixed_precision
    fi
else
    if [ "$MODEL" = resnet20 ]; then
        CUDA_VISIBLE_DEVICES=${GPU_NUM} python main.py \
            --mode fine \
            --epochs 100 \
            --batch 128 \
            --quant_base hawq \
            --arch $MODEL \
            --dataset $DATASET \
            --lr $LEARNING_RATE \
            --act-range-momentum 0.99 \
            --wd 1e-4 \
            --pretrained \
            --channel-wise false \
            --quant-scheme uniform4 \
            --gpu 0 \
            --data $DATASET \
            --batch-size 128
    else
        CUDA_VISIBLE_DEVICES=${GPU_NUM} python main.py \
            --mode fine \
            --epochs 100 \
            --batch 128 \
            --quant_base hawq \
            --arch $MODEL \
            --dataset $DATASET \
            --lr $LEARNING_RATE \
            --act-range-momentum 0.99 \
            --wd 1e-4 \
            --pretrained \
            --channel-wise false \
            --quant-scheme uniform4 \
            --gpu 0 \
            --data $DATASET \
            --batch-size 128 \
            --transfer_param \
            --dnn_path $PRETRAINED_MODEL_PATH/$DATASET/$MODEL/checkpoint.pth
    fi
fi



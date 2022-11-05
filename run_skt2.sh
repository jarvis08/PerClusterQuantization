#! /bin/bash

###################################################
GPU_NUM=${1}
MODEL=${2}         # alexnet / resnet
DATASET=${3}       # cifar10 / cifar100
FIXED_RATIO=${4}
CONST_PORTION=${5} # 1e-4 / 1e-5 / 1e-6 / 1e-7 / 1e-8
QUANTILE=${6}      # 0 / 0.25 / 0.5 / 0.75 / 1

#REDUCE_RATIO=${4} # 0.98 / 0.95 / 0.9 / 0.8 / 0.7

if [ $MODEL = "resnet" ] ; then
  MODEL_DESC=$MODEL'20'
else
  MODEL_DESC=$MODEL
fi

MODEL_PATH="./pretrained_models/$DATASET/$MODEL_DESC/checkpoint.pth"
#echo "$MODEL"
#echo "$MODEL_DESC"
#echo "$MODEL_PATH"
#####################################################

if [ "$MODEL" = alexnet ]; then
  python main.py \
    --arch $MODEL \
    --epoch 100 \
    --lr 1e-3 \
    --batch 128 \
    --dataset $DATASET \
    --dnn_path $MODEL_PATH \
    --smoooth 0.99 \
    --bit 8 \
    --symmetric \
    --mixed_precision \
    --input_grad \
    --percentile 2.0 \
    --channel_epoch 30 \
    --pre_fixed_channel $FIXED_RATIO \
    --reduce_ratio 1.0 \
    --const_portion $CONST_PORTION \
    --quantile $QUANTILE \
    --init_ema 1 \
    --gpu $GPU_NUM
else
    python main.py \
    --arch $MODEL \
    --epoch 100 \
    --lr 1e-3 \
    --batch 128 \
    --dataset $DATASET \
    --dnn_path $MODEL_PATH \
    --smoooth 0.99 \
    --bit 8 \
    --bit_addcat 8 \
    --symmetric \
    --mixed_precision \
    --input_grad \
    --percentile 2.0 \
    --channel_epoch 30 \
    --pre_fixed_channel $FIXED_RATIO \
    --reduce_ratio 1.0 \
    --const_portion $CONST_PORTION \
    --quantile $QUANTILE \
    --init_ema 1 \
    --gpu $GPU_NUM
fi




#! /bin/bash

###################################################
GPU_NUM=${1}
MODEL=${2}         # alexnet / resnet
DATASET=${3}       # cifar10 / cifar100
#INPUT_GRAD=${4}
#FIXED_RATIO=${5}
SCHEDULE_UNIT=${4}
SCHEDULE_COUNT=${5}
CONST_PORTION=${6} # 1e-4 / 1e-5 / 1e-6 / 1e-7 / 1e-8
QUANTILE=${7}      # 0 / 0.25 / 0.5 / 0.75 / 1
#REDUCE_RATIO=${7}  # 0.98 / 0.95 / 0.9 / 0.8 / 0.7

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
    --input_grad True\
    --percentile 2.0 \
    --channel_epoch 100 \
    --pre_fixed_channel 1.0 \
    --reduce_ratio 1.0 \
    --const_portion $CONST_PORTION \
    --quantile $QUANTILE \
    --init_ema 0 \
    --schedule_unit $SCHEDULE_UNIT \
    --schedule_count $SCHEDULE_COUNT \
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
    --input_grad True\
    --percentile 1.8 \
    --channel_epoch 100 \
    --pre_fixed_channel 1.0 \
    --reduce_ratio 1.0 \
    --const_portion $CONST_PORTION \
    --quantile $QUANTILE \
    --init_ema 0 \
    --schedule_unit $SCHEDULE_UNIT \
    --schedule_count $SCHEDULE_COUNT \
    --gpu $GPU_NUM
fi




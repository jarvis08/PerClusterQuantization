#! /bin/bash

###################################################
GPU_NUM=${1}
MODEL=${2}         # alexnet / resnet
DATASET=${3}       # cifar10 / cifar100
MODE=${4}
W_PERCENTILE=${5}  # 0 < p <= 1
ACT_QUANTILE=${6}    # 0 ~ 1

#REDUCE_RATIO=${4}  # 0.98 / 0.95 / 0.9 / 0.8 / 0.7
#CONST_PORTION=${5} # 1e-4 / 1e-5 / 1e-6 / 1e-7 / 1e-8
#RECORD_VAL=${6}    # true / false
#QUANTILE=${7}      # 0 / 0.25 / 0.5 / 0.75 / 1

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
if [ "$MODE" = paper ]; then
  if [ "$MODEL" = alexnet ]; then
    python main.py \
      --arch $MODEL \
      --epoch 100 \
      --lr 1e-3 \
      --batch 128 \
      --dataset $DATASET \
      --dnn_path $MODEL_PATH \
      --run_mode $MODE \
      --smoooth 0.99 \
      --bit 8 \
      --symmetric \
      --per_channel \
      --channel_epoch 10 \
      --percentile $W_PERCENTILE \
      --quantile $ACT_QUANTILE \
      --gpu $GPU_NUM
  else
    python main.py \
    --arch $MODEL \
    --epoch 100 \
    --lr 1e-3 \
    --batch 128 \
    --dataset $DATASET \
    --dnn_path $MODEL_PATH \
    --run_mode $MODE \
    --smoooth 0.99 \
    --bit 8 \
    --bit_addcat 8 \
    --symmetric \
    --per_channel \
    --channel_epoch 10 \
    --percentile $W_PERCENTILE \
    --quantile $ACT_QUANTILE \
    --gpu $GPU_NUM
  fi

elif [ "$MODE" = uniform ]; then
  if [ "$MODEL" = alexnet ]; then
    python main.py \
      --arch $MODEL \
      --epoch 100 \
      --lr 1e-3 \
      --batch 128 \
      --dataset $DATASET \
      --dnn_path $MODEL_PATH \
      --run_mode $MODE \
      --smoooth 0.99 \
      --bit 8 \
      --symmetric \
      --gpu $GPU_NUM
  else
    python main.py \
    --arch $MODEL \
    --epoch 100 \
    --lr 1e-3 \
    --batch 128 \
    --dataset $DATASET \
    --dnn_path $MODEL_PATH \
    --run_mode $MODE \
    --smoooth 0.99 \
    --bit 8 \
    --bit_addcat 8 \
    --symmetric \
    --per_channel \
    --gpu $GPU_NUM
  fi

else
  if [ "$MODEL" = alexnet ]; then
    python main.py \
      --arch $MODEL \
      --epoch 100 \
      --lr 1e-3 \
      --batch 128 \
      --dataset $DATASET \
      --dnn_path $MODEL_PATH \
      --run_mode $MODE \
      --smoooth 0.99 \
      --bit 32 \
      --gpu $GPU_NUM
  else
    python main.py \
    --arch $MODEL \
    --epoch 100 \
    --lr 1e-3 \
    --batch 128 \
    --dataset $DATASET \
    --dnn_path $MODEL_PATH \
    --run_mode $MODE \
    --smoooth 0.99 \
    --bit 32 \
    --bit_addcat 32 \
    --gpu $GPU_NUM
  fi
fi






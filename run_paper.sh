#! /bin/bash

###################################################
GPU_NUM=${1}
MODEL=${2}         # alexnet / resnet
DATASET=${3}       # cifar10 / cifar100
MODE=${4}
QUANTILE=${5}    # 0 ~ 1
COMPRESSION=${6} # PERCENTAGE

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
      --channel_epoch 100 \
      --quantile $QUANTILE \
      --compression_ratio $COMPRESSION \
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
    --fold_convbn \
    --bit 8 \
    --symmetric \
    --channel_epoch 100 \
    --quantile $QUANTILE \
    --compression_ratio $COMPRESSION \
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
    --fold_convbn \
    --bit 8 \
    --symmetric \
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






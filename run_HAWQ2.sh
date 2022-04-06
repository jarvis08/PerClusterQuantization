#! /bin/bash

export TORCH_WARN_ONCE

PRETRAINED_MODEL_PATH="/workspace/pretrained_models"
CLUSTERING_MODEL_PATH="result/kmeans/cifar100/nnac_resnet20.k4.sub8.part2.mean.topk_3.sim_0.7/"

MODEL="resnet20"
DATASET="cifar100"
PRETRAINED_MODEL="resnet20"

BATCH=128

CUDA_VISIBLE_DEVICES=0 python main.py \
    --mode fine \
    --epochs 100 \
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
    --cluster 4 \
    --sub_cluster 8 \
    --clustering_path $CLUSTERING_MODEL_PATH \
    --repr_method mean \
    --similarity_method jaccard \
    --data $DATASET \
    --batch-size $BATCH \
    --transfer_param \
    --dnn_path $PRETRAINED_MODEL_PATH/$DATASET/$PRETRAINED_MODEL/checkpoint.pth \

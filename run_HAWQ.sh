#! /bin/bash

PRETRAINED_MODEL_PATH="/workspace/pretrained_models"




###################################################

GPU_NUM=0

# alexnet / resnet20 / resnet50 / densenet121
MODEL=""
 # svhn / cifar10 / cifar100 / imagenet           
DATASET=""

CLUSTER=                # 16 / 8 / 4 / 2
SUB_CLUSTER=            # 32 / 16 / 8 / 4
SIM_METHOD=""           # and / jaccard
REPR_METHOD="max"       # FIXED TO MAX

FIRST_RUN=true          # true / false

BATCH=128               # 128 / 64 / 32
LEARNING_RATE=0.001     # 0.001 / 0.0001      

#####################################################


if [ -z ${CLUSTER} ]; then
    CUDA_VISIBLE_DEVICES=${GPU_NUM} python main.py \
        --mode fine \
        --epochs 100 \
        --batch $BATCH \
        --quant_base hawq \
        --arch $MODEL \
        --dataset $DATASET \
        --lr $LEARNING_RATE \
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
        --dnn_path $PRETRAINED_MODEL_PATH/$DATASET/$MODEL/checkpoint.pth
else
    if [ -z ${SUB_CLUSTER} ]; then
        if [ "$FIRST_RUN" = true ]; then
            CUDA_VISIBLE_DEVICES=${GPU_NUM} python main.py \
                --mode fine \
                --epochs 100 \
                --batch $BATCH \
                --quant_base hawq \
                --arch $MODEL \
                --dataset $DATASET \
                --lr $LEARNING_RATE \
                --act-range-momentum 0.99 \
                --wd 1e-4 \
                --fix-BN \
                --pretrained \
                --channel-wise true \
                --quant-scheme uniform4 \
                --gpu 0 \
                --cluster ${CLUSTER} \
                --repr_method ${REPR_METHOD} \
                --data $DATASET \
                --batch-size $BATCH \
                --transfer_param \
                --dnn_path $PRETRAINED_MODEL_PATH/$DATASET/$MODEL/checkpoint.pth
        else
            CLUSTERING_MODEL_PATH="result/kmeans/$MODEL/$DATASET/k${CLUSTER}.part2.${REPR_METHOD}"
            CUDA_VISIBLE_DEVICES=${GPU_NUM} python main.py \
                --mode fine \
                --epochs 100 \
                --batch $BATCH \
                --quant_base hawq \
                --arch $MODEL \
                --dataset $DATASET \
                --lr $LEARNING_RATE \
                --act-range-momentum 0.99 \
                --wd 1e-4 \
                --fix-BN \
                --pretrained \
                --channel-wise true \
                --quant-scheme uniform4 \
                --gpu 0 \
                --cluster ${CLUSTER} \
                --repr_method ${REPR_METHOD} \
                --clustering_path ${CLUSTERING_MODEL_PATH} \
                --data $DATASET \
                --batch-size $BATCH \
                --transfer_param \
                --dnn_path $PRETRAINED_MODEL_PATH/$DATASET/$MODEL/checkpoint.pth
        fi
    else
        if [ "$FIRST_RUN" = true ]; then            
            CUDA_VISIBLE_DEVICES=${GPU_NUM} python main.py \
                --mode fine \
                --epochs 100 \
                --batch $BATCH \
                --quant_base hawq \
                --arch $MODEL \
                --dataset $DATASET \
                --lr $LEARNING_RATE \
                --act-range-momentum 0.99 \
                --wd 1e-4 \
                --fix-BN \
                --pretrained \
                --channel-wise true \
                --quant-scheme uniform4 \
                --gpu 0 \
                --cluster ${CLUSTER} \
                --repr_method ${REPR_METHOD} \
                --sub_cluster ${SUB_CLUSTER} \
                --nnac true \
                --similarity_method ${SIM_METHOD} \
                --data $DATASET \
                --batch-size $BATCH \
                --transfer_param \
                --dnn_path $PRETRAINED_MODEL_PATH/$DATASET/$MODEL/checkpoint.pth \
        else
            CLUSTERING_MODEL_PATH="result/kmeans/$MODEL/$DATASET/k${CLUSTER}.part2.${REPR_METHOD}.sub${SUB_CLUSTER}.topk_3.sim_0.7.${SIM_METHOD}/"
            CUDA_VISIBLE_DEVICES=${GPU_NUM} python main.py \
                --mode fine \
                --epochs 100 \
                --batch $BATCH \
                --quant_base hawq \
                --arch $MODEL \
                --dataset $DATASET \
                --lr $LEARNING_RATE \
                --act-range-momentum 0.99 \
                --wd 1e-4 \
                --fix-BN \
                --pretrained \
                --channel-wise true \
                --quant-scheme uniform4 \
                --gpu 0 \
                --cluster ${CLUSTER} \
                --repr_method ${REPR_METHOD} \
                --clustering_path ${CLUSTERING_MODEL_PATH} \
                --sub_cluster ${SUB_CLUSTER} \
                --nnac true \
                --similarity_method ${SIM_METHOD} \
                --data $DATASET \
                --batch-size $BATCH \
                --transfer_param \
                --dnn_path $PRETRAINED_MODEL_PATH/$DATASET/$MODEL/checkpoint.pth
        fi
    fi
fi
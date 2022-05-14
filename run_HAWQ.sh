#! /bin/bash

PRETRAINED_MODEL_PATH="/workspace/pretrained_models"




###################################################

GPU_NUM=${1}

# alexnet / resnet20 / resnet50 / densenet121
MODEL=${2}
# svhn / cifar10 / cifar100 / imagenet           
DATASET=${3}

BATCH=${4}               # 128 / 64 / 32
LEARNING_RATE=${5}     # 0.001 / 0.0001      

FIRST_RUN=${6}          # true / false

CLUSTER=${7}                # 16 / 8 / 4 / 2
SUB_CLUSTER=${8}            # 32 / 16 / 8 / 4
SIM_METHOD=${9}           # and / jaccard
REPR_METHOD="max"       # FIXED TO MAX

#####################################################


if [ -z ${CLUSTER} ]; then
    if [ "$MODEL" = resnet20 ]; then
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
            --batch-size $BATCH
    else 
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
    fi
else
    if [ -z ${SUB_CLUSTER} ]; then
        if [ "$FIRST_RUN" = true ]; then            
            if [ "$MODEL" = resnet20 ]; then
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
                    --batch-size $BATCH
            else
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
            fi
        else
            CLUSTERING_MODEL_PATH="/workspace/PerClusterQuantization/result/kmeans/$MODEL/$DATASET/k${CLUSTER}.part2.${REPR_METHOD}/"
            if [ "$MODEL" = resnet20 ]; then
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
                    --batch-size $BATCH
            else
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
        fi
    else
        if [ "$FIRST_RUN" = true ]; then            
            if [ "$MODEL" = resnet20 ]; then
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
		            --max_method tmp \
                    --sub_cluster ${SUB_CLUSTER} \
                    --nnac true \
                    --similarity_method ${SIM_METHOD} \
                    --data $DATASET \
                    --batch-size $BATCH
            else
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
		            --max_method tmp \
                    --sub_cluster ${SUB_CLUSTER} \
                    --nnac true \
                    --similarity_method ${SIM_METHOD} \
                    --data $DATASET \
                    --batch-size $BATCH \
                    --transfer_param \
                    --dnn_path $PRETRAINED_MODEL_PATH/$DATASET/$MODEL/checkpoint.pth
            fi
        else
            CLUSTERING_MODEL_PATH="/workspace/PerClusterQuantization/result/kmeans/$MODEL/$DATASET/k${CLUSTER}.part2.${REPR_METHOD}.sub${SUB_CLUSTER}.topk_3.sim_0.7.${SIM_METHOD}/"
            if [ "$MODEL" = resnet20 ]; then
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
            else
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
fi

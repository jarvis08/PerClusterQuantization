#! /bin/bash

PRETRAINED_MODEL_PATH="/workspace/pretrained_models"
IMAGENET_PATH="/home/work/.Datasets/Imagenet2012"
# IMAGENET_PATH="/workspace/dataset"



###################################################

GPU_NUM=${1}

# alexnet / resnet20 / resnet50 / densenet121
MODEL=${2}
# svhn / cifar10 / cifar100 / imagenet           
DATASET=${3}
BATCH=${4}             # 256 / 128 / 64 / 32
LEARNING_RATE=${5}     # 0.001 / 0.0001      

FIRST_RUN=${6}         # true / false
CLUSTER=${7}           # Final Clusters : 1024 / 512 / 256 / 128 / 64 / 32 / 16 / 8 / 4 / 2

SUB_CLUSTER=${8}       # Initial Clusters : 1024 / 512 / 256 / 128 / 64 / 32 / 16 / 8 / 4 / 2
SIM_METHOD=${9}        # and / jaccard
REPR_METHOD="max"      # FIXED TO MAX
MERGE_METHOD="mean"
MERGED=${10}           # true / false


#####################################################

if [ -z ${CLUSTER} ]; then
    if [ "$DATASET" = imagenet ]; then
        CUDA_VISIBLE_DEVICES=${GPU_NUM} python main.py \
            --mode fine \
            --epochs 40 \
            --batch $BATCH \
            --quant_base hawq \
            --arch $MODEL \
            --dataset $DATASET \
            --lr $LEARNING_RATE \
            --act-range-momentum 0.99 \
            --wd 1e-4 \
            --pretrained \
            --channel-wise true \
            --quant-scheme uniform4 \
            --gpu 0 \
            --data $DATASET \
            --imagenet $IMAGENET_PATH
    elif [ "$MODEL" = resnet20 ]; then
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
    if [ "$FIRST_RUN" = false ]; then   
        if [ "$MERGED" = true ]; then
            CLUSTERING_MODEL_PATH="/workspace/PerClusterQuantization/result/kmeans/$MODEL/$DATASET/k${SUB_CLUSTER}.part2.${REPR_METHOD}/__.k${CLUSTER}.sub${SUB_CLUSTER}.topk_3.sim_0.7.${SIM_METHOD}/"
        else
            if [ -z ${SUB_CLUSTER} ]; then
                CLUSTERING_MODEL_PATH="/workspace/PerClusterQuantization/result/kmeans/$MODEL/$DATASET/k${CLUSTER}.part2.${REPR_METHOD}/"
            else
                CLUSTERING_MODEL_PATH="/workspace/PerClusterQuantization/result/kmeans/$MODEL/$DATASET/k${SUB_CLUSTER}.part2.${REPR_METHOD}/"
            fi
        fi
    fi

    if [ -z ${SUB_CLUSTER} ]; then
        if [ "$FIRST_RUN" = true ]; then            
            if [ "$DATASET" = imagenet ]; then
                CUDA_VISIBLE_DEVICES=${GPU_NUM} python main.py \
                    --mode fine \
                    --epochs 40 \
                    --batch $BATCH \
                    --quant_base hawq \
                    --arch $MODEL \
                    --dataset $DATASET \
                    --lr $LEARNING_RATE \
                    --act-range-momentum 0.99 \
                    --wd 1e-4 \
                    --pretrained \
                    --channel-wise true \
                    --quant-scheme uniform4 \
                    --gpu 0 \
                    --cluster ${CLUSTER} \
                    --repr_method ${REPR_METHOD} \
                    --data $DATASET \
                    --batch-size $BATCH \
                    --imagenet $IMAGENET_PATH
            elif [ "$MODEL" = resnet20 ]; then
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
            if [ "$DATASET" = imagenet ]; then
                CUDA_VISIBLE_DEVICES=${GPU_NUM} python main.py \
                    --mode fine \
                    --epochs 40 \
                    --batch $BATCH \
                    --quant_base hawq \
                    --arch $MODEL \
                    --dataset $DATASET \
                    --lr $LEARNING_RATE \
                    --act-range-momentum 0.99 \
                    --wd 1e-4 \
                    --pretrained \
                    --channel-wise true \
                    --quant-scheme uniform4 \
                    --gpu 0 \
                    --cluster ${CLUSTER} \
                    --repr_method ${REPR_METHOD} \
                    --clustering_path ${CLUSTERING_MODEL_PATH} \
                    --data $DATASET \
                    --batch-size $BATCH \
                    --imagenet $IMAGENET_PATH
            elif [ "$MODEL" = resnet20 ]; then
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
            if [ "$DATASET" = imagenet ]; then  
                CUDA_VISIBLE_DEVICES=${GPU_NUM} python main.py \
                    --mode fine \
                    --epochs 40 \
                    --batch $BATCH \
                    --quant_base hawq \
                    --arch $MODEL \
                    --dataset $DATASET \
                    --lr $LEARNING_RATE \
                    --act-range-momentum 0.99 \
                    --wd 1e-4 \
                    --pretrained \
                    --channel-wise true \
                    --quant-scheme uniform4 \
                    --gpu 0 \
                    --cluster ${CLUSTER} \
                    --repr_method ${REPR_METHOD} \
                    --max_method ${MERGE_METHOD} \
                    --sub_cluster ${SUB_CLUSTER} \
                    --nnac true \
                    --similarity_method ${SIM_METHOD} \
                    --data $DATASET \
                    --batch-size $BATCH \
                    --imagenet $IMAGENET_PATH
            elif [ "$MODEL" = resnet20 ]; then
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
                    --pretrained \
                    --channel-wise true \
                    --quant-scheme uniform4 \
                    --gpu 0 \
                    --cluster ${CLUSTER} \
                    --repr_method ${REPR_METHOD} \
                    --max_method ${MERGE_METHOD} \
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
                    --pretrained \
                    --channel-wise true \
                    --quant-scheme uniform4 \
                    --gpu 0 \
                    --cluster ${CLUSTER} \
                    --repr_method ${REPR_METHOD} \
                    --max_method ${MERGE_METHOD} \
                    --sub_cluster ${SUB_CLUSTER} \
                    --nnac true \
                    --similarity_method ${SIM_METHOD} \
                    --data $DATASET \
                    --batch-size $BATCH \
                    --transfer_param \
                    --dnn_path $PRETRAINED_MODEL_PATH/$DATASET/$MODEL/checkpoint.pth
            fi
        else
            if [ "$DATASET" = imagenet ]; then
                CUDA_VISIBLE_DEVICES=${GPU_NUM} python main.py \
                    --mode fine \
                    --epochs 40 \
                    --batch $BATCH \
                    --quant_base hawq \
                    --arch $MODEL \
                    --dataset $DATASET \
                    --lr $LEARNING_RATE \
                    --act-range-momentum 0.99 \
                    --wd 1e-4 \
                    --pretrained \
                    --channel-wise true \
                    --quant-scheme uniform4 \
                    --gpu 0 \
                    --cluster ${CLUSTER} \
                    --repr_method ${REPR_METHOD} \
                    --clustering_path ${CLUSTERING_MODEL_PATH} \
                    --sub_cluster ${SUB_CLUSTER} \
                    --max_method ${MERGE_METHOD} \
                    --nnac true \
                    --similarity_method ${SIM_METHOD} \
                    --data $DATASET \
                    --batch-size $BATCH \
                    --imagenet $IMAGENET_PATH
            elif [ "$MODEL" = resnet20 ]; then
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
                    --pretrained \
                    --channel-wise true \
                    --quant-scheme uniform4 \
                    --gpu 0 \
                    --cluster ${CLUSTER} \
                    --repr_method ${REPR_METHOD} \
                    --clustering_path ${CLUSTERING_MODEL_PATH} \
                    --sub_cluster ${SUB_CLUSTER} \
                    --max_method ${MERGE_METHOD} \
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
                    --pretrained \
                    --channel-wise true \
                    --quant-scheme uniform4 \
                    --gpu 0 \
                    --cluster ${CLUSTER} \
                    --repr_method ${REPR_METHOD} \
                    --clustering_path ${CLUSTERING_MODEL_PATH} \
                    --sub_cluster ${SUB_CLUSTER} \
                    --max_method ${MERGE_METHOD} \
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

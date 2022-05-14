#! /bin/bash

PRETRAINED_MODEL_PATH="/workspace/pretrained_models"

IMAGENET_DATASET_PATH="/workspace/dataset"


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
    if [ "$DATASET" = "imagenet" ]; then
        CUDA_VISIBLE_DEVICES=${GPU_NUM} python main.py \
            --mode fine \
            --epochs 100 \
            --batch $BATCH \
            --quant_base qat \
            --arch $MODEL \
            --dataset $DATASET \
            --lr $LEARNING_RATE \
            --smooth 0.99 \
            --bit 4 \
            --bit_first 8 \
            --bit_classifier 8 \
            --bit_addcat 4 \
            --per_channel \
	        --undo_gema \
            --imagenet $IMAGENET_DATASET_PATH 
    elif [ "$MODEL" = resnet20 ]; then
        CUDA_VISIBLE_DEVICES=${GPU_NUM} python main.py \
            --mode fine \
            --epochs 100 \
            --batch $BATCH \
            --quant_base qat \
            --arch $MODEL \
            --dataset $DATASET \
            --lr $LEARNING_RATE \
            --smooth 0.99 \
            --bit 4 \
            --bit_first 8 \
            --bit_classifier 8 \
            --bit_addcat 4 \
	        --undo_gema \
            --torchcv
    else
        CUDA_VISIBLE_DEVICES=${GPU_NUM} python main.py \
            --mode fine \
            --epochs 100 \
            --batch $BATCH \
            --quant_base qat \
            --arch $MODEL \
            --dataset $DATASET \
            --lr $LEARNING_RATE \
            --smooth 0.99 \
            --bit 4 \
            --bit_first 8 \
            --bit_classifier 8 \
            --bit_addcat 4 \
	        --undo_gema \
            --dnn_path $PRETRAINED_MODEL_PATH/$DATASET/$MODEL/checkpoint.pth
    fi
else
    if [ -z ${SUB_CLUSTER} ]; then
        if [ "$FIRST_RUN" = true ]; then            
            if [ "$DATASET" = "imagenet" ]; then
                CUDA_VISIBLE_DEVICES=${GPU_NUM} python main.py \
                    --mode fine \
                    --epochs 100 \
                    --batch $BATCH \
                    --quant_base qat \
                    --arch $MODEL \
                    --dataset $DATASET \
                    --lr $LEARNING_RATE \
                    --smooth 0.99 \
                    --bit 4 \
                    --bit_first 8 \
                    --bit_classifier 8 \
                    --bit_addcat 4 \
                    --undo_gema \
                    --cluster ${CLUSTER} \
                    --repr_method ${REPR_METHOD} \
                    --imagenet $IMAGENET_DATASET_PATH 
            elif [ "$MODEL" = resnet20 ]; then
                CUDA_VISIBLE_DEVICES=${GPU_NUM} python main.py \
                    --mode fine \
                    --epochs 100 \
                    --batch $BATCH \
                    --quant_base qat \
                    --arch $MODEL \
                    --dataset $DATASET \
                    --lr $LEARNING_RATE \
                    --smooth 0.99 \
                    --bit 4 \
                    --bit_first 8 \
                    --bit_classifier 8 \
                    --bit_addcat 4 \
                    --undo_gema \
                    --torchcv \
                    --cluster ${CLUSTER} \
                    --repr_method ${REPR_METHOD}
            else
                CUDA_VISIBLE_DEVICES=${GPU_NUM} python main.py \
                    --mode fine \
                    --epochs 100 \
                    --batch $BATCH \
                    --quant_base qat \
                    --arch $MODEL \
                    --dataset $DATASET \
                    --lr $LEARNING_RATE \
                    --smooth 0.99 \
                    --bit 4 \
                    --bit_first 8 \
                    --bit_classifier 8 \
                    --bit_addcat 4 \
                    --undo_gema \
                    --cluster ${CLUSTER} \
                    --repr_method ${REPR_METHOD}
                    --dnn_path $PRETRAINED_MODEL_PATH/$DATASET/$MODEL/checkpoint.pth
            fi
        else
            CLUSTERING_MODEL_PATH="/workspace/PerClusterQuantization/result/kmeans/$MODEL/$DATASET/k${CLUSTER}.part2.${REPR_METHOD}/"
            if [ "$DATASET" = "imagenet" ]; then
                CUDA_VISIBLE_DEVICES=${GPU_NUM} python main.py \
                    --mode fine \
                    --epochs 100 \
                    --batch $BATCH \
                    --quant_base qat \
                    --arch $MODEL \
                    --dataset $DATASET \
                    --lr $LEARNING_RATE \
                    --smooth 0.99 \
                    --bit 4 \
                    --bit_first 8 \
                    --bit_classifier 8 \
                    --bit_addcat 4 \
                    --undo_gema \
                    --clustering_path ${CLUSTERING_MODEL_PATH} \
                    --cluster ${CLUSTER} \
                    --repr_method ${REPR_METHOD}
                    --imagenet $IMAGENET_DATASET_PATH 
            elif [ "$MODEL" = resnet20 ]; then
                CUDA_VISIBLE_DEVICES=${GPU_NUM} python main.py \
                    --mode fine \
                    --epochs 100 \
                    --batch $BATCH \
                    --quant_base qat \
                    --arch $MODEL \
                    --dataset $DATASET \
                    --lr $LEARNING_RATE \
                    --smooth 0.99 \
                    --bit 4 \
                    --bit_first 8 \
                    --bit_classifier 8 \
                    --bit_addcat 4 \
                    --undo_gema \
                    --torchcv \
                    --clustering_path ${CLUSTERING_MODEL_PATH} \
                    --cluster ${CLUSTER} \
                    --repr_method ${REPR_METHOD}
            else
                CUDA_VISIBLE_DEVICES=${GPU_NUM} python main.py \
                    --mode fine \
                    --epochs 100 \
                    --batch $BATCH \
                    --quant_base qat \
                    --arch $MODEL \
                    --dataset $DATASET \
                    --lr $LEARNING_RATE \
                    --smooth 0.99 \
                    --bit 4 \
                    --bit_first 8 \
                    --bit_classifier 8 \
                    --bit_addcat 4 \
                    --undo_gema \
                    --clustering_path ${CLUSTERING_MODEL_PATH} \
                    --cluster ${CLUSTER} \
                    --repr_method ${REPR_METHOD}
                    --dnn_path $PRETRAINED_MODEL_PATH/$DATASET/$MODEL/checkpoint.pth
            fi
        fi
    else
        if [ "$FIRST_RUN" = true ]; then            
            if [ "$DATASET" = "imagenet" ]; then
                CUDA_VISIBLE_DEVICES=${GPU_NUM} python main.py \
                    --mode fine \
                    --epochs 100 \
                    --batch $BATCH \
                    --quant_base qat \
                    --arch $MODEL \
                    --dataset $DATASET \
                    --lr $LEARNING_RATE \
                    --smooth 0.99 \
                    --bit 4 \
                    --bit_first 8 \
                    --bit_classifier 8 \
                    --bit_addcat 4 \
                    --undo_gema \
                    --cluster ${CLUSTER} \
                    --repr_method ${REPR_METHOD} \
                    --sub_cluster ${SUB_CLUSTER} \
                    --nnac true \
                    --similarity_method ${SIM_METHOD} \
                    --imagenet $IMAGENET_DATASET_PATH 
            elif [ "$MODEL" = resnet20 ]; then
                CUDA_VISIBLE_DEVICES=${GPU_NUM} python main.py \
                    --mode fine \
                    --epochs 100 \
                    --batch $BATCH \
                    --quant_base qat \
                    --arch $MODEL \
                    --dataset $DATASET \
                    --lr $LEARNING_RATE \
                    --smooth 0.99 \
                    --bit 4 \
                    --bit_first 8 \
                    --bit_classifier 8 \
                    --bit_addcat 4 \
                    --undo_gema \
                    --torchcv \
                    --cluster ${CLUSTER} \
                    --repr_method ${REPR_METHOD} \
                    --sub_cluster ${SUB_CLUSTER} \
                    --nnac true \
                    --similarity_method ${SIM_METHOD}
            else
                CUDA_VISIBLE_DEVICES=${GPU_NUM} python main.py \
                    --mode fine \
                    --epochs 100 \
                    --batch $BATCH \
                    --quant_base qat \
                    --arch $MODEL \
                    --dataset $DATASET \
                    --lr $LEARNING_RATE \
                    --smooth 0.99 \
                    --bit 4 \
                    --bit_first 8 \
                    --bit_classifier 8 \
                    --bit_addcat 4 \
                    --undo_gema \
                    --cluster ${CLUSTER} \
                    --repr_method ${REPR_METHOD} \
                    --sub_cluster ${SUB_CLUSTER} \
                    --nnac true \
                    --similarity_method ${SIM_METHOD} \
                    --dnn_path $PRETRAINED_MODEL_PATH/$DATASET/$MODEL/checkpoint.pth
            fi
        else
            CLUSTERING_MODEL_PATH="/workspace/PerClusterQuantization/result/kmeans/$MODEL/$DATASET/k${CLUSTER}.part2.${REPR_METHOD}.sub${SUB_CLUSTER}.topk_3.sim_0.7.${SIM_METHOD}/"
            if [ "$DATASET" = "imagenet" ]; then
                CUDA_VISIBLE_DEVICES=${GPU_NUM} python main.py \
                    --mode fine \
                    --epochs 100 \
                    --batch $BATCH \
                    --quant_base qat \
                    --arch $MODEL \
                    --dataset $DATASET \
                    --lr $LEARNING_RATE \
                    --smooth 0.99 \
                    --bit 4 \
                    --bit_first 8 \
                    --bit_classifier 8 \
                    --bit_addcat 4 \
                    --undo_gema \
                    --clustering_path ${CLUSTERING_MODEL_PATH} \
                    --cluster ${CLUSTER} \
                    --repr_method ${REPR_METHOD} \
                    --sub_cluster ${SUB_CLUSTER} \
                    --nnac true \
                    --similarity_method ${SIM_METHOD} \
                    --imagenet $IMAGENET_DATASET_PATH 
            elif [ "$MODEL" = resnet20 ]; then
                CUDA_VISIBLE_DEVICES=${GPU_NUM} python main.py \
                    --mode fine \
                    --epochs 100 \
                    --batch $BATCH \
                    --quant_base qat \
                    --arch $MODEL \
                    --dataset $DATASET \
                    --lr $LEARNING_RATE \
                    --smooth 0.99 \
                    --bit 4 \
                    --bit_first 8 \
                    --bit_classifier 8 \
                    --bit_addcat 4 \
                    --undo_gema \
                    --torchcv \
                    --clustering_path ${CLUSTERING_MODEL_PATH} \
                    --cluster ${CLUSTER} \
                    --repr_method ${REPR_METHOD} \
                    --sub_cluster ${SUB_CLUSTER} \
                    --nnac true \
                    --similarity_method ${SIM_METHOD}
            else
                CUDA_VISIBLE_DEVICES=${GPU_NUM} python main.py \
                    --mode fine \
                    --epochs 100 \
                    --batch $BATCH \
                    --quant_base qat \
                    --arch $MODEL \
                    --dataset $DATASET \
                    --lr $LEARNING_RATE \
                    --smooth 0.99 \
                    --bit 4 \
                    --bit_first 8 \
                    --bit_classifier 8 \
                    --bit_addcat 4 \
                    --undo_gema \
                    --clustering_path ${CLUSTERING_MODEL_PATH} \
                    --cluster ${CLUSTER} \
                    --repr_method ${REPR_METHOD} \
                    --sub_cluster ${SUB_CLUSTER} \
                    --nnac true \
                    --similarity_method ${SIM_METHOD} \
                    --dnn_path $PRETRAINED_MODEL_PATH/$DATASET/$MODEL/checkpoint.pth
            fi
        fi
    fi
fi

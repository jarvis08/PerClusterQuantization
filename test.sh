#! /bin/bash

SCRIPT_PATH="./run_HAWQ.sh"

GPU=0

MODELS=(resnet18 resnet50 densenet121 mobilenetv2_w1 inceptionv3)
DATAS=(imagenet)
LEARNING_RATES=(0.0001)
BATCH_SIZES=(128)

CLUSTERS=(128 256 512 1024)

for MODEL in ${MODELS[@]}; do
    for DATA in ${DATAS[@]}; do
        for BATCH_SIZE in ${BATCH_SIZES[@]}; do
            for LEARNING_RATE in ${LEARNING_RATES[@]}; do
                for CLUSTER in ${CLUSTERS[@]}; do
                    /bin/bash ${SCRIPT_PATH} ${GPU} ${MODEL} ${DATA} ${BATCH_SIZE} ${LEARNING_RATE} false ${CLUSTER}
                done
            done
        done
    done
done

# SUB_CLUSTERS=(128 256 512 1024)
# FIN_CLUSTERS=(15 14 13 12)

# for MODEL in ${MODELS[@]}; do
#     for DATA in ${DATAS[@]}; do
#         for BATCH_SIZE in ${BATCH_SIZES[@]}; do
#             for LEARNING_RATE in ${LEARNING_RATES[@]}; do
#                 for SUB_CLUSTER in ${SUB_CLUSTERS[@]}; do
#                     for FIN_CLUSTER in ${FIN_CLUSTERS[@]}; do
#                         /bin/bash ${SCRIPT_PATH} ${GPU} ${MODEL} ${DATA} ${BATCH_SIZE} ${LEARNING_RATE} false $((SUB_CLUSTER/16*FIN_CLUSTER)) ${SUB_CLUSTER} jaccard true & wait
#                     done
#                 done
#             done
#         done
#     done
# done

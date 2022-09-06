#! /bin/bash

SCRIPT_PATH="/workspace/PerClusterQuantization/run_HAWQ.sh"

GPU=${1}

MODELS=(alexnet resnet20)
DATAS=(cifar10 cifar100 svhn)
CLUSTERS=(2 4 8 16 32 64 128 256 512 1024)
LEARNING_RATES=(0.0001)
BATCH_SIZES=(128)

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
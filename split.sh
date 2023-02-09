#! /bin/bash

SCRIPT_PATH="/workspace/PerClusterQuantization/run_SKT.sh"

GPU=${1}

MODELS=(resnet20)
DATAS=(cifar10)
LEARNING_RATES=(0.0001)
ITERS=(iter)
INTERVALS=(1)
RANGES=(0.5 0.55 0.6 0.65 0.7)
MANIPULATIONS=${2}

for MODEL in ${MODELS[@]}; do
    for DATA in ${DATAS[@]}; do
        for LEARNING_RATE in ${LEARNING_RATES[@]}; do
            for ITER in ${ITERS[@]}; do
                for INTERVAL in ${INTERVALS[@]}; do
                    for RANGE in ${RANGES[@]}; do
                        for MANIPULATION in ${MANIPULATIONS[@]}; do
                            /bin/bash ${SCRIPT_PATH} ${GPU} ${MODEL} ${DATA} ${LEARNING_RATE} true ${ITER} ${INTERVAL} ${RANGE} ${MANIPULATION}
                        done
                    done
                done
            done
        done
    done
done

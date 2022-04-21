###################################################
PRETRAINED_MODEL_PATH="pretrained_models"
GPU_NUM=0

# alexnet / resnet20 / resnet50 / densenet121
MODEL="alexnet"
# svhn / cifar10 / cifar100 / imagenet           
DATASET="cifar10"

CLUSTER=4                # 16 / 8 / 4 / 2
SUB_CLUSTER=8            # 32 / 16 / 8 / 4
SIM_METHOD="and"           # and / jaccard
REPR_METHOD="max"       # FIXED TO MAX

FIRST_RUN=true          # true / false

BATCH=128               # 128 / 64 / 32
LEARNING_RATE=0.001     # 0.001 / 0.0001      

#####################################################

CUDA_VISIBLE_DEVICES=${GPU_NUM} python main.py \
                --mode fine \
                --epochs 1 \
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
                --checkpoint-iter -1 \
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

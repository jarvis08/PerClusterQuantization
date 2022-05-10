python main.py --mode fine --arch alexnet --epoch 100 --lr 1e-4 --batch 128 --dataset cifar10 --smooth 0.99 --bit 4 --repr_method max --cluster 8 --bit_addcat 4 --bit_first 8 --bit_classifier 8 --dnn_path pretrained_models/cifar10/alexnet/checkpoint.pth
python main.py --mode fine --arch alexnet --epoch 100 --lr 1e-4 --batch 128 --dataset cifar10 --smooth 0.99 --bit 4 --repr_method mean --cluster 8 --bit_addcat 4 --bit_first 8 --bit_classifier 8 --dnn_path pretrained_models/cifar10/alexnet/checkpoint.pth

python main.py --mode fine --arch alexnet --epoch 100 --lr 1e-4 --batch 128 --dataset cifar100 --smooth 0.99 --bit 4 --repr_method max --cluster 8 --bit_addcat 4 --bit_first 8 --bit_classifier 8 --dnn_path pretrained_models/cifar100/alexnet/checkpoint.pth
python main.py --mode fine --arch alexnet --epoch 100 --lr 1e-4 --batch 128 --dataset cifar100 --smooth 0.99 --bit 4 --repr_method mean --cluster 8 --bit_addcat 4 --bit_first 8 --bit_classifier 8 --dnn_path pretrained_models/cifar100/alexnet/checkpoint.pth

python main.py --mode fine --arch alexnet --epoch 100 --lr 1e-4 --batch 128 --dataset svhn --smooth 0.99 --bit 4 --repr_method max --cluster 8 --bit_addcat 4 --bit_first 8 --bit_classifier 8 --dnn_path pretrained_models/svhn/alexnet/checkpoint.pth
python main.py --mode fine --arch alexnet --epoch 100 --lr 1e-4 --batch 128 --dataset svhn --smooth 0.99 --bit 4 --repr_method mean --cluster 8 --bit_addcat 4 --bit_first 8 --bit_classifier 8 --dnn_path pretrained_models/svhn/alexnet/checkpoint.pth

python main.py --mode fine --arch resnet20 --epoch 100 --lr 1e-4 --batch 128 --dataset cifar10 --smooth 0.99 --bit 4 --repr_method max --cluster 8 --bit_addcat 4 --bit_first 8 --bit_classifier 8 --torchcv
python main.py --mode fine --arch resnet20 --epoch 100 --lr 1e-4 --batch 128 --dataset cifar10 --smooth 0.99 --bit 4 --repr_method mean --cluster 8 --bit_addcat 4 --bit_first 8 --bit_classifier 8 --torchcv

python main.py --mode fine --arch resnet20 --epoch 100 --lr 1e-4 --batch 128 --dataset cifar100 --smooth 0.99 --bit 4 --repr_method max --cluster 8 --bit_addcat 4 --bit_first 8 --bit_classifier 8 --torchcv
python main.py --mode fine --arch resnet20 --epoch 100 --lr 1e-4 --batch 128 --dataset cifar100 --smooth 0.99 --bit 4 --repr_method mean --cluster 8 --bit_addcat 4 --bit_first 8 --bit_classifier 8 --torchcv

python main.py --mode fine --arch resnet20 --epoch 100 --lr 1e-4 --batch 128 --dataset svhn --smooth 0.99 --bit 4 --repr_method max --cluster 8 --bit_addcat 4 --bit_first 8 --bit_classifier 8 --torchcv
python main.py --mode fine --arch resnet20 --epoch 100 --lr 1e-4 --batch 128 --dataset svhn --smooth 0.99 --bit 4 --repr_method mean --cluster 8 --bit_addcat 4 --bit_first 8 --bit_classifier 8 --torchcv

#python main.py --mode fine --arch resnet --lr 0.00001 --weight_decay 0.0 --epoch 2 --fq 1 --batch 64 --bit 4 --cluster 10 --kmeans_path kmeans/checkpoint.pkl --dnn_path pretrained/checkpoint.pth
#python main.py --mode fine --arch resnet --lr 0.00001 --weight_decay 0.0 --epoch 10 --fq 1 --batch 64 --bit 4 --cluster 10 --kmeans_path kmeans/checkpoint.pkl --dnn_path pretrained/checkpoint.pth
#python main.py --mode fine --arch resnet --lr 0.00001 --weight_decay 0.0 --epoch 20 --fq 1 --batch 64 --bit 4 --cluster 10 --kmeans_path kmeans/checkpoint.pkl --dnn_path pretrained/checkpoint.pth
#python main.py --mode fine --arch resnet --lr 0.00001 --weight_decay 0.0 --epoch 30 --fq 1 --batch 64 --bit 4 --cluster 10 --kmeans_path kmeans/checkpoint.pkl --dnn_path pretrained/checkpoint.pth
#python main.py --mode fine --arch resnet --lr 0.00001 --weight_decay 0.0 --epoch 40 --fq 1 --batch 64 --bit 4 --cluster 10 --kmeans_path kmeans/checkpoint.pkl --dnn_path pretrained/checkpoint.pth
#python main.py --mode fine --arch resnet --lr 0.00001 --weight_decay 0.0 --epoch 50 --fq 1 --batch 64 --bit 4 --cluster 10 --kmeans_path kmeans/checkpoint.pkl --dnn_path pretrained/checkpoint.pth
#python main.py --mode fine --arch resnet --lr 0.00001 --weight_decay 0.0 --epoch 60 --fq 1 --batch 64 --bit 4 --cluster 10 --kmeans_path kmeans/checkpoint.pkl --dnn_path pretrained/checkpoint.pth

#python main.py --mode eval --arch resnet --quantized True --bit 4  --cluster 10 --kmeans_path kmeans/checkpoint.pkl --dnn_path ./result/fine/ResNet20_4bit/07-22-2356/quantized/checkpoint.pth
#python main.py --mode eval --arch resnet --quantized True --bit 4  --cluster 10 --kmeans_path kmeans/checkpoint.pkl --dnn_path ./result/fine/ResNet20_4bit/07-23-0003/quantized/checkpoint.pth
#python main.py --mode eval --arch resnet --quantized True --bit 4  --cluster 10 --kmeans_path kmeans/checkpoint.pkl --dnn_path ./result/fine/ResNet20_4bit/07-23-0044/quantized/checkpoint.pth
#python main.py --mode eval --arch resnet --quantized True --bit 4  --cluster 10 --kmeans_path kmeans/checkpoint.pkl --dnn_path ./result/fine/ResNet20_4bit/07-23-0207/quantized/checkpoint.pth
#python main.py --mode eval --arch resnet --quantized True --bit 4  --cluster 10 --kmeans_path kmeans/checkpoint.pkl --dnn_path ./result/fine/ResNet20_4bit/07-23-0414/quantized/checkpoint.pth
#python main.py --mode eval --arch resnet --quantized True --bit 4  --cluster 10 --kmeans_path kmeans/checkpoint.pkl --dnn_path ./result/fine/ResNet20_4bit/07-23-0658/quantized/checkpoint.pth
#python main.py --mode eval --arch resnet --quantized True --bit 4  --cluster 10 --kmeans_path kmeans/checkpoint.pkl --dnn_path ./result/fine/ResNet20_4bit/07-23-1026/quantized/checkpoint.pth

#python main.py --mode fine --arch alexnet --lr 0.0001 --weight_decay 0.0 --epoch 20 --fq 1 --batch 8 --bit 4 --dnn_path ./alexnet_pretrained/checkpoint.pth
#python main.py --mode fine --arch alexnet --lr 0.0001 --weight_decay 0.0 --epoch 20 --fq 1 --batch 8 --bit 4 --dnn_path ./alexnet_pretrained/checkpoint.pth
#python main.py --mode fine --arch alexnet --lr 0.0001 --weight_decay 0.0 --epoch 20 --fq 1 --batch 8 --bit 4 --dnn_path ./alexnet_pretrained/checkpoint.pth
#python main.py --mode fine --arch alexnet --lr 0.0001 --weight_decay 0.0 --epoch 20 --fq 1 --batch 8 --bit 4 --dnn_path ./alexnet_pretrained/checkpoint.pth
#python main.py --mode fine --arch alexnet --lr 0.0001 --weight_decay 0.0 --epoch 20 --fq 1 --batch 8 --bit 4 --dnn_path ./alexnet_pretrained/checkpoint.pth

#python main.py --mode fine --arch alexnet --lr 0.0001 --weight_decay 0.0 --epoch 20 --fq 1 --batch 32 --bit 4 --cluster 10 --kmeans_path kmeans/checkpoint.pkl --dnn_path alexnet_pretrained/checkpoint.pth
#python main.py --mode fine --arch alexnet --lr 0.0001 --weight_decay 0.0 --epoch 20 --fq 1 --batch 32 --bit 4 --cluster 10 --kmeans_path kmeans/checkpoint.pkl --dnn_path alexnet_pretrained/checkpoint.pth
#python main.py --mode fine --arch alexnet --lr 0.0001 --weight_decay 0.0 --epoch 20 --fq 1 --batch 32 --bit 4 --cluster 10 --kmeans_path kmeans/checkpoint.pkl --dnn_path alexnet_pretrained/checkpoint.pth
#python main.py --mode fine --arch alexnet --lr 0.0001 --weight_decay 0.0 --epoch 20 --fq 1 --batch 32 --bit 4 --cluster 10 --kmeans_path kmeans/checkpoint.pkl --dnn_path alexnet_pretrained/checkpoint.pth
#python main.py --mode fine --arch alexnet --lr 0.0001 --weight_decay 0.0 --epoch 20 --fq 1 --batch 32 --bit 4 --cluster 10 --kmeans_path kmeans/checkpoint.pkl --dnn_path alexnet_pretrained/checkpoint.pth

#python main.py --mode fine --arch resnet --lr 0.0001 --weight_decay 0.0 --epoch 5 --fq 1 --batch 8 --bit 4 --dnn_path resnet_pretrained/checkpoint.pth
#python main.py --mode fine --arch resnet --lr 0.0001 --weight_decay 0.0 --epoch 50 --fq 1 --batch 64 --bit 4 --cluster 10 --kmeans_path kmeans/checkpoint.pkl --dnn_path resnet_pretrained/checkpoint.pth
#python main.py --mode fine --arch resnet --lr 0.0001 --weight_decay 0.0 --epoch 50 --fq 1 --batch 64 --bit 4 --cluster 10 --kmeans_path kmeans/checkpoint.pkl --dnn_path resnet_pretrained/checkpoint.pth
#python main.py --mode fine --arch resnet --lr 0.0001 --weight_decay 0.0 --epoch 50 --fq 1 --batch 32 --bit 4 --cluster 8 --kmeans_path kmeans/cluster_8/checkpoint.pkl --dnn_path resnet_pretrained/checkpoint.pth

python main.py --mode fine --arch resnet --lr 0.0001 --weight_decay 0.0 --epoch 5 --batch 16 --bit 4 --cluster 8 --kmeans_path /nvme/ken/mnt/PerClusterQuantization/kmeans/cluster_8/ --dnn_path /nvme/ken/mnt/PerClusterQuantization/resnet_pretrained/checkpoint.pth


#python main.py --mode fine --arch resnet --lr 0.0001 --weight_decay 0.0 --epoch 15 --batch 8 --bit 4 --dnn_path /nvme/ken/mnt/PerClusterQuantization/resnet_pretrained/checkpoint.pth
#python main.py --mode eval --arch resnet --quantized True --bit 4 --dnn_path ./result/fine/ResNet20_4bit/08-05-0143/quantized/checkpoint.pth

#python main.py --mode eval --arch resnet --quantized True --bit 4 --cluster 8 --kmeans_path /nvme/ken/mnt/PerClusterQuantization/kmeans/cluster_8/ --dnn_path /nvme/ken/mnt/PerClusterQuantization/result/fine/ResNet20_4bit/07-30-2139/quantized/checkpoint.pth
#python main.py --mode eval --arch resnet --dnn_path /nvme/ken/mnt/PerClusterQuantization/result/fine/ResNet20_4bit/07-30-2139/quantized/checkpoint.pth
#python main.py --mode eval --arch resnet --val_batch 256 --dnn_path resnet_pretrained/checkpoint.pth

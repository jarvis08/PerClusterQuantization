import argparse
import os
from utils.torch_dataset import get_data_loaders

parser = argparse.ArgumentParser(description='[PyTorch] Per Cluster Quantization')
parser.add_argument('--worker', default=4, type=int, help='Number of workers for input data loader')
parser.add_argument('--mode', default='fine', type=str, help="pre/fine/eval/lip")
parser.add_argument('--imagenet', default='', type=str, help="ImageNet dataset path")
parser.add_argument('--dataset', default='cifar10', type=str, help='Dataset to use')
parser.add_argument('--batch', default=128, type=int, help='Mini-batch size')
parser.add_argument('--val_batch', default=0, type=int, help='Validation batch size')

parser.add_argument('--quant_base', default='qat', type=str,
                    help='Among qat/qn/hawq, choose fine-tuning method to apply DAQ')
parser.add_argument('--nnac', action='store_true', help="Use Neural Network Aware Clustering")
parser.add_argument('--exclude', action='store_true', help="exclude less important layers during nnac")
parser.add_argument('--mixrate', default=3, type=int,
                    help='Number of epochs to mix augmented dataset to non-augmented dataset in training of clustering')
parser.add_argument('--sim_threshold', default=0.7, type=float,
                    help='Similarity threshold of ratio for considering similar clusters in nnac')
parser.add_argument('--max_method', default='', type=str,
                    help='max nnac')
parser.add_argument('--clustering_method', default='kmeans', type=str, help="Clustering method(K-means or BIRCH)")
parser.add_argument('--topk', default=3, type=int, help='Number of cluster combination candidates to choose in nnac')
parser.add_argument('--cluster', default=1, type=int, help='Number of clusters')
parser.add_argument('--sub_cluster', default=0, type=int, help='Number of sub-clusters in NN-aware Clustering')
parser.add_argument('--partition', default=2, type=int,
                    help="Number of partitions to divide per channel for clustering's input")
parser.add_argument('--partition_method', default='square', type=str, help="How to divide image into partitions")
parser.add_argument('--repr_method', default='mean', type=str, help="How to get representation per partition")
parser.add_argument('--similarity_method', default='and', type=str, help="How to measure similarity score")
parser.add_argument('--clustering_path', default='', type=str, help="Trained K-means clustering model's path")
parser.add_argument('--undo_gema', action='store_true',
                    help='Undo gema for DAQ models')

parser.add_argument('--kmeans_epoch', default=300, type=int, help='Max epoch of K-means model to train')
parser.add_argument('--kmeans_tol', default=0.0001, type=float, help="K-means model's tolerance to detect convergence")
parser.add_argument('--kmeans_init', default=10, type=int, help="Train K-means model n-times, and use the best model")
parser.add_argument('--visualize_clustering', action='store_true',
                    help="Visualize clustering result with PCA-ed training dataset")
parser.add_argument('--darknet', default=False, type=bool, help="Evaluate with dataset preprocessed in darknet")
parser.add_argument('--horovod', default=False, type=bool, help="Use distributed training with horovod")
args_daq, tmp = parser.parse_known_args()

arch = None
if '--arch' in tmp:
    arch = tmp[tmp.index('--arch') + 1]

if args_daq.imagenet:
    args_daq.dataset = 'imagenet'
if args_daq.dataset == 'cifar':
    args_daq.dataset = 'cifar10'
if not args_daq.val_batch:
    args_daq.val_batch = 256 if args_daq.dataset != 'imagenet' else 128

# # NN-aware Clustering
if args_daq.cluster > 1 and args_daq.sub_cluster:
    args_daq.nnac = True


if __name__ == '__main__':
    data_loaders = get_data_loaders(args_daq)
    clustering_model = None
    if args_daq.cluster > 1:
        from Clustering import get_clustering_model
        if not args_daq.clustering_path:
            from utils.misc import set_clustering_dir
            args_daq.clustering_path = set_clustering_dir(args_daq, arch)
            clustering_model = get_clustering_model(args_daq, data_loaders)
        else:
            clustering_model = get_clustering_model(args_daq)

    if args_daq.quant_base == 'qat':
        from QAT.qat import main
        main(args_daq, data_loaders, clustering_model)
    else:
        from HAWQ.quant_train import main
        main(args_daq, data_loaders, clustering_model)

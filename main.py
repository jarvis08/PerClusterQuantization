import argparse
import os

# from QAT.models import *
from Clustering import *
from pretrain import _pretrain
# from evaluate import _evaluate
# from utils.lipschitz import check_lipschitz

parser = argparse.ArgumentParser(description='[PyTorch] Per Cluster Quantization')
parser.add_argument('--quant_base', default='qat', type=str,
                    help='Among qat/qn/hawq, choose fine-tuning method to apply DAQ')
parser.add_argument('--nnac', action='store_true', help="Use Neural Network Aware Clustering")
parser.add_argument('--sim_threshold', default=0.7, type=float,
                    help='Similarity threshold of ratio for considering similar clusters in nnac')
parser.add_argument('--clustering_method', default='kmeans', type=str, help="Clustering method(K-means or BIRCH)")
parser.add_argument('--mixrate', default=2, type=int,
                    help='Number of epochs to mix augmented dataset to non-augmented dataset in training of clustering')
parser.add_argument('--cluster', default=1, type=int, help='Number of clusters')
parser.add_argument('--sub_cluster', default=0, type=int, help='Number of sub-clusters in NN-aware Clustering')
parser.add_argument('--partition', default=2, type=int,
                    help="Number of partitions to divide per channel for clustering's input")
parser.add_argument('--partition_method', default='square', type=str, help="How to divide image into partitions")
parser.add_argument('--repr_method', default='minmax', type=str, help="How to get representation per partition")
parser.add_argument('--clustering_path', default='', type=str, help="Trained K-means clustering model's path")

parser.add_argument('--kmeans_epoch', default=300, type=int, help='Max epoch of K-means model to train')
parser.add_argument('--kmeans_tol', default=0.0001, type=float, help="K-means model's tolerance to detect convergence")
parser.add_argument('--kmeans_init', default=10, type=int, help="Train K-means model n-times, and use the best model")
parser.add_argument('--visualize_clustering', action='store_true',
                    help="Visualize clustering result with PCA-ed training dataset")
parser.add_argument('--darknet', default=False, type=bool, help="Evaluate with dataset preprocessed in darknet")
parser.add_argument('--horovod', default=False, type=bool, help="Use distributed training with horovod")
args_daq, _ = parser.parse_known_args()

# # NN-aware Clustering
if args_daq.cluster > 1 and args_daq.sub_cluster:
    args_daq.nnac = True


if __name__ == '__main__':
    if args_daq.quant_base == 'qat':
        from QAT.qat import main
        main(args_daq)
    else:
        from HAWQ.quant_train import main
        main(args_daq)

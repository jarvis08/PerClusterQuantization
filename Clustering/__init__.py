from .kmeans import KMeansClustering
from .birch import BIRCH
from .mm_dist import MinMaxDistClustering


def get_clustering_model(args, data_loaders=None, index=None):
    if args.clustering_method == 'kmeans':
        clustering_model = KMeansClustering(args)
    elif args.clustering_method == 'dist':
        clustering_model = MinMaxDistClustering(args)
    else:
        clustering_model = BIRCH(args)

    if data_loaders is None:
        clustering_model.load_clustering_model()
    # else:
    #     clustering_model.train_clustering_model(data_loaders['non_aug_train'])
    return clustering_model

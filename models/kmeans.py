import torch

from sklearn.cluster import KMeans, MiniBatchKMeans
import numpy as np

from tqdm import tqdm
import joblib
from copy import deepcopy
import json
import os


class KMeansClustering(object):
    def __init__(self, args):
        self.args = args
        self.model = None

    @torch.no_grad()
    def get_partitioned_batch(self, data):
        # Under the premise that images are in the form of square matrix
        batch = data.size(0)
        channel = data.size(1)
        _size = data.size(2)
        if self.args.partition_method == 'square':
            n_part = self.args.partition
            n_data = int(_size / n_part)  # Per part

            data = data.view(batch, channel, n_part, n_data, _size).transpose(3, 4)
            data = data.reshape(batch, channel, n_part * n_part, -1)

            if self.args.repr_method == 'mean':
                rst = data.mean(-1, keepdim=True)
            else:
                _min = data.min(-1, keepdim=True).values
                _max = data.max(-1, keepdim=True).values
                rst = torch.cat((_min, _max), dim=-1)
            return rst.view(rst.size(0), -1).numpy()
        else:
            # To make clustering model more robust about augmentation's horizontal flip
            n_part = 4
            n_data = int(_size / n_part)
            rst = None
            for c in range(n_part):
                c_start = n_data * c
                part_data = data[:, :, c_start:c_start + n_data, :].reshape(batch, channel, -1)
                _min = part_data.min(-1, keepdim=True).values
                _max = part_data.max(-1, keepdim=True).values

                part_rst = torch.cat([_min, _max], dim=-1)
                if rst is None:
                    rst = part_rst
                else:
                    rst = torch.cat([rst, part_rst], dim=-1)
            return rst.view(rst.size(0), -1).numpy()

    def load_clustering_model(self):
        # Load k-means model's hparams, and check dependencies
        with open(os.path.join(self.args.clustering_path, 'params.json'), 'r') as f:
            saved_args = json.load(f)
        assert self.args.cluster == saved_args['k'], \
            "Current model's target # of clusters must be same with K-means model's it" \
            "\n(K-means model's = {}, Current model's = {}".format(saved_args['k'], self.args.cluster)
        assert self.args.partition == saved_args['num_partitions'], \
            "Current model's target # of partitions to divide an channel must be same with K-means model's it" \
            "\n(K-means model's = {}, Current model's = {}".format(saved_args['num_partitions'], self.args.partition)
        self.model = joblib.load(os.path.join(self.args.clustering_path, 'checkpoint.pkl'))
    
    def predict_cluster_of_batch(self, input):
        kmeans_input = self.get_partitioned_batch(input)
        cluster_info = self.model.predict(kmeans_input)
        return torch.LongTensor(cluster_info)

    def train_clustering_model(self, train_loader):
        best_model = None
        if self.args.dataset == 'imagenet':
            print(">> Use Mini-batch K-means Clustering for ImageNet dataset")
            def check_convergence(prev, cur, tol):
                """
                    Relative tolerance with regards to Frobenius norm of the difference in the cluster centers
                    of two consecutive iterations to declare convergence.
                """
                diff = np.subtract(prev, cur)
                normed = np.linalg.norm(diff)
                if normed > tol:
                    return False
                return True

            prev_centers = None
            is_converged = False
            best_model_inertia = 9999999999999999
            print("Train K-means model 10 times, and choose the best model")
            for trial in range(1, 11):
                model = MiniBatchKMeans(n_clusters=self.args.cluster, batch_size=self.args.batch, tol=self.args.kmeans_tol, random_state=0)
                early_stopped = False
                t_epoch = tqdm(total=self.args.kmeans_epoch, desc="Trial-{}, Epoch".format(trial), position=0, ncols=90)
                for e in range(self.args.kmeans_epoch):
                    for image, _ in train_loader:
                        train_data = self.get_partitioned_batch(image)
                        model = model.partial_fit(train_data)

                        if prev_centers is not None:
                            is_converged = check_convergence(prev_centers, model.cluster_centers_, model.tol)
                            if is_converged:
                                break
                        prev_centers = deepcopy(model.cluster_centers_)
                    t_epoch.update(1)
                    if is_converged:
                        early_stopped = True
                        if model.inertia_ < best_model_inertia:
                            best_model = model
                            best_model_inertia = model.inertia_
                        break
                t_epoch.close()
                if early_stopped:
                    print("Early stop training trial-{} kmeans model".format(trial))
        else:
            x = None
            print(">> Load dataset & get representations for clustering")
            with tqdm(train_loader, unit="batch", ncols=90) as t:
                for i, (input, _) in enumerate(t):
                    batch = torch.tensor(self.get_partitioned_batch(input))
                    if x is None:
                        x = batch
                    else:
                        x = torch.cat((x, batch))
            best_model = KMeans(n_clusters=self.args.cluster, random_state=0).fit(x)

        path = self.args.clustering_path
        joblib.dump(best_model, os.path.join(path + '/checkpoint.pkl'))
        with open(os.path.join(path, "params.json"), 'w') as f:
            args_to_save = {'repr_method': self.args.repr_method,
                            'partition_method': self.args.partition_method, 'num_partitions': self.args.partition, 
                            'k': self.args.cluster, 'tol': self.args.kmeans_tol,
                            'n_inits': self.args.kmeans_init, 'epoch': self.args.kmeans_epoch, 'batch': self.args.batch}
            json.dump(args_to_save, f, indent=4)
        exit()
        self.model = best_model


def check_cluster_distribution(kmeans, train_loader):
    n_data = kmeans.args.data_per_cluster * kmeans.args.cluster * len(train_loader)
    n_data_per_cluster = dict()
    for c in range(kmeans.args.cluster):
        n_data_per_cluster[c] = 0
    for i, (input, target) in enumerate(train_loader):
        batch_cluster = kmeans.predict_cluster_of_batch(input)
        for c in batch_cluster:
            n_data_per_cluster[c.item()] += 1

    assert sum(n_data_per_cluster.values()) == n_data,\
        "Total # of data doesn't match (n_data: {}, calc: {})".format(n_data, sum(n_data_per_cluster.values()))

    ratio = np.zeros((kmeans.args.cluster))
    for c in range(kmeans.args.cluster):
        ratio[c] = n_data_per_cluster[c] / n_data * 100

    for c in range(kmeans.args.cluster):
        print("{},{:.2f} %".format(n_data_per_cluster[c], ratio[c]))
    print(">> [#Data] Mean, Var, Std")
    d = list((n_data_per_cluster.values()))
    print("{}, {:.2f}, {:.2f}".format(np.mean(d), np.var(d), np.std(d)))
    print(">> [Ratio] Mean, Var, Std")
    print("{:.2f} %, {:.4f}, {:.4f}".format(np.mean(ratio), np.var(ratio), np.std(ratio)))
    centroids = kmeans.model.cluster_centers_
    print(">> [Centroids] Var, Std")
    print("var: {:.4f}, std: {:.4f}".format(np.var(centroids), np.std(centroids)))

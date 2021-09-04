import torch
from sklearn.cluster import Birch
import numpy as np

import tqdm
import joblib
from copy import deepcopy
import json
import os


class BIRCH(object):
    def __init__(self, args):
        self.args = args
        self.model = None

    def get_partitioned_batch(self, data):
        # Under the premise that images are in the form of square matrix
        _size = data.shape[-1]
        n_part = int((self.args.partition / 2) if self.args.partition % 2 == 0 else (self.args.partition / 3))  # Per row or col
        n_data = int(_size / n_part)  # Per part
        rst = None
        for i in range(n_part):
            r_start = n_data * i
            for j in range(n_part):
                c_start = n_data * j
                _min = torch.min(data[:, :, r_start:r_start + n_data, c_start:c_start + n_data], -1).values
                _max = torch.max(data[:, :, r_start:r_start + n_data, c_start:c_start + n_data], -1).values
                _min = torch.min(_min, -1, keepdim=True).values
                _max = torch.max(_max, -1, keepdim=True).values
                tmp = torch.cat([_min, _max], dim=-1)
                if rst is None:
                    rst = tmp
                else:
                    rst = torch.cat([rst, tmp], dim=-1)
        return rst.view(rst.size(0), -1).numpy()

    def load_clustering_model(self):
        # Load k-means model's hparams, and check dependencies
        with open(os.path.join(self.args.clustering_path, 'params.json'), 'r') as f:
            saved_args = json.load(f)
        assert self.args.cluster == saved_args['k'], \
            "Loaded model's target # of clusters must be same with initialized model" \
            "\n(Loaded model's = {}, Initialized model's = {}".format(saved_args['k'], self.args.cluster)
        assert self.args.partition == saved_args['num_partitions'], \
            "Loaded model's target # of partitions to divide an channel must be same with initialized model" \
            "\n(Loaded model's = {}, Initialized model's = {}".format(saved_args['num_partitions'], self.args.partition)
        self.model = joblib.load(os.path.join(self.args.clustering_path, 'checkpoint.pkl'))
    
    def predict_cluster_of_batch(self, input):
        partitioned_input = self.get_partitioned_batch(input)
        cluster_info = self.model.predict(partitioned_input)
        return torch.cuda.LongTensor(cluster_info)

    def train_clustering_model(self, train_loader):
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

        model = None
        print("Train BIRCH model")
        t_epoch = tqdm.tqdm(total=1, desc="BIRCH", position=0, ncols=90)
        for image, _ in train_loader:
            train_data = self.get_partitioned_batch(image)
            model = Birch(n_clusters=self.args.cluster)
            model.fit(train_data)
            t_epoch.update(1)
            t_epoch.close()
        print("Birch n_cluster: {}".format(model.subcluster_centers_.shape))
        #joblib.dump(model, os.path.join(self.args.kmeans_path + '/checkpoint.pkl'))
        self.model = model


def check_cluster_distribution(kmeans, train_loader):
    #n_data = kmeans.args.data_per_cluster * kmeans.args.cluster * len(train_loader)
    n_data = 50000
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
    #centroids = kmeans.model.cluster_centers_
    #print(">> [Centroids] Var, Std")
    #print("var: {:.4f}, std: {:.4f}".format(np.var(centroids), np.std(centroids)))

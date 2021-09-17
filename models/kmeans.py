import torch
from sklearn.cluster import MiniBatchKMeans
import numpy as np

import tqdm
import joblib
from copy import deepcopy
import json
import os


class KMeans(object):
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
                #_min = torch.topk(data[:, :, r_start:r_start + n_data, c_start:c_start + n_data], k=2, dim=-1, largest=False, sorted=False).values
                #_max = torch.topk(data[:, :, r_start:r_start + n_data, c_start:c_start + n_data], k=2, dim=-1, largest=True, sorted=False).values
                #_min = _min.reshape(_min.shape[0], _min.shape[1], -1)
                #_max = _max.reshape(_max.shape[0], _max.shape[1], -1)
                #_min = torch.topk(_min, k=2, dim=2, largest=False, sorted=False).values
                #_max = torch.topk(_max, k=2, dim=2, largest=True, sorted=False).values
                #_min = torch.mean(_min, dim=2, keepdim=True)
                #_max = torch.mean(_max, dim=2, keepdim=True)
                _min = torch.min(data[:, :, r_start:r_start + n_data, c_start:c_start + n_data], -1).values
                _max = torch.max(data[:, :, r_start:r_start + n_data, c_start:c_start + n_data], -1).values
                _min = torch.min(_min, -1, keepdim=True).values
                _max = torch.max(_max, -1, keepdim=True).values
                tmp = torch.cat([_min, _max], dim=-1)
                if rst is None:
                    rst = tmp
                else:
                    rst = torch.cat([rst, tmp], dim=-1)
        return rst.view(rst.size(0), -1).cpu().numpy()

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
        return torch.cuda.LongTensor(cluster_info).cuda()

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

        prev_centers = None
        is_converged = False
        best_model = None
        best_model_inertia = 9999999999999999
        print("Train K-means model 10 times, and choose the best model")
        for trial in range(1, 11):
            model = MiniBatchKMeans(n_clusters=self.args.cluster, batch_size=self.args.batch, tol=self.args.kmeans_tol, random_state=0)
            early_stopped = False
            t_epoch = tqdm.tqdm(total=self.args.kmeans_epoch, desc="Trial-{}, Epoch".format(trial), position=0, ncols=90)
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
        joblib.dump(best_model, os.path.join(self.args.clustering_path + '/checkpoint.pkl'))
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

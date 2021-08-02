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

    @staticmethod
    def get_partitioned_batch(data, partition):
        channel = data.shape[1]
        width = data.shape[2]
        height = data.shape[3]
        data = data.reshape((data.shape[0], channel * width * height))
        n_row = int(width / 2)
        n_col = int(height / (partition / 2))

        rst = np.array([])
        for i in range(channel):
            chanel_start = i * (width * height)
            for j in range(partition):
                if j < partition / 2:
                    # The upper half of an Image
                    part_start = chanel_start + (j * n_col)
                else:
                    # The rest(half) of an Image
                    part_start = chanel_start + (width * height / 2) + (j - int(partition / 2)) * n_col

                part = np.array([])
                start = part_start
                for k in range(n_row):
                    # A row of current part
                    end = start + n_col
                    if not k:
                        part = np.copy(data[:, start:end])
                    else:
                        part = np.concatenate((part, data[:, start:end]), axis=1)
                    start += width

                part_min = np.min(part, axis=1).reshape(part.shape[0], 1)
                part_max = np.max(part, axis=1).reshape(part.shape[0], 1)
                tmp = np.append(part_min, part_max, axis=1)
                if not i and not j:
                    rst = np.copy(tmp)
                else:
                    rst = np.append(rst, tmp, axis=1)
        return rst

    def load_kmeans_model(self):
        # Load k-means model's hparams, and check dependencies
        with open(os.path.join(self.args.kmeans_path, 'params.json'), 'r') as f:
            saved_args = json.load(f)
        assert self.args.cluster == saved_args['k'], \
            "Current model's target # of clusters must be same with K-means model's it" \
            "\n(K-means model's = {}, Current model's = {}".format(saved_args['k'], self.args.cluster)
        assert self.args.partition == saved_args['num_partitions'], \
            "Current model's target # of partitions to divide an channel must be same with K-means model's it" \
            "\n(K-means model's = {}, Current model's = {}".format(saved_args['num_partitions'], self.args.partition)
        self.model = joblib.load(os.path.join(self.args.kmeans_path, 'checkpoint.pkl'))
    
    def get_batch(self, input, target):
        kmeans_input = self.get_partitioned_batch(input.numpy(), self.args.partition)
        cluster_info = self.model.predict(kmeans_input)

        num_data_per_cluster = []
        input_ordered_by_cluster = torch.zeros(input.shape)
        target_ordered_by_cluster = torch.zeros(target.shape, dtype=torch.long)
        existing_clusters, counts = np.unique(cluster_info, return_counts=True)
        ordered = 0
        for cluster, n in zip(existing_clusters, counts):
            num_data_per_cluster.append([cluster, n])
            data_indices = (cluster_info == cluster).nonzero()[0]
            input_ordered_by_cluster[ordered:ordered + n] = input[data_indices].clone().detach()
            target_ordered_by_cluster[ordered:ordered + n] = target[data_indices].clone().detach()
            ordered += n
        return input_ordered_by_cluster, target_ordered_by_cluster, torch.ByteTensor(num_data_per_cluster)

    def train_kmeans_model(self, train_loader):
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

        model = MiniBatchKMeans(n_clusters=self.args.cluster, batch_size=self.args.batch, tol=self.args.kmeans_tol, random_state=0)

        prev_centers = None
        is_converged = False
        t_epoch = tqdm.tqdm(total=self.args.kmeans_epoch, desc='Epoch', position=0, ncols=90)
        for e in range(self.args.kmeans_epoch):
            for image, _ in train_loader:
                train_data = self.get_partitioned_batch(image.numpy(), self.args.partition)
                model = model.partial_fit(train_data)

                if prev_centers is not None:
                    is_converged = check_convergence(prev_centers, model.cluster_centers_, model.tol)
                    if is_converged:
                        break
                prev_centers = deepcopy(model.cluster_centers_)
            t_epoch.update(1)
            if is_converged:
                print("\nEarly stop training kmeans model")
                break
        joblib.dump(model, os.path.join(self.args.kmeans_path + '/checkpoint.pkl'))
        self.model = model


def check_cluster_distribution(kmeans, train_loader):
    n_data = kmeans.args.batch * len(train_loader)
    n_data_per_cluster = dict()
    for c in range(kmeans.args.cluster):
        n_data_per_cluster[c] = 0
    for i, (input, target) in enumerate(train_loader):
        _, _, batch_cluster = kmeans.get_batch(input, target)
        for c, n in batch_cluster:
            n_data_per_cluster[c.item()] += n.item()
    for c in range(kmeans.args.cluster):
        print("C{}: {}, \t{:.2f}%".format(c, n_data_per_cluster[c], n_data_per_cluster[c] / n_data * 100))

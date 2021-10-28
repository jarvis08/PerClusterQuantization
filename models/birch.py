import torch
from sklearn.cluster import Birch
import numpy as np

from tqdm import tqdm
import joblib
from copy import deepcopy
import json
import os


class BIRCH(object):
    def __init__(self, args):
        self.args = args
        self.model = None

    @torch.no_grad()
    def get_partitioned_batch(self, data):
        # Under the premise that images are in the form of square matrix
        batch = data.size(0)
        channel = data.size(1)
        _size = data.size(2)
        n_part = int((self.args.partition / 2)
                     if self.args.partition % 2 == 0
                     else (self.args.partition / 3))  # Per row & col
        n_data = int(_size / n_part)  # Per part
        rst = None
        for i in range(n_part):
            c_start = n_data * i
            for j in range(n_part):
                r_start = n_data * j
                part = data[:, :, c_start:c_start + n_data, r_start:r_start + n_data].reshape(batch, channel, -1)
                _min = part.min(-1, keepdim=True).values
                _max = part.max(-1, keepdim=True).values
                part_rst = torch.cat([_min, _max], dim=-1)
                if rst is None:
                    rst = part_rst
                else:
                    rst = torch.cat([rst, part_rst], dim=-1)
        return rst.view(rst.size(0), -1).cpu().numpy()

    def predict_cluster_of_batch(self, input):
        kmeans_input = self.get_partitioned_batch(input)
        cluster_info = self.model.predict(kmeans_input)
        return torch.cuda.LongTensor(cluster_info).cuda()

    def train_clustering_model(self, train_loader):
        model = Birch(n_clusters=self.args.cluster)
        with tqdm(train_loader, unit="batch", ncols=90) as t:
            for i, (image, _) in enumerate(t):
                t.set_description("BIRCH")
                train_data = self.get_partitioned_batch(image)
                model = model.partial_fit(train_data)
                t.update(1)
        print("Birch n_cluster: {}".format(model.subcluster_centers_.shape))
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

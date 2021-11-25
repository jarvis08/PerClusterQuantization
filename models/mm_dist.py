import torch

from tqdm import tqdm
import json
import os


class MinMaxDistClustering(object):
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

        data = data.view(batch, channel, n_part, n_data, _size).transpose(3, 4)
        data = data.reshape(batch, channel, n_part * n_part, -1)
        _min = data.min(-1, keepdim=True).values
        _max = data.max(-1, keepdim=True).values
        rst = torch.cat((_min, _max), dim=-1)
        return rst.view(rst.size(0), -1)

    @torch.no_grad()
    def predict(self, data):
        found = set()
        rst = torch.full(data.size(0), self.args.cluster - 1, dtype=torch.int64)
        for c in range(self.args.cluster - 1):
            cluster_key = str(c)
            dim = self.model[cluster_key]['index']
            value = self.model[cluster_key]['value']
            indices = set((data[:, dim] < value).nonzero(as_tuple=True)[0].tolist())
            newly_found = indices - found
            found.update(newly_found)
            rst[list(newly_found)] = c
        return rst

    def load_clustering_model(self):
        # Load k-means model's hparams, and check dependencies
        with open(os.path.join(self.args.clustering_path, 'params.json'), 'r') as f:
            saved_args = json.load(f)
        assert self.args.cluster == saved_args['k'], \
            "Current model's target # of clusters must be same with clustering model's it" \
            "\n(Clustering model's = {}, Current model's = {}".format(saved_args['k'], self.args.cluster)
        assert self.args.partition == saved_args['num_partitions'], \
            "Current model's target # of partitions to divide an channel must be same with clustering model's it" \
            "\n(Clustering model's = {}, Current model's = {}".format(saved_args['num_partitions'], self.args.partition)
        with open(os.path.join(self.args.clustering_path, 'model.json'), 'r') as f:
            self.model = json.load(f)

    def predict_cluster_of_batch(self, input):
        partitioned_minmax = self.get_partitioned_batch(input)
        cluster_info = self.predict(partitioned_minmax)
        return torch.LongTensor(cluster_info)

    def train_clustering_model(self, train_loader):
        print("Making clustering model by parsing index of representation whose var is the largest among the left data")
        model = dict()
        for i in range(self.args.cluster):
            model[i] = {'index': 0, 'value': 0.0}

        dataset = None
        with tqdm(train_loader, unit="batch", ncols=90) as t:
            for i, (input, target) in enumerate(t):
                batch = self.get_partitioned_batch(input)
                if dataset is None:
                    dataset = batch
                else:
                    dataset = torch.cat((dataset, batch))

        n_points = self.args.cluster - 1
        to_percentile = 1 / self.args.cluster

        var_per_dim = torch.var(dataset, dim=0)
        topk_dims = torch.topk(var_per_dim, n_points).indices
        for dim in range(n_points):
            model[dim]['index'] = topk_dims[dim].item()
            model[dim]['value'] = torch.quantile(dataset[:, topk_dims[dim]], [to_percentile]).item()
            indices = (dataset[:, model[dim]['index']] > model[dim]['value']).nonzero(as_tuple=True)[0]

            if dim != n_points - 1:
                dataset[indices] = dataset
                var_per_dim = torch.var(dataset, dim=0)
                topk_dims = torch.topk(var_per_dim, n_points).indices

        path = self.args.clustering_path
        with open(os.path.join(path + 'model.json'), "w") as f:
            json.dump(model, f, indent=4)
        with open(os.path.join(path, "params.json"), 'w') as f:
            args_to_save = {'k': self.args.cluster, 'partition_method': self.args.partition_method,
                            'num_partitions': self.args.partition}
            json.dump(args_to_save, f, indent=4)
        self.model = model

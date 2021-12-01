import torch

from tqdm import tqdm
import json
import os

import pandas as pd
import seaborn as sns
sns.set(style="darkgrid", font_scale=1.2)
import matplotlib.pyplot as plt


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

        n_part = self.args.partition // 2      \
                 if self.args.partition % 2 == 0 \
                 else self.args.partition // 3  # Per row & col
        n_data = _size // n_part  # Per part

        data = data.view(batch, channel, n_part, n_data, _size).transpose(3, 4)
        data = data.reshape(batch, channel, n_part * n_part, -1)
        _min = data.min(-1, keepdim=True).values
        _max = data.max(-1, keepdim=True).values
        rst = torch.cat((_min, _max), dim=-1)
        return rst.view(rst.size(0), -1)

    @torch.no_grad()
    def predict(self, data):
        rst = torch.zeros(data.size(0), dtype=torch.int64)
        indices = torch.arange(data.size(0))
        wip_indices = indices

        path_per_cluster = self.model['path']
        model_ptr = self.model['root']
        for path in path_per_cluster:
            for step, p in enumerate(path[1:]):
                dim = model_ptr['dim']
                value = model_ptr['value']

                if p == 'gt':
                    found = (data[wip_indices, dim] >= value).nonzero(as_tuple=True)[0]
                else:
                    found = (data[wip_indices, dim] < value).nonzero(as_tuple=True)[0]

                if found.size(0) == 0:
                    break

                wip_indices = wip_indices[found]
                if step == len(path[1:]) - 1:
                    rst[wip_indices] = model_ptr[p]['cluster']
                else:
                    model_ptr = model_ptr[p]
            model_ptr = self.model['root']
            wip_indices = indices
        _, counts = torch.unique(rst, return_counts=True)
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

    @torch.no_grad()
    def train_clustering_model(self, train_loader, aug_loader):
        def check_dividable_cluster(level, identity):
            if level['divided']:
                gt_rst = check_dividable_cluster(level['gt'], 'gt')
                lt_rst = check_dividable_cluster(level['lt'], 'lt')

                paths = gt_rst + lt_rst
                for i in range(len(paths)):
                    paths[i] = [identity] + paths[i]
                return paths
            else:
                return [[identity]]

        print("Making clustering model by parsing index of representation whose var is the largest among the left data")
        dataset = None
        with tqdm(train_loader, unit="batch", ncols=90) as t:
            for i, (input, target) in enumerate(t):
                batch = self.get_partitioned_batch(input)
                if dataset is None:
                    dataset = batch
                else:
                    dataset = torch.cat((dataset, batch))

        with tqdm(aug_loader, unit="batch", ncols=90) as t:
            for i, (input, target) in enumerate(t):
                batch = self.get_partitioned_batch(input)
                dataset = torch.cat((dataset, batch))

        with tqdm(aug_loader, unit="batch", ncols=90) as t:
            for i, (input, target) in enumerate(t):
                batch = self.get_partitioned_batch(input)
                dataset = torch.cat((dataset, batch))

        dataset = dataset[torch.randperm(dataset.size()[0])]  # shuffle batch

        ##########################################################
        ################### Input distribution ###################
        ##########################################################
        # total = dataset.numpy()
        # total = pd.DataFrame(total)
        # total.columns = [f'dim{d}' for d in range(24)]
        # pwd = '/home/ken/Documents/Lab/Quantization/PerClusterQuantization'
        # for d in range(24):
        #     kind = 'hist'
        #     sns.displot(
        #         data=total,
        #         x=f'dim{d}',
        #         kind=kind,
        #         aspect=1.4,
        #         # bins=100
        #         bins=100
        #     )
        #     if d % 2:
        #         min_or_max = 'max'
        #     else:
        #         min_or_max = 'min'
        #     plt.title(f"[SVHN] NumData={dataset.size(0)}, Dim={d}({min_or_max}), Std={total.std(axis=0)[d]:.4f}")
        #     plt.savefig(f"{pwd}/figs/[SVHN-WithAug] {kind}-dim{d}.png", format="png", dpi=200, bbox_inches='tight')
        #     plt.cla()
        ##########################################################
        ##########################################################
        ##########################################################

        num_data = dataset.size(0)
        min_data_per_cluster = num_data // (self.args.cluster * 4)
        n_dims_to_choose = self.args.cluster - 1

        builder = dict()
        builder['root'] = {'index': torch.arange(num_data), 'num_data': num_data, 'divided': False}

        used_dims = []  # len(chosen) + 1 == found_num_clusters
        indivisible_cluster_paths = []
        while True:
            candidate_paths = check_dividable_cluster(builder['root'], 'root')
            candidate_ptrs = []
            candidate_vars = []
            candidate_dim = []
            for c in range(len(candidate_paths)):
                cdd_ptr = builder
                for p in candidate_paths[c]:
                    cdd_ptr = cdd_ptr[p]

                if cdd_ptr['num_data'] < min_data_per_cluster * 2:
                    continue

                candidate_ptrs.append(cdd_ptr)
                cdd_data = dataset[cdd_ptr['index']]

                var_per_dim = torch.var(cdd_data, dim=0)
                variables, indices = torch.topk(var_per_dim, n_dims_to_choose)
                for var, dim in zip(variables, indices):
                    if dim not in used_dims:
                        candidate_vars.append(var.item())
                        candidate_dim.append(dim.item())
                        break
            assert len(candidate_vars), f"Can't divide dataset into {self.args.cluster} clusters. Change clustering criteria."

            candidate_vars = torch.tensor(candidate_vars)
            _, cdd_indices = torch.sort(candidate_vars, dim=0)

            target_dict = None
            target_dim = None
            value = None
            ratio = None
            gt_indices = None
            lt_indices = None
            found = False
            for target in cdd_indices:
                if "".join(candidate_paths[target]) in indivisible_cluster_paths:
                    continue

                target_dim = candidate_dim[target]
                target_dict = candidate_ptrs[target]
                target_idx = target_dict['index']
                target_data = dataset[target_idx]

                is_max = True if target_dim % 2 != 0 else False
                n_bins = 100
                n_data = target_data.size(0)
                bins = torch.histc(target_data[:, target_dim], bins=n_bins)  # In ascending order
                max_bin_value, max_bin_index = torch.max(bins, dim=0)
                topk_bin_values, topk_bin_index = torch.topk(bins, 10)

                """
                    dist_type == 1 :: Absolutely biased
                    dist_type == 2 :: Gaussian
                    dist_type == 3 :: Gaussian + Abs.biased
                    dist_type == 4 :: Etc. (including diagonal shape)
                """
                mid_sum = torch.sum(bins[45:55])
                included = False
                type1 = max_bin_index in [0, n_bins - 1] and topk_bin_values[0] > topk_bin_values[1] * 10
                type2 = bins[0] + bins[-1] < topk_bin_values[0] * 0.4
                type3 = mid_sum > bins[0] * 5 or mid_sum > bins[-1] * 5
                ratio = None

                if type1 or type3:
                    included = True
                    if is_max:
                        last_3_bins = torch.sum(bins[-3:])
                        ratio = 1 - last_3_bins / n_data 
                    else:
                        first_3_bins = torch.sum(bins[:3])
                        ratio = first_3_bins / n_data

                # Type-2
                if type2:
                    from_left, from_right = [], []
                    to_check = 4
                    for i in range(to_check + 1):
                        idx1, idx2 = i * 10, (i + 1) * 10
                        from_left.append(torch.sum(bins[idx1:idx2]).item())
                        from_right.append(torch.sum(bins[- idx2:- idx1]).item())

                    avg_left_increased, avg_right_increased = sum(from_left) / to_check, sum(from_right) / to_check
                    if avg_left_increased >  avg_right_increased:
                        ratio = 0.90
                    else:
                        ratio = 0.10

                    # all_positive = True
                    # left_increased, right_increased = [], []
                    # for i in range(to_check):
                    #     left = from_left[i] - from_left[i + 1]
                    #     right = from_right[i] - from_right[i + 1]
                    #     if left < 0 or right < 0:
                    #         all_positive = False
                    #         break
                    #     else:
                    #         left_increased.append(left.item())
                    #         right_increased.append(right.item())

                    # if all_positive:
                    #     type2, included = True, True
                    #     avg_left_increased, avg_right_increased = sum(left_increased) / to_check, sum(right_increased) / to_check
                    #     if avg_left_increased >  avg_right_increased:
                    #         ratio = 0.90
                    #     else:
                    #         ratio = 0.10

                if ratio is None:
                    if is_max:
                        last_3_bins = torch.sum(bins[-3:])
                        ratio = 1 - last_3_bins / n_data 
                    else:
                        first_3_bins = torch.sum(bins[:3])
                        ratio = first_3_bins / n_data

                if type1: dist_type = 1
                elif type2: dist_type = 2
                elif type3: dist_type = 3
                else: dist_type = 4
                print(f"Type-{dist_type}")

                while ratio * n_data < min_data_per_cluster:
                    ratio += 0.01
                while (1 - ratio) * n_data < min_data_per_cluster:
                    ratio -= 0.01

                value = torch.quantile(target_data[:, target_dim], ratio)
                target_idx = set(target_idx.tolist())
                gt_indices = set((dataset[:, target_dim] >= value).nonzero(as_tuple=True)[0].tolist())
                lt_indices = set((dataset[:, target_dim] < value).nonzero(as_tuple=True)[0].tolist())
                gt_indices = torch.tensor(list(gt_indices.intersection(target_idx)), dtype=torch.int64)
                lt_indices = torch.tensor(list(lt_indices.intersection(target_idx)), dtype=torch.int64)
                if gt_indices.size(0) < min_data_per_cluster or lt_indices.size(0) < min_data_per_cluster:
                    print(f"[Path] {''.join(candidate_paths[target])}\n[Min-req-data] {min_data_per_cluster}\n[GT-data] {gt_indices.size(0)} [LT-data] {lt_indices.size(0)}")
                    indivisible_cluster_paths.append(''.join(candidate_paths[target]))
                else:
                    found = True
                    break
            assert found, f"Can't divide dataset into {self.args.cluster} clusters. Change clustering criteria."

            target_dict['divided'] = True
            target_dict['dim'] = target_dim
            target_dict['value'] = value
            target_dict['gt'] = dict({'index': gt_indices, 'num_data': gt_indices.size(0), 'divided': False})
            target_dict['lt'] = dict({'index': lt_indices, 'num_data': lt_indices.size(0), 'divided': False})
            used_dims.append(target_dim)

            ####################################################################
            ################### Dist comparison in splitting ###################
            ####################################################################
            kind = 'hist'
            d = target_dict['dim']
            d_name = f'dim{d}'
            n_found = len(used_dims)

            min_or_max = 'max' if d % 2 != 0 else 'min'
            pwd = '/home/ken/Documents/Lab/Quantization/PerClusterQuantization/figs'
            clustered = '-'.join(candidate_paths[target])
            f_base = f'{pwd}/{self.args.dataset}.k{self.args.cluster}.{n_found}.({clustered}).{min_or_max}.dim{d}.type{dist_type}'

            total = pd.DataFrame(dataset[target_dict['index'], d].numpy())
            total.columns = [d_name]
            sns.displot(data=total, x=d_name, kind=kind, aspect=1.4, bins=100)
            plt.axvline(target_dict['value'], color='r')
            plt.title(f"[All] NumData={total.shape[0]}, SplitValue={target_dict['value']:.4f}({ratio}%),"
                      f" Var={total.var().loc[d_name]:.4f}")
            plt.savefig(f"{f_base}.All.png", format="png", dpi=200, bbox_inches='tight')
            plt.cla()

            gt = pd.DataFrame(dataset[target_dict['gt']['index'], d].numpy())
            gt.columns = [d_name]
            sns.displot(data=gt, x=d_name, kind=kind, aspect=1.4, bins=100)
            plt.title(f"[GreaterThan] NumData={gt.shape[0]}(Min={min_data_per_cluster}), Var={gt.var().loc[d_name]:.4f}")
            plt.savefig(f"{f_base}.GT.png", format="png", dpi=200, bbox_inches='tight')
            plt.cla()

            lt = pd.DataFrame(dataset[target_dict['lt']['index'], d].numpy())
            lt.columns = [d_name]
            sns.displot(data=lt, x=d_name, kind=kind, aspect=1.4, bins=100)
            plt.title(f"[LessThan] NumData={lt.shape[0]}(Min={min_data_per_cluster}), Var={lt.var().loc[d_name]:.4f}")
            plt.savefig(f"{f_base}.LT.png", format="png", dpi=200, bbox_inches='tight')
            plt.cla()
            ####################################################################
            ####################################################################
            ####################################################################

            if len(used_dims) == n_dims_to_choose:
                break

        model = dict()
        cluster_id = -1
        clusters_info = check_dividable_cluster(builder['root'], 'root')
        model['path'] = clusters_info
        for path in clusters_info:
            depth = len(path)
            model_ptr = model
            builder_ptr = builder

            for d in range(depth):
                p = path[d]
                builder_ptr = builder_ptr[p]
                if model_ptr.get(p) is None:
                    model_ptr[p] = dict()
                    model_ptr = model_ptr[p]

                    if d == depth - 1:
                        cluster_id += 1
                        model_ptr['cluster'] = cluster_id
                        model_ptr['num_training_data'] = builder_ptr['num_data']
                    else:
                        model_ptr['dim'] = builder_ptr['dim']
                        model_ptr['value'] = builder_ptr['value'].item()
                else:
                    model_ptr = model_ptr[p]

        path = self.args.clustering_path
        with open(os.path.join(path, 'model.json'), "w") as f:
            json.dump(model, f, indent=4)
        with open(os.path.join(path, "params.json"), 'w') as f:
            args_to_save = {'k': self.args.cluster, 'partition_method': self.args.partition_method,
                            'num_partitions': self.args.partition}
            json.dump(args_to_save, f, indent=4)
        self.model = model

    @torch.no_grad()
    def predict_v1(self, data):
        found = set()
        rst = torch.full((data.size(0), 1), self.args.cluster - 1, dtype=torch.int64)
        for c in range(self.args.cluster - 1):
            cluster_key = str(c)
            dim = self.model[cluster_key]['index']
            value = self.model[cluster_key]['value']
            indices = set((data[:, dim] < value).nonzero(as_tuple=True)[0].tolist())
            newly_found = indices - found
            found.update(newly_found)
            rst[list(newly_found), 0] = c
        return rst.view(-1)

    @torch.no_grad()
    def train_clustering_model_v1(self, train_loader):
        print("Making clustering model by parsing index of representation whose var is the largest among the left data")
        model = dict()
        for c in range(self.args.cluster):
            cluster_key = str(c)
            model[cluster_key] = {'index': 0, 'value': 0.0}

        dataset = None
        with tqdm(train_loader, unit="batch", ncols=90) as t:
            for i, (input, target) in enumerate(t):
                batch = self.get_partitioned_batch(input)
                if dataset is None:
                    dataset = batch
                else:
                    dataset = torch.cat((dataset, batch))

        n_dims_to_choose = self.args.cluster - 1
        used_dims = []
        left_n_cluster = self.args.cluster
        var_per_dim = torch.var(dataset, dim=0)
        topk_dims = torch.topk(var_per_dim, n_dims_to_choose).indices
        for c in range(n_dims_to_choose):
            idx = -1
            while True:
                idx += 1
                dim = topk_dims[idx].item()
                if dim not in used_dims:
                    used_dims.append(dim)
                    break

            target_dim = used_dims[-1]
            percentage = 1 / left_n_cluster
            cluster_key = str(c)

            model[cluster_key]['index'] = target_dim
            model[cluster_key]['value'] = torch.quantile(dataset[:, target_dim], percentage).item()
            indices = (dataset[:, model[cluster_key]['index']] > model[cluster_key]['value']).nonzero(as_tuple=True)[0]

            if c != n_dims_to_choose - 1:
                dataset = dataset[indices]
                var_per_dim = torch.var(dataset, dim=0)
                topk_dims = torch.topk(var_per_dim, n_dims_to_choose).indices
                left_n_cluster -= 1

        path = self.args.clustering_path
        with open(os.path.join(path, 'model.json'), "w") as f:
            json.dump(model, f, indent=4)
        with open(os.path.join(path, "params.json"), 'w') as f:
            args_to_save = {'k': self.args.cluster, 'partition_method': self.args.partition_method,
                            'num_partitions': self.args.partition}
            json.dump(args_to_save, f, indent=4)
        self.model = model

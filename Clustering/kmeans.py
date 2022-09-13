import torch

from cuml.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from audtorch.metrics.functional import pearsonr as pearson_correlation
import numpy as np

from tqdm import tqdm
import joblib
from copy import deepcopy
import json
import os
import csv

class KMeansClustering(object):
    def __init__(self, args):
        self.args = args
        self.model = None
        self.feature_index = None
        self.final_cluster = None  # Used in NN-aware Clustering only

    @torch.no_grad()
    def get_partitioned_batch(self, data):
        # Under the premise that images are in the form of square matrix
        batch = data.size(0)
        channel = data.size(1)
        _size = data.size(2)
        # n_part = self.args.partition
        n_part = 2
        if _buffer := ((n_part - _size % n_part) % n_part):
            m = torch.nn.ZeroPad2d((0, _buffer, 0, _buffer))
            data = m(data)
            _size = data.size(2)
        if self.args.partition_method == 'square':
            n_data = int(_size / n_part)  # Per part

            # data = data.amax(dim=1, keepdim=True)
            # channel = 1
            data = data.view(batch, channel, n_part,
                             n_data, _size).transpose(3, 4).contiguous()
            data = data.view(batch, channel, n_part * n_part, -1)

            if self.args.repr_method == 'max':
                rst, _ = data.topk(k=3, dim=-1)
                rst = rst.mean(-1, keepdim=True)
            elif self.args.repr_method == 'mean':
                rst = data.mean(-1, keepdim=True)
            else:
                _min = data.min(-1, keepdim=True).values
                _max = data.max(-1, keepdim=True).values
                rst = torch.cat((_min, _max), dim=-1)

            rst = rst.view(rst.size(0), -1)
            if self.feature_index is not None:
                return torch.index_select(rst, dim=1, index=self.feature_index)
            return rst

        else:
            # To make clustering model more robust about augmentation's horizontal flip
            n_part = 4
            n_data = int(_size / n_part)
            rst = None
            for c in range(n_part):
                c_start = n_data * c
                part_data = data[:, :, c_start:c_start +
                                 n_data, :].reshape(batch, channel, -1)
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
            loaded_args = json.load(f)

        assert self.args.dataset == loaded_args['dataset'], \
            f"Dataset of model doesn't match (args: {self.args.dataset}, loaded: {loaded_args['dataset']})"
        assert self.args.clustering_method == loaded_args['clustering_method'], \
            f"# of clusters doesn't match (args: {self.args.clustering_method}, loaded: {loaded_args['clustering_method']})"
        if self.args.nnac and loaded_args.get('nnac') is None:
            assert self.args.sub_cluster == loaded_args['k'], \
                f"# of sub_clusters doesn't match (args: {self.args.sub_cluster}, loaded: {loaded_args['k']})"
        else:
            assert self.args.cluster == loaded_args['k'], \
                f"# of clusters doesn't match (args: {self.args.cluster}, loaded: {loaded_args['k']})"
        assert self.args.partition == loaded_args['num_partitions'], \
            f"# of partitions doesn't match (args: {self.args.partition}, loaded: {loaded_args['num_partitions']})"
        assert self.args.repr_method == loaded_args['repr_method'], \
            f"Representation method doesn't match (arg: {self.args.repr_method}, loaded: {loaded_args['repr_method']}"

        if self.args.sub_cluster:
            if loaded_args.get('nnac') is None:
                print(
                    "Warning: NN-aware Clustering mode is on now, but loaded model doesn't have final cluster info.")
            else:
                assert self.args.sub_cluster == loaded_args['sub_k'], \
                    f"Number of sub-clusters doesn't match (arg: {self.args.sub_cluster}, loaded: {loaded_args['sub_k']}"
                self.final_cluster = torch.zeros(
                    self.args.sub_cluster, dtype=torch.int64)
                print(">>> NN-aware Clustering mode")
                print(loaded_args['nnac'])
                for sub, final in loaded_args['nnac'].items():
                    self.final_cluster[int(sub)] = int(final)

        self.model = joblib.load(os.path.join(
            self.args.clustering_path, 'checkpoint.pkl'))

        self.feature_index = torch.load(os.path.join(self.args.clustering_path, 'index.pth'))

    @torch.no_grad()
    def predict_cluster_of_batch(self, input):
        kmeans_input = self.get_partitioned_batch(input)
        cluster_info = self.model.predict(kmeans_input)
        if self.final_cluster is not None:  # make output as merged cluster form
            return torch.index_select(self.final_cluster, 0, torch.LongTensor(cluster_info))
        return torch.LongTensor(cluster_info)


    @torch.no_grad()
    def get_high_corr_features(self, dnn_model, nonaug_loader):
        print('Start Observing Max Values from Each Data...')
        total_features = None
        total_maxs = None

        dnn_model.toggle_full_precision()
        with tqdm(nonaug_loader, desc="Step", ncols=80) as t:
            for i, (images, target) in enumerate(t):
                images = images.cuda(non_blocking=True)
                features = self.get_partitioned_batch(images)

                if total_features is not None:
                    total_features = torch.cat((total_features, features), dim=0)
                else:
                    total_features = features

                maxs = dnn_model.get_max_activations(images)

                if total_maxs is not None:
                    total_maxs = torch.cat((total_maxs, maxs), dim=0)
                else:
                    total_maxs = maxs
        dnn_model.toggle_full_precision()

        print('start calculating correlations...')
        coef_per_features = torch.zeros(total_features.size(1)).cuda()
        for feature_idx in range(total_features.size(1)):
            # print(f'feature index : {feature_idx} ')
            coef_per_features[feature_idx] = pearson_correlation(total_features[:, feature_idx], total_maxs[:, 0])

        _, idx = coef_per_features.topk(k=4)
        torch.save(idx, os.path.join(self.args.clustering_path, "index.pth"))

        return idx


    def train_clustering_model(self, nonaug_loader):
        print('Train K-means clustering model..')
        best_model = None
        x = None
        print(">> Load Non-augmented dataset & get representations for clustering..")
        with tqdm(nonaug_loader, unit="batch", ncols=90) as t:
            for image, _ in t:
                batch = self.get_partitioned_batch(image.cuda()).clone()
                if x is None:
                    x = batch
                else:
                    x = torch.cat((x, batch))
                    
        n_prediction_cluster = self.args.sub_cluster if self.args.sub_cluster else self.args.cluster
        best_model_inertia = 9999999999999999
        x = x.cuda()
        print("Train K-means model 5 times, and choose the best model")
        for trial in range(5):
            model = KMeans(n_clusters=n_prediction_cluster, random_state=0).fit(x)
            if model.inertia_ < best_model_inertia:
                best_model = model
                best_model_inertia = model.inertia_
            print("Trial-{} done".format(trial))

        path = self.args.clustering_path
        joblib.dump(best_model, os.path.join(path + '/checkpoint.pkl'))
        with open(os.path.join(path, "params.json"), 'w') as f:
            args_to_save = {
                'dataset': self.args.dataset,
                'clustering_method': self.args.clustering_method,
                'repr_method': self.args.repr_method,
                'partition_method': self.args.partition_method,
                'num_partitions': self.args.partition,
                'k': self.args.cluster
            }
            if self.args.dataset == 'imagenet':
                args_to_save.update({
                    'tol': self.args.kmeans_tol,
                    'n_inits': self.args.kmeans_init,
                    'epoch': self.args.kmeans_epoch,
                })
            json.dump(args_to_save, f, indent=4)
        self.model = best_model

    @torch.no_grad()
    def zero_max_nn_aware_clustering(self, dnn_model, train_loader, arch):
        print('\n>>> zero-max NN-aware Clustering..')
        from utils.misc import InputContainer

        n_sub_clusters = self.args.sub_cluster
        container = InputContainer(
            train_loader, self, n_sub_clusters, self.args.dataset, arch, self.args.batch)
        container.initialize_generator()
        container.set_next_batch()

        print('Count Max Values and Zero Counter per cluster about dataset..')
        n_per_sub = [0 for _ in range(n_sub_clusters)]
        dnn_model.eval()
        with tqdm(range(len(train_loader)), desc="Merge Clusters", ncols=90) as t:
            for i, _ in enumerate(t):
                input, _, cluster = container.get_batch()

                n_per_sub[cluster] += self.args.batch
                dnn_model.get_output_max_distribution(
                    input, cluster, n_sub_clusters)
                dnn_model.count_zeros_per_index(
                    input, cluster, n_sub_clusters)

                container.set_next_batch()
                if container.ready_cluster is None:
                    break
            container.check_leftover()
            for c in range(container.num_clusters):
                if container.leftover_cluster_data[c]:
                    input, _, cluster = container.leftover_batch[c][0], \
                        container.leftover_batch[c][1], c
                    n_per_sub[cluster] += input.size(0)
                    dnn_model.get_output_max_distribution(
                        input, cluster, n_sub_clusters)
                    dnn_model.count_zeros_per_index(
                        input, cluster, n_sub_clusters)

        # for i, max_counters in enumerate(dnn_model.max_counter):
        #     size_counter = []
        #     for max in max_counters:
        #         size_counter.append(len(max))
        #     total_max_per_layer = torch.cat(max_counters)
            
        #     # Normalize
        #     # normalized_max_per_layer = torch.nn.functional.normalize(total_max_per_layer, p=torch.inf, dim=0)

        #     # Standardize
        #     std, mean = torch.std_mean(total_max_per_layer, dim=0, unbiased=False)
        #     normalized_max_per_layer = (total_max_per_layer - mean) / std

        #     dnn_model.max_counter[i] = list(torch.split(normalized_max_per_layer, size_counter))

        # # Handle Empty Clusters
        # for c in range(n_sub_clusters):
        #     if dnn_model.max_counter[0][c] == []:
        #         for l in range(len(dnn_model.max_counter)):
        #             dnn_model.max_counter[l][c] = torch.zeros(1).cuda()



        print("\n>>> [Original] Number of data per cluster")
        for c in range(n_sub_clusters):
            print(f"C{c}: {n_per_sub[c]}")

        n_layers = len(dnn_model.zero_counter)
        n_per_sub = torch.tensor(n_per_sub).cuda()

        # [layer_idx] .. cluster size
        n_features = torch.tensor([dnn_model.zero_counter[i].size(1) for i in range(n_layers)]).cuda()
        # [cluster_idx, layer_idx, neurons] .. zero count
        dnn_model.zero_counter = torch.transpose(torch.stack([torch.nn.ConstantPad1d((0, torch.amax(n_features) - n_features[i]), 0)(dnn_model.zero_counter[i]) for i in range(n_layers)]), 0, 1)

        def check_merged_groups(merged_groups, cluster_id, cur_idx):
            for idx in range(len(merged_groups)):
                if idx == cur_idx:
                    continue
                merged_group = merged_groups[idx][0]
                if cluster_id in merged_group:
                    return idx
            return -1

        def calc_cross_similarity(zero_ratio, _from, _to, sim_method):
            if sim_method == 'and':
                similarity = torch.logical_and(zero_ratio[_from], zero_ratio[_to]).sum(dim=1) / \
                        n_features
            else:
                similarity = torch.logical_and(zero_ratio[_from], zero_ratio[_to]).sum(dim=1) / \
                        torch.logical_or(zero_ratio[_from], zero_ratio[_to]).sum(dim=1)
            return torch.nan_to_num(similarity)

        merged_clusters = []

        to_merge = n_sub_clusters - self.args.cluster
        n_merged = 0
        similarity_threshold = self.args.sim_threshold

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        while n_merged < to_merge:
            print(f'\n>>> Number of clusters to be merged: {to_merge - n_merged}')

            n_candidates_per_layer = int(
                ((n_sub_clusters - n_merged) * (n_sub_clusters - n_merged - 1))/2 * 0.8)  # discard low 20%
            zero_ratio = deepcopy(dnn_model.zero_counter)

            # Set True / False by Zero Count Threshold
            zero_ratio = torch.div(zero_ratio, n_per_sub.view(-1, 1, 1))
            zero_ratio = torch.where(zero_ratio > similarity_threshold, True, False)

            # Exclude merged clusters except 1 left
            exclude = set()
            for group in merged_clusters:
                exclude.update(group[0] - {min(group[0])})

            cross_similarity = torch.zeros(n_sub_clusters, n_sub_clusters, n_layers, device='cuda')

            current_num_clusters = n_sub_clusters - len(exclude)

            # record cross_similarity
            for _from in range(n_sub_clusters):
                if _from in exclude:
                    continue
                for _to in range(_from + 1, n_sub_clusters):
                    if _to in exclude:
                        continue
                    cross_similarity[_from][_to] = calc_cross_similarity(zero_ratio, _from, _to, self.args.similarity_method)

            # [cluster_idx, cluster_idx, layer_idx] -> [layer_idx, cluster_idx, cluster_idx] .. similarity
            cross_similarity = torch.transpose(torch.transpose(cross_similarity, 0, 2), 1, 2)

            count_duplicated_candidates = dict()
            dist = cross_similarity.view(cross_similarity.size(0), -1)
            _similarity, _cluster_idx = torch.topk(dist, n_candidates_per_layer)

            zero_sim_layer = torch.count_nonzero(_similarity.size(1) - torch.count_nonzero(_similarity, dim=1))
            candidates, counts = torch.unique(_cluster_idx, return_counts=True)
            candidates = torch.where(counts >= (n_layers-zero_sim_layer), candidates, 0)
            candidates = candidates[candidates.nonzero()]

            print("number of candidates : ", candidates.size(0))

            # finding best pair from choosen candidates
            # measure median or mean of individual cluster's max values
            n_layers = len(dnn_model.max_counter)
            cur_max_counter = deepcopy(dnn_model.max_counter)
            max_ratio = torch.zeros((n_layers, n_sub_clusters), device='cuda')


            # Clipping value approximation
            for l in range(n_layers):
                for c in range(n_sub_clusters):
                    max_ratio[l][c] = torch.quantile(cur_max_counter[l][c], 0.997)

            max_ratio = torch.transpose(max_ratio, 0, 1)

            # measure L2 distance between clusters
            cross_similarity = torch.full(
                (n_sub_clusters, n_sub_clusters), float('inf'), device='cuda')


            # Distance Metric
            dist = torch.nn.PairwiseDistance(p=2)

            for candidate in candidates:
                _from = int(candidate // n_sub_clusters)
                _to = int(candidate % n_sub_clusters)
                cross_similarity[_from][_to] = dist(
                    max_ratio[_from], max_ratio[_to])

            # choose pair of clusters with smallest L2 distance
            pair = (cross_similarity == (torch.min(cross_similarity))
                    ).nonzero(as_tuple=True)

            pair = [cluster[0].item() for cluster in pair]

            # merge clusters
            print(f'Merge', end='')
            c1, c2 = pair[0], pair[1]
            n_c1, n_c2 = n_per_sub[c1], n_per_sub[c2]
            summed = n_c1 + n_c2
            for g in range(len(merged_clusters)):
                group = merged_clusters[g][0]
                if c1 in group and c2 in group:
                    break
                elif c1 in group:
                    print(f' {c1}&{c2}')
                    group_id = check_merged_groups(
                        merged_clusters, c2, g)
                    if group_id == -1:
                        group.add(c2)
                    else:
                        group.update(merged_clusters[group_id][0])

                    merged_clusters[g][1] += n_c2
                    n_per_sub[list(group)] = merged_clusters[g][1]

                    for l in range(n_layers):
                        max_merged_count = torch.cat(
                            [dnn_model.max_counter[l][c1], dnn_model.max_counter[l][c2]])
                        dnn_model.max_counter[l][c1] = max_merged_count
                        dnn_model.max_counter[l][c2] = max_merged_count

                    zero_merged_count = dnn_model.zero_counter[c1] + dnn_model.zero_counter[c2]
                    dnn_model.zero_counter[c1] = zero_merged_count
                    dnn_model.zero_counter[c2] = zero_merged_count
                    if group_id != -1:
                        del merged_clusters[group_id]
                    break
                elif c2 in group:
                    print(f' {c1}&{c2}')
                    group_id = check_merged_groups(
                        merged_clusters, c1, g)
                    if group_id == -1:
                        group.add(c1)
                    else:
                        group.update(merged_clusters[group_id][0])

                    merged_clusters[g][1] += n_c1
                    n_per_sub[list(group)] = merged_clusters[g][1]
                    for l in range(n_layers):
                        max_merged_count = torch.cat(
                            [dnn_model.max_counter[l][c1], dnn_model.max_counter[l][c2]])
                        dnn_model.max_counter[l][c1] = max_merged_count
                        dnn_model.max_counter[l][c2] = max_merged_count

                    zero_merged_count = dnn_model.zero_counter[c1] + dnn_model.zero_counter[c2]
                    dnn_model.zero_counter[c1] = zero_merged_count
                    dnn_model.zero_counter[c2] = zero_merged_count
                    if group_id != -1:
                        del merged_clusters[group_id]
                    break
            else:
                print(f' {c1}&{c2}')
                merged_clusters.append([{c1, c2}, summed])
                n_per_sub[[c1, c2]] = summed
                for l in range(n_layers):
                    max_merged_count = torch.cat(
                        [dnn_model.max_counter[l][c1], dnn_model.max_counter[l][c2]])
                    dnn_model.max_counter[l][c1] = max_merged_count
                    dnn_model.max_counter[l][c2] = max_merged_count

                zero_merged_count = dnn_model.zero_counter[c1] + dnn_model.zero_counter[c2]
                dnn_model.zero_counter[c1] = zero_merged_count
                dnn_model.zero_counter[c2] = zero_merged_count

            n_merged = 0
            for group in merged_clusters:
                n_merged += len(group[0]) - 1

            print("distance : ", torch.min(cross_similarity).item())

        final_clusters = dict()
        n_per_final = [0 for _ in range(self.args.cluster)]

        k = 0  # Final cluster ID
        leftover_clusters = set(range(n_sub_clusters))
        print("\n>>> Merged clusters")
        for merged_single_cluster in merged_clusters:
            group = merged_single_cluster[0]
            print(f"C{k}: {tuple(group)}")

            leftover_clusters = leftover_clusters.difference(group)
            for cluster in group:
                final_clusters[str(cluster)] = k
            n_per_final[k] = merged_single_cluster[1]
            k += 1

        for cluster in leftover_clusters:
            final_clusters[str(cluster)] = k
            n_per_final[k] = n_per_sub[cluster]
            k += 1

        print(
            f"\n>>> [Final] Number of data per cluster")
        for c in range(self.args.cluster):
            print(f"C{c}: {n_per_final[c]}")

        with open(os.path.join(self.args.clustering_path, 'params.json'), 'r') as f:
            args_without_nnac = json.load(f)
            if args_without_nnac['k'] != self.args.cluster:
                path = self.args.clustering_path + \
                    f'__.k{self.args.cluster}.sub{self.args.sub_cluster}.topk_{self.args.topk}.sim_{self.args.sim_threshold}.{self.args.similarity_method}'
                print(
                    f"Copy json and pkl file from {self.args.clustering_path} to {path}")
                if not os.path.exists(path):
                    os.makedirs(path)
                import shutil
                shutil.copyfile(os.path.join(self.args.clustering_path, 'checkpoint.pkl'),
                                os.path.join(path, 'checkpoint.pkl'))
                self.args.clustering_path = path
                args_without_nnac['k'] = self.args.cluster

        with open(os.path.join(self.args.clustering_path, "params.json"), 'w') as f:
            args_without_nnac['sub_k'] = self.args.sub_cluster
            args_without_nnac['nnac'] = final_clusters
            json.dump(args_without_nnac, f, indent=4)

        torch.save(self.feature_index, os.path.join(self.args.clustering_path, "index.pth"))

        self.final_cluster = torch.zeros(
            self.args.sub_cluster, dtype=torch.int64)
        for sub, final in final_clusters.items():
            self.final_cluster[int(sub)] = final
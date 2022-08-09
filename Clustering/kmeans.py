import torch

from sklearn.cluster import MiniBatchKMeans
from cuml.cluster import KMeans
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
        self.final_cluster = None  # Used in NN-aware Clustering only

    @torch.no_grad()
    def get_partitioned_batch(self, data):
        # Under the premise that images are in the form of square matrix
        batch = data.size(0)
        channel = data.size(1)
        _size = data.size(2)
        if _size % 2:
            m = torch.nn.ZeroPad2d((0, 1, 0, 1))
            data = m(data)
            _size = data.size(2)
        if self.args.partition_method == 'square':
            n_part = self.args.partition
            n_data = int(_size / n_part)  # Per part

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
            # return rst.view(rst.size(0), -1).numpy()
            return rst.view(rst.size(0), -1)
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

    @torch.no_grad()
    def predict_cluster_of_batch(self, input):
        kmeans_input = self.get_partitioned_batch(input)
        cluster_info = self.model.predict(kmeans_input)
        if self.final_cluster is not None:  # make output as merged cluster form
            return torch.index_select(self.final_cluster, 0, torch.LongTensor(cluster_info))
        return torch.LongTensor(cluster_info)

    def train_clustering_model(self, nonaug_loader, aug_loader):
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

        print(">> Load augmented datasets & mix dataset..")
        for _ in range(self.args.mixrate):
            with tqdm(aug_loader, unit="batch", ncols=90) as t:
                for image, _ in t:
                    batch = self.get_partitioned_batch(image.cuda()).clone()
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

        # Handle Empty Clusters
        for c in range(n_sub_clusters):
            if dnn_model.max_counter[0][c] == []:
                for l in range(len(dnn_model.max_counter)):
                    dnn_model.max_counter[l][c] = torch.zeros(1).cuda()

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
            # if len(pairs) == 0:
            #     self.args.cluster = n_sub_clusters - n_merged
            #     print("no candidates to merge... stop at ", self.args.cluster)
            #     break

            # finding best pair from choosen candidates
            # measure median or mean of individual cluster's max values
            n_layers = len(dnn_model.max_counter)
            cur_max_counter = deepcopy(dnn_model.max_counter)
            max_ratio = torch.zeros((n_layers, n_sub_clusters), device='cuda')

            if self.args.max_method == 'median':
                percentile_tensor = torch.tensor([0.5], device='cuda')
                for l in range(n_layers):
                    for c in range(n_sub_clusters):
                        max_ratio[l][c] = torch.quantile(
                            cur_max_counter[l][c], percentile_tensor)
            elif self.args.max_method == 'mean':
                for l in range(n_layers):
                    for c in range(n_sub_clusters):
                        max_ratio[l][c] = cur_max_counter[l][c].mean()
            else:
                raise Exception('max method not implemented')

            max_ratio = torch.transpose(max_ratio, 0, 1)

            # measure L2 distance between clusters
            cross_similarity = torch.full(
                (n_sub_clusters, n_sub_clusters), float('inf'), device='cuda')

            l2_dist = torch.nn.PairwiseDistance(p=2)

            for candidate in candidates:
                # with torch.cuda.stream(streams[torch.randint(num_streams, (1,))]):
                _from = int(candidate // n_sub_clusters)
                _to = int(candidate % n_sub_clusters)
                cross_similarity[_from][_to] = l2_dist(
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

        self.final_cluster = torch.zeros(
            self.args.sub_cluster, dtype=torch.int64)
        for sub, final in final_clusters.items():
            self.final_cluster[int(sub)] = final

    @torch.no_grad()
    def max_nn_aware_clustering(self, dnn_model, train_loader, arch):
        print('\n>>> max NN-aware Clustering..')
        from utils.misc import InputContainer

        n_sub_clusters = self.args.sub_cluster
        container = InputContainer(
            train_loader, self, n_sub_clusters, self.args.dataset, self.args.batch)
        container.initialize_generator()
        container.set_next_batch()

        print('Count Max Values per cluster about dataset..')
        n_per_sub = [0 for _ in range(n_sub_clusters)]
        dnn_model.eval()
        with tqdm(range(len(train_loader)), desc="Merge Clusters", ncols=90) as t:
            for i, _ in enumerate(t):
                input, _, cluster = container.get_batch()

                n_per_sub[cluster] += self.args.batch
                dnn_model.get_output_max_distribution(
                    input.cuda(), cluster, n_sub_clusters)

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
                        input.cuda(), cluster, n_sub_clusters)

        print("\n>>> [Original] Number of data per cluster")
        for c in range(n_sub_clusters):
            print(f"C{c}: {n_per_sub[c]}")
        n_per_sub = torch.tensor(n_per_sub)
        # max_data_num_per_merged_cluster = sum(n_per_sub) / (self.args.cluster - 1)
        max_data_num_per_merged_cluster = sum(n_per_sub) / 2
        threshold_per_merged_cluster = max_data_num_per_merged_cluster

        def check_merged_groups(groups, cluster_id, cur_idx):
            for idx in range(len(groups)):
                if idx == cur_idx:
                    continue
                nxt_group = groups[idx][0]
                if cluster_id in nxt_group:
                    return idx
            return -1

        n_layers = len(dnn_model.max_counter)
        n_candidates_per_layer = self.args.topk
        merged_clusters = []

        to_merge = n_sub_clusters - self.args.cluster
        n_merged = 0

        while n_merged < to_merge:
            print(
                f'\n>>> Number of clusters to be merged: {to_merge - n_merged}')
            indices = [i for i in range(n_layers)]
            # Normalize with n_data of cluster, and make 1 if greater than 80 %
            cur_max_counter = deepcopy(dnn_model.max_counter)
            max_ratio = torch.zeros(
                (len(cur_max_counter), n_sub_clusters), device='cuda')

            if self.args.max_method == 'median':
                percentile_tensor = torch.tensor([0.5], device='cuda')
                for l in range(n_layers):
                    for c in range(n_sub_clusters):
                        max_ratio[l][c] = torch.quantile(
                            cur_max_counter[l][c], percentile_tensor)
            else:
                for l in range(n_layers):
                    for c in range(n_sub_clusters):
                        max_ratio[l][c] = cur_max_counter[l][c].mean()

            print('Calc. Distance between clusters.. (`1` means both zero)')
            # Exclude merged clusters except 1 left
            exclude = set()
            for group in merged_clusters:
                exclude.update(group[0] - {min(group[0])})

            cross_similarity = torch.ones(
                n_layers, n_sub_clusters, n_sub_clusters, device='cuda')
            for l in range(n_layers):
                if l not in indices:
                    continue
                for _from in range(n_sub_clusters):
                    if _from in exclude:
                        continue
                    for _to in range(_from + 1, n_sub_clusters):
                        if _to in exclude:
                            continue
                        cross_similarity[l][_from][_to] = abs(
                            max_ratio[l][_from] - max_ratio[l][_to])

            sorted_dist, sorted_indices = torch.sort(
                cross_similarity, dim=2, descending=False)

            candidates_per_layer = [[] for _ in range(n_layers)]
            count_duplicated_candidates = dict()
            for l in range(n_layers):
                if l not in indices:
                    continue
                l_dist = sorted_dist[l].view(-1)
                l_idx = sorted_indices[l].view(-1)

                v_of_sorted, i_of_sorted = torch.topk(
                    l_dist, n_candidates_per_layer, largest=False)
                for c in range(n_candidates_per_layer):
                    if v_of_sorted[c] != 0.0:
                        row = i_of_sorted[c] // n_sub_clusters
                        col = l_idx[i_of_sorted[c]]
                        i_of_original = (row.item(), col.item())

                        candidates_per_layer[l].append(i_of_original)
                        if count_duplicated_candidates.get(i_of_original):
                            count_duplicated_candidates[i_of_original][0] += 1
                            count_duplicated_candidates[i_of_original][1] += n_candidates_per_layer - c
                            count_duplicated_candidates[i_of_original][2].append(
                                l)
                        else:
                            count_duplicated_candidates[i_of_original] = [
                                1, n_candidates_per_layer - c, [l]]

            counted = count_duplicated_candidates.items()
            similar_cluster_pairs = sorted(
                counted, key=lambda x: (x[1][0], x[1][1]), reverse=True)

            print(f'Merge', end='')
            for p in range(len(similar_cluster_pairs)):
                merged = False
                pair = similar_cluster_pairs[p]
                c1, c2 = pair[0][0], pair[0][1]
                n_c1, n_c2 = n_per_sub[c1], n_per_sub[c2]
                summed = n_c1 + n_c2
                if summed < threshold_per_merged_cluster:
                    for g in range(len(merged_clusters)):
                        group = merged_clusters[g][0]
                        if c1 in group and c2 in group:
                            break
                        elif c1 in group:
                            merged = True
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
                                merged_count = torch.cat(
                                    [dnn_model.max_counter[l][c1], dnn_model.max_counter[l][c2]])
                                dnn_model.max_counter[l][c1] = merged_count
                                dnn_model.max_counter[l][c2] = merged_count
                            if group_id != -1:
                                del merged_clusters[group_id]
                            break
                        elif c2 in group:
                            merged = True
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
                                merged_count = torch.cat(
                                    [dnn_model.max_counter[l][c1], dnn_model.max_counter[l][c2]])
                                dnn_model.max_counter[l][c1] = merged_count
                                dnn_model.max_counter[l][c2] = merged_count
                            if group_id != -1:
                                del merged_clusters[group_id]
                            break
                    else:
                        merged = True
                        print(f' {c1}&{c2}')
                        merged_clusters.append([{c1, c2}, summed])
                        n_per_sub[[c1, c2]] = summed
                        for l in range(n_layers):
                            merged_count = torch.cat(
                                [dnn_model.max_counter[l][c1], dnn_model.max_counter[l][c2]])
                            dnn_model.max_counter[l][c1] = merged_count
                            dnn_model.max_counter[l][c2] = merged_count
                if merged:
                    # threshold_per_merged_cluster = max_data_num_per_merged_cluster
                    break
                # else :
                #     threshold_per_merged_cluster = int(threshold_per_merged_cluster * 1.05)

            n_merged = 0
            for group in merged_clusters:
                n_merged += len(group[0]) - 1

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
            f"\n>>> [Final] Number of data per cluster (Max.limit: {max_data_num_per_merged_cluster})")
        for c in range(self.args.cluster):
            print(f"C{c}: {n_per_final[c]}")

        # with open(os.path.join(self.args.clustering_path, 'params.json'), 'r') as f:
        #     args_without_nnac = json.load(f)
        # with open(os.path.join(self.args.clustering_path, "params.json"), 'w') as f:
        #     args_without_nnac['sub_k'] = self.args.sub_cluster
        #     args_without_nnac['nnac'] = final_clusters
        #     json.dump(args_without_nnac, f, indent=4)
        with open(os.path.join(self.args.clustering_path, 'params.json'), 'r') as f:
            args_without_nnac = json.load(f)
            if args_without_nnac['k'] != self.args.cluster:
                path = self.args.clustering_path + \
                    f'__.nnac_{arch}_k{self.args.cluster}_sub{self.args.sub_cluster}_topk_{self.args.topk}_sim_{self.args.sim_threshold}'
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

        self.final_cluster = torch.zeros(
            self.args.sub_cluster, dtype=torch.int64)
        for sub, final in final_clusters.items():
            self.final_cluster[int(sub)] = final

    @torch.no_grad()
    def nn_aware_clustering(self, dnn_model, train_loader, arch):
        print('\n>>> NN-aware Clustering..')
        from utils.misc import InputContainer

        n_sub_clusters = self.args.sub_cluster
        container = InputContainer(
            train_loader, self, n_sub_clusters, self.args.dataset, self.args.batch)
        container.initialize_generator()
        container.set_next_batch()

        print('Count zero indices per cluster about dataset..')
        n_per_sub = [0 for _ in range(n_sub_clusters)]
        dnn_model.eval()
        with tqdm(range(len(train_loader)), desc="Merge Clusters", ncols=90) as t:
            for i, _ in enumerate(t):
                input, _, cluster = container.get_batch()

                n_per_sub[cluster] += self.args.batch
                dnn_model.count_zeros_per_index(
                    input.cuda(), cluster, n_sub_clusters)

                container.set_next_batch()
                if container.ready_cluster is None:
                    break
            container.check_leftover()
            for c in range(container.num_clusters):
                if container.leftover_cluster_data[c]:
                    input, _, cluster = container.leftover_batch[c][0], \
                        container.leftover_batch[c][1], c
                    n_per_sub[cluster] += input.size(0)
                    dnn_model.count_zeros_per_index(
                        input.cuda(), cluster, n_sub_clusters)

        print("\n>>> [Original] Number of data per cluster")
        for c in range(n_sub_clusters):
            print(f"C{c}: {n_per_sub[c]}")
        n_per_sub = torch.tensor(n_per_sub)
        # max_data_num_per_merged_cluster = sum(n_per_sub) / (self.args.cluster - 1)
        max_data_num_per_merged_cluster = sum(n_per_sub) / 2
        threshold_per_merged_cluster = max_data_num_per_merged_cluster

        def check_merged_groups(merged_groups, cluster_id, cur_idx):
            for idx in range(len(merged_groups)):
                if idx == cur_idx:
                    continue
                merged_group = merged_groups[idx][0]
                if cluster_id in merged_group:
                    return idx
            return -1

        n_layers = len(dnn_model.zero_counter)
        n_candidates_per_layer = self.args.topk
        merged_clusters = []

        to_merge = n_sub_clusters - self.args.cluster
        n_merged = 0
        similarity_threshold = self.args.sim_threshold

        while n_merged < to_merge:
            print(
                f'\n>>> Number of clusters to be merged: {to_merge - n_merged}')
            indices = [i for i in range(n_layers)]
            zero_ratio = deepcopy(dnn_model.zero_counter)
            for l in range(n_layers):
                for c in range(n_sub_clusters):
                    zero_ratio[l][c] /= n_per_sub[c]
                zero_ratio[l] = torch.where(
                    zero_ratio[l] > similarity_threshold, 1, 0)

            print('Calc. `And` between clusters.. (`1` means both zero)')
            # Exclude merged clusters except 1 left
            exclude = set()
            for group in merged_clusters:
                exclude.update(group[0] - {min(group[0])})

            if self.args.exclude:
                zero_ratio_per_layer = torch.zeros(n_layers, device='cuda')
                for l in range(n_layers):
                    for c in range(n_sub_clusters):
                        if c in exclude:
                            continue
                        zero_ratio_per_layer[l] += zero_ratio[l][c].sum()
                    zero_ratio_per_layer[l] /= zero_ratio[l].size(
                        1) * (n_sub_clusters - len(exclude))
                indices = (zero_ratio_per_layer > 0.25).nonzero(
                    as_tuple=True)[0]

            cross_similarity = torch.zeros(
                n_layers, n_sub_clusters, n_sub_clusters, device='cuda')
            for l in range(n_layers):
                if l not in indices:
                    continue
                n_features = zero_ratio[l].size(1)
                for _from in range(n_sub_clusters):
                    if _from in exclude:
                        continue
                    for _to in range(_from + 1, n_sub_clusters):
                        if _to in exclude:
                            continue
                        if self.args.similarity_method == 'and':
                            n_commonly_zero = torch.logical_and(
                                zero_ratio[l][_from], zero_ratio[l][_to]).sum()
                            similarity = n_commonly_zero / n_features
                        else:
                            similarity = torch.logical_and(zero_ratio[l][_from], zero_ratio[l][_to]).sum() / \
                                torch.logical_or(
                                    zero_ratio[l][_from], zero_ratio[l][_to]).sum()
                        cross_similarity[l][_from][_to] = similarity

            sorted_dist, sorted_indices = torch.sort(
                cross_similarity, dim=2, descending=True)

            candidates_per_layer = [[] for _ in range(n_layers)]
            count_duplicated_candidates = dict()
            for l in range(n_layers):
                if l not in indices:
                    continue
                l_dist = sorted_dist[l].view(-1)
                l_idx = sorted_indices[l].view(-1)

                v_of_sorted, i_of_sorted = torch.topk(
                    l_dist, n_candidates_per_layer)
                for c in range(n_candidates_per_layer):
                    if v_of_sorted[c] != 0.0:
                        row = i_of_sorted[c] // n_sub_clusters
                        col = l_idx[i_of_sorted[c]]
                        i_of_original = (row.item(), col.item())

                        candidates_per_layer[l].append(i_of_original)
                        if count_duplicated_candidates.get(i_of_original):
                            count_duplicated_candidates[i_of_original][0] += 1
                            count_duplicated_candidates[i_of_original][1] += n_candidates_per_layer - c
                            count_duplicated_candidates[i_of_original][2].append(
                                l)
                        else:
                            count_duplicated_candidates[i_of_original] = [
                                1, n_candidates_per_layer - c, [l]]

            counted = count_duplicated_candidates.items()
            similar_cluster_pairs = sorted(
                counted, key=lambda x: (x[1][0], x[1][1]), reverse=True)

            print(f'Merge', end='')
            for p in range(len(similar_cluster_pairs)):
                merged = False
                pair = similar_cluster_pairs[p]
                c1, c2 = pair[0][0], pair[0][1]
                n_c1, n_c2 = n_per_sub[c1], n_per_sub[c2]
                summed = n_c1 + n_c2
                if summed < threshold_per_merged_cluster:
                    for g in range(len(merged_clusters)):
                        group = merged_clusters[g][0]
                        if c1 in group and c2 in group:
                            break
                        elif c1 in group:
                            merged = True
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
                                merged_count = dnn_model.zero_counter[l][c1] + \
                                    dnn_model.zero_counter[l][c2]
                                dnn_model.zero_counter[l][list(
                                    group)] = merged_count
                            if group_id != -1:
                                del merged_clusters[group_id]
                            break
                        elif c2 in group:
                            merged = True
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
                                merged_count = dnn_model.zero_counter[l][c1] + \
                                    dnn_model.zero_counter[l][c2]
                                dnn_model.zero_counter[l][list(
                                    group)] = merged_count
                            if group_id != -1:
                                del merged_clusters[group_id]
                            break
                    else:
                        merged = True
                        print(f' {c1}&{c2}')
                        merged_clusters.append([{c1, c2}, summed])
                        n_per_sub[[c1, c2]] = summed
                        for l in range(n_layers):
                            merged_count = dnn_model.zero_counter[l][c1] + \
                                dnn_model.zero_counter[l][c2]
                            dnn_model.zero_counter[l][[c1, c2]] = merged_count
                if merged:
                    # threshold_per_merged_cluster = max_data_num_per_merged_cluster
                    break
                # else :
                #     threshold_per_merged_cluster = int(threshold_per_merged_cluster * 1.05)

            n_merged = 0
            for group in merged_clusters:
                n_merged += len(group[0]) - 1

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
            f"\n>>> [Final] Number of data per cluster (Max.limit: {max_data_num_per_merged_cluster})")
        for c in range(self.args.cluster):
            print(f"C{c}: {n_per_final[c]}")

        with open(os.path.join(self.args.clustering_path, 'params.json'), 'r') as f:
            args_without_nnac = json.load(f)
            if args_without_nnac['k'] != self.args.cluster:
                path = self.args.clustering_path + \
                    f'__.nnac_{arch}_k{self.args.cluster}_sub{self.args.sub_cluster}_topk_{self.args.topk}_sim_{self.args.sim_threshold}'
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

        self.final_cluster = torch.zeros(
            self.args.sub_cluster, dtype=torch.int64)
        for sub, final in final_clusters.items():
            self.final_cluster[int(sub)] = final


def check_cluster_distribution(kmeans, train_loader):
    n_data = kmeans.args.data_per_cluster * \
        kmeans.args.cluster * len(train_loader)
    n_data_per_cluster = dict()
    for c in range(kmeans.args.cluster):
        n_data_per_cluster[c] = 0
    for i, (input, target) in enumerate(train_loader):
        batch_cluster = kmeans.predict_cluster_of_batch(input)
        for c in batch_cluster:
            n_data_per_cluster[c.item()] += 1

    assert sum(n_data_per_cluster.values()) == n_data, \
        "Total # of data doesn't match (n_data: {}, calc: {})".format(
            n_data, sum(n_data_per_cluster.values()))

    ratio = np.zeros((kmeans.args.cluster))
    for c in range(kmeans.args.cluster):
        ratio[c] = n_data_per_cluster[c] / n_data * 100

    for c in range(kmeans.args.cluster):
        print("{},{:.2f} %".format(n_data_per_cluster[c], ratio[c]))
    print(">> [#Data] Mean, Var, Std")
    d = list((n_data_per_cluster.values()))
    print("{}, {:.2f}, {:.2f}".format(np.mean(d), np.var(d), np.std(d)))
    print(">> [Ratio] Mean, Var, Std")
    print("{:.2f} %, {:.4f}, {:.4f}".format(
        np.mean(ratio), np.var(ratio), np.std(ratio)))
    centroids = kmeans.model.cluster_centers_
    print(">> [Centroids] Var, Std")
    print("var: {:.4f}, std: {:.4f}".format(
        np.var(centroids), np.std(centroids)))

import torch

from sklearn.cluster import KMeans, MiniBatchKMeans
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
        if self.args.partition_method == 'square':
            n_part = self.args.partition
            n_data = int(_size / n_part)  # Per part

            data = data.view(batch, channel, n_part, n_data, _size).transpose(3, 4)
            data = data.reshape(batch, channel, n_part * n_part, -1)

            if self.args.repr_method == 'max':
                rst, _ = data.topk(k=3, dim=-1)
                rst = rst.mean(-1, keepdim=True)
            elif self.args.repr_method == 'mean':
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
                print("Warning: NN-aware Clustering mode is on now, but loaded model doesn't have final cluster info.")
            else:
                assert self.args.sub_cluster == loaded_args['sub_k'], \
                    f"Number of sub-clusters doesn't match (arg: {self.args.sub_cluster}, loaded: {loaded_args['sub_k']}"
                self.final_cluster = torch.zeros(self.args.sub_cluster, dtype=torch.int64)
                print(">>> NN-aware Clustering mode")
                print(loaded_args['nnac'])
                for sub, final in loaded_args['nnac'].items():
                    self.final_cluster[int(sub)] = int(final)

        self.model = joblib.load(os.path.join(self.args.clustering_path, 'checkpoint.pkl'))

    def predict_cluster_of_batch(self, input):
        kmeans_input = self.get_partitioned_batch(input)
        cluster_info = self.model.predict(np.float64(kmeans_input))
        if self.final_cluster is not None:  # make output as merged cluster form
            return torch.index_select(self.final_cluster, 0, torch.LongTensor(cluster_info))
        return torch.LongTensor(cluster_info)

    def train_clustering_model(self, nonaug_loader, aug_loader):
        print('Train K-means clustering model..')
        best_model = None
        if self.args.dataset == 'imagenet':
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

            print(">> Use Mini-batch K-means Clustering for ImageNet dataset")
            prev_centers = None
            is_converged = False
            best_model_inertia = 9999999999999999
            print("Train K-means model 10 times, and choose the best model")
            for trial in range(10):
                n_clusters = self.args.cluster if not self.args.nnac else self.args.sub_cluster
                model = MiniBatchKMeans(n_clusters=n_clusters, batch_size=self.args.batch,
                                        tol=self.args.kmeans_tol, random_state=0)
                for epoch in range(self.args.kmeans_epoch):
                    with tqdm(nonaug_loader, desc="Trial-{} Epoch {}".format(trial, epoch), position=0, ncols=90) as t:
                        for image, _ in t:
                            train_data = torch.tensor(self.get_partitioned_batch(image))
                            model = model.partial_fit(train_data)

                            if prev_centers is not None:
                                is_converged = check_convergence(prev_centers, model.cluster_centers_, model.tol)
                                if is_converged:
                                    break
                            prev_centers = deepcopy(model.cluster_centers_)
                        if is_converged:
                            break
                if model.inertia_ < best_model_inertia:
                    best_model = model
                    best_model_inertia = model.inertia_
        else:
            x = None
            print(">> Load Non-augmented dataset & get representations for clustering..")
            with tqdm(nonaug_loader, unit="batch", ncols=90) as t:
                for image, _ in t:
                    batch = torch.tensor(self.get_partitioned_batch(image))
                    if x is None:
                        x = batch
                    else:
                        x = torch.cat((x, batch))

            print(">> Load augmented datasets & mix dataset..")
            for _ in range(self.args.mixrate):
                with tqdm(aug_loader, unit="batch", ncols=90) as t:
                    for image, _ in t:
                        batch = torch.tensor(self.get_partitioned_batch(image))
                        x = torch.cat((x, batch))

            n_prediction_cluster = self.args.sub_cluster if self.args.sub_cluster else self.args.cluster
            best_model = KMeans(n_clusters=n_prediction_cluster, random_state=0).fit(x)

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
    def nn_aware_clustering(self, dnn_model, train_loader, arch):
        print('\n>>> NN-aware Clustering..')
        from utils.misc import InputContainer

        n_sub_clusters = self.args.sub_cluster
        container = InputContainer(train_loader, self, n_sub_clusters, self.args.dataset, self.args.batch)
        container.initialize_generator()
        container.set_next_batch()

        print('Count zero indices per cluster about dataset..')
        n_per_sub = [0 for _ in range(n_sub_clusters)]
        dnn_model.eval()
        with tqdm(range(len(train_loader)), desc="Merge Clusters", ncols=90) as t:
            for i, _ in enumerate(t):
                input, _, cluster = container.get_batch()

                n_per_sub[cluster] += self.args.batch
                dnn_model.count_zeros_per_index(input.cuda(), cluster, n_sub_clusters)

                container.set_next_batch()
                if container.ready_cluster is None:
                    break
            container.check_leftover()
            for c in range(container.num_clusters):
                if container.leftover_cluster_data[c]:
                    input, _, cluster = container.leftover_batch[c][0], \
                                             container.leftover_batch[c][1], c
                    n_per_sub[cluster] += input.size(0)
                    dnn_model.count_zeros_per_index(input.cuda(), cluster, n_sub_clusters)

        print("\n>>> [Original] Number of data per cluster")
        for c in range(n_sub_clusters):
            print(f"C{c}: {n_per_sub[c]}")
        n_per_sub = torch.tensor(n_per_sub)
        max_data_num_per_merged_cluster = sum(n_per_sub) / (self.args.cluster - 1)
        threshold_per_merged_cluster = max_data_num_per_merged_cluster

        def check_other_groups(groups, cluster_id, cur_idx):
            for idx in range(len(groups)):
                if idx == cur_idx:
                    continue
                nxt_group = groups[idx][0]
                if cluster_id in nxt_group:
                    return idx
            return -1

        n_layers = len(dnn_model.zero_counter)
        n_candidates_per_layer = self.args.topk
        merged_clusters = []

        to_merge = n_sub_clusters - self.args.cluster
        n_merged = 0
        similarity_threshold = self.args.sim_threshold

        while n_merged < to_merge:
            print(f'\n>>> Number of clusters to be merged: {to_merge - n_merged}')
            indices = [i for i in range(n_layers)]
            # Normalize with n_data of cluster, and make 1 if greater than 80 %
            zero_ratio = deepcopy(dnn_model.zero_counter)
            for l in range(n_layers):
                for c in range(n_sub_clusters):
                    zero_ratio[l][c] /= n_per_sub[c]  # Normalize counts by number of data in cluster
                zero_ratio[l] = torch.where(zero_ratio[l] > similarity_threshold, 1, 0)

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
                    zero_ratio_per_layer[l] /= zero_ratio[l].size(1) * (n_sub_clusters - len(exclude))
                indices = (zero_ratio_per_layer > 0.25).nonzero(as_tuple=True)[0]

            cross_similarity = torch.zeros(n_layers, n_sub_clusters, n_sub_clusters, device='cuda')
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
                            n_commonly_zero = torch.logical_and(zero_ratio[l][_from], zero_ratio[l][_to]).sum()
                            similarity = n_commonly_zero / n_features
                        else : 
                            similarity = torch.logical_and(zero_ratio[l][_from], zero_ratio[l][_to]).sum() / \
                                    torch.logical_or(zero_ratio[l][_from], zero_ratio[l][_to]).sum()
                        cross_similarity[l][_from][_to] = similarity

            sorted_dist, sorted_indices = torch.sort(cross_similarity, dim=2, descending=True)

            candidates_per_layer = [[] for _ in range(n_layers)]
            count_duplicated_candidates = dict()
            for l in range(n_layers):
                if l not in indices:
                    continue
                l_dist = sorted_dist[l].view(-1)
                l_idx = sorted_indices[l].view(-1)

                v_of_sorted, i_of_sorted = torch.topk(l_dist, n_candidates_per_layer)
                for c in range(n_candidates_per_layer):
                    if v_of_sorted[c] != 0.0:
                        row = i_of_sorted[c] // n_sub_clusters
                        col = l_idx[i_of_sorted[c]]
                        i_of_original = (row.item(), col.item())

                        candidates_per_layer[l].append(i_of_original)
                        if count_duplicated_candidates.get(i_of_original):
                            count_duplicated_candidates[i_of_original][0] += 1
                            count_duplicated_candidates[i_of_original][1] += n_candidates_per_layer - c
                            count_duplicated_candidates[i_of_original][2].append(l)
                        else:
                            count_duplicated_candidates[i_of_original] = [1, n_candidates_per_layer - c, [l]]

            counted = count_duplicated_candidates.items()
            similar_cluster_pairs = sorted(counted, key=lambda x: (x[1][0], x[1][1]), reverse=True)

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
                            group_id = check_other_groups(merged_clusters, c2, g)
                            if group_id == -1:
                                group.add(c2)
                            else:
                                group.update(merged_clusters[group_id][0])

                            merged_clusters[g][1] += n_c2
                            n_per_sub[list(group)] = merged_clusters[g][1]
                            for l in range(n_layers):
                                merged_count = dnn_model.zero_counter[l][c1] + dnn_model.zero_counter[l][c2]
                                dnn_model.zero_counter[l][list(group)] = merged_count
                            if group_id != -1:
                                del merged_clusters[group_id]
                            break
                        elif c2 in group:
                            merged = True
                            print(f' {c1}&{c2}')
                            group_id = check_other_groups(merged_clusters, c1, g)
                            if group_id == -1:
                                group.add(c1)
                            else:
                                group.update(merged_clusters[group_id][0])

                            merged_clusters[g][1] += n_c1
                            n_per_sub[list(group)] = merged_clusters[g][1]
                            for l in range(n_layers):
                                merged_count = dnn_model.zero_counter[l][c1] + dnn_model.zero_counter[l][c2]
                                dnn_model.zero_counter[l][list(group)] = merged_count
                            if group_id != -1:
                                del merged_clusters[group_id]
                            break
                    else:
                        merged = True
                        print(f' {c1}&{c2}')
                        merged_clusters.append([{c1, c2}, summed])
                        n_per_sub[[c1, c2]] = summed
                        for l in range(n_layers):
                            merged_count = dnn_model.zero_counter[l][c1] + dnn_model.zero_counter[l][c2]
                            dnn_model.zero_counter[l][[c1, c2]] = merged_count
                if merged:
                    threshold_per_merged_cluster = max_data_num_per_merged_cluster
                    break
                else :
                    threshold_per_merged_cluster = int(threshold_per_merged_cluster * 1.05)

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

        print(f"\n>>> [Final] Number of data per cluster (Max.limit: {max_data_num_per_merged_cluster})")
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
                path = self.args.clustering_path + f'__.nnac_{arch}_k{self.args.cluster}_sub{self.args.sub_cluster}_topk_{self.args.topk}_sim_{self.args.sim_threshold}'
                print(f"Copy json and pkl file from {self.args.clustering_path} to {path}")
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

        self.final_cluster = torch.zeros(self.args.sub_cluster, dtype=torch.int64)
        for sub, final in final_clusters.items():
            self.final_cluster[int(sub)] = final

    # @torch.no_grad()
    # def nn_aware_clutering(self, dnn_model, train_loader):
    #     print('\n>>> NN-aware Clustering..')
    #     from utils.misc import InputContainer
    #
    #     n_sub_clusters = self.args.sub_cluster
    #     container = InputContainer(train_loader, self, n_sub_clusters, self.args.dataset, self.args.batch)
    #     container.initialize_generator()
    #     container.set_next_batch()
    #
    #     print('Count zero indices per cluster about dataset..')
    #     n_per_sub = [0 for _ in range(n_sub_clusters)]
    #     dnn_model.eval()
    #     with tqdm(range(len(train_loader)), desc="Merge Clusters", ncols=90) as t:
    #         for i, _ in enumerate(t):
    #             input, _, cluster = container.get_batch()
    #             n_per_sub[cluster] += self.args.batch
    #
    #             dnn_model.count_zeros_per_index(input.cuda(), cluster, n_sub_clusters)
    #
    #             container.set_next_batch()
    #             if container.ready_cluster is None:
    #                 break
    #
    #     # Normalize with n_data of cluster, and make 1 if greater than 80 %
    #     print('Normalize counter..')
    #     n_layers = len(dnn_model.zero_counter)
    #     for l in range(n_layers):
    #         for c in range(n_sub_clusters):
    #             dnn_model.zero_counter[l][c] /= n_per_sub[c]  # Normalize counts by number of data in cluster
    #         dnn_model.zero_counter[l] = torch.where(dnn_model.zero_counter[l] > self.args.sim_threshold, 1, 0)
    #     print('Calc. `And` between clusters.. (`1` means both zero)')
    #     cross_similarity = torch.zeros(n_layers, n_sub_clusters, n_sub_clusters, device='cuda')
    #     for l in range(n_layers):
    #         n_features = dnn_model.zero_counter[l].size(1)
    #         for _from in range(n_sub_clusters):
    #             for _to in range(_from + 1, n_sub_clusters):
    #                 n_commonly_zero = torch.logical_and(dnn_model.zero_counter[l][_from],
    #                                                     dnn_model.zero_counter[l][_to]).sum()
    #                 similarity = n_commonly_zero / n_features
    #                 cross_similarity[l][_from][_to] = similarity
    #         # with open(f'./sims/sim{int(self.args.sim_threshold * 100)}/Similarity-{self.args.arch}-{self.args.dataset}-k{n_sub_clusters}-l{l}.csv', 'w') as f:
    #         #     for _from in range(n_sub_clusters):
    #         #         for _to in range(n_sub_clusters):
    #         #             if _to == n_sub_clusters - 1:
    #         #                 f.write(f'{cross_similarity[l][_from][_to].item():.4f}\n')
    #         #             else:
    #         #                 f.write(f'{cross_similarity[l][_from][_to].item():.4f},')
    #
    #     # Using Inner-prodcut for caculating similarity
    #     # n_layers = len(dnn_model.zero_counter)
    #     # for l in range(n_layers):
    #     #     for c in range(n_sub_clusters):
    #     #         dnn_model.zero_counter[l][c] /= n_per_sub[c]  # Normalize counts by number of data in cluster
    #     # print('Calc. `Dot-product` between clusters..')
    #     # cross_similarity = torch.zeros(n_layers, n_sub_clusters, n_sub_clusters, device='cuda')
    #     # for l in range(n_layers):
    #     #     for _from in range(n_sub_clusters):
    #     #         for _to in range(_from + 1, n_sub_clusters):
    #     #             cross_similarity[l][_from][_to] = torch.dot(dnn_model.zero_counter[l][_from],
    #     #                                                         dnn_model.zero_counter[l][_to])
    #
    #     sorted_dist, sorted_indices = torch.sort(cross_similarity, dim=2, descending=True)
    #     to_merge = n_sub_clusters - self.args.cluster
    #     n_candidates_per_layer = to_merge + 2
    #     candidates_per_layer = [[] for _ in range(n_layers)]
    #     count_duplicated_candidates = dict()
    #     for l in range(n_layers):
    #         l_dist = sorted_dist[l].view(-1)
    #         l_idx = sorted_indices[l].view(-1)
    #
    #         v_of_sorted, i_of_sorted = torch.topk(l_dist, n_candidates_per_layer)
    #         for c in range(n_candidates_per_layer):
    #             if v_of_sorted[c] != 0.0:
    #                 row = i_of_sorted[c] // n_sub_clusters
    #                 col = l_idx[i_of_sorted[c]]
    #                 i_of_original = (row.item(), col.item())
    #
    #                 candidates_per_layer[l].append(i_of_original)
    #                 if count_duplicated_candidates.get(i_of_original):
    #                     count_duplicated_candidates[i_of_original] += 1
    #                 else:
    #                     count_duplicated_candidates[i_of_original] = 1
    #
    #     print("\n>>> Number of similar cluster pairs among layers")
    #     counted = count_duplicated_candidates.items()
    #     similar_cluster_pairs = sorted(counted, key=lambda x: x[1], reverse=True)
    #     for pair in range(len(similar_cluster_pairs)):
    #         print(f"Cluster {similar_cluster_pairs[pair][0][0]}&{similar_cluster_pairs[pair][0][1]}, "
    #               f"in {similar_cluster_pairs[pair][1]} layers")
    #
    #     print("\n>>> [Sub] Number of data per cluster")
    #     for c in range(n_sub_clusters):
    #         print(f"C{c}: {n_per_sub[c]}")
    #
    #     def check_other_groups(groups, cluster_id, cur_idx):
    #         for idx in range(len(groups)):
    #             if idx == cur_idx:
    #                 continue
    #             nxt_group = groups[idx][0]
    #             if cluster_id in nxt_group:
    #                 return idx
    #         return -1
    #
    #     max_data_num_per_merged_cluster = sum(n_per_sub) / 2
    #     print(f"\n>>> Merge similar clusters..")
    #     merged_clusters = []
    #     n_per_sub = torch.tensor(n_per_sub)
    #     print(f'Merge', end='')
    #     for p in range(len(similar_cluster_pairs)):
    #         pair = similar_cluster_pairs[p]
    #         c1, c2 = pair[0][0], pair[0][1]
    #         n_c1, n_c2 = n_per_sub[c1], n_per_sub[c2]
    #         summed = n_c1 + n_c2
    #         if summed < max_data_num_per_merged_cluster:
    #             print(f' {c1}&{c2}', end='')
    #             for g in range(len(merged_clusters)):
    #                 group = merged_clusters[g][0]
    #                 if c1 in group and c2 in group:
    #                     break
    #                 elif c1 in group:
    #                     group_id = check_other_groups(merged_clusters, c2, g)
    #                     if group_id == -1:
    #                         group.add(c2)
    #                     else:
    #                         group.update(merged_clusters[group_id][0])
    #
    #                     merged_clusters[g][1] += n_c2
    #                     n_per_sub[list(group)] = merged_clusters[g][1]
    #                     if group_id != -1:
    #                         del merged_clusters[group_id]
    #                     break
    #                 elif c2 in group:
    #                     group_id = check_other_groups(merged_clusters, c1, g)
    #                     if group_id == -1:
    #                         group.add(c1)
    #                     else:
    #                         group.update(merged_clusters[group_id][0])
    #
    #                     merged_clusters[g][1] += n_c1
    #                     n_per_sub[list(group)] = merged_clusters[g][1]
    #                     if group_id != -1:
    #                         del merged_clusters[group_id]
    #                     break
    #             else:
    #                 merged_clusters.append([{c1, c2}, summed])
    #
    #         n_merged = 0
    #         for group in merged_clusters:
    #             n_merged += len(group[0]) - 1
    #
    #         if n_merged == to_merge:
    #             print()
    #             break
    #
    #     final_clusters = dict()
    #     n_per_final = [0 for _ in range(self.args.cluster)]
    #
    #     k = 0  # Final cluster ID
    #     leftover_clusters = set(range(n_sub_clusters))
    #     print("\n>>> Merged clusters")
    #     for merged_single_cluster in merged_clusters:
    #         group = merged_single_cluster[0]
    #         print(f"C{k}: {tuple(group)}")
    #
    #         leftover_clusters = leftover_clusters.difference(group)
    #         for cluster in group:
    #             final_clusters[str(cluster)] = k
    #         n_per_final[k] = merged_single_cluster[1]
    #         k += 1
    #
    #     for cluster in leftover_clusters:
    #         final_clusters[str(cluster)] = k
    #         n_per_final[k] = n_per_sub[cluster]
    #         k += 1
    #
    #     print(f"\n>>> [Final] Number of data per cluster (Max.limit: {max_data_num_per_merged_cluster})")
    #     for c in range(self.args.cluster):
    #         print(f"C{c}: {n_per_final[c]}")
    #
    #     with open(os.path.join(self.args.clustering_path, 'params.json'), 'r') as f:
    #         args_without_nnac = json.load(f)
    #     with open(os.path.join(self.args.clustering_path, "params.json"), 'w') as f:
    #         args_without_nnac['sub_k'] = self.args.sub_cluster
    #         args_without_nnac['nnac'] = final_clusters
    #         json.dump(args_without_nnac, f, indent=4)
    #
    #     self.final_cluster = torch.zeros(self.args.sub_cluster, dtype=torch.int64)
    #     for sub, final in final_clusters.items():
    #         self.final_cluster[int(sub)] = final
    #     exit()

    # @torch.no_grad()
    # def nn_aware_clutering(self, dnn_model, train_loader):
    #     print('\n>>> NN-aware Clustering..')
    #     from utils.misc import InputContainer
    #
    #     n_sub_clusters = self.args.sub_cluster
    #     container = InputContainer(train_loader, self, n_sub_clusters, self.args.dataset, self.args.batch)
    #     container.initialize_generator()
    #     container.set_next_batch()
    #
    #     print('Count zero indices per cluster about dataset..')
    #     n_per_sub = torch.zeros(n_sub_clusters, device='cuda')
    #     dnn_model.eval()
    #     with tqdm(range(len(train_loader)), desc="Merge Clusters", ncols=90) as t:
    #         for i, _ in enumerate(t):
    #             input, _, cluster = container.get_batch()
    #             n_per_sub[cluster] += self.args.batch
    #
    #             dnn_model.count_zeros_per_index(input.cuda(), cluster, n_sub_clusters)
    #
    #             container.set_next_batch()
    #             if container.ready_cluster is None:
    #                 break
    #
    #     # Normalize with n_data of cluster, and make 1 if greater than 80 %
    #     print('Calc. Inner-product between clusters..')
    #     print('Normalize counter..')
    #     n_layers = len(dnn_model.zero_counter)
    #     zero_counter = dnn_model.zero_counter[0]
    #     for l in range(1, n_layers):
    #         zero_counter = torch.cat((zero_counter, dnn_model.zero_counter[l]), dim=1)
    #     zero_counter = zero_counter / n_per_sub[:, None]  # Normalize counts by number of data in cluster
    #
    #     cross_similarity = torch.zeros(n_sub_clusters, n_sub_clusters, device='cuda')
    #     for _from in range(n_sub_clusters):
    #         for _to in range(_from + 1, n_sub_clusters):
    #             cross_similarity[_from, _to] = torch.dot(zero_counter[_from], zero_counter[_to])
    #
    #     # print('Calc. `And` between clusters.. (`1` means both zero)')
    #     # n_layers = len(dnn_model.zero_counter)
    #     # zero_counter = dnn_model.zero_counter[0]
    #     # for l in range(1, n_layers):
    #     #     zero_counter = torch.cat((zero_counter, dnn_model.zero_counter[l]), dim=1)
    #     # zero_counter = zero_counter / n_per_sub[:, None]  # Normalize counts by number of data in cluster
    #     # zero_counter = torch.where(zero_counter > self.args.sim_threshold, 1, 0)
    #     # cross_similarity = torch.zeros(n_sub_clusters, n_sub_clusters, device='cuda')
    #     # n_features = zero_counter.size(1)
    #     # for _from in range(n_sub_clusters):
    #     #     for _to in range(_from + 1, n_sub_clusters):
    #     #         n_commonly_zero = torch.logical_and(zero_counter[_from], zero_counter[_to]).sum()
    #     #         cross_similarity[_from][_to] = n_commonly_zero / n_features
    #
    #     def check_other_groups(groups, cluster_id, cur_idx):
    #         for idx in range(len(groups)):
    #             if idx == cur_idx:
    #                 continue
    #             other_group = groups[idx][0]
    #             if cluster_id in other_group:
    #                 return idx
    #         return -1
    #
    #     to_merge = n_sub_clusters - self.args.cluster
    #     _, max_sim_idx = torch.sort(cross_similarity.flatten(), descending=True)
    #     max_num_per_cluster = torch.sum(n_per_sub) / 2
    #     print(f"\n>>> Merge similar clusters..")
    #     merged_clusters = []
    #     print(f'Merge', end='')
    #     for idx in max_sim_idx:
    #         c1 = idx // n_sub_clusters
    #         c2 = idx % n_sub_clusters
    #         c1, c2 = c1.item(), c2.item()
    #         n_c1, n_c2 = n_per_sub[c1], n_per_sub[c2]
    #         summed = n_c1 + n_c2
    #         if summed < max_num_per_cluster:
    #             print(f' {c1}&{c2}', end='')
    #             for g in range(len(merged_clusters)):
    #                 group = merged_clusters[g][0]
    #                 if c1 in group and c2 in group:
    #                     break
    #                 elif c1 in group:
    #                     group_id = check_other_groups(merged_clusters, c2, g)
    #                     if group_id == -1:
    #                         group.add(c2)
    #                     else:
    #                         group.update(merged_clusters[group_id][0])
    #
    #                     merged_clusters[g][1] += n_c2
    #                     n_per_sub[list(group)] = merged_clusters[g][1]
    #                     if group_id != -1:
    #                         del merged_clusters[group_id]
    #                     break
    #                 elif c2 in group:
    #                     group_id = check_other_groups(merged_clusters, c1, g)
    #                     if group_id == -1:
    #                         group.add(c1)
    #                     else:
    #                         group.update(merged_clusters[group_id][0])
    #
    #                     merged_clusters[g][1] += n_c1
    #                     n_per_sub[list(group)] = merged_clusters[g][1]
    #                     if group_id != -1:
    #                         del merged_clusters[group_id]
    #                     break
    #             else:
    #                 merged_clusters.append([{c1, c2}, summed])
    #
    #         n_merged = 0
    #         for group in merged_clusters:
    #             n_merged += len(group[0]) - 1
    #
    #         if n_merged == to_merge:
    #             print()
    #             break
    #
    #     final_clusters = dict()
    #     n_per_final = [0 for _ in range(self.args.cluster)]
    #
    #     k = 0  # Final cluster ID
    #     leftover_clusters = set(range(n_sub_clusters))
    #     print("\n>>> Merged clusters")
    #     for merged_single_cluster in merged_clusters:
    #         group = merged_single_cluster[0]
    #         print(f"C{k}: {tuple(group)}")
    #
    #         leftover_clusters = leftover_clusters.difference(group)
    #         for cluster in group:
    #             final_clusters[str(cluster)] = k
    #         n_per_final[k] = merged_single_cluster[1]
    #         k += 1
    #
    #     for cluster in leftover_clusters:
    #         final_clusters[str(cluster)] = k
    #         n_per_final[k] = n_per_sub[cluster]
    #         k += 1
    #
    #     print(f"\n>>> [Final] Number of data per cluster (Max.limit: {max_num_per_cluster})")
    #     for c in range(self.args.cluster):
    #         print(f"C{c}: {n_per_final[c]}")
    #
    #     with open(os.path.join(self.args.clustering_path, 'params.json'), 'r') as f:
    #         args_without_nnac = json.load(f)
    #     with open(os.path.join(self.args.clustering_path, "params.json"), 'w') as f:
    #         args_without_nnac['sub_k'] = self.args.sub_cluster
    #         args_without_nnac['nnac'] = final_clusters
    #         json.dump(args_without_nnac, f, indent=4)
    #
    #     self.final_cluster = torch.zeros(self.args.sub_cluster, dtype=torch.int64)
    #     for sub, final in final_clusters.items():
    #         self.final_cluster[int(sub)] = final
    #     exit()


def check_cluster_distribution(kmeans, train_loader):
    n_data = kmeans.args.data_per_cluster * kmeans.args.cluster * len(train_loader)
    n_data_per_cluster = dict()
    for c in range(kmeans.args.cluster):
        n_data_per_cluster[c] = 0
    for i, (input, target) in enumerate(train_loader):
        batch_cluster = kmeans.predict_cluster_of_batch(input)
        for c in batch_cluster:
            n_data_per_cluster[c.item()] += 1

    assert sum(n_data_per_cluster.values()) == n_data, \
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

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
            loaded_args = json.load(f)

        assert self.args.dataset == loaded_args['dataset'], \
            f"Dataset of model doesn't match (args: {self.args.dataset}, loaded: {loaded_args['dataset']})"
        assert self.args.clustering_method == loaded_args['clustering_method'], \
            f"# of clusters doesn't match (args: {self.args.clustering_method}, loaded: {loaded_args['clustering_method']})"
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
            for trial in range(1, 11):
                model = MiniBatchKMeans(n_clusters=self.args.cluster, batch_size=self.args.batch, tol=self.args.kmeans_tol, random_state=0)
                early_stopped = False
                t_epoch = tqdm(total=self.args.kmeans_epoch, desc="Trial-{}, Epoch".format(trial), position=0, ncols=90)
                for e in range(self.args.kmeans_epoch):
                    for image, _ in nonaug_loader:
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
    def nn_aware_clutering(self, dnn_model, train_loader):
        print('\n>>> NN-aware Clustering..')
        from utils.misc import InputContainer

        n_sub_clusters = self.args.sub_cluster
        container = InputContainer(train_loader, self, n_sub_clusters, self.args.dataset, self.args.batch)
        container.initialize_generator()
        container.set_next_batch()

        print('Count zero indices per cluster about dataset..')
        n_data_per_sub_cluster = [0 for _ in range(n_sub_clusters)]
        dnn_model.eval()
        with tqdm(range(len(train_loader)), desc="Merge Clusters", ncols=90) as t:
            for i, _ in enumerate(t):
                input, _, cluster = container.get_batch()
                n_data_per_sub_cluster[cluster] += self.args.batch

                dnn_model.count_zeros_per_index(input.cuda(), cluster, n_sub_clusters)

                container.set_next_batch()
                if container.ready_cluster is None:
                    break

        # Normalize with n_data of cluster, and make 1 if greater than 80 %
        print('Normalize counter..')
        n_layers = len(dnn_model.zero_counter)
        for l in range(n_layers):
            for c in range(n_sub_clusters):
                dnn_model.zero_counter[l][c] /= n_data_per_sub_cluster[c]  # Normalize counts by number of data in cluster
            dnn_model.zero_counter[l] = torch.where(dnn_model.zero_counter[l] > self.args.sim_threshold, 1, 0)

        # distances :: [[dist. to other clusters] per cluster] per layer
        print('Calc. `And` between clusters.. (`1` means both zero)')
        distances = [[[0 for _ in range(n_sub_clusters)] for _ in range(n_sub_clusters)] for _ in range(n_layers)]
        for l in range(n_layers):
            n_features = dnn_model.zero_counter[l].size(1)
            for _from in range(n_sub_clusters):
                for _to in range(_from + 1, n_sub_clusters):
                    n_commonly_zero = torch.logical_and(dnn_model.zero_counter[l][_from], dnn_model.zero_counter[l][_to]).sum()
                    similarity = n_commonly_zero / n_features
                    distances[l][_from][_to] = similarity
                    # distances[l][_to][_from] = similarity  # Comment this line to make duplicated values 0
            with open(f'./sims/sim{int(self.args.sim_threshold * 100)}/Similarity-{self.args.arch}-{self.args.dataset}-k{n_sub_clusters}-l{l}.csv', 'w') as f:
                for _from in range(n_sub_clusters):
                    for _to in range(n_sub_clusters):
                        if _to == n_sub_clusters - 1:
                            f.write(f'{distances[l][_from][_to]:.4f}\n')
                        else:
                            f.write(f'{distances[l][_from][_to]:.4f},')

        distances = torch.tensor(distances)
        sorted_dist, sorted_indices = torch.sort(distances, dim=2, descending=True)
        to_merge = n_sub_clusters - self.args.cluster
        n_candidates_per_layer = to_merge + 1
        candidates_per_layer = [[] for _ in range(n_layers)]
        count_duplicated_candidates = dict()
        for l in range(n_layers):
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
                        count_duplicated_candidates[i_of_original] += 1
                    else:
                        count_duplicated_candidates[i_of_original] = 1

        print("\n>>> Number of similar cluster pairs among layers")
        counted = count_duplicated_candidates.items()
        most_similar_cluster_pairs = sorted(counted, key=lambda x: x[1], reverse=True)
        for pair in range(len(most_similar_cluster_pairs)):
            print(f"Pair {most_similar_cluster_pairs[pair][0][0]}-{most_similar_cluster_pairs[pair][0][1]}, "
                  f"in {most_similar_cluster_pairs[pair][1]}")

        print("\n>>> [Sub] Number of data per cluster")
        for c in range(n_sub_clusters):
            print(f"C{c}: {n_data_per_sub_cluster[c]}")

        max_data_num_per_merged_cluster = sum(n_data_per_sub_cluster) // 2
        print(f"\n>>> Merge similar clusters..")
        n_merged = 0
        merged_clusters = []
        print(f'Merge', end='')
        for pair in most_similar_cluster_pairs:
            c1, c2 = pair[0][0], pair[0][1]
            c1_n_data, c2_n_data = n_data_per_sub_cluster[c1], n_data_per_sub_cluster[c2]
            summed = c1_n_data + c2_n_data
            if summed < max_data_num_per_merged_cluster:
                print(f' {c1}&{c2}', end='')
                n_merged += 1
                for ctr in merged_clusters:
                    clusters = ctr[0]
                    if c1 in clusters and c2 in clusters:
                        n_merged -= 1
                        break
                    elif c1 in clusters:
                        ctr[0].append(c2)
                        ctr[1] += n_data_per_sub_cluster[c2]
                        break
                    elif c2 in clusters:
                        ctr[0].append(c1)
                        ctr[1] += n_data_per_sub_cluster[c1]
                        break
                else:
                    merged_clusters.append([[c1, c2], n_data_per_sub_cluster[c1] + n_data_per_sub_cluster[c2]])

            if n_merged == to_merge:
                print()
                break

        final_clusters = dict()
        n_data_per_final_cluster = [0 for _ in range(self.args.cluster)]

        # Update the number of data per cluster with merged number
        k = 0
        leftover_clusters = set(range(n_sub_clusters))
        print("\n>>> Final clusters")
        for merged in merged_clusters:
            clusters = merged[0]
            print(tuple(clusters))

            leftover_clusters = leftover_clusters.difference(clusters)
            for c in clusters:
                final_clusters[str(c)] = k
            n_data_per_final_cluster[k] = merged[1]
            k += 1

        leftover_clusters = sorted(list(leftover_clusters))
        for c in range(len(leftover_clusters)):
            final_clusters[leftover_clusters[c]] = k
            n_data_per_final_cluster[k] = n_data_per_sub_cluster[leftover_clusters[c]]
            k += 1

        print(f"[Final] Number of data per cluster (Max.limit: {max_data_num_per_merged_cluster})")
        for c in range(self.args.cluster):
            print(f"C{c}: {n_data_per_final_cluster[c]}")

        with open(os.path.join(self.args.clustering_path, 'params.json'), 'r') as f:
            args_without_nnac = json.load(f)
        with open(os.path.join(self.args.clustering_path, "params.json"), 'w') as f:
            args_without_nnac['sub_k'] = self.args.sub_cluster
            args_without_nnac['nnac'] = final_clusters
            json.dump(args_without_nnac, f, indent=4)

        self.final_cluster = torch.zeros(self.args.sub_cluster, dtype=torch.int64)
        for sub, final in final_clusters.items():
            self.final_cluster[int(sub)] = final


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

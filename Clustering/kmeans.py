import torch

from cuml.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from audtorch.metrics.functional import pearsonr as pearson_correlation
import numpy as np
import networkx as nx

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

        if os.path.isfile(index_path := os.path.join(self.args.clustering_path, 'index.pth')):
            self.feature_index = torch.load(index_path)

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
    def ema_nn_aware_clustering(self, dnn_model, train_loader, arch, print_log=True):
        print('\n>>> NN-aware Clustering..')        
        ema = torch.transpose(dnn_model.get_ema_per_layer().cuda(), 0, 1)

        n_per_sub = torch.zeros([self.args.sub_cluster], dtype=torch.int).cuda()
        dnn_model.eval()
        with tqdm(train_loader, desc="Collecting Cluster Information", ncols=95) as t:
            for i, (images, _) in enumerate(t):
                images = images.cuda()
                cluster = self.predict_cluster_of_batch(images).cuda()
                
                indices, counts = torch.unique(cluster, return_counts=True)
                n_per_sub[indices] += counts
                
        merged_clusters, n_per_sub = clustering_aggregation(self.args, dnn_model, ema, n_per_sub, print_log) # NEW WAY
        
        final_clusters = print_merged_clusters(merged_clusters, n_per_sub, self.args.sub_cluster, self.args.cluster)
        self.args, self.final_cluster = save_cluster_info(self.args, self.feature_index, final_clusters)
        

    @torch.no_grad()
    def max_nn_aware_clustering(self, dnn_model, train_loader, arch, print_log=True):
        print('\n>>> NN-aware Clustering..')
        
        n_per_sub = torch.zeros([self.args.sub_cluster], dtype=torch.int).cuda()
        dnn_model.eval()
        with tqdm(train_loader, desc="Collecting Cluster Information", ncols=95) as t:
            for i, (images, _) in enumerate(t):
                images = images.cuda()
                cluster = self.predict_cluster_of_batch(images).cuda()
                
                indices, counts = torch.unique(cluster, return_counts=True)
                n_per_sub[indices] += counts
                dnn_model.accumulate_output_max_distribution(images, cluster, self.args.sub_cluster)
        
        dnn_model.max_accumulator = torch.transpose(dnn_model.max_accumulator, 0, 1)

        if print_log:
            print("\n>>> [Original] Number of data per cluster")
            for c in range(self.args.sub_cluster):
                print(f"{{C{c}: {n_per_sub[c]}}}", end=' ')
            print()
            
        merged_clusters, n_per_sub = clustering_aggregation(self.args, dnn_model, dnn_model.max_accumulator, n_per_sub, print_log) # NEW WAY
        # merged_clusters, n_per_sub = average_link_clustering(self.args, dnn_model, n_per_sub, print_log) # OLD WAY

        final_clusters = print_merged_clusters(merged_clusters, n_per_sub, self.args.sub_cluster, self.args.cluster)
        self.args, self.final_cluster = save_cluster_info(self.args, self.feature_index, final_clusters)
        
        dnn_model.delete_counters()
            

    @torch.no_grad()
    def zero_max_nn_aware_clustering(self, dnn_model, train_loader, arch, print_log=True):
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
        with tqdm(range(len(train_loader)*2), desc="Collecting Cluster Informations", ncols=90) as t:
            for i, _ in enumerate(t):
                input, _, cluster = container.get_batch()

                n_per_sub[cluster] += self.args.batch
                dnn_model.get_output_max_distribution(
                    input, cluster, n_sub_clusters)
                # dnn_model.count_zeros_per_index(
                #     input, cluster, n_sub_clusters)

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
                    # dnn_model.count_zeros_per_index(
                    #     input, cluster, n_sub_clusters)

        # # Handle Empty Clusters
        # for c in range(n_sub_clusters):
        #     if dnn_model.max_counter[0][c] == []:
        #         for l in range(len(dnn_model.max_counter)):
        #             dnn_model.max_counter[l][c] = torch.zeros(1).cuda()

        import pdb
        pdb.set_trace()

        if print_log:
            print("\n>>> [Original] Number of data per cluster")
            for c in range(n_sub_clusters):
                print(f"{{C{c}: {n_per_sub[c]}}}", end=' ')
            print()

        n_layers = len(dnn_model.max_counter)
        n_per_sub = torch.tensor(n_per_sub).cuda()

        # # shape(n_features) = [layer_idx]
        # n_features = torch.tensor([dnn_model.zero_counter[i].size(1) for i in range(n_layers)]).cuda()
        # # shape(dnn_model.zero_counter) = [cluster_size, layer_size, neurons]
        # dnn_model.zero_counter = torch.transpose(torch.stack([torch.nn.ConstantPad1d((0, torch.amax(n_features) - n_features[i]), 0)(dnn_model.zero_counter[i]) for i in range(n_layers)]), 0, 1)
        
        # distance_score, num_clusters = test_cases(dnn_model, 
        #                                           n_sub_clusters, 
        #                                           task=1,   # 0 : quantile / 1 : max / 2 : mean / 3 : median
        #                                           from_=0.,
        #                                           to_=4.01,
        #                                           gap=0.01)

        # Robustness of clustering (outlier)
        max_ratio = get_max_ratio(dnn_model, self.args.n_sub_clusters, task=1)
        merged_clusters, n_per_sub = clustering_aggregation(self.args, dnn_model, max_ratio, n_per_sub, print_log) # NEW WAY
        # merged_clusters, n_per_sub = average_link_clustering(self.args, dnn_model, n_per_sub, print_log) # OLD WAY

        final_clusters = print_merged_clusters(merged_clusters, n_per_sub, n_sub_clusters, self.args.cluster)
        self.args, self.final_cluster = save_cluster_info(self.args, self.feature_index, final_clusters)
        
        dnn_model.delete_counters()


    @torch.no_grad()
    def get_cluster_score(self, dnn_model, data_loader, ema, arch):
        from utils.misc import InputContainer
        n_clusters = self.args.cluster
        container = InputContainer(
            data_loader, self, self.args.cluster, self.args.dataset, arch, self.args.batch)
        container.initialize_generator()
        container.set_next_batch()

        print('Count Max Values per cluster about dataset..')
        dnn_model.eval()
        dnn_model.delete_counters()
        with tqdm(range(len(data_loader)*2), desc="Collecting Cluster Informations", ncols=90) as t:
            for i, _ in enumerate(t):
                input, _, cluster = container.get_batch()

                dnn_model.get_output_max_distribution(
                    input, cluster, self.args.cluster)

                container.set_next_batch()
                if container.ready_cluster is None:
                    break
            container.check_leftover()
            for c in range(container.num_clusters):
                if container.leftover_cluster_data[c]:
                    input, _, cluster = container.leftover_batch[c][0], \
                        container.leftover_batch[c][1], c
                    dnn_model.get_output_max_distribution(
                        input, cluster, self.args.cluster)
            
        score = torch.zeros_like(ema)
        for layer_idx in range(ema.size(0)):
            for cluster_idx in range(ema.size(1)):
                score[layer_idx][cluster_idx] = torch.mean(torch.abs(dnn_model.max_counter[layer_idx][cluster_idx] - ema[layer_idx][cluster_idx])) if dnn_model.max_counter[layer_idx][cluster_idx] != [] else 0
                
        dnn_model.delete_counters()
        return score.clone()


    @torch.no_grad()
    def measure_cluster_score(self, dnn_model, aug_loader, nonaug_loader, test_loader, arch, print_log=True):
        print('\n>>> measure cluster score..')

        ema = dnn_model.get_ema_per_layer().cuda()

        aug_score = self.get_cluster_score(dnn_model, aug_loader, ema, arch)
        nonaug_score = self.get_cluster_score(dnn_model, nonaug_loader, ema, arch)
        test_score = self.get_cluster_score(dnn_model, test_loader, ema, arch)

        aug_score_refined = (aug_score.T)[aug_score.sum(dim=0) != 0].T
        nonaug_score_refined = (nonaug_score.T)[nonaug_score.sum(dim=0) != 0].T
        test_score_refined = (test_score.T)[test_score.sum(dim=0) != 0].T
        
        aug_score_avg = torch.mean(aug_score_refined, dim=1)
        nonaug_score_avg = torch.mean(nonaug_score_refined, dim=1)
        test_score_avg = torch.mean(test_score_refined, dim=1)
        
        return aug_score_avg.clone(), nonaug_score_avg.clone(), test_score_avg.clone()


    @torch.no_grad()
    def measure_cluster_distance(self, dnn_model, train_loader, arch, print_log=True):
        print('\n>>> measure cluster score..')
        from utils.misc import InputContainer

        n_clusters = self.args.cluster
        container = InputContainer(
            train_loader, self, self.args.cluster, self.args.dataset, arch, self.args.batch)
        container.initialize_generator()
        container.set_next_batch()

        print('Count Max Values per cluster about dataset..')
        n_per_sub = [0 for _ in range(self.args.cluster)]
        dnn_model.eval()
        
        dnn_model.toggle_full_precision()
        with tqdm(range(len(train_loader)*2), desc="Collecting Cluster Informations", ncols=90) as t:
            for i, _ in enumerate(t):
                input, _, cluster = container.get_batch()

                n_per_sub[cluster] += self.args.batch
                dnn_model.get_output_max_distribution(
                    input, cluster, self.args.cluster)

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
                        input, cluster, self.args.cluster)
        dnn_model.toggle_full_precision()

        n_layers = len(dnn_model.max_counter)
        n_per_sub = torch.tensor(n_per_sub).cuda()
        ema = dnn_model.get_ema_per_layer().cuda()
        
        # First Layer
        first_layer_ema = ema[0].unsqueeze(1)
        first_layer_distance = torch.triu(torch.cdist(first_layer_ema, first_layer_ema, p=2), diagonal=1)
        first_layer_distance = first_layer_distance[torch.nonzero(first_layer_distance, as_tuple=True)]
        first_layer_score = torch.mean(first_layer_distance).item()
        
        # Middle Layer
        middle_layer_ema = ema[int(ema.size(0)/2)].unsqueeze(1)
        middle_layer_distance = torch.triu(torch.cdist(middle_layer_ema, middle_layer_ema, p=2), diagonal=1)
        middle_layer_distance = middle_layer_distance[torch.nonzero(middle_layer_distance, as_tuple=True)]
        middle_layer_distance = torch.mean(middle_layer_distance).item()
        
        # Last Layer
        last_layer_ema = ema[-1].unsqueeze(1)
        last_layer_distance = torch.triu(torch.cdist(last_layer_ema, last_layer_ema, p=2), diagonal=1)
        last_layer_distance = last_layer_distance[torch.nonzero(last_layer_distance, as_tuple=True)]
        last_layer_distance = torch.mean(last_layer_distance).item()
        
        # Total Average Layer
        total_distance = torch.triu(torch.cdist(ema.T, ema.T, p=2), diagonal=1)
        total_distance = total_distance[torch.nonzero(total_distance, as_tuple=True)]
        total_distance = torch.mean(total_distance).item()

        return torch.tensor([first_layer_score, middle_layer_distance, last_layer_distance, total_distance]).cuda()


##############################################################

def get_group_id(index, clusters_group):
    group = [index in clusters for clusters in clusters_group]
    return group.index(True) if sum(group) != 0 else None

def calc_cross_similarity(zero_ratio, _from, _to, sim_method):
    if sim_method == 'and':
        similarity = torch.logical_and(zero_ratio[_from], zero_ratio[_to]).sum(dim=1) / \
                n_features
    else:
        similarity = torch.logical_and(zero_ratio[_from], zero_ratio[_to]).sum(dim=1) / \
                torch.logical_or(zero_ratio[_from], zero_ratio[_to]).sum(dim=1)
    return torch.nan_to_num(similarity)

# get_candidates_from_
def get_candidates_from_lipschitz_bound(args, dnn_model, n_sub_clusters, n_merged, n_per_sub, exclude):
    n_candidates_per_layer = int(
        ((n_sub_clusters - n_merged) * (n_sub_clusters - n_merged - 1))/2 * 0.8)  # discard low 20%
    zero_ratio = deepcopy(dnn_model.zero_counter)
    n_layers = len(zero_ratio[2])

    # Set True / False by Zero Count Threshold
    zero_ratio = torch.div(zero_ratio, n_per_sub.view(-1, 1, 1))
    zero_ratio = torch.where(zero_ratio > args.sim_threshold, True, False)

    # record cross_similarity
    # predict lipschitz 
    cross_similarity = torch.zeros(n_sub_clusters, n_sub_clusters, n_layers, device='cuda')
    for _from in [index for index in range(n_sub_clusters) if index not in exclude]:
        for _to in [index for index in range(_from + 1, n_sub_clusters) if index not in exclude]:
            cross_similarity[_from][_to] = calc_cross_similarity(zero_ratio, _from, _to, args.similarity_method)

    # [cluster_idx, cluster_idx, layer_idx] -> [layer_idx, cluster_idx, cluster_idx] .. similarity
    cross_similarity = torch.transpose(torch.transpose(cross_similarity, 0, 2), 1, 2)

    dist = cross_similarity.view(cross_similarity.size(0), -1)
    _similarity, _cluster_idx = torch.topk(dist, n_candidates_per_layer)

    zero_sim_layer = torch.count_nonzero(_similarity.size(1) - torch.count_nonzero(_similarity, dim=1))
    candidates, counts = torch.unique(_cluster_idx, return_counts=True)
    candidates = torch.where(counts >= (n_layers-zero_sim_layer), candidates, 0)
    candidates = candidates[candidates.nonzero()]

    print("number of candidates : ", candidates.size(0))
    return candidates


def get_max_ratio(dnn_model, n_sub_clusters, task=None):
    # finding best pair from choosen candidates
    cur_max_counter = deepcopy(dnn_model.max_counter)
    n_layers = len(cur_max_counter)
    max_ratio = torch.zeros((n_layers, n_sub_clusters), device='cuda')

    # Clipping value approximation
    for l in range(n_layers):
        for c in range(n_sub_clusters):
            if task == 0:
                max_ratio[l][c] = torch.quantile(cur_max_counter[l][c], 0.95) # 0.9986
            elif task == 1:
                max_ratio[l][c] = torch.amax(cur_max_counter[l][c])
            elif task == 2:
                max_ratio[l][c] = torch.mean(cur_max_counter[l][c])
            else:
                max_ratio[l][c] = torch.quantile(cur_max_counter[l][c], 0.5)

    return torch.transpose(max_ratio, 0, 1)

def get_pairwise_distance(max_ratio, mask=True):
    distance = torch.cdist(max_ratio, max_ratio, p=2)
    if mask:
        distance = torch.triu(distance, diagonal=1)
    return distance

def get_pairs_and_similarity_from_candidates(candidates, distance, n_sub_clusters, exclude):
    # measure L2 distance between clusters
    cross_similarity = torch.full(
        (n_sub_clusters, n_sub_clusters), float('inf'), device='cuda')
    cross_similarity_candidate = torch.full(
        (n_sub_clusters, n_sub_clusters), float('inf'), device='cuda')

    # Set distance 0 to infinity to represent their distance is far enough to not get considered
    cross_similarity = torch.where(distance != 0., distance, cross_similarity)

    for index in exclude:
        cross_similarity[:, index] = torch.inf
        cross_similarity[index, :] = torch.inf

    if candidates is not None:
        for candidate in candidates:
            _from = int(candidate // n_sub_clusters)
            _to = int(candidate % n_sub_clusters)
            cross_similarity_candidate[_from][_to] = cross_similarity[_from][_to]
    else:
        cross_similarity_candidate = cross_similarity.clone()

    # choose pair of clusters with smallest L2 distance
    pair = (cross_similarity_candidate == (torch.min(cross_similarity_candidate))
            ).nonzero(as_tuple=True)

    return [cluster[0].item() for cluster in pair], cross_similarity_candidate




############# NEW WAY #############
def clustering_aggregation(args, dnn_model, max_ratio, n_per_sub, print_log):
    n_sub_clusters = args.sub_cluster
    print(">>> Start measuring pairwise distance")
    distance = get_pairwise_distance(max_ratio, mask=True)
    print(">>> Find Splittable cluster set")
    clusters = get_splitted_cluster_sets(distance, target_cluster=args.cluster)
    print(">>> Start Merging cluster pairs")
    return merge_clustered_pairs(dnn_model, clusters, n_per_sub, print_log)

############# OLD WAY #############
def average_link_clustering(args, dnn_model, n_per_sub, print_log):
    n_sub_clusters = args.sub_cluster
    merged_clusters = []

    to_merge = n_sub_clusters - args.cluster
    n_merged = 0

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    while n_merged < to_merge:
        print(f'\n>>> Number of clusters to be merged: {to_merge - n_merged}')

        # Exclude merged clusters except 1 left
        exclude = set()
        for group in merged_clusters:
            exclude.update(set(group) - {min(group)})

        # get candidates using zero count matrix which consider lipschitz boundary of layers
        # candidates = get_candidates_from_lipschitz_bound(self.args, dnn_model, n_sub_clusters, n_merged, n_per_sub, exclude)
        candidates = None
        
        # get pairwise distance using approximated clipping values from each clusters
        max_ratio = get_max_ratio(dnn_model, n_sub_clusters, task=1)
        distance, max_ratio = get_pairwise_distance(max_ratio, mask=True)
        
        # get a most appropriate merging pair from candidates using distance metric
        # if no candidates are given, choose a most appropriate merging pair from all sets
        pair, cross_similarity = get_pairs_and_similarity_from_candidates(candidates, distance, n_sub_clusters, exclude)

        # Merge pairs
        merged_clusters, n_per_sub = merge(dnn_model, pair, merged_clusters, n_per_sub, print_log)
        
        n_merged += 1

        check = sum([len(group) - 1 for group in merged_clusters])
        assert check == n_merged, "WTF??"
        print("distance : ", torch.min(cross_similarity).item())
    return merged_clusters, n_per_sub




##############################################################
def merge_clustered_pairs(dnn_model, clusters, n_per_sub=None, print_log=True):
    merged_clusters = []
    for cluster in clusters:
        cluster = list(cluster)
        pair1 = cluster[0]
        for pair2 in cluster[1:]:
            pair = [pair1, pair2]
            merged_clusters, n_per_sub = merge(dnn_model, pair, merged_clusters, n_per_sub, print_log)
    return merged_clusters, n_per_sub

def merge(dnn_model, pair, merged_clusters, n_per_sub=None, print_log=True):
    c1, c2 = pair[0], pair[1]
    if n_per_sub is not None:
        n_c1, n_c2 = n_per_sub[c1].item(), n_per_sub[c2].item()

    group_id_c1 = get_group_id(c1, merged_clusters)
    group_id_c2 = get_group_id(c2, merged_clusters)

    if group_id_c1 is None and group_id_c2 is None:
        # c1 and c2 have never been merged
        if print_log:
            print(f'Merge {c1}&{c2}')
        merged_clusters.append(sorted([c1, c2]))
        if n_per_sub is not None:
            n_per_sub[[c1, c2]] = n_c1 + n_c2
        
        if hasattr(dnn_model, 'max_accumulator'):
            _max = dnn_model.max_accumulator[c1].max(dnn_model.max_accumulator[c2])
            dnn_model.max_accumulator[c1] = _max.clone()
            dnn_model.max_accumulator[c2] = torch.zeros_like(_max).cuda()
            
        if hasattr(dnn_model, 'max_counter'):
            for l in range(len(dnn_model.max_counter)):
                max_merged_count = torch.cat(
                    [dnn_model.max_counter[l][c1], dnn_model.max_counter[l][c2]])
                dnn_model.max_counter[l][c1] = max_merged_count
                dnn_model.max_counter[l][c2] = max_merged_count

        if hasattr(dnn_model, 'zero_counter'):
            zero_merged_count = dnn_model.zero_counter[c1] + dnn_model.zero_counter[c2]
            dnn_model.zero_counter[c1] = zero_merged_count
            dnn_model.zero_counter[c2] = zero_merged_count
    elif group_id_c1 == group_id_c2:
        # Shouldn't be here
        print("????")
    else:
        if group_id_c1 is not None:
            if print_log:
                print(f'Merge {c1}&{c2}')
            if group_id_c2 is None:
                merged_clusters[group_id_c1].append(c2)
            else:
                merged_clusters[group_id_c1] = sorted(list(set(merged_clusters[group_id_c1]) | set(merged_clusters[group_id_c2])))
                del merged_clusters[group_id_c2]
                group_id_c1 = get_group_id(c1, merged_clusters)
            if n_per_sub is not None:
                n_per_sub[merged_clusters[group_id_c1]] = n_c1 + n_c2
                
            if hasattr(dnn_model, 'max_accumulator'):
                _max = dnn_model.max_accumulator[c1].max(dnn_model.max_accumulator[c2])
                dnn_model.max_accumulator[c1] = _max.clone()
                dnn_model.max_accumulator[c2] = torch.zeros_like(_max).cuda()
                
            if hasattr(dnn_model, 'zero_counter'):
                zero_merged_count = dnn_model.zero_counter[c1] + dnn_model.zero_counter[c2]
                dnn_model.zero_counter[c1] = zero_merged_count.clone()
                dnn_model.zero_counter[c2] = zero_merged_count.clone()

            if hasattr(dnn_model, 'max_counter'):
                for l in range(len(dnn_model.max_counter)):
                    max_merged_count = torch.cat(
                        [dnn_model.max_counter[l][c1], dnn_model.max_counter[l][c2]])
                    dnn_model.max_counter[l][c1] = max_merged_count.clone()
                    dnn_model.max_counter[l][c2] = max_merged_count.clone()

        else:
            if print_log:  
                print(f'Merge {c1}&{c2}')
            if group_id_c1 is None:
                merged_clusters[group_id_c2].append(c1)
            else:
                merged_clusters[group_id_c2] = sorted(list(set(merged_clusters[group_id_c1]) | set(merged_clusters[group_id_c2])))
                del merged_clusters[group_id_c1]
                group_id_c2 = get_group_id(c2, merged_clusters)
            if n_per_sub is not None:
                n_per_sub[merged_clusters[group_id_c2]] = n_c1 + n_c2
            
            if hasattr(dnn_model, 'max_accumulator'):
                _max = dnn_model.max_accumulator[c1].max(dnn_model.max_accumulator[c2])
                dnn_model.max_accumulator[c1] = _max.clone()
                dnn_model.max_accumulator[c2] = torch.zeros_like(_max).cuda()
                
            # zero_count_per_layer
            if hasattr(dnn_model, 'zero_counter'):
                zero_merged_count = dnn_model.zero_counter[c1] + dnn_model.zero_counter[c2]
                dnn_model.zero_counter[c1] = zero_merged_count.clone()
                dnn_model.zero_counter[c2] = zero_merged_count.clone()

            if hasattr(dnn_model, 'max_counter'):
                for l in range(len(dnn_model.max_counter)):
                    # max_per_layer_per_datapoint
                    max_merged_count = torch.cat(
                        [dnn_model.max_counter[l][c1], dnn_model.max_counter[l][c2]])
                    dnn_model.max_counter[l][c1] = max_merged_count.clone()
                    dnn_model.max_counter[l][c2] = max_merged_count.clone()

    return merged_clusters, n_per_sub



##############################################################

def print_merged_clusters(merged_clusters, n_per_sub, n_sub_clusters, n_fin_clusters):
    final_clusters = dict()
    n_per_final = [0 for _ in range(n_fin_clusters)]
    k = 0  # Final cluster ID
    leftover_clusters = set(range(n_sub_clusters))
    print("\n>>> Merged clusters")
    for i, merged_single_cluster in enumerate(merged_clusters):
        print(f"C{k}: {tuple(merged_single_cluster)}")

        leftover_clusters = leftover_clusters.difference(merged_single_cluster)
        for cluster in merged_single_cluster:
            final_clusters[str(cluster)] = k
        n_per_final[k] = n_per_sub[merged_single_cluster[0]]
        k += 1

    for cluster in leftover_clusters:
        final_clusters[str(cluster)] = k
        n_per_final[k] = n_per_sub[cluster]
        k += 1

    print(f"\n>>> [Final] Number of data per cluster")
    for c in range(n_fin_clusters):
        print(f"C{c}: {n_per_final[c]}")

    return final_clusters

def save_cluster_info(args, feature_index, final_clusters):
    with open(os.path.join(args.clustering_path, 'params.json'), 'r') as f:
        args_without_nnac = json.load(f)
        if args_without_nnac['k'] != args.cluster:
            path = args.clustering_path + \
                f'__.k{args.cluster}.sub{args.sub_cluster}.topk_{args.topk}.sim_{args.sim_threshold}.{args.similarity_method}'
            print(
                f"Copy json and pkl file from {args.clustering_path} to {path}")
            if not os.path.exists(path):
                os.makedirs(path)
            import shutil
            shutil.copyfile(os.path.join(args.clustering_path, 'checkpoint.pkl'),
                            os.path.join(path, 'checkpoint.pkl'))
            args.clustering_path = path
            args_without_nnac['k'] = args.cluster

    with open(os.path.join(args.clustering_path, "params.json"), 'w') as f:
        args_without_nnac['sub_k'] = args.sub_cluster
        args_without_nnac['nnac'] = final_clusters
        json.dump(args_without_nnac, f, indent=4)

    torch.save(feature_index, os.path.join(args.clustering_path, "index.pth"))

    final_cluster = torch.zeros(
        args.sub_cluster, dtype=torch.int64)
    for sub, final in final_clusters.items():
        final_cluster[int(sub)] = final
        
    return args, final_cluster



def count_num_clusters(distance, THRESHOLD):
    graph = nx.from_numpy_array(torch.where(distance > THRESHOLD, torch.zeros(1).cuda(), distance).cpu().numpy())
    return len(list(nx.connected_components(graph)))

def count_num_edges(A, THRESHOLD):
    A = torch.triu(A, diagonal=1)
    A = torch.where(A > THRESHOLD, torch.zeros(1).cuda(), A)
    dist = A[A.nonzero(as_tuple=True)]

    bucket = torch.tensor([x for x in np.arange(0, torch.amax(A).item(), 0.01)]).cuda()
    num_edges = torch.bincount(torch.squeeze(torch.bucketize(dist, bucket)))
    return num_edges

def measure_weighted_edges(distance, max_ratio, THRESHOLD):
    graph = nx.from_numpy_array(torch.where(distance > THRESHOLD, torch.zeros(1).cuda(), distance).cpu().numpy())
    merge_clusters = list(nx.connected_components(graph))
    attributes = torch.zeros([len(merge_clusters), max_ratio.size(1)]).cuda()
    for i, cluster in enumerate(merge_clusters):
        index = list(cluster)
        attributes[i] = torch.mean(max_ratio[index], dim=0)
    distance = torch.triu(torch.cdist(attributes, attributes, p=2), diagonal=1)
    return distance[distance.nonzero(as_tuple=True)].mean()

def measure_average_distance_from_candidates(dnn_model, distance, THRESHOLD):
    cur_dnn_model = deepcopy(dnn_model)
    graph = nx.from_numpy_array(torch.where(distance > THRESHOLD, torch.zeros(1).cuda(), distance).cpu().numpy())
    merge_clusters = list(nx.connected_components(graph))
    merged_clusters = []
    for i, clusters in enumerate(merge_clusters):
        indices = list(clusters)
        merge_clusters[i] = indices
        for index in indices[1:]:
            pair = [indices[0], index]
            merged_clusters, _ = merge(cur_dnn_model, pair, merged_clusters, print_log=False)

    attributes = torch.zeros([len(merge_clusters), len(cur_dnn_model.max_counter)]).cuda()
    for l in range(len(cur_dnn_model.max_counter)):
        for i, c in enumerate(merge_clusters):
            attributes[i][l] = torch.quantile(cur_dnn_model.max_counter[l][c[0]], 0.9986)
    distance = torch.triu(torch.cdist(attributes, attributes, p=2), diagonal=1)
    return distance[distance.nonzero(as_tuple=True)].mean().item()
    # return distance[distance.nonzero(as_tuple=True)].amax().item()

    

    

def get_splitted_cluster_sets(distance, threshold=None, target_cluster=None):
    import math
    if threshold is None:
        threshold = torch.quantile(distance[torch.nonzero(distance, as_tuple=True)], 0.0668).item() # 2 sigma : 0.02275 / 1.5 sigma : 0.0668 / 1 sigma : 0.158655
    if target_cluster is None:
        graph = nx.from_numpy_array(torch.where(distance > threshold, torch.zeros(1).cuda(), distance).cpu().numpy())
        merge_clusters = list(nx.connected_components(graph))
    else:
        while True:
            graph = nx.from_numpy_array(torch.where(distance > threshold, torch.zeros(1).cuda(), distance).cpu().numpy())
            merge_clusters = list(nx.connected_components(graph))
            
            if (delta := target_cluster - len(merge_clusters)):
                threshold -= 0.000001 * delta
            else:
                break
    # print("applying threshold : ", threshold)
    return merge_clusters

def test_cases(dnn_model, n_sub_clusters, task=1, from_=0., to_=4.01, gap=0.01):
    print('start')
    distance, max_ratio = get_pairwise_distance(dnn_model, n_sub_clusters, mask=True, task=task)
    test = []
    test_num_clusters = []
    for threshold in np.arange(from_, to_, gap):
        test.append(measure_average_distance_from_candidates(dnn_model, distance, threshold))
        test_num_clusters.append(count_num_clusters(distance, threshold))
    print('done')
    return test, test_num_clusters




# TODO
# def get_splitted_cluster_sets(distance, target_cluster):
#     THRESHOLD = torch.quantile(distance, 0.0228)
#     while True:
#         graph = nx.from_numpy_array(torch.where(distance > THRESHOLD, torch.zeros(1).cuda(), distance).cpu().numpy())
#         merge_clusters = list(nx.connected_components(graph))
        
#         if (delta := torch.mean(distance) * 0.0001):
#             THRESHOLD = THRESHOLD - delta
#             print("diff : ", diff, " threshold :", THRESHOLD)
#         else:
#             break

#         if break_condition:
#             break

#         if (delta := target_cluster - len(merge_clusters)):
#             THRESHOLD = THRESHOLD - delta
#             print("diff : ", diff, " threshold :", THRESHOLD)
#         else:
#             break
#     return merge_clusters

    # TODO
    # use torch.fx.experimental.optimization.UnionFold instead of networkx connected_components
    # 
    # too slow since it does not support adjacency matrix
    # x = UnionFind(512)
    # A = torch.triu(A, diagonal=1)
    # for i in range(512):
    #     x.make_set(i)
    # for pair in A.nonzero():
    #     tmp = x.join(pair[0], pair[1])
    # (unique in x.size) - 1
    
    # TODO
    # use scipy.sparce.csgraph.connected_components
    # 
    # you don't get the indexes of each clustered group, though it is still fast

    # import scipy
    # from scipy.sparse import csr_matrix
    # from scipy.sparse.csgraph import connected_components

##############################################################

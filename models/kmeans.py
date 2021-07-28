import torch
from sklearn.cluster import MiniBatchKMeans
import numpy as np

import tqdm
import joblib
from copy import deepcopy


def load_kmeans_model(path):
    return joblib.load(path)


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
                part_start = chanel_start + (j * n_col)
            else:
                part_start = chanel_start + 512 + (j - int(partition / 2)) * n_col

            part = np.array([])
            start = part_start
            for k in range(n_row):
                end = start + n_col
                if not k:
                    part = np.copy(data[:, start:end])
                else:
                    part = np.concatenate((part, data[:, start:end]), axis=1)
                start += 32

            part_min = np.min(part, axis=1).reshape(part.shape[0], 1)
            part_max = np.max(part, axis=1).reshape(part.shape[0], 1)

            tmp = np.append(part_min, part_max, axis=1)
            if not i and not j:
                rst = np.copy(tmp)
            else:
                rst = np.append(rst, tmp, axis=1)
    return rst


def train_kmeans(args, train_loader):
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

    model = MiniBatchKMeans(n_clusters=args.cluster, batch_size=args.batch, tol=args.kmeans_tol, random_state=0)

    prev_centers = None
    is_converged = False
    t_epoch = tqdm.tqdm(total=args.kmeans_epoch, desc='Epoch', position=0, ncols=90)
    for e in range(args.kmeans_epoch):
        for image, _ in train_loader:
            train_data = get_partitioned_batch(image.numpy(), args.partition)
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
    joblib.dump(model, args.kmeans_path + '/checkpoint.pkl')
    return model


def get_pcq_batch(model, input, target, num_partitions):
    kmeans_input = get_partitioned_batch(input.numpy(), num_partitions)
    cluster_info = model.predict(kmeans_input)

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


# class PCQtools(object):
#     def __init__(self):
#         self.batch_cluster = None
#
#     def set_pcq_batch(self, info):
#         self.batch_cluster = info

# class PCQtools(object):
#     def __init__(self, kmeans_path=None):
#         self.batch_cluster = None
#         if kmeans_path:
#             self.kmeans_model = self.load_kmeans_model(kmeans_path)
#
#     def get_pcq_batch(self, input, target):
#         kmeans_input = self.get_partitioned_batch(input.numpy(), self.args.partition)
#         cluster_info = self.kmeans_model.predict(kmeans_input)
#
#         num_data_per_cluster = []
#         input_ordered_by_cluster = torch.zeros(input.shape)
#         target_ordered_by_cluster = torch.zeros(target.shape, dtype=torch.long)
#         existing_clusters, counts = np.unique(cluster_info, return_counts=True)
#         ordered = 0
#         for cluster, n in zip(existing_clusters, counts):
#             num_data_per_cluster.append([cluster, n])
#             data_indices = (cluster_info == cluster).nonzero()[0]
#             input_ordered_by_cluster[ordered:ordered + n] = input[data_indices].clone().detach()
#             target_ordered_by_cluster[ordered:ordered + n] = target[data_indices].clone().detach()
#             ordered += n
#         self.batch_cluster = torch.ByteTensor(num_data_per_cluster)
#         return input_ordered_by_cluster, target_ordered_by_cluster
#
#     def train_kmeans(self, args, train_loader):
#         def check_convergence(prev, cur, tol):
#             """
#                 Relative tolerance with regards to Frobenius norm of the difference in the cluster centers
#                 of two consecutive iterations to declare convergence.
#             """
#             diff = np.subtract(prev, cur)
#             normed = np.linalg.norm(diff)
#             if normed > tol:
#                 return False
#             return True
#
#         model = MiniBatchKMeans(n_clusters=args.cluster, batch_size=args.batch, tol=args.kmeans_tol, random_state=0)
#
#         prev_centers = None
#         is_converged = False
#         t_epoch = tqdm.tqdm(total=args.kmeans_epoch, desc='Epoch', position=0, ncols=90)
#         for e in range(args.kmeans_epoch):
#             for image, _ in train_loader:
#                 train_data = self.get_partitioned_batch(image.numpy(), args.partition)
#                 model = model.partial_fit(train_data)
#
#                 if prev_centers is not None:
#                     is_converged = check_convergence(prev_centers, model.cluster_centers_, model.tol)
#                     if is_converged:
#                         break
#                 prev_centers = deepcopy(model.cluster_centers_)
#             t_epoch.update(1)
#             if is_converged:
#                 print("\nEarly stop training kmeans model")
#                 break
#         joblib.dump(model, args.kmeans_path + '/checkpoint.pkl')
#         self.kmeans_model = model
#
#     def get_partitioned_batch(self, data, partition):
#         channel = data.shape[1]
#         width = data.shape[2]
#         height = data.shape[3]
#         data = data.reshape((data.shape[0], channel * width * height))
#         n_row = int(width / 2)
#         n_col = int(height / (partition / 2))
#
#         rst = np.array([])
#         for i in range(channel):
#             chanel_start = i * (width * height)
#             for j in range(partition):
#                 if j < partition / 2:
#                     part_start = chanel_start + (j * n_col)
#                 else:
#                     part_start = chanel_start + 512 + (j - int(partition / 2)) * n_col
#
#                 part = np.array([])
#                 start = part_start
#                 for k in range(n_row):
#                     end = start + n_col
#                     if not k:
#                         part = np.copy(data[:, start:end])
#                     else:
#                         part = np.concatenate((part, data[:, start:end]), axis=1)
#                     start += 32
#
#                 part_min = np.min(part, axis=1).reshape(part.shape[0], 1)
#                 part_max = np.max(part, axis=1).reshape(part.shape[0], 1)
#
#                 tmp = np.append(part_min, part_max, axis=1)
#                 if not i and not j:
#                     rst = np.copy(tmp)
#                 else:
#                     rst = np.append(rst, tmp, axis=1)
#         return rst
#
#     def load_kmeans_model(path):
#         return joblib.load(path)

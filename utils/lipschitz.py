import torch

from utils import *
from tqdm import tqdm
import numpy as np


def l2_dist(a, b):
    return torch.norm(a - b, p=2)


def batch_l2_dist(a, b):
    return torch.norm(a.view(a.size(0), -1) - b.view(b.size(0), -1), p=2, dim=1)


def range_l2_dist(a, b):
    a_min = a.min()
    a_max = a.max()
    b_min = b.min()
    b_max = b.max()
    return torch.sqrt(a_max.sub(b_max).square() + a_min.sub(b_min).square())


def batch_range_l2_dist(a, b):
    _a = a.view(a.size(0), -1)
    a_min = _a.min(dim=1).values
    a_max = _a.max(dim=1).values

    _b = b.view(b.size(0), -1)
    b_min = _b.min(dim=1).values
    b_max = _b.max(dim=1).values
    return torch.sqrt(a_max.sub(b_max).square() + a_min.sub(b_min).square())


def partitioned_range_l2_dist(a, b, clustering_model):
    a_data = clustering_model.get_partitioned_batch(a)
    b_data = clustering_model.get_partitioned_batch(b)
    return torch.norm(torch.tensor(a_data - b_data), p=2, dim=1)


def norm_per_data(a, b):
    return abs(torch.norm(a, p=2) - torch.norm(b, p=2))


class Simple32Network(torch.nn.Module):
    def __init__(self):
        super(Simple32Network, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.fc = torch.nn.Linear(32 * 32 * 32, 10, bias=False)
        # self.conv1 = torch.nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=True)
        # self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=True)
        # self.fc = torch.nn.Linear(32 * 32 * 32, 10, bias=True)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class Simple224Network(torch.nn.Module):
    def __init__(self):
        super(Simple224Network, self).__init__()
        # self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # self.conv2 = torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        # self.fc = torch.nn.Linear(802816, 1000, bias=False)
        self.conv1 = torch.nn.Conv2d(3, 16, kernel_size=7, stride=2, padding=3, bias=True)
        self.conv2 = torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.fc = torch.nn.Linear(802816, 10000, bias=True)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def qat_lipschitz(model, clustering_model, loader, x_metric, y_metric):
    test_iter = iter(loader)
    model.eval()
    lipschitz_list = []
    with tqdm(loader, unit="batch", ncols=90) as t:
        for i in range(22500):
            input, target = next(test_iter)

            x1, x1_target = input[0].unsqueeze(0).cuda(), target[0].cuda()
            x2, x2_target = input[1].unsqueeze(0).cuda(), target[1].cuda()

            y1 = model(x1)
            y2 = model(x2)

            if x_metric == partitioned_range_l2_dist:
                x_rst = x_metric(x1, x2, clustering_model)
            else:
                x_rst = x_metric(x1, x2)

            if y_metric == partitioned_range_l2_dist:
                y_rst = y_metric(y1, y2, clustering_model)
            else:
                y_rst = y_metric(y1, y2)

            cur_lipschitz = y_rst / x_rst
            lipschitz_list.append(cur_lipschitz.item())
    lipschitz_list = np.array(lipschitz_list)
    print("QAT max: {:.4f}, avg: {:.4f}, std: {:.4f}".format(np.max(lipschitz_list), np.mean(lipschitz_list), np.std(lipschitz_list)))


def pcq_lipschitz(model, clustering_model, loader, runtime_helper, x_metric, y_metric):
    model.eval()
    container = [[] for _ in range(runtime_helper.num_clusters)]
    image_container = []
    lipschitz_list = [[] for _ in range(runtime_helper.num_clusters)]
    with tqdm(loader, unit="batch", ncols=90) as t:
        for i, (images, targets) in enumerate(t):
            cluster_info = clustering_model.predict_cluster_of_batch(images)
            for c in range(runtime_helper.num_clusters):
                indices = (cluster_info == c).nonzero(as_tuple=True)[0]
                container[c].append(images[indices])

    for c in range(runtime_helper.num_clusters):
        image_container.append(torch.cat(container[c]))

    for c in range(runtime_helper.num_clusters):
        for i in range(image_container[c].size(0) // 2):
            runtime_helper.batch_cluster = c

            x1 = image_container[c][i * 2].unsqueeze(0).cuda()
            x2 = image_container[c][i * 2 + 1].unsqueeze(0).cuda()

            y1 = model(x1)
            y2 = model(x2)

            if x_metric == partitioned_range_l2_dist:
                x_rst = x_metric(x1, x2, clustering_model)
            else:
                x_rst = x_metric(x1, x2)

            if y_metric == partitioned_range_l2_dist:
                y_rst = y_metric(y1, y2, clustering_model)
            else:
                y_rst = y_metric(y1, y2)

            cur_lipschitz = y_rst / x_rst
            lipschitz_list[c].append(cur_lipschitz.item())

    for c in range(runtime_helper.num_clusters):
        print("[cluster {}] max: {:.4f}, avg: {:.4f}, std: {:.4f}"
              .format(c, np.max(lipschitz_list[c]), np.mean(lipschitz_list[c]), np.std(lipschitz_list[c])))


def qat_lipschitz_with_zero(model, clustering_model, loader, x_metric, y_metric):
    model.eval()
    candidates = torch.zeros(0)
    x2 = torch.zeros((256, 3, 32, 32), device='cuda')
    y2 = None
    with tqdm(loader, unit="batch", ncols=90) as t:
        for i, (input, _) in enumerate(t):
            t.set_description("QAT")
            x1 = input.cuda()
            y1 = model(x1)
            if y2 is None:
                y2 = model(x2)
            if y2.size(0) > y1.size(0):
                x2 = x2[:y1.size(0)]
                y2 = y2[:y1.size(0)]

            if x_metric == partitioned_range_l2_dist:
                x_rst = x_metric(x1, x2, clustering_model)
            else:
                x_rst = x_metric(x1, x2)

            if y_metric == partitioned_range_l2_dist:
                y_rst = y_metric(y1, y2, clustering_model)
            else:
                y_rst = y_metric(y1, y2)

            cur_lipschitz = y_rst.cpu() / x_rst.cpu()
            candidates = torch.cat([candidates, cur_lipschitz.detach()])
    print("QAT max: {:.5f}, avg: {:.5f}, std: {:.5f}"
          .format(candidates.max(), candidates.mean(), candidates.std(unbiased=False)))


def pcq_lipschitz_with_zero(model, clustering_model, loader, runtime_helper, x_metric, y_metric):
    model.eval()
    container = [torch.zeros((0, 3, 32, 32)) for _ in range(runtime_helper.num_clusters)]
    candidates = [torch.zeros(0) for _ in range(runtime_helper.num_clusters)]
    with tqdm(loader, unit="batch", ncols=90) as t:
        for i, (images, targets) in enumerate(t):
            cluster_info = clustering_model.predict_cluster_of_batch(images)
            for c in range(runtime_helper.num_clusters):
                indices = (cluster_info == c).nonzero(as_tuple=True)[0]
                container[c] = torch.cat([container[c], images[indices]])

    batch_size = 256
    x2 = torch.zeros((batch_size, 3, 32, 32), device='cuda')
    y2 = None
    for c in range(runtime_helper.num_clusters):
        n_batch = container[c].size(0) // batch_size
        for i in range(n_batch):
            x1 = container[c][i * batch_size: (i + 1) * batch_size].cuda()
            y1 = model(x1)
            if y2 is None:
                y2 = model(x2)

            if x_metric == partitioned_range_l2_dist:
                x_rst = x_metric(x1, x2, clustering_model)
            else:
                x_rst = x_metric(x1, x2)

            if y_metric == partitioned_range_l2_dist:
                y_rst = y_metric(y1, y2, clustering_model)
            else:
                y_rst = y_metric(y1, y2)

            cur_lipschitz = y_rst.cpu() / x_rst.cpu()
            candidates[c] = torch.cat([candidates[c], cur_lipschitz.detach()])

        leftover = container[c].size(0) - n_batch * batch_size
        if leftover:
            x1 = container[c][n_batch * batch_size:].cuda()
            y1 = model(x1)
            leftover_x2 = x2[:leftover]
            leftover_y2 = y2[:leftover]

            if x_metric == partitioned_range_l2_dist:
                x_rst = x_metric(x1, leftover_x2, clustering_model)
            else:
                x_rst = x_metric(x1, leftover_x2)
            if y_metric == partitioned_range_l2_dist:
                y_rst = y_metric(y1, leftover_y2, clustering_model)
            else:
                y_rst = y_metric(y1, leftover_y2)
            cur_lipschitz = y_rst.cpu() / x_rst.cpu()
            candidates[c] = torch.cat([candidates[c], cur_lipschitz.detach()])

    for c in range(runtime_helper.num_clusters):
        print("[cluster {} ({})] max: {:.5f}, avg: {:.5f}, std: {:.5f}"
              .format(c, candidates[c].size(0), candidates[c].max(), candidates[c].mean(), candidates[c].std(unbiased=False)))


def check_lipschitz(args, tools):
    normalizer = get_normalizer(args.dataset)
    train_dataset = get_non_augmented_train_dataset(args, normalizer)
    train_dataset, _ = split_dataset_into_train_and_val(train_dataset, args.dataset)

    runtime_helper = RuntimeHelper()
    runtime_helper.set_pcq_arguments(args)

    if args.dataset != 'imagenet':
        model = Simple32Network()
    else:
        model = Simple224Network()
    model.cuda()

    clustering_model = tools.clustering_method(args)
    clustering_model.load_clustering_model()

    # Dist between 2 Data
    # x_metric = norm_per_data
    # y_metric = norm_per_data
    # x_metric = l2_dist
    # y_metric = l2_dist
    # x_metric = range_l2_dist
    # y_metric = range_l2_dist
    # x_metric = partitioned_range_l2_dist
    # y_metric = partitioned_range_l2_dist
    # pcq_loader = get_data_loader(train_dataset, batch_size=128, shuffle=True, workers=args.worker)
    # qat_loader = get_data_loader(train_dataset, batch_size=2, shuffle=True, workers=args.worker)
    # pcq_lipschitz(model, clustering_model, pcq_loader, runtime_helper, x_metric, y_metric)
    # qat_lipschitz(model, clustering_model, qat_loader, x_metric, y_metric)

    # Dist between Zero & Data
    # x_metric = batch_l2_dist
    # y_metric = batch_l2_dist
    # x_metric = batch_range_l2_dist
    # y_metric = batch_range_l2_dist
    # x_metric = partitioned_range_l2_dist
    # y_metric = partitioned_range_l2_dist
    data_loader = get_data_loader(train_dataset, batch_size=args.val_batch, shuffle=False, workers=args.worker)

    x_metric = batch_l2_dist
    y_metric = batch_range_l2_dist
    qat_lipschitz_with_zero(model, clustering_model, data_loader, x_metric, y_metric)
    pcq_lipschitz_with_zero(model, clustering_model, data_loader, runtime_helper, x_metric, y_metric)

    x_metric = partitioned_range_l2_dist
    y_metric = batch_range_l2_dist
    qat_lipschitz_with_zero(model, clustering_model, data_loader, x_metric, y_metric)
    pcq_lipschitz_with_zero(model, clustering_model, data_loader, runtime_helper, x_metric, y_metric)

    x_metric = batch_l2_dist
    y_metric = batch_l2_dist
    qat_lipschitz_with_zero(model, clustering_model, data_loader, x_metric, y_metric)
    pcq_lipschitz_with_zero(model, clustering_model, data_loader, runtime_helper, x_metric, y_metric)

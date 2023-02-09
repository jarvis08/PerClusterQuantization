import os

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

def get_normalizer(dataset_name):
    if dataset_name == 'imagenet':
        return transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    elif dataset_name == 'cifar10':
        return transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    elif dataset_name == 'cifar100':
        return transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    else:
        return transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1201, 0.1231, 0.1052))


def get_augmented_train_dataset(args, normalizer, model):
    train_resolution = 224
    if model == "inceptionv3":
        train_resolution = 299
    if args.dataset == 'imagenet':
        transformer = transforms.Compose([transforms.RandomResizedCrop(train_resolution),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         normalizer])
        return datasets.ImageFolder(root=os.path.join(args.imagenet, 'train'), transform=transformer)
    else:
        transformer = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          normalizer])
        if args.dataset == 'cifar10':
            dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transformer)
        elif args.dataset == 'cifar100':
            dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transformer)
        else:
            dataset = datasets.SVHN(root='./data', split='train', download=True, transform=transformer)
    return dataset


def get_non_augmented_train_dataset(args, normalizer, model):
    train_resolution, test_resolution = 224, 256
    if model == "inceptionv3":
        train_resolution, test_resolution = 299, 342
    if args.dataset == 'imagenet':
        transformer = transforms.Compose([transforms.Resize(test_resolution),
                                          transforms.CenterCrop(train_resolution),
                                          transforms.ToTensor(),
                                          normalizer])
        imagenet_dataset = datasets.ImageFolder(root=os.path.join(args.imagenet, 'train'), transform=transformer)
        ### partition data
        dataset_length = int(len(imagenet_dataset) * 0.1)           
        dataset, _ = torch.utils.data.random_split(imagenet_dataset, [dataset_length, len(imagenet_dataset) - dataset_length])
    else:
        transformer = transforms.Compose([transforms.ToTensor(), normalizer])
        if args.dataset == 'cifar10':
            dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transformer)
        elif args.dataset == 'cifar100':
            dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transformer)
        else:
            dataset = datasets.SVHN(root='./data', split='train', download=True, transform=transformer)
    return dataset


def get_test_dataset(args, normalizer, model):
    train_resolution, test_resolution = 224, 256
    if model == "inceptionv3":
        train_resolution, test_resolution = 299, 342
    if args.dataset == 'imagenet':
        transformer = transforms.Compose([transforms.Resize(test_resolution),
                                          transforms.CenterCrop(train_resolution),
                                          transforms.ToTensor(),
                                          normalizer])
        test_dataset = datasets.ImageFolder(root=os.path.join(args.imagenet, 'val'), transform=transformer)
    else:
        transformer = transforms.Compose([transforms.ToTensor(), normalizer])
        if args.dataset == 'cifar10':
            test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transformer)
        elif args.dataset == 'cifar100':
            test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transformer)
        else:
            test_dataset = datasets.SVHN(root='./data', split='test', download=True, transform=transformer)
    return test_dataset


def get_data_loader(dataset, batch_size=128, shuffle=False, workers=4):
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers, pin_memory=True)
    # return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True)


def split_dataset_into_train_and_val(full_dataset, dataset_name):
    n_data = len(full_dataset)
    indices = list(range(n_data))

    if dataset_name == 'svhn':
        train_idx = indices[:n_data - 6000]
        val_idx = indices[n_data - 6000:]
    else:
        train_size = int(n_data * 0.9)
        train_idx = indices[:train_size]
        val_idx = indices[train_size:]

    train_dataset = torch.utils.data.Subset(full_dataset, train_idx)
    val_dataset = torch.utils.data.Subset(full_dataset, val_idx)
    return train_dataset, val_dataset


def get_data_loaders(args, model):
    if args.dataset != 'imagenet':
        normalizer = get_normalizer(args.dataset)

        test_dataset = get_test_dataset(args, normalizer, model)
        test_loader = get_data_loader(test_dataset, batch_size=args.val_batch, shuffle=False, workers=args.worker)
        if args.mode == 'eval':
            return test_loader

        train_dataset = get_augmented_train_dataset(args, normalizer, model)

        train_loader = get_data_loader(train_dataset, batch_size=args.batch, shuffle=True, workers=args.worker)
    else:
        from .dali import get_dali_dataloader
        train_loader = get_dali_dataloader(128, os.path.join(args.imagenet, 'train'), is_training=True, num_workers=4)
        test_loader = get_dali_dataloader(256, os.path.join(args.imagenet, 'val'), is_training=False, num_workers=4)
        clustering_train_loader = get_dali_dataloader(512, os.path.join(args.imagenet, 'train'), is_training=False, num_workers=4)
    return {'train': train_loader, 'test': test_loader}

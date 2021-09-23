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


def get_augmented_train_dataset(args, normalizer):
    if args.dataset == 'imagenet':
        transformer = transforms.Compose([transforms.RandomResizedCrop(224),
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


def get_non_augmented_train_dataset(args, normalizer):
    if args.dataset == 'imagenet':
        transformer = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          normalizer])
        return datasets.ImageFolder(root=os.path.join(args.imagenet, 'train'), transform=transformer)
    else:
        transformer = transforms.Compose([transforms.ToTensor(), normalizer])
        if args.dataset == 'cifar10':
            dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transformer)
        elif args.dataset == 'cifar100':
            dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transformer)
        else:
            dataset = datasets.SVHN(root='./data', split='train', download=True, transform=transformer)
    return dataset


def get_test_dataset(args, normalizer):
    if args.dataset == 'imagenet':
        transformer = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
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


def get_sequential_loader(args, dataset):
    return torch.utils.data.DataLoader(dataset, batch_size=args.val_batch, shuffle=False, num_workers=args.worker)


def get_shuffled_loader(args, dataset):
    return torch.utils.data.DataLoader(dataset, batch_size=args.batch, shuffle=True, num_workers=args.worker)


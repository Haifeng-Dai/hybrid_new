import torch
import random
import numpy as np

from copy import deepcopy
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision.datasets import MNIST, CIFAR10, CIFAR100
from torchvision import transforms


class GetDataset:
    def __init__(self,
                 dataset_name: str = 'mnist',
                 n_public: int = 0) -> None:
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        if dataset_name == 'mnist':
            self.__mnist_set()
        elif dataset_name == 'cifar10':
            self.__cifar10_set()
        elif dataset_name == 'cifar100':
            self.__cifar100_set()
        else:
            raise ValueError('dataset error.')

        if n_public:
            self.train_set, self.public_set = random_split(
                dataset=self.train_set,
                lengths=[len(self.train_set)-n_public, n_public]
            )

    def __trans(self, mean, std):
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    def __mnist_set(self):
        self.train_set = MNIST(
            root='./data',
            train=True,
            download=True,
            transform=self.__trans((0.1307,), (0.3081,))
        )
        self.test_set = MNIST(
            root='./data',
            train=False,
            transform=self.__trans((0.1326,), (0.3106,))
        )
        self.n_targets = 10
        self.in_channel = 1

    def __cifar10_set(self):
        self.train_set = CIFAR10(
            root='./data',
            train=True,
            download=True,
            transform=self.__trans(mean=(0.4914, 0.4822, 0.4465),
                                   std=(0.2470, 0.2435, 0.2616)),
        )
        self.test_set = CIFAR10(
            root='./data',
            train=False,
            transform=self.__trans(mean=(0.4940, 0.4850, 0.4504),
                                   std=(0.2467, 0.2429, 0.2616)),
        )
        self.n_targets = 10
        self.in_channel = 3

    def __cifar100_set(self):
        self.train_set = CIFAR100(
            root='./data',
            train=True,
            download=True,
            transform=self.__trans(mean=(0.5071, 0.4865, 0.4409),
                                   std=(0.2009, 0.1984, 0.2023)),
        )
        self.test_set = CIFAR100(
            root='./data',
            train=False,
            transform=self.__trans(mean=(0.5088, 0.4874, 0.4419),
                                   std=(0.2019, 0.2000, 0.2037)),
        )
        self.n_targets = 100
        self.in_channel = 3


# def get_dataset(dataset_name: str = 'mnist') -> list:
#     transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#     ])
#     if dataset_name == 'mnist':
#         train_set = MNIST(root='./data',
#                           train=True,
#                           download=True,
#                           transform=transforms.ToTensor())
#         test_set = MNIST(root='./data',
#                          train=False,
#                          transform=transforms.ToTensor())
#         train_set_, public_set = random_split(dataset=train_set,
#                                               lengths=[50000, 10000])
#     elif dataset_name == 'cifar10':
#         train_set = CIFAR10(root='./data',
#                             train=True,
#                             download=True,
#                             transform=transform)
#         test_set = CIFAR10(root='./data',
#                            train=False,
#                            transform=transform)
#         train_set_, public_set = random_split(dataset=train_set,
#                                               lengths=[50000, 10000])
#     elif dataset_name == 'cifar100':
#         train_set = CIFAR100(root='./data',
#                              train=True,
#                              download=True,
#                              transform=transform)
#         test_set = CIFAR100(root='./data',
#                             train=False,
#                             transform=transform)
#         train_set_, public_set = random_split(dataset=train_set,
#                                               lengths=[59990, 10])
#     else:
#         raise ValueError('dataset error.')
#     return train_set, train_set_, public_set, test_set


def get_dataloader(
        dataset_name: str = 'mnist',
        splite: bool = False,
) -> list:
    # dataset = get_dataset(dataset_name)
    dataset = GetDataset(dataset_name)
    train_set = dataset.train_set
    public_set = dataset.public_set
    test_set = dataset.test_set
    train_loader = DataLoader(
        dataset=dataset.train_set,
        shuffle=True,
        batch_size=160,
        num_workers=8,
    )
    test_loader = DataLoader(
        dataset=test_set,
        shuffle=False,
        batch_size=160,
        num_workers=8,
    )
    if splite:
        targets = set(test_set.targets.tolist())
        dataset_splited = data_split(train_set, targets)
        return dataset_splited, public_set, test_loader
    return train_loader, public_set, test_loader


def data_split(
        dataset: Dataset | list,
        targets: list | set,
) -> dict:

    data_set = dict.fromkeys(targets)
    for target in targets:
        data_set[target] = []
    for data, target_ in dataset:
        data_set[target_].append((data, target_))
    return data_set


def non_iid(
        # dataset: list | Dataset,
        # alpha=100.,
        pro: torch.Tensor,
        targets: list,
        all_idx: list,
        all_target: list,
        all_data: list,
        n_client_data: int = 1000,
        n_client: int = 3,
):
    '''根据Dirichlet分布分割数据集'''
    # alpha = float(alpha)
    # all_idx = []
    # all_data = []
    # all_target = []
    # for idx, (data, target) in enumerate(dataset):
    #     all_idx.append(idx)
    #     all_data.append(data)
    #     all_target.append(target)
    # random.shuffle(all_idx)

    # targets = set(all_target)
    # num_targets = len(targets)
    # pro = Dirichlet(torch.full((num_targets,), alpha)).sample([num_client])
    num = (pro * n_client_data).round().int()

    idx_max = num.argmax(dim=1)
    num_sum = num.sum(dim=1)
    target_data = {target: [] for target in targets}
    for idx in all_idx:
        target_data[all_target[idx]].append((all_data[idx], all_target[idx]))

    data_client = {}
    for client in range(n_client):
        data_client[client] = []
        err = num_sum[client] - n_client_data
        if err:
            num[client, idx_max[client]] -= err
        idx_ = 0
        for target_ in targets:
            num_new_data = num[client, target_]
            new_data = target_data[target_][:idx_+num_new_data]
            data_client[client].extend(new_data)
            target_data[target_] = target_data[target_][idx_+num_new_data:]
    for client in range(n_client):
        if len(data_client[client]) != n_client_data:
            raise Warning(
                f'number of data in client {client} less than {n_client_data}')

    return data_client


def split_non_iid(
        dataset: list | Dataset,
        pro: torch.Tensor,
        # alpha: float = 100,
        n_data: int = 1000,
        n_client: int = 3,
):
    # alpha = float(alpha)
    all_idx = []
    all_data = []
    all_target = []
    for idx, (data, target) in enumerate(dataset):
        all_idx.append(idx)
        all_data.append(data)
        all_target.append(target)
    random.shuffle(all_idx)

    targets = set(all_target)
    # n_targets = len(targets)
    # pro = Dirichlet(torch.full((n_targets,), alpha)).sample([n_client])
    dataset_ = non_iid(pro=pro, targets=targets, all_idx=all_idx,
                       all_target=all_target, all_data=all_data, n_client_data=n_data, n_client=n_client)
    return dataset_


def dirichlet_split_noniid(train_labels, n_clients, n_classes, label_distribution):
    class_idcs = [np.argwhere(np.array(train_labels) == y).flatten()
                  for y in range(n_classes)]
    client_idcs = [[] for _ in range(n_clients)]
    for k_idcs, fracs in zip(class_idcs, label_distribution):
        split_point = np.split(
            k_idcs, (np.cumsum(fracs)[:-1]*len(k_idcs)).astype(int))
        for i, idcs in enumerate(split_point):
            # client_idcs[i] += [idcs]
            client_idcs[i] += idcs.tolist()
    # client_idcs = [np.concatenate(idcs).tolist() for idcs in client_idcs]
    return client_idcs


def dirichlet_split(
        dataset_name: str = 'mnist',
        alpha: float or int = 1,
        n_clients: int = 3,
        n_public: int = 50,
        avg: bool = False,
):

    dataset = GetDataset(dataset_name=dataset_name, n_public=n_public)
    # n_targets = dataset.n_targets
    # in_channel = dataset.in_channel
    # train_set = dataset.train_set
    # test_set = dataset.test_set
    # n_classes = dataset.n_targets
    alpha = float(alpha)
    label_distribution = np.random.dirichlet(
        [alpha]*n_clients, dataset.n_targets)

    train_targets = [data[1] for data in dataset.train_set]
    train_idx = dirichlet_split_noniid(
        train_labels=train_targets,
        n_clients=n_clients,
        n_classes=dataset.n_targets,
        label_distribution=label_distribution
    )
    if avg:
        test_set = dataset.test_set
    else:
        test_targets = [data[1] for data in dataset.test_set]
        test_idx = dirichlet_split_noniid(
            train_labels=test_targets,
            n_clients=n_clients,
            n_classes=dataset.n_targets,
            label_distribution=label_distribution
        )
        test_set = {}

    train_set = {}
    for i in range(n_clients):
        train_set[i] = [dataset.train_set[j] for j in train_idx[i]]
        if not avg:
            test_set[i] = [dataset.test_set[j] for j in test_idx[i]]

    return train_set, test_set, dataset.n_targets, dataset.in_channel, dataset.public_set


def iid_split(dataset, num, n_classes):
    # print(type(dataset), len(dataset))
    all_target = []
    for _, target in dataset:
        all_target.append(target)
    # all_target = dataset.targets
    client_idx = {i: [] for i in range(num)}
    class_idcs = [np.argwhere(np.array(all_target) == y).flatten()
                  for y in range(n_classes)]
    # [print(i, len(idxs)) for i, idxs in enumerate(class_idcs)]
    for idxs in class_idcs:
        frac = int((1 / num) * len(idxs))
        fracs = np.cumsum([frac,] * (num - 1))
        splited_idx = np.split(idxs, fracs)
        for client in range(num):
            client_idx[client] += splited_idx[client].tolist()
    client_dataset = {i: [] for i in range(num)}
    for i in range(num):
        client_dataset[i] = [dataset[j] for j in client_idx[i]]
    return client_dataset


# if __name__ == '__main__':
    # dataset = GetDataset().test_set
    # dataset_ = iid_split(dataset, 3, 10)
    # print(dataset_[0][0])
    # print(len(dataset_))
    # [print(len(dataset_[i])) for i in range(3)]
    # for i in range(3):
    #     k = 0
    #     for _, target in dataset_[i]:
    #         if target == 1:
    #             k += 1
    #     print(k)
    # train_set, test_set, n_targets, in_channel = dirichlet_split(
    #     dataset_name='cifar10', alpha=1, n_clients=3)
    # print(train_set[0][0])

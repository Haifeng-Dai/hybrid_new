import torch
import random

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

    def __mnist_set(self):
        trans_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        trans_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1326,), (0.3106,))
        ])
        self.train_set = MNIST(root='./data',
                               train=True,
                               download=True,
                               transform=trans_train)
        self.test_set = MNIST(root='./data',
                              train=False,
                              transform=trans_test)
        self.n_targets = 10
        self.in_channel = 1

    def __cifar10_set(self):
        trans_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])
        trans_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4940, 0.4850, 0.4504), (0.2467, 0.2429, 0.2616))
        ])
        self.train_set = CIFAR10(root='./data',
                                 train=True,
                                 download=True,
                                 transform=trans_train)
        self.test_set = CIFAR10(root='./data',
                                train=False,
                                transform=trans_test)
        self.n_targets = 10
        self.in_channel = 3

    def __cifar100_set(self):
        trans_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2009, 0.1984, 0.2023))
        ])
        trans_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5088, 0.4874, 0.4419), (0.2019, 0.2000, 0.2037))
        ])
        self.train_set = CIFAR100(root='./data',
                                  train=True,
                                  download=True,
                                  transform=trans_train)
        self.test_set = CIFAR100(root='./data',
                                 train=False,
                                 transform=trans_test)
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

import torch
import random

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

def list_same_term(len_list, term=[]):
    # 返回一个全是空列表的列表
    list_return = []
    for _ in range(len_list):
        list_return.append(deepcopy(term))
    return list_return

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

class SplitData:
    '''
    分割数据集
    '''

    def __init__(self, dataset):
        self.initial_dataset = dataset
        self.targets = self.get_target()
        self.num_target = len(self.targets)

    def get_target(self):
        # 获取所有标签
        if isinstance(self.initial_dataset.targets, list):
            targets = set(self.initial_dataset.targets)
        elif torch.is_tensor(self.initial_dataset.targets):
            targets = set(self.initial_dataset.targets.numpy().tolist())
        else:
            raise ValueError('dataset.targets is not tensor or list.')
        targets = list(targets)
        targets.sort()
        return targets

    def split_data(self):
        # 将数据集按标签分割
        targets = self.targets
        splited_data = dict.fromkeys(targets)
        for key in splited_data.keys():
            splited_data[key] = []
        for data in self.initial_dataset:
            splited_data[data[1]].append(data)
        for key in splited_data.keys():
            random.shuffle(splited_data[key])
        return splited_data

    def num_data_target(self):
        # 获取每个标签对数据集的数量
        num_data_target_all = []
        splited_data = self.split_data()
        for target in self.targets:
            num_data_target_all.append(len(splited_data[target]))
        return num_data_target_all

    def all_iid(self, num_client, num_client_data):
        # 按照客户端数量和每个客户端的数据量分配数据
        # if num_client_data * num_client > self.num_target * min(self.num_data_target()):
        #     raise ValueError('too large num_client_data * num_client.')
        # if num_client_data % self.num_target != 0:
        #     raise ValueError('num_client_data \% num_targets != 0.')

        num_data_target = num_client_data // self.num_target
        client_data = list_same_term(num_client)
        splited_data = self.split_data()
        for target in self.targets:
            data_target = deepcopy(splited_data[target])
            random.shuffle(data_target)
            idx = 0
            for client in range(num_client):
                add_data = data_target[idx: idx + num_data_target]
                client_data[client].extend(add_data)
                idx += num_data_target
        return client_data

    def all_non_iid(self, num_client, num_client_data, client_main_target, proportion=None):
        # if num_client_data * num_client > self.num_target * min(self.num_data_target()):
        #     raise ValueError('too large num_client_data * num_client.')
        if not proportion:
            proportion = 2 / self.num_target
        # if num_client_data * num_client * proportion > min(self.num_data_target()):
        #     raise Warning(
        #         'maybe too large num_client_data * num_client * proportion.')
        # if num_client_data % self.num_target != 0:
        #     raise ValueError('num_client_data \% num_targets != 0.')

        num_client_data_minor = int(
            (1 - proportion) * num_client_data // (self.num_target - 1))
        num_client_data_mian = num_client_data - \
            num_client_data_minor * (self.num_target - 1)
        splited_data = self.split_data()
        client_data = list_same_term(num_client)
        for target in self.targets:
            data_target = deepcopy(splited_data[target])
            random.shuffle(data_target)
            idx = 0
            for client in range(num_client):
                if client_main_target[client] == target:
                    add_data = data_target[idx: idx + num_client_data_mian]
                    client_data[client].extend(add_data)
                    idx += num_client_data_mian
                    continue
                add_data = data_target[idx: idx + num_client_data_minor]
                client_data[client].extend(add_data)
                idx += num_client_data_minor
        return client_data

    def server_non_iid(self, num_server, num_server_client, num_client_data, client_main_target, proportion=None):
        # 按照客户端数量和每个客户端的数据量分配数据
        # if num_client_data * num_server * num_server_client > self.num_target * min(self.num_data_target()):
        #     raise ValueError(
        #         'too large num_client_data * num_server * num_server_client.')
        if not proportion:
            proportion = 2 / self.num_target
        num_data_server = num_server_client * num_client_data
        # if num_server_client * num_client_data * num_server * proportion > min(self.num_data_target()):
        #     raise Warning(
        #         'maybe too large num_server * num_server_client * num_client_data.')
        # if num_client_data % self.num_target != 0:
        #     raise ValueError('num_client_data \% num_targets != 0.')

        server_data = self.all_non_iid(
            num_server, num_data_server, client_main_target, proportion)
        server_client_data = list_same_term(num_server)
        for server in range(num_server):
            server_data_ = server_data[server]
            random.shuffle(server_data_)
            idx = 0
            server_client_data[server] = list_same_term(num_server_client)
            for client in range(num_server_client):
                add_data = server_data_[idx: idx + num_client_data]
                server_client_data[server][client].extend(add_data)
                idx += num_client_data
        client_data = []
        for server in range(num_server):
            for client in range(num_server_client):
                client_data.append(server_client_data[server][client])
        return client_data

    def client_non_iid(self, num_server, num_server_client, num_client_data, client_main_target, proportion=None):
        # if num_client_data * num_server * num_server_client > self.num_target * min(self.num_data_target()):
        #     raise ValueError(
        #         'too large num_client_data * num_server * num_server_client.')
        if not proportion:
            proportion = 2 / self.num_target
        # if num_server_client * num_client_data * num_server * proportion > min(self.num_data_target()):
        #     raise Warning(
        #         'maybe too large num_server * num_server_client * num_client_data.')
        # if num_client_data % self.num_target != 0:
        #     raise ValueError('num_client_data \% num_targets != 0.')

        num_client_data_minor = int(
            (1 - proportion) * num_client_data // (self.num_target - 1))
        num_client_data_mian = num_client_data - \
            num_client_data_minor * (self.num_target - 1)
        splited_data = deepcopy(self.split_data())
        server_client_data = list_same_term(
            num_server, list_same_term(num_server_client))
        for target in range(self.num_target):
            data_target = splited_data[target]
            idx = 0
            for server in range(num_server):
                for client in range(num_server_client):
                    if client_main_target[client] == target:
                        add_data = data_target[idx: idx + num_client_data_mian]
                        server_client_data[server][client].extend(add_data)
                        idx += num_client_data_mian
                        continue
                    add_data = data_target[idx: idx + num_client_data_minor]
                    server_client_data[server][client].extend(add_data)
                    idx += num_client_data_minor
        client_data = []
        for server in range(num_server):
            for client in range(num_server_client):
                client_data.append(server_client_data[server][client])
        return client_data

    def server_target_niid(self, num_server, num_server_client, target_list, num_client_data):
        server_data = list_same_term(num_server)
        splited_data = deepcopy(self.split_data())
        num_server_data = num_client_data * num_server_client
        for server in range(num_server):
            num_data_each_target = num_server_data // len(target_list[server])
            for target in target_list[server]:
                add_data = splited_data[target][
                    : num_data_each_target]
                server_data[server].extend(add_data)
                splited_data[target] = splited_data[target][
                    num_data_each_target:]
        client_data = list_same_term(num_server * num_server_client)
        client_idx = 0
        for server in range(num_server):
            random.shuffle(server_data[server])
            for _ in range(num_server_client):
                client_data[client_idx] = server_data[server][: num_client_data]
                server_data[server] = server_data[server][num_client_data:]
                client_idx += 1
        return client_data
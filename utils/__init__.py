import torch
import os
import sys
import logging
import numpy
import argparse
import torchvision
from torch.distributed import init_process_group

from copy import deepcopy
from .data import *
from .model import *
from .train import *


def model_init(
        num_client: int,
        model_structure: str = 'mlp',
        num_target: int = 10,
        in_channel: int = 1
) -> list[torch.nn.Module]:

    if model_structure == 'mlp':
        model_ = MLP(n_class=num_target)
    elif model_structure == 'resnet18':
        model_ = torchvision.models.resnet18(weights=None, num_classes=num_target)
    elif model_structure == 'cnn1':
        model_ = CNN1(in_channel=in_channel, n_class=num_target)
    elif model_structure == 'cnn2':
        model_ = CNN1(in_channel=in_channel, n_class=num_target)
    elif model_structure == 'lenet5':
        model_ = CNN1(in_channel=in_channel, n_class=num_target)
    model_list = []
    for i in range(num_client):
        model_list.append(deepcopy(model_))
    return model_list


def eval_model(
        model: torch.nn.Module,
        test_loader: DataLoader,
) -> float:

    if torch.cuda.is_available():
        device = torch.cuda.current_device()
    # elif torch.backends.mps.is_available():
    #     device = 'mps'
    else:
        device = 'cpu'
    model_copy = deepcopy(model).to(device)
    model_copy.eval()
    correct = 0
    len_data = 0
    for data, targets in test_loader:
        outputs = model_copy(data.to(device))
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == targets.to(device)).sum()
        len_data += len(targets)
    accuracy = correct / len_data
    return accuracy.item()


def setup_seed(seed: int = 0):
    if not seed:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    numpy.random.seed(seed)
    random.seed(seed)


def ddp_setup(rank: int,
              world_size: int,
              port: int = 20001):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    # os.environ["MASTER_PORT"] = "12355"
    os.environ["MASTER_PORT"] = str(port)
    init_process_group(backend="nccl",
                       #    init_method=f'tcp://localhost:{port}',
                       rank=rank,
                       world_size=world_size)
    torch.cuda.set_device(rank)


def get_args():
    parser = argparse.ArgumentParser(description='test')
    parser.add_argument('--a', type=float or int, default=1.,
                        help='weight of kl loss in distilling trainning, default=1.0')
    parser.add_argument('--temperature', type=float or int, default=6.,
                        help='temperature of distillation, default=6')
    parser.add_argument('--n_client', type=int, default=9,
                        help='number of clients, default=9')
    parser.add_argument('--n_train_data', type=int, default=1000,
                        help='number of client\'s train data, default=1000')
    parser.add_argument('--n_public_data', type=int, default=100,
                        help='number of publci data, default=100')
    parser.add_argument('--n_test_data', type=int, default=200,
                        help='number of client\'s test data, default=200')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed, default=0')
    parser.add_argument('--local_epochs', type=int, default=10,
                        help='epochs of training in local data, default=10')
    parser.add_argument('--distill_epochs', type=int, default=10,
                        help='epochs of distilling in public data, default=10')
    parser.add_argument('--batch_size', type=int, default=160,
                        help='batch size, default=160')
    parser.add_argument('--server_epochs', type=int, default=10,
                        help='epochs of server communication, default=10')
    parser.add_argument('--alpha', type=float or int, default=10.,
                        help='parameter of Dirichlet distribution, default=10.0')
    parser.add_argument('--dataset', type=str, default='mnist',
                        help='select a dataset from mnist, cifar10 and cifar100, default=mnist.')
    parser.add_argument('--model_structure', type=str, default='mlp',
                        help='select a model to train, default=mlp.')
    parser.add_argument('--device', type=int, default=0,
                        help='select the GPU to use, default=0.')
    args = parser.parse_args()
    return args


def get_logger(filename, mode='w'):
    '''log setiing'''
    log_formatter = logging.Formatter(
        fmt='[%(asctime)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    # File logger
    file_handler = logging.FileHandler(filename, mode=mode)
    file_handler.setFormatter(log_formatter)
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    # Stdout logger
    std_handler = logging.StreamHandler(sys.stdout)
    std_handler.setFormatter(log_formatter)
    std_handler.setLevel(logging.DEBUG)
    logger.addHandler(std_handler)
    return logger

# if __name__ == '__main__':
#     args = get_args()
#     print(args.a)

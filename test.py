import time
import torch
import torch.nn as nn
import os

from copy import deepcopy
from torch.utils.data import DataLoader
from mpi4py import MPI

from utils import *

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

t_start = time.time()
torch.set_printoptions(precision=3,
                       threshold=1000,
                       edgeitems=5,
                       linewidth=1000,
                       sci_mode=False)
t = time.localtime()
log_path = f'./log/{t.tm_year}-{t.tm_mon}-{t.tm_mday}/'
if not os.path.exists(log_path):
    os.makedirs(log_path)
log_path += f'{t.tm_hour}-{t.tm_min}-{t.tm_sec}-{rank}.log'
log = get_logger(log_path)


# %% 1. basic parameters
args = get_args()
args.device = rank
if rank == 0:
    args.temperature = 5
elif rank == 1:
    args.temperature = 10
elif rank == 2:
    args.temperature = 15
else:
    raise ValueError('error.')
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
setup_seed(args.seed)

# %% 2. data preparation
server_train_set, server_test_set, n_targets, in_channel, public_data = dirichlet_split(
    dataset_name=args.dataset,
    alpha=args.alpha,
    n_clients=args.n_server,
    n_public=args.n_public_data,
    avg=False
)

print(server_train_set.keys())
print(server_test_set.keys())
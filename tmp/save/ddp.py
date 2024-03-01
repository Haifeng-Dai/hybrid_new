import torch
import torchvision
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from utils import *

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group, is_available
import os


def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "30003"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def prepare_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        pin_memory=True,
        sampler=DistributedSampler(dataset),
        num_workers=8,
    )


def main(rank, train_set, world_size, acc):
    ddp_setup(rank, world_size)
    data_loader = prepare_dataloader(train_set, 160)
    data_loader.sampler.set_epoch(1)
    for i, (data, target) in enumerate(data_loader):
        if i == 0:
            print(target)
    destroy_process_group()


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    train_set, _, _, test_set = get_dataset()
    # cuda_id = torch.cuda.current_device()
    # print(cuda_id)
    acc = []
    mp.spawn(main, args=(train_set, world_size, acc), nprocs=world_size)

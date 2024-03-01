import torch, random
import torchvision
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from utils import *

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group, barrier
import os


def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "30001"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        gpu_id: int,
        save_every: int,
        test_loader: DataLoader,
        i: int
    ) -> None:
        self.gpu_id = gpu_id
        # self.model = model.cuda()
        self.train_data = train_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.model = DDP(model.cuda(), device_ids=[gpu_id])
        self.test_loader = test_loader
        self.i = i

    def _run_batch(self, source, targets):
        self.optimizer.zero_grad()
        output = self.model(source)
        loss = F.cross_entropy(output, targets)
        loss.backward()
        self.optimizer.step()

    def _run_epoch(self, epoch):
        # b_sz = len(next(iter(self.train_data))[0])
        if self.gpu_id == 1:
            msg = "[Client {} Local epoch {}] | Steps: {}"
            print(msg.format(self.i, epoch, len(self.train_data)))
        self.train_data.sampler.set_epoch(epoch)
        for source, targets in self.train_data:
            source = source.cuda()
            targets = targets.cuda()
            self._run_batch(source, targets)

    def _save_checkpoint(self, epoch):
        acc = eval_model(self.model, self.test_loader)
        PATH = "res/tmp/checkpoint.pt"
        # torch.save(self.model.state_dict(), PATH)
        msg = "Epoch {} | acc={}, Training checkpoint saved at {}"
        print(msg.format(epoch, acc, PATH))

    def train(self, max_epochs: int):
        for epoch in range(max_epochs):
            self._run_epoch(epoch)
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_checkpoint(epoch)
        return self.model


def prepare_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset)
    )


def main(rank: int, i: int, client: torch.nn.Module, world_size: int, save_every: int, total_epochs: int, batch_size: int, client_set: list[Dataset], test_loader: DataLoader):
    ddp_setup(rank, world_size)
    train_data = prepare_dataloader(client_set[i], batch_size)
    optimizer = torch.optim.Adam(client.parameters(), lr=1e-3)
    trainer = Trainer(client, train_data, optimizer, rank,
                      save_every, test_loader=test_loader, i=i)
    model_ = trainer.train(total_epochs)
    if rank == 1:
        torch.save(model_.module.state_dict(), f'res/tmp/{i}.pt')
    destroy_process_group()


if __name__ == "__main__":
    save_every, total_epochs, batch_size, agg_epochs = 5, 10, 160, 10
    world_size = torch.cuda.device_count()

    client_list = model_init(
        num_client=9,
        model='mlp',
        num_target=10
    )
    setup_seed(20)

    train_set, _, _, test_set = get_dataset()
    client_set = non_iid(
        dataset=train_set,
        alpha=1,
        num_client_data=2000,
        num_client=9
    )
    test_loader = DataLoader(
        dataset=test_set,
        batch_size=160,
        shuffle=True,
        pin_memory=True,
        num_workers=8
    )

    init_model = MLP(n_class=10)
    acc = []
    for epoch in range(agg_epochs):
        for i, client in enumerate(client_list):
            mp.spawn(fn=main,
                     args=(i, client, world_size, save_every,
                           total_epochs, batch_size, client_set, test_loader),
                     nprocs=world_size)
        client_list_ = []
        for j, _ in enumerate(client_list):
            model_para = torch.load(
                f'res/tmp/{j}.pt', map_location='cuda:1')
            model_ = deepcopy(init_model)
            model_.load_state_dict(model_para)
            client_list_.append(model_)
        client_list = aggregate(client_list_)
        acc_ = eval_model(client_list[0], test_loader)
        acc.append(acc_)
        print(f'Epoch {epoch} | server: {acc_}.')
    torch.save({'acc': acc,
                'model': client_list[0]}, 'res/avg_para.pt')

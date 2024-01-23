import sys
from utils import *

import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

torch.set_printoptions(
    precision=3,
    linewidth=100000,
    sci_mode=False,
)


# def ddp_setup(rank, world_size, port):
#     """
#     Args:
#         rank: Unique identifier of each process
#         world_size: Total number of processes
#     """
#     os.environ["MASTER_ADDR"] = "localhost"
#     os.environ["MASTER_PORT"] = str(port)
#     init_process_group(backend="nccl", rank=rank, world_size=world_size)
#     torch.cuda.set_device(rank)


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        train_data: DataLoader,
        gpu_id: int,
        test_loader: DataLoader,
        i: int,
        client_list: list = [],
        way: str = 'local',
        a: float = 1,
    ):
        self.gpu_id = gpu_id
        self.train_data = train_data
        self.optimizer = optimizer
        self.model = DDP(model.cuda(), device_ids=[gpu_id])
        self.test_loader = test_loader
        self.i = i
        self.way = way
        self.model_list = [model.cuda() for model in client_list]
        if way == 'local':
            self.loss_fun = CE_Loss.cuda()
        elif way == 'dist':
            self.ce_loss = CE_Loss.cuda()
            self.kl_loss = KL_Loss.cuda()
            self.model_list.pop(gpu_id)
        self.a = a

    def _run_batch(self, source: torch.Tensor, targets: torch.Tensor):
        self.optimizer.zero_grad()
        output = self.model(source)
        batch_size = output.shape[0]
        if self.way == 'local':
            loss = self.loss_fun(output, targets)
        elif self.way == 'dist':
            loss_ce = self.ce_loss(output, targets)
            logits = torch.zeros([batch_size, 10], device=self.gpu_id)
            for model__ in self.model_list:
                logits += model__(source).detach()
            logits /= len(self.model_list)
            loss_kl = self.kl_loss(output, logits)
            loss = (1 - self.a) * loss_ce + self.a * loss_kl
        loss.backward()
        self.optimizer.step()

    def _run_epoch(self, epoch: int):
        self.train_data.sampler.set_epoch(epoch)
        for source, targets in self.train_data:
            source = source.cuda()
            targets = targets.cuda()
            self._run_batch(source, targets)

    def _save_checkpoint(self, epoch: int):
        acc = eval_model(self.model, self.test_loader)
        # PATH = "res/tmp/checkpoint.pt"
        # torch.save(self.model.state_dict(), PATH)
        print("[Client {}, {} Epoch {}/{}] acc={:.4f}".format(
            self.i+1, self.way.capitalize(), epoch+1, self.max_epochs, acc))
        return acc

    def train(self, max_epochs: int):
        self.max_epochs = max_epochs
        acc = []
        for epoch in range(max_epochs):
            self._run_epoch(epoch)
            # if self.gpu_id == 0 and epoch % self.save_every == 0:
            if self.gpu_id == 1:
                acc_ = self._save_checkpoint(epoch)
                acc.append(acc_)
        return self.model, acc


def prepare_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=8,
        sampler=DistributedSampler(dataset)
    )


def main_local(
        rank: int,
        i: int,
        client: torch.nn.Module,
        world_size: int,
        data_set: Dataset,
        test_loader: DataLoader,
        batch_size: int = 32,
        train_epochs: int = 1,
        port: int = 20001
):

    ddp_setup(rank, world_size, port)
    train_data = prepare_dataloader(data_set, batch_size)
    optimizer = torch.optim.Adam(params=client.parameters(),
                                 lr=1e-3,
                                 weight_decay=1e-4)
    trainer = Trainer(model=client,
                      optimizer=optimizer,
                      train_data=train_data,
                      gpu_id=rank,
                      test_loader=test_loader,
                      i=i)
    model_, acc = trainer.train(train_epochs)
    if rank == 1:
        torch.save(obj={'model': model_.module.state_dict(),
                        'acc': acc},
                   f=f'res/tmp/md{i}.pt')
    destroy_process_group()


def main_distill(
        rank: int,
        i: int,
        client_list: list[torch.nn.Module],
        world_size: int,
        data_set: Dataset,
        test_loader: DataLoader,
        batch_size: int = 32,
        distill_epochs: int = 1,
        a: float = 1,
        port: int = 20001
):

    ddp_setup(rank, world_size, port)
    train_data = prepare_dataloader(data_set, batch_size)
    optimizer = torch.optim.Adam(params=client_list[i].parameters(),
                                 lr=1e-3,
                                 weight_decay=1e-4)
    trainer = Trainer(model=client_list[i],
                      optimizer=optimizer,
                      train_data=train_data,
                      gpu_id=rank,
                      test_loader=test_loader,
                      i=i,
                      client_list=client_list,
                      way='dist',
                      a=a)
    model_, acc = trainer.train(distill_epochs)
    if rank == 1:
        torch.save(obj={'model': model_.module.state_dict(),
                        'acc': acc},
                   f=f'res/tmp/md{i}.pt')
    destroy_process_group()


if __name__ == "__main__":
    args = get_args()
    # a = args.a
    # n_client = args.n_client
    # n_train_data = args.n_train_data
    # n_test_data = args.n_test_data
    # seed = args.seed
    # train_epochs = args.train_epochs
    # distill_epochs = args.distill_epochs
    # batch_size = args.batch_size
    # server_epochs = args.server_epochs
    # alpha = args.alpha

    a = args.a
    n_client = 2
    n_train_data = args.n_train_data
    n_test_data = args.n_test_data
    seed = args.seed
    train_epochs = 1
    distill_epochs = 1
    batch_size = args.batch_size
    server_epochs = 1
    alpha = args.alpha

    world_size = torch.cuda.device_count()
    all_client = range(n_client)
    client_list = model_init(
        num_client=n_client,
        model='mlp',
        num_target=10
    )
    # setup_seed(seed)

    data_set = GetDataset(dataset_name='mnist', n_public=100)
    pro = Dirichlet(torch.full((data_set.n_targets,),
                    float(alpha))).sample([n_client])
    train_set = split_non_iid(
        dataset=data_set.train_set,
        pro=pro,
        n_data=n_train_data,
        n_client=n_client
    )
    test_set = split_non_iid(
        dataset=data_set.test_set,
        pro=pro,
        n_data=n_test_data,
        n_client=n_client
    )
    test_loader = {}
    for i, dataset_ in test_set.items():
        test_loader[i] = DataLoader(
            dataset=dataset_,
            batch_size=10,
            pin_memory=True,
            num_workers=8
        )

    model_str = MLP(n_class=10)
    acc = {i: [] for i in all_client}
    port = str(torch.randint(low=10000, high=50000, size=[
        1], dtype=torch.int).item())
    for epoch in range(server_epochs):
        # local train
        for i, client in enumerate(client_list):
            print('-'*10, f'client {i+1} local epoch {epoch+1}', '-'*10)
            mp.spawn(fn=main_local,
                     args=(i, client, world_size, train_set[i],
                           test_loader[i], batch_size, train_epochs, port),
                     nprocs=world_size)

        client_list_ = []
        print('-'*10, f'local {epoch+1}/{server_epochs} acc', '-'*10)
        for i in all_client:
            model_para = torch.load(
                f'res/tmp/{i}.pt', map_location='cuda:1')
            model_str.load_state_dict(model_para)
            client_list_.append(deepcopy(model_str))
            acc_i = eval_model(model_str, test_loader[i])
            acc[i].append(acc_i)
            print('Epoch {}/{} | client {}/{} local acc: {:.4f}.'.format(
                epoch+1, server_epochs, i+1, n_client, acc_i))
        client_list = client_list_

        # distill
        for i in all_client:
            print('-'*10, f'client {i+1} distill epoch {epoch+1}', '-'*10)
            mp.spawn(fn=main_distill,
                     args=(i, client_list, world_size,
                           train_set[i], test_loader[i], batch_size, distill_epochs, args.a, port),
                     nprocs=world_size)

        client_list_ = []
        print('-'*10, f'distill {epoch+1}/{server_epochs} acc', '-'*10)
        for i in all_client:
            model_para = torch.load(
                f'res/tmp/{i}.pt', map_location='cuda:1')
            model_str.load_state_dict(model_para)
            client_list_.append(deepcopy(model_str))
            acc_i = eval_model(model_str, test_loader[i])
            acc[i].append(acc_i)
            print('Epoch {}/{} | client {}/{} dist acc: {:.4f}.'.format(
                epoch+1, server_epochs, i+1, n_client, acc_i))
        client_list = client_list_

    torch.save(obj={'acc': acc},
               f='res/md_para.pt')
    print('res/md_para.pt saved.')

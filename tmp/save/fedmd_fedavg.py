
# from torch.utils.data import random_split
from utils import *
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

def init_():
    print('client init')
    client_list = model_init(
        num_client=5,
        model='mlp',
        num_target=10
    )
    setup_seed(20)
    return client_list


def data_obtain():
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_set, train_set_, public_set, test_set = get_dataset()
    client_set = non_iid(
        dataset=train_set,
        alpha=1,
        num_client_data=2000,
        num_client=5
    )
    client_dataloader = {}
    for (client, dataset) in client_set.items():
        client_dataloader[client] = DataLoader(
            dataset=dataset,
            batch_size=160,
            shuffle=True,
            pin_memory=True,
            sampler=DistributedSampler(dataset)
        )
    test_loader = DataLoader(
        dataset=test_set,
        batch_size=160,
        shuffle=True,
        pin_memory=True,
        num_workers=8
    )
    return client_dataloader, test_loader


def main(rank: int, client_list: list[torch.nn.Module], client_dataloader: DataLoader, test_loader: DataLoader, world_size: int):
    ddp_setup(rank, world_size)
    client_dataloader, test_loader = data_obtain()
    for epoch in range(100):
        client_list_ = []
        for i, client in enumerate(client_list):
            client_ = local_update(
                model=DDP(client, device_ids=[rank]),
                dataloader=client_dataloader[i],
                device=rank,
                epochs=5,
            )
            client_list_.append(client_)
            if rank == 0:
                acc = eval_model(
                    model=client_,
                    test_loader=test_loader
                )
                distll_acc[i].append(acc)
                print('revisit:', epoch, i, acc)
    client_list = aggregate(client_list_)
    destroy_process_group()


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    client_list = init_()
    distll_acc = {client: [] for client in range(len(client_list))}
    revist_acc = deepcopy(distll_acc)
    # device, client_dataloader, test_loader = data_obtain()
    mp.spawn(main, args=(client_list, world_size), nprocs=world_size)
    torch.save({'distll_acc': distll_acc,
                'revist_acc': revist_acc},
               'res/save_fedavg.pt')

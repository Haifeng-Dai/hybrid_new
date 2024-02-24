import time
import torch
import torch.nn as nn
import os

from copy import deepcopy
from torch.utils.data import DataLoader, ConcatDataset

from utils import *

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
log_path += f'{t.tm_hour}-{t.tm_min}-{t.tm_sec}.log'
log = get_logger(log_path)


# %% 1. basic parameters
args = get_args()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
setup_seed(args.seed)
server_client = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
client_server = [0] * 3 + [1] * 3 + [2] * 3

message = 'fedavg_pub' + f"\n\
    {'n_client':^17}:{args.n_client:^7}\n\
    {'seed':^17}:{args.seed:^7}\n\
    {'local_epochs':^17}:{args.local_epochs:^7}\n\
    {'batch_size':^17}:{args.batch_size:^7}\n\
    {'server_epochs':^17}:{args.server_epochs:^7}\n\
    {'alpha':^17}:{args.alpha:^7}\n\
    {'dataset':^17}:{args.dataset:^7}\n\
    {'model_structure':^17}:{args.model_structure:^7}"
log.info(message)


# %% 2. data preparation
train_set, test_set, n_targets, in_channel, public_data = dirichlet_split(
    dataset_name=args.dataset,
    alpha=args.alpha,
    n_public=args.n_public_data,
    n_clients=args.n_client,
    avg=False
)
train_loader = {}
for i, dataset_ in train_set.items():
    dataset_new = ConcatDataset([dataset_, public_data])
    train_loader[i] = DataLoader(dataset=dataset_,
                                 batch_size=args.batch_size,
                                 num_workers=8,
                                 shuffle=True)
test_loader = {}
for i, dataset_ in test_set.items():
    test_loader[i] = DataLoader(dataset=dataset_,
                                batch_size=1000,
                                pin_memory=True,
                                num_workers=8)

# %% 3. model initialization
client_list = model_init(num_client=args.n_client,
                         model_structure=args.model_structure,
                         num_target=n_targets,
                         in_channel=in_channel)
server_list = deepcopy(client_list[::3])


# %% 4. loss function initialization
CE_Loss = nn.CrossEntropyLoss().cuda()


# %% 5. model training and distillation
acc = {}
for cid, client in enumerate(client_list):
    acc[cid] = [eval_model(client, test_loader[client_server[cid]]),]
acc_server = {}
for sid, server in enumerate(server_list):
    acc_server[sid] = [eval_model(server, test_loader[sid]),]
lr_all = [1e-3, 5e-4, 1e-4, 5e-5, 1e-6]
msg_local = '[server epoch {}, client {}, local train]'
msg_test_local = 'local epoch {}, acc: {:.4f}'
msg_test_server = 'server epoch {}, acc {:.4f}'
for server_epoch in range(args.server_epochs):
    # local train
    client_list_ = []
    # lr = lr_all[server_epoch // (server_epochs // 5)]
    lr = 1e-4
    for cid, client in enumerate(client_list):
        log.info(msg_local.format(server_epoch + 1, cid + 1))
        client_ = client.cuda()
        optimizer = torch.optim.Adam(params=client_.parameters(), lr=lr)
        for local_epoch in range(args.local_epochs):
            for data_, target_ in train_loader[cid]:
                optimizer.zero_grad()
                output_ = client_(data_.cuda())
                loss = CE_Loss(output_, target_.cuda())
                loss.backward()
                optimizer.step()

            # test
            model__ = client_
            acc[cid].append(eval_model(model__, test_loader[client_server[cid]]))
            log.info(msg_test_local.format(local_epoch + 1, acc[cid][-1]))
        client_list_.append(deepcopy(client_))
    client_list = deepcopy(client_list_)

    client_list__ = []
    for sid, clients in enumerate(server_client):
        client_list_ = [client_list[i] for i in clients]
        agg_list = aggregate(client_list_)
        client_list__.extend(agg_list)
        server_list[sid] = deepcopy(agg_list[0])
        acc_server[sid].append(eval_model(server_list[sid], test_loader[sid]))
        log.info(msg_test_server.format(server_epoch + 1, acc_server[sid][-1]))
    client_list = deepcopy(client_list__)

save_path = f'./res/fedavg_pub_seed_{args.seed}_' + \
    f'alpha_{args.alpha}_' + \
    f'dataset_{args.dataset}_' + \
    f'model_structure_{args.model_structure}/'
file_name = save_path + \
    f'local_epochs_{args.local_epochs}_' + \
    f'server_epochs_{args.server_epochs}_' + \
    f'batch_size_{args.batch_size}.pt'
os.makedirs(save_path, exist_ok=True)
torch.save(obj={'acc': acc,
                'acc_server': acc_server},
           f=file_name)
log.info(f'results saved in {file_name}.')
t_end = time.time()
time_cost = time.strftime("%H:%M:%S", time.gmtime(t_end - t_start))
log.info(f'time cost: {time_cost}')

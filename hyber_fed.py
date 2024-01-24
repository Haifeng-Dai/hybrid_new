import time
import torch
import torch.nn as nn

from copy import deepcopy
from torch.distributions.dirichlet import Dirichlet
from torch.utils.data import DataLoader

import os
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
# n_client = args.n_client
# n_train_data = args.n_train_data
# n_public_data = args.n_public_data
# n_test_data = args.n_test_data
# seed = args.seed
# local_epochs = args.local_epochs
# distill_epochs = args.distill_epochs
# batch_size = args.batch_size
# server_epochs = args.server_epochs
# alpha = args.alpha
# dataset = args.dataset
# model_structure = args.model_structure

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)

setup_seed(args.seed)
all_client = range(args.n_client)
server_client = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]

message = 'fed_avg' + f"\n\
    {'n_client':^17}:{args.n_client:^7}\n\
    {'n_train_data':^17}:{args.n_train_data:^7}\n\
    {'n_test_data':^17}:{args.n_test_data:^7}\n\
    {'seed':^17}:{args.seed:^7}\n\
    {'local_epochs':^17}:{args.local_epochs:^7}\n\
    {'distill_epochs':^17}:{args.distill_epochs:^7}\n\
    {'batch_size':^17}:{args.batch_size:^7}\n\
    {'server_epochs':^17}:{args.server_epochs:^7}\n\
    {'alpha':^17}:{args.alpha:^7}\n\
    {'dataset':^17}:{args.dataset:^7}\n\
    {'model_structure':^17}:{args.model_structure:^7}"
log.info(message)


# %% 2. data preparation
train_set, test_set, n_targets, in_channel = dirichlet_split(dataset_name=args.dataset,
                                                             alpha=args.alpha,
                                                             n_clients=args.n_client,)
train_loader = {}
for i, dataset_ in train_set.items():
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


# %% 4. loss function initialization
CE_Loss = nn.CrossEntropyLoss().cuda()
KL_Loss = nn.KLDivLoss(reduction='batchmean').cuda()
Softmax = nn.Softmax(dim=-1).cuda()
LogSoftmax = nn.LogSoftmax(dim=-1).cuda()


# %% 5. model training and distillation
acc = {i: [] for i in all_client}
acc_server = deepcopy(acc)
for server_epoch in range(args.server_epochs):
    # local train
    msg_local = '[server epoch {}, client {}, local train]'
    msg_test_local = 'local epoch {}, acc: {:.4f}'
    client_list_ = []
    lr = 1e-3 / (server_epoch + 1)
    for i, client in enumerate(client_list):
        log.info(msg_local.format(server_epoch + 1, i + 1))
        client_ = client.cuda()
        optimizer = torch.optim.Adam(params=client_.parameters(),
                                     lr=lr,
                                     weight_decay=1e-4)
        for local_epoch in range(args.local_epochs):
            for data_, target_ in train_loader[i]:
                optimizer.zero_grad()
                output_ = client_(data_.cuda())
                loss = CE_Loss(output_, target_.cuda())
                loss.backward()
                optimizer.step()

            # test
            model__ = client_
            acc[i].append(eval_model(model__, test_loader[i]))
            log.info(msg_test_local.format(local_epoch + 1, acc[i][-1]))
        client_list_.append(deepcopy(client_))
    client_list = deepcopy(client_list_)

    client_list_ = []
    for clients in server_client:
        client_models = [client_list[i] for i in clients]
        client_list_.extend(aggregate(model_list=client_models))
    client_list = deepcopy(client_list_)
    server = deepcopy(client_list[::3])

    for i, model_ in enumerate(server):
        acc_server[i].append(eval_model(model_, test_loader[i//3]))

# %% 6. save
save_path = f'./res/fedavg_seed_{args.seed}_alpha_{args.alpha}_dataset_{args.dataset}_model_structure_{args.model_structure}/'
file_name = save_path + \
    f'n_client_{args.n_client}_' + \
    f'n_train_data_{args.n_train_data}_' + \
    f'n_test_data_{args.n_test_data}_' + \
    f'local_epochs_{args.local_epochs}_' + \
    f'server_epochs_{args.server_epochs}_' + \
    f'distill_epochs_{args.distill_epochs}_' + \
    f'batch_size_{args.batch_size}_' + \
    f'dataset_{args.dataset}.pt'
os.makedirs(save_path, exist_ok=True)
torch.save(obj={'acc': acc,
                'acc_server': acc_server,
                'server_model': client_list[0].state_dict()},
           f=file_name)
log.info(f'results saved in {file_name}.')
t_end = time.time()
log.info(f'time cost: {t_end - t_start}')

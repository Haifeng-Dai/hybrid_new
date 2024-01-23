import time
import torch
import torch.nn as nn

from copy import deepcopy
from torch.distributions.dirichlet import Dirichlet
from torch.utils.data import DataLoader

import os
from utils import *

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


'''1. basic parameters'''
args = get_args()
n_client = args.n_client
n_train_data = args.n_train_data
n_public_data = args.n_public_data
n_test_data = args.n_test_data
seed = args.seed
local_epochs = args.local_epochs
distill_epochs = args.distill_epochs
batch_size = args.batch_size
server_epochs = args.server_epochs
alpha = args.alpha
dataset = args.dataset
model_structure = args.model_structure

# n_client = 9
# n_train_data = 1000
# n_test_data = 200
# seed = 0
# local_epochs = 5
# distill_epochs = 5
# batch_size = 160
# server_epochs = 10
# alpha = 1.0
# dataset = 'mnist'
# model_structure = 'cnn1'

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)

setup_seed(seed)
all_client = range(n_client)
acc = {i: [] for i in all_client}
acc_server = []
server_client = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]

message = 'fed_avg' + f"\n\
    {'n_client':^17}:{n_client:^7}\n\
    {'n_train_data':^17}:{n_train_data:^7}\n\
    {'n_test_data':^17}:{n_test_data:^7}\n\
    {'seed':^17}:{seed:^7}\n\
    {'local_epochs':^17}:{local_epochs:^7}\n\
    {'distill_epochs':^17}:{distill_epochs:^7}\n\
    {'batch_size':^17}:{batch_size:^7}\n\
    {'server_epochs':^17}:{server_epochs:^7}\n\
    {'alpha':^17}:{alpha:^7}\n\
    {'dataset':^17}:{dataset:^7}\n\
    {'model_structure':^17}:{model_structure:^7}"
log.info(message)


'''2. data preparation'''
data_set = GetDataset(dataset_name=dataset,
                      n_public=0)
in_channel = data_set.in_channel
n_targets = data_set.n_targets
pro = Dirichlet(torch.full(size=(data_set.n_targets,),
                           fill_value=float(alpha))).sample([n_client])
train_set = split_non_iid(
    dataset=data_set.train_set,
    pro=pro,
    n_data=n_train_data,
    n_client=n_client,
)
test_set = split_non_iid(
    dataset=data_set.test_set,
    pro=pro,
    n_data=n_test_data,
    n_client=n_client,
)
test_loader = {}
for i, dataset_ in test_set.items():
    test_loader[i] = DataLoader(dataset=dataset_,
                                batch_size=10,
                                pin_memory=True,
                                num_workers=8)
if model_structure == 'mlp':
    client_list = model_init(num_client=n_client,
                             model_structure='mlp',
                             num_target=n_targets,
                             in_channel=in_channel)
elif model_structure == 'resnet18':
    client_list = model_init(num_client=n_client,
                             model_structure='resnet18',
                             num_target=n_targets,
                             in_channel=in_channel)
elif model_structure == 'cnn1':
    client_list = model_init(num_client=n_client,
                             model_structure='cnn1',
                             num_target=n_targets,
                             in_channel=in_channel)
elif model_structure == 'cnn2':
    client_list = model_init(num_client=n_client,
                             model_structure='cnn2',
                             num_target=n_targets,
                             in_channel=in_channel)
elif model_structure == 'lenet5':
    client_list = model_init(num_client=n_client,
                             model_structure='lenet5',
                             num_target=n_targets,
                             in_channel=in_channel)
else:
    raise ValueError(f'No such model: {model_structure}')


# dataloader must be defined after DDP initialization
train_loader = {}
for i, dataset_ in train_set.items():
    train_loader[i] = DataLoader(dataset=dataset_,
                                 batch_size=batch_size,
                                 num_workers=8,
                                 shuffle=True)


'''4. DDP: loss function initialization'''
CE_Loss = nn.CrossEntropyLoss().cuda()


'''5. model training and distillation'''
for server_epoch in range(server_epochs):
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
        for local_epoch in range(local_epochs):
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

    client_list = aggregate(model_list=client_list_)
    acc_ = 0
    for i in all_client:
        acc_ += eval_model(client_list[0], test_loader[i])
    acc_server.append(acc_ / n_client)

save_path = f'./res/fedavg_seed{seed}_alpha_{alpha}_dataset_{dataset}_model_structure_{model_structure}/'
file_name = save_path + \
    f'n_client_{n_client}_' + \
    f'n_train_data_{n_train_data}_' + \
    f'n_test_data_{n_test_data}_' + \
    f'local_epochs_{local_epochs}_' + \
    f'server_epochs_{server_epochs}_' + \
    f'distill_epochs_{distill_epochs}_' + \
    f'batch_size_{batch_size}_' + \
    f'dataset_{dataset}.pt'
os.makedirs(save_path, exist_ok=True)
torch.save(obj={'acc': acc,
                'acc_server': acc_server,
                'server_model': client_list[0].state_dict()},
           f=file_name)
log.info(f'results saved in {file_name}.')

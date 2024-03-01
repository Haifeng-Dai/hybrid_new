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
# acc = {i: [] for i in all_client}
# acc_server = {i: [] for i in all_client}

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
msg = '\n' + str(pro) + '\n' + str(pro * n_train_data)
log.info(msg)
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
acc = {}
for i in all_client:
    acc[i] = [eval_model(client_list[0], test_loader[i]),]
acc_server = deepcopy(acc)
lr_all = [1e-3, 5e-4, 1e-4, 5e-5, 1e-6]
msg_local = '[server epoch {}, client {}, local train]'
msg_test_local = 'local epoch {}, acc: {:.4f}'
msg_test_server = 'server epoch {}, acc {:.4f}'
for server_epoch in range(server_epochs):
    # local train
    client_list_ = []
    # lr = lr_all[server_epoch // (server_epochs // 5)]
    lr = 1e-4
    i = 0
    client = client_list[i]
    log.info(msg_local.format(server_epoch + 1, i + 1))
    client_ = client.cuda()
    optimizer = torch.optim.Adam(params=client_.parameters(), lr=lr)
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
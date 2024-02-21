import time
import torch
import torch.nn as nn
import os

from copy import deepcopy
from torch.utils.data import DataLoader

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

message = 'normal' + f"\n\
    {'n_client':^17}:{args.n_client:^7}\n\
    {'seed':^17}:{args.seed:^7}\n\
    {'local_epochs':^17}:{args.local_epochs:^7}\n\
    {'batch_size':^17}:{args.batch_size:^7}\n\
    {'alpha':^17}:{args.alpha:^7}\n\
    {'dataset':^17}:{args.dataset:^7}\n\
    {'model_structure':^17}:{args.model_structure:^7}"
log.info(message)


# %% 2. data preparation
server_train_set, server_test_set, n_targets, in_channel, public_data = dirichlet_split(
    dataset_name=args.dataset,
    alpha=args.alpha,
    n_clients=args.n_server,
    n_public=args.n_public_data,
    avg=False
)

train_set = {}
for i, data_set in server_train_set.items():
    splited_set = iid_split(data_set, 3, n_targets)
    for j, client in enumerate(server_client[i]):
        train_set[client] = splited_set[j]

train_loader = {}
for i, dataset_ in train_set.items():
    train_loader[i] = DataLoader(dataset=dataset_,
                                 batch_size=args.batch_size,
                                 num_workers=8,
                                 shuffle=True)
test_loader = {}
for i, dataset_ in server_test_set.items():
    test_loader[i] = DataLoader(dataset=dataset_,
                                batch_size=1000,
                                pin_memory=True,
                                num_workers=8)

# %% 3. model initialization
client_list = model_init(num_client=args.n_client,
                         model_structure=args.model_structure,
                         num_target=n_targets,
                         in_channel=in_channel)
server = deepcopy(client_list[::3])


# %% 4. loss function initialization
CE_Loss = nn.CrossEntropyLoss().cuda()


# %% 5. model training and distillation
acc = {}
for cid in range(args.n_client):
    acc[cid] = [eval_model(client_list[cid], test_loader[client_server[cid]]),]
msg = 'local epoch {}/{}, acc: {:.4f}'
lr = 1e-4
for cid, client in enumerate(client_list):
    log.info(f'client {cid} training starts.')
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
        acc_ = eval_model(client_, test_loader[client_server[cid]])
        acc[cid].append(acc_)
        log.info(msg.format(local_epoch + 1, str(args.local_epochs), acc_))

save_path = f'./res/normal_seed_{args.seed}_' + \
    f'alpha_{args.alpha}_' + \
    f'dataset_{args.dataset}_' + \
    f'model_structure_{args.model_structure}/'
file_name = save_path + \
    f'local_epochs_{args.local_epochs}_' + \
    f'batch_size_{args.batch_size}.pt'
os.makedirs(save_path, exist_ok=True)
torch.save(obj={'acc': acc},
           f=file_name)
log.info(f'results saved in {file_name}.')
t_end = time.time()
time_cost = time.strftime("%H:%M:%S", time.gmtime(t_end - t_start))
log.info(f'time cost: {time_cost}')

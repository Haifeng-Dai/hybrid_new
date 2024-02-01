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
log_path += f'{t.tm_hour}-{t.tm_min}-{t.tm_sec}.log'
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
server_client = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
client_server = [0] * 3 + [1] * 3 + [2] * 3

message = 'hyper_fed' + f"\n\
    {'n_public_data':^17}:{args.n_public_data:^7}\n\
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
public_loader = DataLoader(dataset=public_data,
                           batch_size=args.n_public_data,
                           pin_memory=True,
                           num_workers=8)


# %% 3. model initialization
client_list = model_init(num_client=args.n_client,
                         model_structure=args.model_structure,
                         num_target=n_targets,
                         in_channel=in_channel)
server = deepcopy(client_list[::3])
# print(type(server), type(server[0]))
# sys.exit()


# %% 4. loss function initialization
CE_Loss = nn.CrossEntropyLoss().cuda()
KL_Loss = nn.KLDivLoss(reduction='batchmean').cuda()
Softmax = nn.Softmax(dim=-1).cuda()
LogSoftmax = nn.LogSoftmax(dim=-1).cuda()


# %% 5. model training and distillation
acc = {}
for cid in range(args.n_client):
    acc[cid] = [eval_model(client_list[cid], test_loader[client_server[cid]]),]
for server_epoch in range(args.server_epochs):
    # local train
    msg_local = '[rank: {}, server epoch {}, client {}, local train]'
    msg_test_local = 'rank: {}, local epoch {}, acc: {:.4f}'
    client_list_ = []
    # lr = 1e-3 / (server_epoch + 1)
    lr = 1e-4
    for cid, client in enumerate(client_list):
        log.info(msg_local.format(rank, server_epoch + 1, cid + 1))
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
            acc[cid].append(eval_model(
                client_, test_loader[client_server[cid]]))
            log.info(msg_test_local.format(rank, local_epoch + 1, acc[cid][-1]))
        client_list_.append(deepcopy(client_))
    client_list = deepcopy(client_list_)

    client_list__ = []
    for sid, clients in enumerate(server_client):
        client_list_ = [client_list[i] for i in clients]
        agg_list = aggregate(client_list_)
        client_list__.extend(agg_list)
        server[sid] = deepcopy(agg_list[0])
    client_list = deepcopy(client_list__)

    # client_list_ = []
    # for clients in server_client:
    #     client_models = [client_list[i] for i in clients]
    #     client_list_.extend(
    #         aggregate(model_list=client_models[client_server[cid]]))
    # client_list = deepcopy(client_list_)
    # server = deepcopy(client_list[::3])
    # for sid, model_ in enumerate(server):
    #     acc_server[sid].append(eval_model(
    #         model_, test_loader[sid]))

    client_list_ = []
    msg_dist = '[rank: {}, server epoch {}, client {}, distill train]'
    msg_test_dist = 'rank: {}, distill epoch {}, acc: {:.4f}'
    for cid, client in enumerate(client_list):
        log.info(msg_dist.format(rank, server_epoch+1, cid+1))
        server_ = deepcopy(server)
        server_.pop(client_server[cid])
        client_ = client.cuda()
        optimizer = torch.optim.Adam(params=client_.parameters(), lr=lr)
        for distill_epoch in range(args.distill_epochs):
            for data_, target_ in public_loader:
                data_ = data_.cuda()
                logits = torch.zeros(size=[data_.shape[0], n_targets]).cuda()
                for model__ in server_:
                    logits += model__(data_).detach()
                logits /= (args.n_server - 1)
                optimizer.zero_grad()
                output_ = client_(data_)
                loss_ce = CE_Loss(output_, target_.cuda())
                loss_kl = KL_Loss(LogSoftmax(output_/args.temperature),
                                  Softmax(logits/args.temperature))
                loss = args.a * loss_kl + (1 - args.a) * loss_ce
                loss.backward()
                optimizer.step()

            # test
            model__ = client_
            acc[cid].append(eval_model(
                model__, test_loader[client_server[cid]]))
            log.info(msg_test_dist.format(rank, distill_epoch + 1, acc[i][-1]))
        client_list_.append(deepcopy(client_))
    client_list = deepcopy(client_list_)

    client_list__ = []
    for sid, clients in enumerate(server_client):
        client_list_ = [client_list[i] for i in clients]
        agg_list = aggregate(client_list_)
        client_list__.extend(agg_list)
        server[sid] = deepcopy(agg_list[0])
    client_list = deepcopy(client_list__)

    # client_list_ = []
    # for clients in server_client:
    #     client_models = [client_list[i] for i in clients]
    #     client_list_.extend(aggregate(model_list=client_models))
    # client_list = deepcopy(client_list_)
    # server = deepcopy(client_list[::3])
    # for sid, model_ in enumerate(server):
    #     acc_server[sid].append(eval_model(
    #         model_, test_loader[sid]))

for cid, client in enumerate(client_list):
    log.info(msg_local.format(rank, server_epoch + 1, cid + 1))
    client_ = client.cuda()
    optimizer = torch.optim.Adam(params=client_.parameters(), lr=lr)
    for local_epoch in range(10):
        for data_, target_ in train_loader[cid]:
            optimizer.zero_grad()
            output_ = client_(data_.cuda())
            loss = CE_Loss(output_, target_.cuda())
            loss.backward()
            optimizer.step()

        # test
        acc[cid].append(eval_model(
            client_, test_loader[client_server[cid]]))
        log.info(msg_test_local.format(local_epoch + 1, acc[cid][-1]))
    client_list_.append(deepcopy(client_))
client_list = deepcopy(client_list_)

# for sid, clients in enumerate(server_client):
#     client_list_ = [client_list[i] for i in clients]
#     server[sid] = aggregate(model_list=client_list_)
#     acc_server[sid].append(eval_model(server[sid], test_loader[sid]))

# %% 6. save
save_path = f'./res/hyper_fed_seed_{args.seed}_' + \
    f'alpha_{args.alpha}_' + \
    f'dataset_{args.dataset}_' + \
    f'model_structure_{args.model_structure}/'
file_name = save_path + \
    f'a_{args.a}_' + \
    f'T_{args.temperature}_' + \
    f'local_epochs_{args.local_epochs}_' + \
    f'server_epochs_{args.server_epochs}_' + \
    f'distill_epochs_{args.distill_epochs}_' + \
    f'batch_size_{args.batch_size}.pt'
os.makedirs(save_path, exist_ok=True)
torch.save(obj={'acc': acc,
                # 'acc_server': acc_server,
                'server_model': [server_.state_dict() for server_ in server]},
           f=file_name)
log.info(f'results saved in {file_name}.')
t_end = time.time()
time_cost = time.strftime("%H:%M:%S", time.gmtime(t_end - t_start))
log.info(f'time cost: {time_cost}')
comm.barrier()

from torch.utils.data import random_split
from utils import *

print('client init')
client_list = model_init(
    num_client=5,
    model='mlp',
    num_target=10
)
setup_seed(20)

device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
train_set, train_set_, public_set, test_set = get_dataset()
client_set = non_iid(
    dataset=train_set_,
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
        num_workers=8,
        pin_memory=True
    )
test_loader = DataLoader(dataset=test_set,
                         batch_size=160,
                         shuffle=True,
                         pin_memory=True,
                         num_workers=8)

distll_acc = {client: [] for client in range(len(client_list))}
revist_acc = deepcopy(distll_acc)

# [train_set, public_dataset, test_loader] = get_dataloader(
#     dataset_name='mnist', splite=True)
# client_datset = [train_set[0] + train_set[1],
#                  train_set[2] + train_set[3],
#                  train_set[4] + train_set[5],
#                  train_set[6] + train_set[7],
#                  train_set[8] + train_set[9]]
# client_dataloader = {}
# for i, dataset in enumerate(client_datset):
#     client_dataloader[i] = DataLoader(
#         dataset=dataset,
#         batch_size=160,
#         shuffle=True,
#         num_workers=8,
#         pin_memory=True
#     )

client_list_ = []
for i, client in enumerate(client_list):
    client_ = local_update(
        model=client,
        dataloader=client_dataloader[i],
        device=device,
        epochs=20,
    )
    client_list_.append(client_)
    acc = eval_model(
        model=client_,
        test_loader=test_loader,
        device=device
    )
    revist_acc[i].append(acc)
    print('client pre-train', i, acc)
client_list = client_list_

a = 0
for epoch in range(100):
    client_list_ = []
    for i, client in enumerate(client_list):
        client_ = distll_train(
            model=client,
            model_list=client_list,
            dataset=public_set,
            device=device,
            epochs=20,
            a=a,
        )
        acc = eval_model(
            model=client_,
            test_loader=test_loader,
            device=device
        )
        revist_acc[i].append(acc)
        print('distll', epoch, i, acc)
        client_list_.append(client_)
    client_list = client_list_

    client_list_ = []
    for i, client in enumerate(client_list):
        client_ = local_update(
            model=client,
            dataloader=client_dataloader[i],
            device=device,
            epochs=5,
        )
        client_list_.append(client_)
        acc = eval_model(
            model=client_,
            test_loader=test_loader,
            device=device
        )
        distll_acc[i].append(acc)
        print('revisit:', epoch, i, acc)
    client_list = client_list_

client_list_ = []
for i, client in enumerate(client_list):
    client_ = distll_train(
        model=client,
        model_list=client_list,
        dataset=public_set,
        device=device,
        epochs=20
    )
    acc = eval_model(
        model=client_,
        test_loader=test_loader,
        device=device
    )
    distll_acc[i].append(acc)
    print('distll', epoch, i, acc)
    client_list_.append(client_)
client_list = client_list_

torch.save({'distll_acc': distll_acc,
            'revist_acc': revist_acc},
            f'res/save_dist-a{a}.pt')
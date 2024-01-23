from torch.utils.data import random_split
from utils import *

print('client init')
client_list = model_init(
    num_client=5,
    model='mlp',
    num_target=10
)

device = torch.device('cuda')
print(device)
[train_loader, public_dataset, test_loader] = get_dataloader(
    dataset_name='mnist')
client_list_ = []
for i, client in enumerate(client_list):
    client_ = local_update(
        model=client,
        dataloader=train_loader,
        device=torch.device('cuda'),
        epochs=1,
    )
    client_list_.append(client_)
    acc = eval_model(
        model=client_,
        test_loader=test_loader,
        device=device
    )
    print('client pre-train', i, acc)
client_list = client_list_

for epoch in range(100):
    client_list_ = []
    for i, client in enumerate(client_list):
        client_ = distll_train(
            model=client,
            model_list=client_list,
            dataset=public_dataset,
            device=device,
            epochs=10
        )
        acc = eval_model(
            model=client_,
            test_loader=test_loader,
            device=device
        )
        print('distll', epoch, i, acc)
        client_list_.append(client_)
    client_list = client_list_

    client_list_ = []
    for client in client_list:
        client_ = local_update(
            model=client,
            dataloader=train_loader,
            device=torch.device('cuda'),
            epochs=5,
        )
        client_list_.append(client_)
        acc = eval_model(
            model=client_,
            test_loader=test_loader,
            device=device
        )
        print('revisit:', epoch, i, acc)
    client_list = client_list_

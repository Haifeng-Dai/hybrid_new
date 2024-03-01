import torch

import torch.nn as nn
from copy import deepcopy
from torch.utils.data import DataLoader, Dataset


Softmax = nn.Softmax(dim=-1)  # softmax函数，只有一个维度输出和为1
LogSoftmax = nn.LogSoftmax(dim=-1)  # log(softmax(.))

KL_Loss = nn.KLDivLoss(reduction='batchmean')  # KL散度，'batchmean'输出总和除以batch规模
CE_Loss = nn.CrossEntropyLoss()  # 交叉熵


def local_update(
        model: torch.nn.Module,
        dataloader: DataLoader,
        # device: torch.device = torch.device('cpu'),
        epochs: int = 5,
) -> torch.nn.Module:

    if torch.cuda.is_available():
        device = torch.cuda.current_device()
    # elif torch.backends.mps.is_available():
    #     device = 'mps'
    else:
        device = 'cpu'
    model_ = deepcopy(model).train().to(device)
    opti = torch.optim.Adam(params=model_.parameters(),
                            lr=1e-4,
                            weight_decay=1e-3)
    loss_func = CE_Loss.to(device)
    for epoch in range(epochs):
        for (data, target) in dataloader:
            opti.zero_grad()
            data_device = data.to(device)
            output = model_(data_device)
            loss = loss_func(output, target.to(device))
            loss.backward()
            opti.step()
    return model_


# def distll_train(
#         model: torch.nn.Module,
#         dataset_logits_target: list[torch.Tensor, torch.Tensor],
#         device: torch.device = torch.device('cpu'),
#         epochs: int = 5,
#         T: int = 6,
# ) -> torch.nn.Module:

#     model_ = deepcopy(model).train().to(device)
#     opti = torch.optim.Adam(params=model_.parameters(),
#                             lr=1e-4,
#                             weight_decay=1e-3)
#     # loss_func = DistillKL(alpha=0, T=1, device=device)
#     for epoch in range(epochs):
#         for data, logits, target in dataset_logits_target:
#             target_ = torch.tensor(target, device=device).unsqueeze(0)
#             opti.zero_grad()
#             data_device = data.to(device)
#             output = model_(data_device)
#             loss = KL_Loss(LogSoftmax(output), Softmax(logits).to(device))
#             # loss = loss_func(output, target_, logits.to(device))
#             loss.backward()
#             opti.step()
#     return model_


def distll_train(
        model: torch.nn.Module,
        model_list: list[torch.nn.Module],
        dataset: Dataset,
        # dataset_logits_target: list[torch.Tensor, torch.Tensor],
        # device: torch.device = torch.device('cpu'),
        epochs: int = 5,
        T: int = 6,
        a: int = 1,
) -> torch.nn.Module:

    if torch.cuda.is_available():
        device = torch.cuda.current_device()
    # elif torch.backends.mps.is_available():
    #     device = 'mps'
    else:
        device = 'cpu'
    dataloader = DataLoader(dataset=dataset,
                            batch_size=160,
                            shuffle=True,
                            pin_memory=True,
                            num_workers=8)
    model_ = deepcopy(model).train().to(device)
    model_list_ = deepcopy(model_list)
    opti = torch.optim.Adam(params=model_.parameters(),
                            lr=1e-4,
                            weight_decay=1e-3)
    # loss_func = DistillKL(alpha=0, T=1, device=device)
    # a = 1
    for epoch in range(epochs):
        for data, target in dataloader:
            batch_size = len(data)
            data = data.to(device)
            logits = torch.zeros([batch_size, 10], device=device)
            for model__ in model_list_:
                logits += model__(data).detach()
            logits /= batch_size
            opti.zero_grad()
            data_device = data.to(device)
            output = model_(data_device)
            loss_kl = KL_Loss(LogSoftmax(output/T), Softmax(logits/T).to(device))
            loss_ce = CE_Loss(output, target.to(device))
            loss = a * loss_kl + (1-a) *loss_ce
            loss.backward()
            opti.step()
    return model_


# def server_aggregate(
#         model_list: list[torch.nn.Module],
#         dataset: Dataset | list[torch.Tensor],
#         device: torch.device = torch.device('cpu'),
#         num_classes: int = 10,
# ) -> list[tuple[torch.Tensor, torch.Tensor]]:

#     model_list = [model.eval().to(device) for model in model_list]
#     # if isinstance(dataset, Dataset):
#     #     dataset = [data[0].to(device) for data in dataset]
#     # elif isinstance(dataset, list):
#     #     dataset = [data.to(device) for data in dataset]
#     num_data = len(dataset)
#     logits = []
#     for data, target in dataset:
#         logits_data = torch.zeros([1, num_classes], device=device)
#         for model in model_list:
#             logits_data += model(data.to(device))
#         logits_data /= num_data
#         logits_data_ = deepcopy(logits_data.detach().cpu())
#         logits.append((data.cpu(), logits_data_, target))
#         # print(logits_data_.argmax(dim=-1), target)
#         # a = input('input: ')
#         # if a:
#         #     break
#     return logits

def aggregate(model_list):
    num_model = len(model_list)
    model_ = deepcopy(model_list[0])
    parameters = deepcopy(model_list[0].state_dict())
    for key in parameters:
        if parameters[key].shape == torch.Size([]):
            continue
        parameters[key] /= num_model
    for model in model_list[1:]:
        for key in parameters:
            if parameters[key].shape == torch.Size([]):
                continue
            parameters[key] += model.state_dict()[key] / num_model
    model_.load_state_dict(parameters)
    models = [deepcopy(model_) for _ in range(num_model)]
    return models

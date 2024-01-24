import torch

from torch.utils.data import DataLoader, ConcatDataset
from utils import GetDataset

torch.set_printoptions(precision=3,
                       threshold=1000,
                       edgeitems=5,
                       linewidth=1000,
                       sci_mode=False)


'''2. data preparation'''
data_set = GetDataset(dataset_name='mnist',
                      n_public=0)

dataset_all = ConcatDataset([data_set.train_set, data_set.test_set])

print(len(dataset_all), type(dataset_all))
data_loader_all = DataLoader(dataset=dataset_all,
                             batch_size=160,
                             shuffle=True,
                             num_workers=8)
for data, target in data_loader_all:
    print(data.shape, target.shape)
    print(data[0], target[0])
    break

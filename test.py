import torch
import torchvision

from torch.utils.data import DataLoader, Dataset
from utils import GetDataset

dataset = torchvision.datasets.MNIST(root='./data',
                                     train=False,
                                     download=True)


class SubDataset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices
        self.targets = dataset[1]

    def __getitem__(self, idx):
        image, label = self.dataset[self.indices[idx]]
        return image, label

    def __len__(self):
        return len(self.indices)


sub_dataset = SubDataset(dataset, [1, 2, 3, 4, 5])

aa = sub_dataset.targets
print(aa)
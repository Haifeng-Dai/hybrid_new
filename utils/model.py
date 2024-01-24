import torch
import math

import torch.nn as nn
import torch.nn.functional as F


# def conv_cal(c, kernel_size, stride=None, padding=0, operation='conv'):
#     '''
#     卷积/池化操作后特征数量计算
#     '''
#     if stride == None:
#         if operation == 'conv':
#             stride = 1
#         else:
#             stride = kernel_size
#     l_return = (c - kernel_size + 2 * padding) / stride + 1
#     if operation == 'conv':
#         return math.ceil(l_return)
#     if operation == 'pool':
#         return math.floor(l_return)


# class LeNet5(torch.nn.Module):
#     '''
#     修改后的LeNet5模型
#     '''

#     def __init__(self, h, w, c, num_classes):
#         super(LeNet5, self).__init__()
#         self.h = h
#         self.w = w
#         self.conv1 = torch.nn.Sequential(
#             torch.nn.Conv2d(c, 6, 5),
#             torch.nn.ReLU(inplace=True),
#             torch.nn.AvgPool2d(kernel_size=2))
#         self.conv2 = torch.nn.Sequential(
#             torch.nn.Conv2d(6, 16, kernel_size=5),
#             torch.nn.ReLU(inplace=True),
#             torch.nn.AvgPool2d(kernel_size=2))
#         h_conv, w_conv = self.len_s()
#         self.full_con = torch.nn.Sequential(
#             torch.nn.Linear(16 * h_conv * w_conv, 120),
#             torch.nn.ReLU(inplace=True),
#             torch.nn.Linear(120, 84),
#             torch.nn.ReLU(inplace=True))
#         self.output = torch.nn.Linear(84, num_classes)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.conv2(x)
#         x = torch.flatten(x, 1)
#         x = self.full_con(x)
#         x = self.output(x)
#         return x

#     # 计算卷积/池化层后的特征数
#     def len_s(self):
#         h_conv = conv_cal(self.h, kernel_size=5)
#         h_conv = conv_cal(h_conv, kernel_size=2, operation='pool')
#         h_conv = conv_cal(h_conv, kernel_size=5)
#         h_conv = conv_cal(h_conv, kernel_size=2, operation='pool')
#         w_conv = conv_cal(self.w, kernel_size=5)
#         w_conv = conv_cal(w_conv, kernel_size=2, operation='pool')
#         w_conv = conv_cal(w_conv, kernel_size=5)
#         w_conv = conv_cal(w_conv, kernel_size=2, operation='pool')
#         return h_conv, w_conv


class LeNet5(nn.Module):

    def __init__(self, in_channel, n_class):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc_shape = 16 * 4 * 4
        self.fc1 = nn.Linear(self.fc_shape, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, n_class)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.fc_shape)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class MLP(nn.Module):

    def __init__(self, n_class):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(784, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, n_class)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class CNN1(nn.Module):

    def __init__(self, in_channel, n_class):
        super(CNN1, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        # reshape for both MNIST and CIFAR based on # of channels
        self.fc_shape = 16 * int(4.5 + in_channel * 0.5) * \
            int(4.5 + in_channel * 0.5)
        self.fc1 = nn.Linear(self.fc_shape, 64)
        self.fc2 = nn.Linear(64, n_class)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.fc_shape)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class CNN2(nn.Module):

    def __init__(self, in_channel, n_class):
        super(CNN2, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, 128, 3)
        self.conv2 = nn.Conv2d(128, 128, 3)
        self.fc_shape = 128 * int(4.5 + in_channel * 0.5) * \
            int(4.5 + in_channel * 0.5)
        self.fc = nn.Linear(self.fc_shape, n_class)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.fc_shape)
        x = self.fc(x)
        return x

class CNN3(nn.Module):
    def __init__(self, in_channel=1, n_class=10, dim=1024, dim1=512):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel,
                        32,
                        kernel_size=5,
                        padding=0,
                        stride=1,
                        bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32,
                        64,
                        kernel_size=5,
                        padding=0,
                        stride=1,
                        bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.fc1 = nn.Sequential(
            nn.Linear(dim, dim1),
            nn.ReLU(inplace=True)
        )
        self.fc = nn.Linear(dim1, n_class)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = torch.flatten(out, 1)
        out = self.fc1(out)
        out = self.fc(out)
        return out
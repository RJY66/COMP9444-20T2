# kuzu.py
# COMP9444, CSE, UNSW
# Python version 3.7.6
# torch version 1.4.0
# torchvision version 0.5.0

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F


class NetLin(nn.Module):
    # linear function followed by log softmax
    def __init__(self):
        super(NetLin, self).__init__()
        self.l = nn.Linear(in_features=28*28, out_features=10, bias=True)

    def forward(self, x):
        temp = x.view(x.shape[0], -1)
        result1 = F.log_softmax(self.l(temp), dim=1)
        return result1


class NetFull(nn.Module):
    # two fully connected tanh layers followed by log softmax
    # maybe take relative long time to run NetFull because of hidden node number
    # in order to reach as high accuracy as possible
    def __init__(self):
        super(NetFull, self).__init__()
        self.l1 = nn.Linear(in_features=28*28, out_features=200, bias=True)
        self.l2 = nn.Linear(in_features=200, out_features=10, bias=True)

    def forward(self, x):
        temp = x.view(x.shape[0], -1)
        temp2 = torch.tanh(self.l1(temp))
        result2 = F.log_softmax(self.l2(temp2), dim=1)
        return result2


class NetConv(nn.Module):
    # two convolutional layers and one fully connected layer,
    # all using relu, followed by log_softmax
    # maybe take relative long time to run the result
    def __init__(self):
        super(NetConv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, padding=2)
        self.l1 = nn.Linear(in_features=1152, out_features=650, bias=True)
        self.l2 = nn.Linear(in_features=650, out_features=10, bias=True)
        self.pool = nn.MaxPool2d(kernel_size=5, padding=2)

    def forward(self, x):
        temp = self.pool(F.relu(self.conv2(F.relu(self.conv1(x)))))
        temp2 = temp.view(temp.shape[0], -1)
        result3 = F.log_softmax(self.l2(F.relu(self.l1(temp2))), dim=1)
        return result3

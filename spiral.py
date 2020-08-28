# spiral.py
# COMP9444, CSE, UNSW
# Python version 3.7.6
# torch version 1.4.0
# torchvision version 0.5.0
# matplotlib version 3.1.3

import torch
import torch.nn as nn
import matplotlib.pyplot as plt


class PolarNet(torch.nn.Module):
    def __init__(self, num_hid):
        super(PolarNet, self).__init__()
        self.l1 = nn.Linear(in_features=2, out_features=num_hid, bias=True)
        self.l2 = nn.Linear(in_features=num_hid, out_features=1, bias=True)

    def forward(self, input):
        x = input[:, 0]
        y = input[:, 1]
        temp = x * x + y * y
        r = torch.sqrt(temp).reshape(-1, 1)
        a = torch.atan2(y, x).reshape(-1, 1)
        pos = torch.cat(tensors=(r, a), dim=1)
        self.hid1 = torch.tanh(self.l1(pos))
        output = torch.sigmoid(self.l2(self.hid1))
        return output


class RawNet(torch.nn.Module):
    def __init__(self, num_hid):
        super(RawNet, self).__init__()
        self.l1 = nn.Linear(in_features=2, out_features=num_hid, bias=True)
        self.l2 = nn.Linear(in_features=num_hid, out_features=num_hid, bias=True)
        self.l3 = nn.Linear(in_features=num_hid, out_features=1, bias=True)

    def forward(self, input):
        self.hid1 = torch.tanh(self.l1(input))
        self.hid2 = torch.tanh(self.l2(self.hid1))
        output = torch.sigmoid(self.l3(self.hid2))
        return output


class ShortNet(torch.nn.Module):
    def __init__(self, num_hid):
        super(ShortNet, self).__init__()
        # firstly, the same as RawNet as a part of ShortNet
        self.in_h1 = nn.Linear(in_features=2, out_features=num_hid, bias=True)
        self.h1_h2 = nn.Linear(in_features=num_hid, out_features=num_hid, bias=True)
        self.h2_out = nn.Linear(in_features=num_hid, out_features=1, bias=True)
        # then, input -> hid2 and hid2 -> output
        self.in_h2 = nn.Linear(in_features=2, out_features=num_hid, bias=True)
        # then, input -> output
        self.in_out = nn.Linear(in_features=2, out_features=1, bias=True)
        # lastly, input -> hid1 and hid1 -> output
        self.h1_out = nn.Linear(in_features=num_hid, out_features=1, bias=True)

    def forward(self, input):
        self.hid1 = torch.tanh(self.in_h1(input))
        self.hid2 = torch.tanh(self.in_h2(input) + self.h1_h2(self.hid1))
        output = torch.sigmoid(self.in_out(input) + self.h1_out(self.hid1) + self.h2_out(self.hid2))
        return output


def graph_hidden(net, layer, node):
    plt.clf()
    xrange = torch.arange(start=-7, end=7.1, step=0.01, dtype=torch.float32)
    yrange = torch.arange(start=-6.6, end=6.7, step=0.01, dtype=torch.float32)
    xcoord = xrange.repeat(yrange.size()[0])
    ycoord = torch.repeat_interleave(yrange, xrange.size()[0], dim=0)
    grid = torch.cat(tensors=(xcoord.unsqueeze(dim=1), ycoord.unsqueeze(dim=1)), dim=1)
    with torch.no_grad():
        net.eval()
        output = net(grid)
        # first hidden layer, all 3 nets have
        if 1 == layer:
            pred = (net.hid1[:, node] >= 0).float()
        # second hidden layer, only ShortNet and RawNet have
        else:
            pred = (net.hid2[:, node] >= 0).float()
        plt.pcolormesh(xrange, yrange, pred.cpu().view(yrange.size()[0], xrange.size()[0]), cmap='Wistia')

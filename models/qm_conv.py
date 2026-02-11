import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class QMConv1x1_S1(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, 1, 1, 0, groups=1, dilation=1, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act =  nn.ReLU()

    def forward(self, x):
        y = self.conv(x)
        y = self.bn(y)
        y = self.act(y)

        return y

class QMConv3x3_S1(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, 3, 1, 0, groups=1, dilation=1, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act =  nn.ReLU()

    def forward(self, x):
        x = F.pad(x, (1, 1, 1, 1))
        y = self.conv(x)
        y = self.bn(y)
        y = self.act(y)

        return y

class QMConv3x3_S2(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, 3, 2, 0, groups=1, dilation=1, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act =  nn.ReLU()

    def forward(self, x):
        x = F.pad(x, (0, 1, 0, 1))
        y = self.conv(x)
        #bn_out_nhwc = np.transpose(y, (0, 2, 3, 1))
        y = self.bn(y)
        #bn_out_nhwc2 = np.transpose(y, (0, 2, 3, 1))
        #print(
        #    "conv output1111:",
        #    "shape =", tuple(y.shape),
        #    "min =", y.min().item(),
        #    "max =", y.max().item(),
        #    "mean =", y.mean().item()
        #)

        y = self.act(y)
        #bn_out_nhwc2 = np.transpose(y, (0, 2, 3, 1))

        return y
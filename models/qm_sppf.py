import torch
import torch.nn as nn
from models.qm_conv import QMConv

class QMSPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1: int, c2: int, k: int = 5):
        """Initialize the SPPF layer with given input/output channels and kernel size.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            k (int): Kernel size.

        Notes:
            This module is equivalent to SPP(k=(5, 9, 13)).
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = QMConv(c1, c_, 1, 1, 0)
        self.cv2 = QMConv(c_ * 4, c2, 1, 1, 0)
        #self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.conv1 = nn.Conv2d(c_, c_, 3, 1, 1, bias=False)
        self.conv2 = nn.Conv2d(c_, c_, 3, 1, 1, bias=False)
        self.conv21 = nn.Conv2d(c_, c_, 3, 1, 1, bias=False)

        self.conv3 = nn.Conv2d(c_, c_, 3, 1, 1, bias=False)
        self.conv31 = nn.Conv2d(c_, c_, 3, 1, 1, bias=False)

        self.conv4 = nn.Conv2d(c_, c_, 3, 1, 1, bias=False)
        self.conv41 = nn.Conv2d(c_, c_, 3, 1, 1, bias=False)

        #self.pool1 = nn.MaxPool2d(2, stride=1, padding=1)
        #self.pool2 = nn.MaxPool2d(2, stride=1, padding=1)
        #self.pool3 = nn.MaxPool2d(2, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply sequential pooling operations to input and return concatenated feature maps."""
        #y = [self.cv1(x)]
        #y.extend(self.m(y[-1]) for _ in range(3))
        #return self.cv2(torch.cat(y, 1))
        #h, w = x.shape[-2:]

        y0 = self.cv1(x)

        y1 = self.conv1(y0)
        y1 = self.conv2(y1)
        y1 = self.conv21(y1)
        #y1 = self.pool1(y1)
        #y1 = self._pool_crop(y1, h, w)

        y2 = self.conv3(y1)
        y2 = self.conv31(y2)
        #y2 = self.pool2(y2)
        #y2 = self._pool_crop(y2, h, w)

        y3 = self.conv4(y2)
        y3 = self.conv41(y3)
        #y3 = self.pool3(y3)
        #y3 = self._pool_crop(y3, h, w)

        y = [y0, y1, y2, y3]
        return self.cv2(torch.cat(y, 1))
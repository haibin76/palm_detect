import torch
import torch.nn as nn
import torch.nn.functional as F

from models.qm_conv import QMConv
from models.qm_sppf import QMSPPF

class QMDetect(nn.Module):
    #dynamic = False  # force grid reconstruction
    #export = False  # export mode
    #format = None  # export format
    #end2end = False  # end2end
    #max_det = 300  # max_det
    #shape = None
    #anchors = torch.empty(0)  # init
    #strides = torch.empty(0)  # init
    #legacy = False  # backward compatibility for v3/v5/v8/v9 models
    #xyxy = False  # xyxy or xywh output

    def __init__(self, nc: int = 80, ch: tuple = ()):
        #nc = 1, ch = [128]
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))  # channels
        self.cv2 = nn.Sequential(QMConv(ch[0], c2, 3), QMConv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1))
        self.cv3 = nn.Sequential(QMConv(ch[0], c3, 3), QMConv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1))

    def forward(self, x):
        boxs = self.cv2(x)
        class_id = self.cv3(x)

        return boxs, class_id

class QMPose(QMDetect):
    def __init__(self, nc: int = 80, kpt_shape: tuple = (4, 3), ch: tuple = ()):
        # nc = 1, ch = [4, 3], ch = 128
        super().__init__(nc, ch)
        self.kpt_shape = kpt_shape  # number of keypoints, number of dims (2 for x,y or 3 for x,y,visible)
        self.nk = kpt_shape[0] * kpt_shape[1]  # number of keypoints total

        c4 = max(ch[0] // 4, self.nk)
        #self.cv4 = nn.ModuleList(nn.Sequential(QMConv(x, c4, 3), QMConv(c4, c4, 3), nn.Conv2d(c4, self.nk, 1)) for x in ch)
        self.cv4 = nn.Sequential(QMConv(ch[0], c4, 3), QMConv(c4, c4, 3), nn.Conv2d(c4, self.nk, 1))

    def forward(self, x):
        """Perform forward pass through YOLO model and return predictions."""
        bs = x.shape[0]  # batch size
        keypoints = self.cv4(x)
        boxs, class_id = QMDetect.forward(self, x)

        return class_id, boxs, keypoints
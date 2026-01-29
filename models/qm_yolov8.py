import torch
import torch.nn as nn
import torch.nn.functional as F

from models.qm_conv import QMConv
from models.qm_sppf import QMSPPF
from models.qm_pose import QMPose

'''
  n: [0.33, 0.25, 1024]
  s: [0.33, 0.50, 1024]
  m: [0.67, 0.75, 768]
  l: [1.00, 1.00, 512]
  x: [1.00, 1.25, 512]
  
backbone:
  # [from, repeats, module, args]
  - [-1, 1, Conv, [64, 3, 2]] # 0-P1/2
  - [-1, 1, Conv, [128, 3, 2]] # 1-P2/4
  - [-1, 1, Conv, [128, 3, 1]] # 1-P2/4 #  - [-1, 3, C2f, [128, True]]
  - [-1, 1, Conv, [256, 3, 2]] # 3-P3/8

  - [-1, 1, Conv, [256, 3, 1]] # [-1, 6, C2f, [256, True]]
  - [-1, 1, Conv, [512, 3, 1]]

  - [-1, 1, Conv, [512, 3, 2]] # 5-P4/16

  - [-1, 1, Conv, [512, 3, 1]] # [-1, 6, C2f, [256, True]]↵
  - [-1, 1, Conv, [512, 3, 1]]

#- [-1, 6, C2f, [512, True]]
  - [-1, 1, Conv, [512, 3, 2]] # 7-P5/32
  - [-1, 1, Conv, [512, 3, 1]] #- [-1, 3, C2f, [512, True]]
  - [-1, 1, SPPF, [512, 2]] # 9

# YOLOv8.0n head
head:
#  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
#  - [[-1, 6], 1, Concat, [1]] # cat backbone P4
#  - [-1, 3, C2f, [512]] # 12

#  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
#  - [[-1, 4], 1, Concat, [1]] # cat backbone P3
#  - [-1, 3, C2f, [256]] # 15 (P3/8-small)

#  - [-1, 1, Conv, [256, 3, 2]]
#  - [[-1, 12], 1, Concat, [1]] # cat head P4
#  - [-1, 3, C2f, [512]] # 18 (P4/16-medium)

#  - [-1, 1, Conv, [512, 3, 2]]
#  - [9, 1, Concat, [1]] # cat head P5
#   - [-1, 3, C2f, [1024]] # 21 (P5/32-large)

- [[10], 1, Pose, [nc, kpt_shape]] # Pose(P3, P4, P5)
'''

class QMYoloV8(nn.Module):
    def __init__(self, k=1, s=1, p=1, g=1, d=1, act=True):
        super().__init__()

        #- [-1, 1, Conv, [64, 3, 2]]  # 0-P1/2
        self.qmcm0 = QMConv(3, 16, 3, 2, 1, 1, 1)

        #- [-1, 1, Conv, [128, 3, 2]]  # 1-P2/4
        self.qmcm1 = QMConv(16, 32, 3, 2, 1, 1, 1)

        #- [-1, 1, Conv, [128, 3, 1]]  # 1-P2/4 #  - [-1, 3, C2f, [128, True]]
        self.qmcm2 = QMConv(32, 32, 3, 1, 1, 1, 1)

        #- [-1, 1, Conv, [256, 3, 2]]  # 3-P3/8
        self.qmcm3 = QMConv(32, 64, 3, 2, 1, 1, 1)

        #- [-1, 1, Conv, [256, 3, 1]]  # [-1, 6, C2f, [256, True]]
        self.qmcm4 = QMConv(64, 64, 3, 1, 1, 1, 1)

        #- [-1, 1, Conv, [512, 3, 1]]
        self.qmcm5 = QMConv(64, 128, 3, 1, 1, 1, 1)

        #- [-1, 1, Conv, [512, 3, 2]]  # 5-P4/16
        self.qmcm6 = QMConv(128, 128, 3, 2, 1, 1, 1)

        #- [-1, 1, Conv, [512, 3, 1]]  # [-1, 6, C2f, [256, True]]↵
        self.qmcm7 = QMConv(128, 128, 3, 1, 1, 1, 1)

        #- [-1, 1, Conv, [512, 3, 1]]
        self.qmcm8 = QMConv(128, 128, 3, 1, 1, 1, 1)

        #- [-1, 1, Conv, [512, 3, 2]]  # 7-P5/32
        self.qmcm9 = QMConv(128, 128, 3, 2, 1, 1, 1)

        #- [-1, 1, Conv, [512, 3, 1]]  # - [-1, 3, C2f, [512, True]]
        self.qmcm10 = QMConv(128, 128, 3, 1, 1, 1, 1)

        #- [-1, 1, SPPF, [512, 2]]  # 9
        self.qmsppf = QMSPPF(128, 128)

        #- [[11], 1, Pose, [nc, kpt_shape]]
        self.qmpose = QMPose(nc = 1, kpt_shape = (4, 3), ch=(128,))

    def forward(self, x):
        y0 = self.qmcm0(x)
        y1 = self.qmcm1(y0)
        y2 = self.qmcm2(y1)
        y3 = self.qmcm3(y2)
        y4 = self.qmcm4(y3)
        y5 = self.qmcm5(y4)
        y6 = self.qmcm6(y5)
        y7 = self.qmcm7(y6)
        y8 = self.qmcm8(y7)
        y9 = self.qmcm9(y8)
        y10 = self.qmcm10(y9)

        y11 = self.qmsppf(y10)

        class_id, boxs, keypoints = self.qmpose(y11)
        return class_id, boxs, keypoints
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.qm_conv import QMConv3x3_S2
from models.qm_conv import QMConv3x3_S1
from models.qm_sppf import QMSPPF
from models.qm_pose import QMPose
from gen_pb.backbone_ztv2 import *

class QMYoloV8(nn.Module):
    def __init__(self, k=1, s=1, p=1, g=1, d=1, act=True):
        super().__init__()

        #- [-1, 1, Conv, [64, 3, 2]]  # 0-P1/2
        self.qmcm0 = QMConv3x3_S2(3, 16)

        #- [-1, 1, Conv, [128, 3, 2]]  # 1-P2/4
        self.qmcm1 = QMConv3x3_S2(16, 32)

        #- [-1, 1, Conv, [128, 3, 1]]  # 1-P2/4 #  - [-1, 3, C2f, [128, True]]
        self.qmcm2 = QMConv3x3_S1(32, 32)

        #- [-1, 1, Conv, [256, 3, 2]]  # 3-P3/8
        self.qmcm3 = QMConv3x3_S2(32, 64)

        #- [-1, 1, Conv, [256, 3, 1]]  # [-1, 6, C2f, [256, True]]
        self.qmcm4 = QMConv3x3_S1(64, 64)

        #- [-1, 1, Conv, [512, 3, 1]]
        self.qmcm5 = QMConv3x3_S1(64, 128)

        #- [-1, 1, Conv, [512, 3, 2]]  # 5-P4/16
        self.qmcm6 = QMConv3x3_S2(128, 128)

        #- [-1, 1, Conv, [512, 3, 1]]  # [-1, 6, C2f, [256, True]]↵
        self.qmcm7 = QMConv3x3_S1(128, 128)

        #- [-1, 1, Conv, [512, 3, 1]]
        self.qmcm8 = QMConv3x3_S1(128, 128)

        #- [-1, 1, Conv, [512, 3, 2]]  # 7-P5/32
        self.qmcm9 = QMConv3x3_S2(128, 128)

        #- [-1, 1, Conv, [512, 3, 1]]  # - [-1, 3, C2f, [512, True]]
        self.qmcm10 = QMConv3x3_S1(128, 128)

        #- [-1, 1, SPPF, [512, 2]]  # 9
        self.qmsppf = QMSPPF(128, 128)

        #- [[11], 1, Pose, [nc, kpt_shape]]
        self.qmpose = QMPose(nc = 1, kpt_shape = (4, 3), ch=(128,))

    def forward(self, x):
        y0 = self.qmcm0(x)
        #print_layer_info("qmcm0", self.qmcm0, x, y0)
        y1 = self.qmcm1(y0)
        #print_layer_info("qmcm1", self.qmcm1, y0, y1)
        y2 = self.qmcm2(y1)
        #print_layer_info("qmcm2", self.qmcm2, y1, y2)
        y3 = self.qmcm3(y2)
        #print_layer_info("qmcm3", self.qmcm3, y2, y3)
        y4 = self.qmcm4(y3)
        #print_layer_info("qmcm4", self.qmcm4, y3, y4)
        y5 = self.qmcm5(y4)
        #print_layer_info("qmcm5", self.qmcm5, y4, y5)
        y6 = self.qmcm6(y5)
        #print_layer_info("qmcm6", self.qmcm6, y5, y6)
        y7 = self.qmcm7(y6)
        #print_layer_info("qmcm7", self.qmcm7, y6, y7)
        y8 = self.qmcm8(y7)
        #print_layer_info("qmcm8", self.qmcm8, y7, y8)
        y9 = self.qmcm9(y8)
        #print_layer_info("qmcm9", self.qmcm9, y8, y9)
        y10 = self.qmcm10(y9)
        #print_layer_info("qmcm10", self.qmcm10, y9, y10)

        y11 = self.qmsppf(y10)

        class_id, boxs, keypoints = self.qmpose(y11)
        return class_id, boxs, keypoints
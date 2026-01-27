import torch
import torch.nn as nn
import torch.nn.functional as F

class DFL(nn.Module):
    """
    Distribution Focal Loss decoder
    [B, 64, H, W] -> [B, 4, H, W]
    """
    def __init__(self, reg_max=16):
        super().__init__()
        self.reg_max = reg_max
        self.register_buffer(
            "project", torch.arange(reg_max, dtype=torch.float32)
        )

    def forward(self, x):
        """
        x: [B, 4*reg_max, H, W]
        return: [B, 4, H, W]  (ltrb, feature-space)
        """
        b, c, h, w = x.shape
        x = x.view(b, 4, self.reg_max, h, w)
        x = x.softmax(dim=2)

        # expectation
        x = (x * self.project.view(1, 1, -1, 1, 1)).sum(dim=2)
        return x

def boxs_loss(pred_boxes_64, gt_boxes, dfl_layer, stride=32):
    device = pred_boxes_64.device
    B, _, H, W = pred_boxes_64.shape

    # 1. 得到解码后的 ltrb (特征图尺度)
    pred_ltrb = dfl_layer(pred_boxes_64)  # [B, 4, H, W]

    target_ltrb = torch.zeros((B, 4, H, W), device=device)
    fg_mask = torch.zeros((B, H, W), dtype=torch.bool, device=device)

    for b in range(B):
        cx, cy, bw, bh = gt_boxes[b]  # 归一化坐标
        if (gt_boxes[b] <= 0).all():  # 如果四个值都 <= 0
            continue

        # 计算目标在特征图上的中心
        f_cx, f_cy = cx * W, cy * H
        f_bw, f_bh = bw * W, bh * H

        gi, gj = int(f_cx), int(f_cy)

        # 3x3 分配
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                ni, nj = gi + dx, gj + dy
                if 0 <= ni < W and 0 <= nj < H:
                    fg_mask[b, nj, ni] = True

                    # 关键修改：相对于当前格子 (ni, nj) 的中心计算 ltrb
                    # 格子中心通常定义为 (ni + 0.5)
                    cell_x, cell_y = ni + 0.5, nj + 0.5

                    l = cell_x - (f_cx - f_bw / 2)
                    t = cell_y - (f_cy - f_bh / 2)
                    r = (f_cx + f_bw / 2) - cell_x
                    b_dist = (f_cy + f_bh / 2) - cell_y

                    target_ltrb[b, :, nj, ni] = torch.tensor([l, t, r, b_dist], device=device).clamp(min=0)

    if fg_mask.any():
        p_ltrb = pred_ltrb.permute(0, 2, 3, 1)[fg_mask]
        t_ltrb = target_ltrb.permute(0, 2, 3, 1)[fg_mask]

        # 建议：CIoU Loss (这里简化为 L1，但逻辑修正了)
        loss_box = F.smooth_l1_loss(p_ltrb, t_ltrb, reduction="mean")

        # 可选：如果想更强，可以加上 DFL 原始分布 Loss (CrossEntropy)
        # 这里为了简单保持 L1，但物理逻辑已正确
        return loss_box

    return torch.zeros((), device=device)


def calculate_iou(box1, box2):
    """计算两个 xyxy 格式框的 IoU"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter + 1e-7
    return inter / union

def evaluate_boxs(pred_cls, pred_boxs_64, gt_boxes, dfl_layer, stride=32, img_size=640, iou_thresh=0.5, conf_thresh=0.5):
    """
    与训练逻辑严格对齐的 BBox 评估函数

    Args:
        pred_cls: [B, 1, H, W] 分类 Logits
        pred_boxs_64: [B, 64, H, W] DFL 输出
        gt_boxes: [B, 4] (cx, cy, w, h) 归一化
        dfl_layer: DFL 解码层对象
    """
    device = pred_boxs_64.device
    B, _, H, W = pred_boxs_64.shape

    # 1. 预先解码 DFL 得到 ltrb (特征图尺度)
    # 注意：这里直接用训练时传进来的 dfl_layer
    pred_ltrb = dfl_layer(pred_boxs_64)  # [B, 4, H, W]

    # 2. 计算分类置信度
    cls_prob = torch.sigmoid(pred_cls)

    TP, FP, FN = 0, 0, 0

    for b in range(B):
        cx, cy, bw, bh = gt_boxes[b]

        # --- 负样本判断 ---
        is_negative = (cx <= 0 and cy <= 0 and bw <= 0 and bh <= 0)

        # --- 预测部分：寻找该 Batch 样本中最自信的预测位 ---
        max_score, max_idx = torch.max(cls_prob[b].view(-1), dim=0)

        if max_score > conf_thresh:
            # 预测存在目标
            pj = int(max_idx // W)
            pi = int(max_idx % W)

            # 解码该位置的预测框
            pl, pt, pr, pb = pred_ltrb[b, :, pj, pi]

            # 物理坐标还原 (与训练逻辑 cell_x = ni + 0.5 互逆)
            p_cell_cx = (pi + 0.5) * stride
            p_cell_cy = (pj + 0.5) * stride

            px1 = p_cell_cx - pl * stride
            py1 = p_cell_cy - pt * stride
            px2 = p_cell_cx + pr * stride
            py2 = p_cell_cy + pb * stride
            pred_xyxy = [px1.item(), py1.item(), px2.item(), py2.item()]

            if not is_negative:
                # 样本有目标，计算 IoU
                gt_x1 = (cx - bw / 2) * img_size
                gt_y1 = (cy - bh / 2) * img_size
                gt_x2 = (cx + bw / 2) * img_size
                gt_y2 = (cy + bh / 2) * img_size
                gt_xyxy = [gt_x1.item(), gt_y1.item(), gt_x2.item(), gt_y2.item()]

                iou = calculate_iou(pred_xyxy, gt_xyxy)
                if iou >= iou_thresh:
                    TP += 1
                else:
                    FP += 1  # 预测太偏或 IoU 不够，记为 FP
            else:
                # 样本没目标但预测了，记为 FP
                FP += 1
        else:
            # 预测没目标
            if not is_negative:
                # 样本有目标但没预测出来，记为 FN
                FN += 1
            else:
                # 样本没目标且没预测，Pass
                pass

    return TP, FP, FN
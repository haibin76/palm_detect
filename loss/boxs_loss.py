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

def boxs_loss(pred_boxes_64, gt_boxes, dfl_layer, stride=32, img_size=640):
    device = pred_boxes_64.device
    B, _, H, W = pred_boxes_64.shape

    # 1. DFL decode
    pred_ltrb = dfl_layer(pred_boxes_64)  # [B, 4, H, W] (feature-space)

    target_ltrb = torch.zeros((B, 4, H, W), device=device)
    fg_mask = torch.zeros((B, H, W), dtype=torch.bool, device=device)

    for b in range(B):
        cx, cy, w, h = gt_boxes[b]

        # ---- image space ----
        cx_img = cx * img_size
        cy_img = cy * img_size
        bw_img = w * img_size
        bh_img = h * img_size

        # ---- feature index ----
        gi = int(cx_img / stride)
        gj = int(cy_img / stride)
        gw = int(bw_img / stride)
        gh = int(bh_img / stride)
        if gi < 0 or gi >= W or gj < 0 or gj >= H or gw <= 0 or gh <= 0 or gw >= W or gh >= H:
            continue

        fg_mask[b, gj, gi] = True

        # ---- cell center (image space) ----
        cell_cx = (gi + 0.5) * stride
        cell_cy = (gj + 0.5) * stride

        # ---- ltrb (image space) ----
        l = cell_cx - (cx_img - bw_img / 2)
        t = cell_cy - (cy_img - bh_img / 2)
        r = (cx_img + bw_img / 2) - cell_cx
        b_dist = (cy_img + bh_img / 2) - cell_cy

        # clamp 防止负数
        l = torch.clamp(l, min=0.0)
        t = torch.clamp(t, min=0.0)
        r = torch.clamp(r, min=0.0)
        b_dist = torch.clamp(b_dist, min=0.0)

        # ---- 转为 feature-space ----
        target_ltrb[b, :, gj, gi] = torch.tensor([l, t, r, b_dist], device=device) / stride
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                ni, nj = gi + dx, gj + dy
                if 0 <= ni < W and 0 <= nj < H:
                    fg_mask[b, nj, ni] = True
                    target_ltrb[b, :, nj, ni] = target_ltrb[b, :, gj, gi]

    # 3. loss
    if fg_mask.any():
        pred = pred_ltrb.permute(0, 2, 3, 1)[fg_mask]
        target = target_ltrb.permute(0, 2, 3, 1)[fg_mask]
        return F.l1_loss(pred, target, reduction="mean")

    return torch.zeros((), device=device)

def calculate_iou(box1, box2):
    """
    box: [x1, y1, x2, y2] (image space)
    """
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    inter_w = max(0.0, xB - xA)
    inter_h = max(0.0, yB - yA)
    inter_area = inter_w * inter_h

    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])

    union = area1 + area2 - inter_area
    return inter_area / (union + 1e-6)

def evaluate_boxs(pred_boxs_64, gt_boxes, stride=32, img_size=640, iou_thresh=0.5):
    """
    简化版 bbox 回归评估（与当前训练方式严格对齐）

    Args:
        pred_boxs_64: [B, 64, H, W]  box head 原始输出
        gt_boxes: [B, 4] (cx, cy, w, h) 归一化
        stride: 特征图 stride
        img_size: 输入图像尺寸
        iou_thresh: IoU 阈值

    Returns:
        TP, FP, FN
        （注意：这是 regression 级指标，不是 detection precision/recall）
    """
    device = pred_boxs_64.device
    B, _, H, W = pred_boxs_64.shape

    # ---------- DFL 解码 ----------
    reg_max = 16
    project = torch.arange(reg_max, device=device, dtype=torch.float32)

    x = pred_boxs_64.view(B, 4, reg_max, H, W).softmax(dim=2)
    pred_ltrb = (x * project.view(1, 1, -1, 1, 1)).sum(dim=2)  # [B, 4, H, W]

    TP, FP, FN = 0, 0, 0
    for b in range(B):
        # ---------- GT box (image space) ----------
        cx, cy, w, h = gt_boxes[b]
        #负样本
        if cx <= 0 and cy <= 0 and w <= 0 and h <= 0:
            continue

        gt_x1 = (cx - w / 2) * img_size
        gt_y1 = (cy - h / 2) * img_size
        gt_x2 = (cx + w / 2) * img_size
        gt_y2 = (cy + h / 2) * img_size

        gt_xyxy = [
            gt_x1.item(),
            gt_y1.item(),
            gt_x2.item(),
            gt_y2.item(),
        ]

        # ---------- 使用 GT 中心点对应的 cell ----------
        cx_img = cx * img_size
        cy_img = cy * img_size

        gi = int(cx_img / stride)
        gj = int(cy_img / stride)

        gi = max(0, min(W - 1, gi))
        gj = max(0, min(H - 1, gj))

        # ---------- 解码预测框 ----------
        l, t, r, b_dist = pred_ltrb[b, :, gj, gi]

        # clamp，防止 early training 崩框
        l = torch.clamp(l, min=0.0)
        t = torch.clamp(t, min=0.0)
        r = torch.clamp(r, min=0.0)
        b_dist = torch.clamp(b_dist, min=0.0)

        cell_cx = (gi + 0.5) * stride
        cell_cy = (gj + 0.5) * stride

        px1 = cell_cx - l * stride
        py1 = cell_cy - t * stride
        px2 = cell_cx + r * stride
        py2 = cell_cy + b_dist * stride

        # 非法框直接算失败
        if px2 <= px1 or py2 <= py1:
            FN += 1
            continue

        pred_xyxy = [
            px1.item(),
            py1.item(),
            px2.item(),
            py2.item(),
        ]

        # ---------- IoU ----------
        iou = calculate_iou(pred_xyxy, gt_xyxy)

        if iou >= iou_thresh:
            TP += 1
        else:
            FN += 1

    return TP, FP, FN
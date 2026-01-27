import torch
import torch.nn.functional as F

@torch.no_grad()
def evaluate_cls(pred_cls, gt_boxes, stride=32, conf_thresh=0.5):
    device = pred_cls.device
    B, _, H, W = pred_cls.shape
    pred_prob = torch.sigmoid(pred_cls)
    TP, FP, FN = 0, 0, 0

    for b in range(B):
        cx, cy, bw, bh = gt_boxes[b]

        # 1. 判定该样本是否为有效正样本
        # 逻辑与 cls_loss 保持一致
        is_positive = (cx > 0 and cy > 0 and bw > 0 and bh > 0)

        # 2. 找到模型最自信的预测点 (与推理 decode 逻辑对齐)
        max_score, max_idx = torch.max(pred_prob[b].view(-1), dim=0)

        # 3. 判定模型是否预测了目标
        pred_exists = (max_score > conf_thresh)

        if is_positive:
            if pred_exists:
                # 进一步判断预测的位置是否在 GT 中心 3x3 范围内 (容错性)
                pgj, pgi = int(max_idx // W), int(max_idx % W)
                gt_gi, gt_gj = int(cx * W), int(cy * H)

                if abs(pgi - gt_gi) <= 1 and abs(pgj - gt_gj) <= 1:
                    TP += 1
                else:
                    # 预测了位置但离 GT 太远
                    FP += 1
            else:
                # 有目标但没预测出来
                FN += 1
        else:
            # 负样本图片
            if pred_exists:
                # 没目标但预测了有，计为 FP
                FP += 1
            else:
                # 没目标也没预测，正确（不需要统计 TN）
                pass

    return TP, FP, FN


def cls_loss(pred_cls, gt_boxes, stride=32):
    device = pred_cls.device
    B, _, H, W = pred_cls.shape
    target = torch.zeros_like(pred_cls)

    # 统一使用特征图尺度进行计算
    for b in range(B):
        cx, cy, bw, bh = gt_boxes[b]

        # 1. 统一负样本过滤逻辑
        if cx <= 0 or cy <= 0 or bw <= 0 or bh <= 0:
            continue

        # 2. 过滤过小的目标（可选，但建议保留以防止单尺度训练崩坏）
        # 如果手部在图像中占据像素小于一个 stride，回归任务很难收敛
        if (bw * W * stride) < stride or (bh * H * stride) < stride:
            continue

        # 3. 确定中心点索引 (与 boxs_loss 严格对齐)
        gi, gj = int(cx * W), int(cy * H)

        # 4. 软 3x3 中心分配
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                ni, nj = gi + dx, gj + dy
                if 0 <= ni < W and 0 <= nj < H:
                    # 曼哈顿距离：0(中心), 1(边), 2(角)
                    dist = abs(dx) + abs(dy)
                    # 计算软标签
                    val = 1.0 - 0.25 * dist
                    # 取 max 是为了防止两个目标重叠时，标签被覆盖
                    target[b, 0, nj, ni] = max(target[b, 0, nj, ni], val)

    # 5. 计算带 Logits 的二元交叉熵
    # reduction="mean" 会除以 B * H * W，这对于单分类任务是标准做法
    return F.binary_cross_entropy_with_logits(
        pred_cls,
        target,
        reduction="mean"
    )
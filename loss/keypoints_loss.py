import torch
import torch.nn.functional as F

@torch.no_grad()
def evaluate_keypoints(pred_cls, pred_kpts, gt_kpts, conf_thresh=0.5, dist_thresh=0.05):
    """
    基于分类中心点评估关键点
    pred_cls: [B, 1, H, W]
    pred_kpts: [B, 12, H, W]
    gt_kpts: [B, 4, 3] (归一化坐标)
    """
    B, _, H, W = pred_cls.shape
    num_kpts = 4
    TP = FP = FN = 0

    # 1. 对分类图做 Sigmoid
    cls_prob = torch.sigmoid(pred_cls)

    for b in range(B):
        # 2. 找到当前样本中信心度最高的格子 (与推理逻辑一致)
        max_score, max_idx = torch.max(cls_prob[b].view(-1), dim=0)

        if max_score > conf_thresh:
            # 预测存在手
            gj = int(max_idx // W)
            gi = int(max_idx % W)

            # 3. 提取该位置的 4 个关键点坐标
            # 注意：这里必须加 sigmoid，因为你 Loss 里用了
            kpt_data = pred_kpts[b].view(num_kpts, 3, H, W)[:, :, gj, gi]

            for k in range(num_kpts):
                gx, gy, gv = gt_kpts[b, k]

                if gv >= 1.0:  # GT 要求该点存在
                    px = (gi + torch.sigmoid(kpt_data[k, 0])) / W
                    py = (gj + torch.sigmoid(kpt_data[k, 1])) / H

                    dist = torch.sqrt((px - gx) ** 2 + (py - gy) ** 2)
                    if dist < dist_thresh:
                        TP += 1
                    else:
                        FP += 1  # 预测偏了
                else:
                    # GT 不存在但预测了（通常这种情况较少，因为 v 指标会处理）
                    pass
        else:
            # 预测不存在手，统计 FN
            for k in range(num_kpts):
                if gt_kpts[b, k, 2] >= 1.0:
                    FN += 1

    return TP, FP, FN

def keypoints_loss(pred_kpts, gt_kpts, gt_boxes, stride):
    """
    增加了 gt_boxes 参数，确保关键点回归位与分类高分位对齐
    gt_boxes: [B, 4] -> (cx, cy, bw, bh) 归一化坐标
    """
    B, _, H, W = pred_kpts.shape
    num_kpts = 4
    device = pred_kpts.device

    pred = pred_kpts.view(B, num_kpts, 3, H, W)
    pred_x, pred_y, pred_v = pred[:, :, 0], pred[:, :, 1], pred[:, :, 2]

    target_v = torch.zeros((B, num_kpts, H, W), device=device)
    target_x = torch.zeros((B, num_kpts, H, W), device=device)
    target_y = torch.zeros((B, num_kpts, H, W), device=device)
    fg_mask = torch.zeros((B, num_kpts, H, W), device=device, dtype=torch.bool)

    num_fg = 0

    for b in range(B):
        # 获取 BBox 的中心点，这对应了 cls_loss 里的 gi, gj
        bcx, bcy, _, _ = gt_boxes[b]
        b_gi, b_gj = int(bcx * W), int(bcy * H)

        for k in range(num_kpts):
            x, y, v = gt_kpts[b, k]
            if int(v.item()) == 2:
                fx, fy = x * W, y * H

                # 配合 cls_loss 的 3x3 策略
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        ni, nj = b_gi + dx, b_gj + dy
                        if 0 <= ni < W and 0 <= nj < H:
                            # 计算关键点相对于【当前格子 ni, nj】的偏移
                            tx = fx - ni
                            ty = fy - nj

                            # 只有偏移在 0-1 之间（即点落在该格子里）或者
                            # 我们允许稍微超出边界回归（但 sigmoid 限制了 tx 必须在 0-1）
                            # 建议：只对关键点所在的真实网格或中心网格做回归
                            if 0 <= tx <= 1 and 0 <= ty <= 1:
                                target_v[b, k, nj, ni] = 1.0
                                target_x[b, k, nj, ni] = tx
                                target_y[b, k, nj, ni] = ty
                                fg_mask[b, k, nj, ni] = True
                                num_fg += 1

    # v loss：BCE
    loss_v = F.binary_cross_entropy_with_logits(pred_v, target_v, reduction='mean')

    # xy loss：仅正样本
    if num_fg > 0:
        # 你用了 sigmoid，所以预测值范围是 [0, 1]
        px = torch.sigmoid(pred_x[fg_mask])
        py = torch.sigmoid(pred_y[fg_mask])
        tx = target_x[fg_mask]
        ty = target_y[fg_mask]
        loss_xy = F.smooth_l1_loss(px, tx, reduction='mean') + \
                  F.smooth_l1_loss(py, ty, reduction='mean')
    else:
        loss_xy = torch.tensor(0., device=device)

    return 10.0 * loss_xy + 1.0 * loss_v
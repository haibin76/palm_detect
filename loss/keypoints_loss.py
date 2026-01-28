import torch
import torch.nn.functional as F

@torch.no_grad()
def evaluate_keypoints(pred_cls, pred_kpts, gt_kpts, gt_boxes,
                       conf_thresh=0.5, dist_thresh=0.05):
    """
    与 keypoints_loss 严格对齐的关键点评估函数

    pred_cls : [B, 1, H, W]
    pred_kpts: [B, K*3, H, W]
    gt_kpts  : [B, K, 3]  (x,y,v) 归一化
    gt_boxes : [B, 4]     (cx,cy,bw,bh) 归一化
    """
    B, _, H, W = pred_cls.shape
    num_kpts = gt_kpts.shape[1]

    TP = FP = FN = 0

    cls_prob = torch.sigmoid(pred_cls)

    for b in range(B):
        # 1. 找分类中心点（与训练 / decode 一致）
        max_score, max_idx = torch.max(cls_prob[b].view(-1), dim=0)

        if max_score < conf_thresh:
            # 预测不存在手 → 所有可见 GT 关键点都是 FN
            for k in range(num_kpts):
                if gt_kpts[b, k, 2] == 2:
                    FN += 1
            continue

        gj = int(max_idx // W)
        gi = int(max_idx % W)

        # 2. box 左上角（归一化）
        cx, cy, bw, bh = gt_boxes[b]
        bw = bw.clamp(min=1e-6)
        bh = bh.clamp(min=1e-6)

        x1 = cx - bw / 2
        y1 = cy - bh / 2

        # 3. 取该 grid 的关键点预测
        kpt_pred = pred_kpts[b].view(num_kpts, 3, H, W)[:, :, gj, gi]

        for k in range(num_kpts):
            gx, gy, gv = gt_kpts[b, k]

            if gv != 2:
                continue

            # ---- 与 loss 完全一致 ----
            px = torch.sigmoid(kpt_pred[k, 0])
            py = torch.sigmoid(kpt_pred[k, 1])

            # box 内 → 全图归一化
            px = x1 + px * bw
            py = y1 + py * bh

            # 欧氏距离（归一化坐标）
            dist = torch.sqrt((px - gx) ** 2 + (py - gy) ** 2)

            if dist < dist_thresh:
                TP += 1
            else:
                FP += 1

    return TP, FP, FN

def keypoints_loss(pred_kpts, gt_kpts, gt_boxes, stride=32):
    """
    pred_kpts: [B, K*3, H, W]
    gt_kpts : [B, K, 3]  (x,y,v) 归一化到 [0,1]
    gt_boxes: [B, 4] (cx, cy, bw, bh) 归一化
    """
    B, _, H, W = pred_kpts.shape
    num_kpts = gt_kpts.shape[1]
    device = pred_kpts.device

    pred = pred_kpts.view(B, num_kpts, 3, H, W)

    loss_xy, loss_v = 0.0, 0.0
    num_vis = 0

    for b in range(B):
        cx, cy, bw, bh = gt_boxes[b]
        gi, gj = int(cx * W), int(cy * H)

        # bbox 像素坐标
        x1 = (cx - bw / 2)
        y1 = (cy - bh / 2)

        for k in range(num_kpts):
            x, y, v = gt_kpts[b, k]
            if int(v.item()) != 2:
                continue

            # GT：box 内归一化
            tx = (x - x1) / bw
            ty = (y - y1) / bh

            tx = tx.clamp(0, 1)
            ty = ty.clamp(0, 1)

            px = torch.sigmoid(pred[b, k, 0, gj, gi])
            py = torch.sigmoid(pred[b, k, 1, gj, gi])
            pv = pred[b, k, 2, gj, gi]

            loss_xy += F.smooth_l1_loss(px, tx) + F.smooth_l1_loss(py, ty)

            loss_v += F.binary_cross_entropy_with_logits(pv, torch.ones_like(pv))

            num_vis += 1

    if num_vis == 0:
        return torch.tensor(0., device=device)

    return 10.0 * loss_xy + 1.0 * loss_v
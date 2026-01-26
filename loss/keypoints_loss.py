import torch
import torch.nn.functional as F

@torch.no_grad()
def evaluate_keypoints(pred_kpts, gt_kpts, stride, conf_thresh=0.5, dist_thresh=0.05):

    B, _, H, W = pred_kpts.shape
    num_kpts = 4

    pred = pred_kpts.view(B, num_kpts, 3, H, W)

    TP = FP = FN = 0

    for b in range(B):
        for k in range(num_kpts):
            gx, gy, gv = gt_kpts[b, k]

            conf_map = torch.sigmoid(pred[b, k, 2])
            mask = conf_map > conf_thresh
            indices = mask.nonzero()

            preds = []
            for gj, gi in indices:
                px = (gi + torch.sigmoid(pred[b, k, 0, gj, gi])) / W
                py = (gj + torch.sigmoid(pred[b, k, 1, gj, gi])) / H
                preds.append((px, py))

            if gv >= 1.0:  # GT 存在
                if len(preds) == 0:
                    FN += 1
                    continue

                # 找最近的预测
                dists = [torch.sqrt((px - gx)**2 + (py - gy)**2) for px, py in preds]
                min_dist = min(dists)

                if min_dist < dist_thresh:
                    TP += 1
                    FP += (len(preds) - 1)  # 多余预测算 FP
                else:
                    FN += 1
                    FP += len(preds)

            else:  # GT 不存在
                FP += len(preds)

    return TP, FP, FN

def keypoints_loss(pred_kpts, gt_kpts, stride):
    """
    pred_kpts: [B, 12, H, W] -> (x, y, v) per keypoint
    gt_kpts:   [B, 4, 3]
        x, y: normalized to [0,1] w.r.t letterboxed image
        v: 0 = not exist, 2 = exist & supervised
    """

    B, _, H, W = pred_kpts.shape
    num_kpts = 4
    device = pred_kpts.device

    pred = pred_kpts.view(B, num_kpts, 3, H, W)
    pred_x, pred_y, pred_v = pred[:, :, 0], pred[:, :, 1], pred[:, :, 2]

    target_v = torch.zeros((B, num_kpts, H, W), device=device)
    target_x = torch.zeros((B, num_kpts, H, W), device=device)
    target_y = torch.zeros((B, num_kpts, H, W), device=device)
    fg_mask  = torch.zeros((B, num_kpts, H, W), device=device, dtype=torch.bool)

    num_fg = 0

    for b in range(B):
        for k in range(num_kpts):
            x, y, v = gt_kpts[b, k]

            # 只对 v == 2 的关键点做正样本
            if int(v.item()) == 2:
                fx = x * W
                fy = y * H
                gi = int(fx)
                gj = int(fy)

                if gi < 0 or gi >= W or gj < 0 or gj >= H:
                    continue

                target_v[b, k, gj, gi] = 1.0
                target_x[b, k, gj, gi] = fx - gi
                target_y[b, k, gj, gi] = fy - gj
                fg_mask[b, k, gj, gi] = True
                num_fg += 1

    # v loss：所有位置
    loss_v = F.binary_cross_entropy_with_logits(pred_v, target_v, reduction='mean')

    # xy loss：仅正样本
    if num_fg > 0:
        px = torch.sigmoid(pred_x[fg_mask])
        py = torch.sigmoid(pred_y[fg_mask])
        tx = target_x[fg_mask]
        ty = target_y[fg_mask]

        loss_xy = (F.smooth_l1_loss(px, tx, reduction='sum') + F.smooth_l1_loss(py, ty, reduction='sum') ) / num_fg
    else:
        loss_xy = torch.tensor(0., device=device)

    return loss_xy + 1.0 * loss_v
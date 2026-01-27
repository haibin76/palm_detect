import torch
import torch.nn.functional as F

@torch.no_grad()
def evaluate_cls(pred_cls, gt_boxes, stride=32, conf_thresh=0.5):
    device = pred_cls.device
    B, _, H, W = pred_cls.shape
    pred_prob = torch.sigmoid(pred_cls)
    img_size = W * stride
    TP = FP = FN = 0

    for b in range(B):
        cx, cy, bw, bh = gt_boxes[b]

        # skip small hands (same as training)
        if bw * img_size < stride or bh * img_size < stride:
            continue

        gi = int(cx * W)
        gj = int(cy * H)

        if not (0 <= gi < W and 0 <= gj < H):
            continue

        pred_mask = pred_prob[b, 0] > conf_thresh

        # center-based evaluation (objectness semantics)
        if pred_mask[gj, gi]:
            TP += 1
        else:
            FN += 1

        # count extra positives as FP
        FP += pred_mask.sum().item() - int(pred_mask[gj, gi])

    precision = TP / (TP + FP + 1e-6)
    recall = TP / (TP + FN + 1e-6)

    return TP, FP, FN

def cls_loss(pred_cls, gt_boxes, stride=32):
    """
    Objectness loss for single-class (hand / no-hand)
    """

    device = pred_cls.device
    B, _, H, W = pred_cls.shape
    target = torch.zeros_like(pred_cls)
    img_size = W * stride

    for b in range(B):
        cx, cy, bw, bh = gt_boxes[b]

        # filter small hands (critical for single-scale)
        if bw * img_size < stride or bh * img_size < stride:
            continue

        gi = int(cx * W)
        gj = int(cy * H)

        if not (0 <= gi < W and 0 <= gj < H):
            continue

        # soft 3x3 center assign
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                ni, nj = gi + dx, gj + dy
                if 0 <= ni < W and 0 <= nj < H:
                    dist = abs(dx) + abs(dy)
                    target[b, 0, nj, ni] = max(
                        target[b, 0, nj, ni],
                        1.0 - 0.25 * dist
                    )

    return F.binary_cross_entropy_with_logits(
        pred_cls,
        target,
        reduction="mean"
    )
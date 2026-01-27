import torch
import torch.nn.functional as F

def unletterbox(x, y, scale, pad_left, pad_top, orig_w, orig_h):
    """
    将 letterbox 后坐标映射回原图坐标（YOLOv8 对齐）

    Args:
        x, y      : Tensor 或 float，letterbox 后坐标
        scale     : resize 比例 (new / original)
        pad_left  : 左侧 padding（像素）
        pad_top   : 顶部 padding（像素）
        orig_w    : 原图宽
        orig_h    : 原图高

    Returns:
        x, y      : 原图坐标（已 clamp）
    """

    # 1. 反算 padding + resize
    x = (x - pad_left) / scale
    y = (y - pad_top) / scale

    # 2. Clamp 到原图范围
    if isinstance(x, torch.Tensor):
        x = x.clamp(0, orig_w - 1)
        y = y.clamp(0, orig_h - 1)
    else:
        x = max(0, min(x, orig_w - 1))
        y = max(0, min(y, orig_h - 1))

    return x, y


def decode_hand_bundle(pred_cls, pred_boxs_64, pred_kpts, stride=32, conf_thresh=0.5, reg_max=16, num_kpts=4, scale=1.0,
                       pad_left=0, pad_top=0, orig_w=None, orig_h=None):
    device = pred_cls.device
    _, _, H, W = pred_cls.shape

    # 1. 找中心点
    cls_prob = torch.sigmoid(pred_cls)
    max_score, max_idx = torch.max(cls_prob.view(-1), dim=0)
    if max_score < conf_thresh:
        return None

    gj = int(max_idx // W)
    gi = int(max_idx % W)

    # 2. BBox 解码
    project = torch.arange(reg_max, device=device, dtype=torch.float32)
    box_dist = pred_boxs_64[0, :, gj, gi].view(4, reg_max)
    box_prob = F.softmax(box_dist, dim=1)
    ltrb = torch.matmul(box_prob, project)
    l, t, r, b = ltrb

    # 【微调点】：这里加上 0.5，确保与 boxs_loss 里的计算基准点完全重合
    cell_cx, cell_cy = gi + 0.5, gj + 0.5

    x1 = (cell_cx - l) * stride
    y1 = (cell_cy - t) * stride
    x2 = (cell_cx + r) * stride
    y2 = (cell_cy + b) * stride

    # 3. Keypoints 解码 (修正偏移量逻辑)
    kpt_raw = pred_kpts[0, :, gj, gi].view(num_kpts, 3)
    kpts = []
    for i in range(num_kpts):
        # 注意：YOLOv8 官方关键点回归不一定用 sigmoid
        # 如果你训练时用了 sigmoid，保持现状；
        # 但通常回归 dx, dy 是线性输出，表示相对于当前网格中心的偏移倍数
        dx = torch.sigmoid(kpt_raw[i, 0])  # 必须加 sigmoid，因为你 Loss 里用了
        dy = torch.sigmoid(kpt_raw[i, 1])
        v = torch.sigmoid(kpt_raw[i, 2])

        # 修正公式：kx = (grid_x + dx * 2) * stride (取决于你 Loss 里的写法)
        # 这里先按常见的“相对坐标”还原：
        kx = (gi + dx) * stride
        ky = (gj + dy) * stride
        kpts.append([kx, ky, v])

    # --------------------------------------------------
    # 4. 反算 Letterbox
    # --------------------------------------------------
    if orig_w is not None and orig_h is not None:
        x1, y1 = unletterbox(
            x1, y1,
            scale=scale,
            pad_left=pad_left,
            pad_top=pad_top,
            orig_w=orig_w,
            orig_h=orig_h,
        )
        x2, y2 = unletterbox(
            x2, y2,
            scale=scale,
            pad_left=pad_left,
            pad_top=pad_top,
            orig_w=orig_w,
            orig_h=orig_h,
        )

        final_kpts = []
        for kx, ky, v in kpts:
            kx, ky = unletterbox(
                kx, ky,
                scale=scale,
                pad_left=pad_left,
                pad_top=pad_top,
                orig_w=orig_w,
                orig_h=orig_h,
            )
            final_kpts.append([int(kx), int(ky), float(v)])
    else:
        final_kpts = [[int(kx), int(ky), float(v)] for kx, ky, v in kpts]

    # --------------------------------------------------
    # 5. 输出
    # --------------------------------------------------
    return {
        "score": float(max_score),
        "bbox": [int(x1), int(y1), int(x2), int(y2)],
        "kpts": final_kpts,
        "center": (gi, gj),
    }
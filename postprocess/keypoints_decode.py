import torch
import numpy as np

def decode_keypoints(pred_kpts, stride=32, conf_thresh=0.5, img_size=640,
                     scale=1.0, pad_left=0, pad_top=0, orig_w=None, orig_h=None):
    """
    pred_kpts: [1, 12, H, W]
    return: List[(x, y, conf)]
    """

    _, _, H, W = pred_kpts.shape
    num_kpts = 4

    pred = pred_kpts.view(num_kpts, 3, H, W)
    results = []

    for k in range(num_kpts):
        conf_map = torch.sigmoid(pred[k, 2])
        max_conf, max_idx = torch.max(conf_map.view(-1), dim=0)

        if max_conf < conf_thresh:
            results.append((0, 0, 0.0))
            continue

        gj = int(max_idx // W)
        gi = int(max_idx % W)

        dx = torch.sigmoid(pred[k, 0, gj, gi])
        dy = torch.sigmoid(pred[k, 1, gj, gi])

        # --- 1️⃣ 映射到 640×640 ---
        x_in = (gi + dx) * stride
        y_in = (gj + dy) * stride

        # --- 2️⃣ 去掉 letterbox padding ---
        x_unpad = x_in - pad_left
        y_unpad = y_in - pad_top

        # --- 3️⃣ 映射回原图 ---
        if orig_w is not None and orig_h is not None:
            x = x_unpad / scale
            y = y_unpad / scale
            if not (0 <= x < orig_w and 0 <= y < orig_h):
                results.append((0, 0, 0.0))
                continue
        else:
            x = x_in
            y = y_in

        results.append((int(x), int(y), max_conf.item()))

    return results
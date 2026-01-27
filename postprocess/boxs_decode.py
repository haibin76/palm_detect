import torch
import numpy as np

def decode_boxs(pred_cls, pred_boxs_64, stride=32, conf_thresh=0.5, img_size=640,
                scale=1.0, pad_left=0, pad_top=0, orig_w=None, orig_h=None):
    """
    pred_cls: [1, num_classes, H, W]  分类头输出 (Logits)
    pred_boxs_64: [1, 64, H, W]       BBox DFL输出
    """
    device = pred_cls.device
    _, _, H, W = pred_cls.shape

    # 1️⃣ 分类头解码：获取置信度
    # YOLOv8 默认使用 Sigmoid 处理分类输出
    cls_score = torch.sigmoid(pred_cls)

    # 获取最大分值及其位置
    # 假设我们只取全图最自信的一个目标 (Top-1)
    max_score, max_idx = torch.max(cls_score.view(-1), dim=0)

    if max_score < conf_thresh:
        return None  # 未发现目标

    # 换算回特征图坐标 (gi, gj)
    # 注意：根据你的输出维度 [1, nc, H, W]，max_idx 需要先除以 nc
    nc = pred_cls.shape[1]
    cell_idx = max_idx // nc
    gj = int(cell_idx // W)
    gi = int(cell_idx % W)

    # 2️⃣ BBox 解码 (使用之前的 DFL 逻辑)
    reg_max = 16
    project = torch.arange(reg_max, dtype=torch.float32, device=device)
    x_box = pred_boxs_64.view(1, 4, reg_max, H, W).softmax(dim=2)
    pred_ltrb = (x_box * project.view(1, 1, -1, 1, 1)).sum(dim=2)

    l, t, r, b_dist = pred_ltrb[0, :, gj, gi]

    # 3️⃣ 映射到全图像素
    cell_cx = (gi + 0.5) * stride
    cell_cy = (gj + 0.5) * stride

    x1_in = cell_cx - l * stride
    y1_in = cell_cy - t * stride
    x2_in = cell_cx + r * stride
    y2_in = cell_cy + b_dist * stride

    # 4️⃣ 去掉 padding 并缩放回原图
    if orig_w is not None and orig_h is not None:
        x1 = (x1_in - pad_left) / scale
        y1 = (y1_in - pad_top) / scale
        x2 = (x2_in - pad_left) / scale
        y2 = (y2_in - pad_top) / scale
        x1, x2 = np.clip([x1, x2], 0, orig_w)
        y1, y2 = np.clip([y1, y2], 0, orig_h)
    else:
        x1, y1, x2, y2 = x1_in, y1_in, x2_in, y2_in

    return {
        "bbox": [int(x1), int(y1), int(x2), int(y2)],
        "score": max_score.item(),
        "gi": gi, "gj": gj  # 传递给关键点解码使用
    }
import os
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np

def letterbox(img, new_size=640, color=(255, 255, 255)):
    h, w = img.shape[:2]

    scale = min(new_size / w, new_size / h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_w = new_size - new_w
    pad_h = new_size - new_h

    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top

    img_out = cv2.copyMakeBorder(
        img_resized,
        pad_top, pad_bottom,
        pad_left, pad_right,
        cv2.BORDER_CONSTANT,
        value=color
    )
    return img_out, scale, (pad_left, pad_top)

class HandKeypointsDataset(Dataset):
    def __init__(self, root, split="train", img_size=640):
        self.img_dir = os.path.join(root, "images", split)
        self.lbl_dir = os.path.join(root, "labels", split)
        self.img_size = img_size

        self.files = sorted([
            f for f in os.listdir(self.img_dir)
            if f.endswith(".jpg")
        ])

    def __transform_box_and_kpts__(self, box, keypoints, orig_w, orig_h, ratio, pad):
        pad_x, pad_y = pad

        # ---- box: norm -> pixel ----
        cx, cy, w, h = box
        cx *= orig_w
        cy *= orig_h
        w *= orig_w
        h *= orig_h

        # scale
        cx *= ratio
        cy *= ratio
        w *= ratio
        h *= ratio

        # pad
        cx += pad_x
        cy += pad_y

        # back to normalized (640x640)
        cx /= 640
        cy /= 640
        w /= 640
        h /= 640

        new_box = torch.tensor([cx, cy, w, h])

        # ---- keypoints ----
        new_kpts = []
        for x, y, c in keypoints:
            x *= orig_w
            y *= orig_h

            x = x * ratio + pad_x
            y = y * ratio + pad_y

            x /= 640
            y /= 640

            new_kpts.append([x, y, c])

        new_kpts = torch.tensor(new_kpts)
        return new_box, new_kpts

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name = self.files[idx]
        img_path = os.path.join(self.img_dir, name)
        lbl_path = os.path.join(self.lbl_dir, name.replace(".jpg", ".txt"))

        # ---------------- image ----------------
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h0, w0 = img.shape[:2]

        img, ratio, pad = letterbox(img)

        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)  # [3,640,640]

        # ---------------- label ----------------
        # 默认：负样本
        class_id = torch.tensor(0, dtype=torch.long)
        box = torch.zeros(4, dtype=torch.float32)          # cx cy w h
        keypoints = torch.zeros((4, 3), dtype=torch.float32)  # x y v

        # 如果 label 文件存在且非空 → 正样本
        if os.path.exists(lbl_path) and os.path.getsize(lbl_path) > 0:
            with open(lbl_path, "r") as f:
                data = list(map(float, f.readline().strip().split()))

            if len(data) != 17:
                raise ValueError(f"{lbl_path} format error, got {len(data)} values")

            class_id = torch.tensor(int(data[0]), dtype=torch.long)

            box = torch.tensor(data[1:5], dtype=torch.float32)
            keypoints = torch.tensor(data[5:], dtype=torch.float32).view(4, 3)

            # 做 letterbox 后的坐标变换
            box, keypoints = self.__transform_box_and_kpts__(
                box,
                keypoints,
                orig_w=w0,
                orig_h=h0,
                ratio=ratio,
                pad=pad
            )

        return {
            "images": img,
            "class_id": class_id,
            "boxs": box,
            "keypoints": keypoints,
        }
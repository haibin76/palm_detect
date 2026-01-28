import os
import torch
from torch.utils.data import DataLoader
from dataset.data import HandKeypointsDataset
import cv2
import numpy as np
from tqdm import tqdm

from loss.cls_loss import cls_loss
from models.qm_yolov8 import QMYoloV8
from loss.keypoints_loss import keypoints_loss
from loss.keypoints_loss import evaluate_keypoints

from loss.boxs_loss import boxs_loss
from loss.boxs_loss import evaluate_boxs
from loss.boxs_loss import DFL

from loss.cls_loss import cls_loss
from loss.cls_loss import evaluate_cls

def visualize_batch(batch):
    """
    逐张显示一个 batch 中的样本
    """
    images = batch["image"]        # [B,3,640,640]
    boxes = batch["box"]           # [B,4]  (cx,cy,w,h) normalized
    keypoints = batch["keypoints"] # [B,4,3] normalized

    B = images.shape[0]

    for i in range(B):
        img = images[i].permute(1, 2, 0).cpu().numpy()
        img = (img * 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        h, w = img.shape[:2]

        # ---------- box ----------
        cx, cy, bw, bh = boxes[i]
        cx, cy, bw, bh = cx*w, cy*h, bw*w, bh*h

        x1 = int(cx - bw / 2)
        y1 = int(cy - bh / 2)
        x2 = int(cx + bw / 2)
        y2 = int(cy + bh / 2)

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # ---------- keypoints ----------
        for (x, y, c) in keypoints[i]:
            if c > 0:  # 可见
                px = int(x * w)
                py = int(y * h)
                cv2.circle(img, (px, py), 4, (0, 0, 255), -1)

        cv2.imshow(f"sample {i+1}/{B}", img)

        key = cv2.waitKey(0)  # 阻塞，等你按键或关窗口
        cv2.destroyAllWindows()

        if key == 27:  # ESC 直接退出
            break

def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0
    pbar = tqdm(dataloader, total=len(dataloader), desc="Train", ncols=120)
    dfl_decoder = DFL(reg_max=16).to(device)

    for step, batch in enumerate(pbar):
        images = batch["images"].to(device)
        targets_kpts = batch["keypoints"].to(device)
        targets_bbox = batch["boxs"].to(device)  # 确保你的 Dataloader 返回了 bbox 字段

        # 1️⃣ forward
        cls, boxs, keypoints = model(images)

        # 2️⃣ 计算多任务 Loss
        # 计算关键点 Loss
        loss_kpt = keypoints_loss(pred_kpts=keypoints, gt_kpts=targets_kpts, gt_boxes=targets_bbox, stride=32)

        # 计算检测框 Loss
        loss_box = boxs_loss(pred_boxes_64=boxs, gt_boxes=targets_bbox, dfl_layer=dfl_decoder, stride=32)

        # 计算分类ID Loss
        loss_cls = cls_loss(pred_cls = cls, gt_boxes=targets_bbox, stride=32)

        # 合并 Loss (通常权重比例为 1:1 或根据收敛情况调整)
        loss = (10.0 * loss_kpt + 7.0 * loss_box + 1.0 * loss_cls)

        # 3️⃣ backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # 更新进度条，显示两个 loss 方便观察哪个不收敛
        pbar.set_postfix(
            total=f"{loss.item():.3f}",
            cls=f"{loss_cls.item():.3f}",
            kpt=f"{loss_kpt.item():.3f}",
            box=f"{loss_box.item():.3f}",
            step=f"{step}"
        )

    return total_loss / len(dataloader)

def val_one_epoch(model, dataloader, device):
    # 1. 切换到验证模式 (关键！)
    model.eval()

    total_loss = 0.0
    KPTS_TP = KPTS_FP = KPTS_FN = 0
    BOXS_TP = BOXS_FP = BOXS_FN = 0
    CLS_TP= CLS_FP = CLS_FN = 0

    pbar = tqdm(dataloader, total=len(dataloader), desc="Val", ncols=120)
    dfl_decoder = DFL(reg_max=16).to(device)

    # 2. 禁用梯度计算 (关键！省显存)
    with torch.no_grad():
        for step, batch in enumerate(pbar):
            images = batch["images"].to(device)
            kpts_targets = batch["keypoints"].to(device)
            boxs_targets = batch["boxs"].to(device)

            # 这里的推理必须对应你模型的 output 顺序
            pred_cls, pred_boxs, pred_kpts = model(images)

            # --- Keypoints 评估 ---
            loss_kpts = keypoints_loss(pred_kpts, kpts_targets, boxs_targets, stride=32)
            kpts_tp, kpts_fp, kpts_fn = evaluate_keypoints(pred_cls=pred_cls, pred_kpts=pred_kpts, gt_kpts=kpts_targets, gt_boxes=boxs_targets, conf_thresh=0.5, dist_thresh=0.05)
            KPTS_TP += kpts_tp
            KPTS_FP += kpts_fp
            KPTS_FN += kpts_fn

            # --- BBox 评估 ---
            loss_boxs = boxs_loss(pred_boxs, boxs_targets, dfl_decoder, stride=32)
            boxs_tp, boxs_fp, boxs_fn = evaluate_boxs(pred_cls=pred_cls, pred_boxs_64=pred_boxs, gt_boxes=boxs_targets, dfl_layer=dfl_decoder, stride=32, img_size = 640, iou_thresh = 0.5, conf_thresh = 0.5)
            BOXS_TP += boxs_tp
            BOXS_FP += boxs_fp
            BOXS_FN += boxs_fn

            #计算class_id的分数
            loss_cls = cls_loss(pred_cls, boxs_targets, stride=32)
            cls_tp, cls_fp, cls_fn = evaluate_cls(pred_cls, boxs_targets, stride=32, conf_thresh=0.5)
            CLS_TP += cls_tp
            CLS_FP += cls_fp
            CLS_FN += cls_fn

            loss = (10.0 * loss_kpts + 7.0 * loss_boxs + 1.0 * loss_cls)

            total_loss += loss.item()

            pbar.set_postfix(
                total=f"{loss.item():.3f}",
                cls=f"{loss_cls.item():.3f}",
                box=f"{loss_boxs.item():.3f}",
                pkts=f"{loss_kpts.item():.3f}"
            )

    # 3. 计算最终指标 (防止除零)
    safe_div = lambda n, d: n / (d + 1e-6)

    metrics = {
        "loss": total_loss / len(dataloader),
        "kpts_precision": safe_div(KPTS_TP, KPTS_TP + KPTS_FP),
        "kpts_recall": safe_div(KPTS_TP, KPTS_TP + KPTS_FN),
        "boxs_precision": safe_div(BOXS_TP, BOXS_TP + BOXS_FP),
        "boxs_recall": safe_div(BOXS_TP, BOXS_TP + BOXS_FN),
        "cls_precision": safe_div(CLS_TP, CLS_TP + CLS_FP),
        "cls_recall": safe_div(CLS_TP, CLS_TP + CLS_FN),
        # 直接把这些值放出来，不要包在 details 里
        "KPTS_TP": KPTS_TP,
        "KPTS_FP": KPTS_FP,
        "KPTS_FN": KPTS_FN,
        "BOXS_TP": BOXS_TP,
        "BOXS_FP": BOXS_FP,
        "BOXS_FN": BOXS_FN,
        "CLS_TP": CLS_TP,
        "CLS_FP": CLS_FP,
        "CLS_FN": CLS_FN
    }
    return metrics

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\nthis device:", device)
    start_epoch = 0
    end_epochs = 50
    batch_size = 64

    train_dataset = HandKeypointsDataset(root="data/hand-keypoints", split="train", img_size=640)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True )

    val_dataset = HandKeypointsDataset(root="data/hand-keypoints", split="val", img_size=640)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    #batch = next(iter(train_loader))
    #print(batch["image"].shape)
    #visualize_batch(batch)

    checkpoint_path = "weights/last.pt"  # 你的权重文件路径
    model = QMYoloV8().to(device)
    optimizer = torch.optim.AdamW( model.parameters(), lr=1e-3, weight_decay=1e-4)
    best_val_loss = float("inf")

    # 2. 加载权重
    if os.path.exists(checkpoint_path):
        print(f"正在加载权重: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # 加载模型参数
        model.load_state_dict(checkpoint['model_state_dict'])

        # 如果是接着训练（Resume），还需要加载优化器状态和 Epoch
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            print(f"成功恢复训练，将从第 {start_epoch} Epoch 开始")
    else:
        print("未找到预训练权重，将从零开始训练")

    for epoch in range(start_epoch, end_epochs):

        print(f"\n==== Epoch {epoch} ====")
        model.train()
        avg_train_loss = train_one_epoch(model, train_loader, optimizer, device)

        model.eval()
        with torch.no_grad():
            val_stats = val_one_epoch(model, val_loader, device)

            print(
                f"\n" + "=" * 50 +
                f"\n[Epoch {epoch}] Summary:"
                f"\nLoss:      Train={avg_train_loss:.4f} | Val={val_stats['loss']:.4f}"
                f"\nCls: Prec={val_stats['cls_precision']:.4f} | Rec={val_stats['cls_recall']:.4f} "
                f"(TP={val_stats['CLS_TP']}, FP={val_stats['CLS_FP']}, FN={val_stats['CLS_FN']})"
                f"\nKeypoints: Prec={val_stats['kpts_precision']:.4f} | Rec={val_stats['kpts_recall']:.4f} "
                f"(TP={val_stats['KPTS_TP']}, FP={val_stats['KPTS_FP']}, FN={val_stats['KPTS_FN']})"
                f"\nBBox:      Prec={val_stats['boxs_precision']:.4f} | Rec={val_stats['boxs_recall']:.4f} "
                f"(TP={val_stats['BOXS_TP']}, FP={val_stats['BOXS_FP']}, FN={val_stats['BOXS_FN']})"
                f"\n" + "=" * 50
            )

        # 保存
        if val_stats["loss"] < best_val_loss:
            best_val_loss = val_stats["loss"]
            torch.save(model.state_dict(), "weights/best.pt")
            print(">> saved best model")

    if start_epoch < end_epochs:
        # 保存当前的进度
        save_dict = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_train_loss,
        }
        torch.save(save_dict, checkpoint_path)

if __name__ == "__main__":
    main()

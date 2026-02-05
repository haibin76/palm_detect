import torch
import cv2
import numpy as np
from dataset.data import letterbox
from postprocess.decode import decode_hand_bundle
from postprocess.keypoints_decode import decode_keypoints
from postprocess.boxs_decode import decode_boxs
from models.qm_yolov8 import QMYoloV8
import onnxruntime as ort
import torch.nn.functional as F

def main_image():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL_PATH = "weights/best.pt"
    IMG_SIZE = 640
    CONF_THRESH = 0.5

    model = QMYoloV8().to(DEVICE)

    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt if isinstance(ckpt, dict) and "model" not in ckpt else ckpt["model"])
    model.eval()

    print("✅ model loaded & ready for inference")

    frame = cv2.imread(
        r"data/hand-keypoints/images/val/81750923-7c33-41ec-9fda-73f30a531bc0.jpg"
    )
    assert frame is not None, "❌ image load failed"

    h0, w0 = frame.shape[:2]

    img_lb, scale, pad = letterbox(frame, IMG_SIZE)

    img = img_lb[:, :, ::-1]
    img = img.astype(np.float32) / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

    x = np.ones((1, 3, 640, 640), dtype=np.float32)
    x = torch.from_numpy(x).float()

    with torch.no_grad():
        pred_cls, pred_boxs, pred_kpts = model(x)

    result = decode_hand_bundle(
        pred_cls, pred_boxs, pred_kpts,
        stride=32,
        conf_thresh=CONF_THRESH,
        reg_max=16,
        num_kpts=4,
        scale=scale,
        pad_left=pad[0],
        pad_top=pad[1],
        orig_w=w0,
        orig_h=h0
    )

    if result:
        x1, y1, x2, y2 = result["bbox"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        for x, y, v in result["kpts"]:
            if v > 0.1:
                cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

    cv2.imshow("Keypoints Inference", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main_camera():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL_PATH = "weights/best.pt"
    IMG_SIZE = 640
    CONF_THRESH = 0.5

    # === load model ===
    model = QMYoloV8().to(DEVICE)

    # 2️⃣ 加载权重（注意：不是直接 model = torch.load）
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)

    # 3️⃣ 切换到 eval
    model.eval()

    print("✅ model loaded & ready for inference")

    # === camera ===
    cap = cv2.VideoCapture(0)
    assert cap.isOpened(), "❌ Camera open failed"

    print("📷 Camera started (press q to quit)")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h0, w0 = frame.shape[:2]

        # --- letterbox ---
        img_lb, scale, pad = letterbox(frame, IMG_SIZE)

        # --- preprocess ---
        img = img_lb[:, :, ::-1]  # BGR → RGB
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

        # --- inference ---
        with torch.no_grad():
            pred_cls, pred_boxs, pred_kpts = model(img)

        result = decode_hand_bundle(pred_cls, pred_boxs, pred_kpts, 32, CONF_THRESH, 16, 4, scale, pad[0], pad[1], w0, h0)
        # 4. 绘制
        if result:
            x1, y1, x2, y2 = result["bbox"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            for x, y, v in result["kpts"]:
                if v > 0.1:
                    cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

            cv2.imshow("Keypoints Inference", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.imshow("Keypoints Inference", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main_image()
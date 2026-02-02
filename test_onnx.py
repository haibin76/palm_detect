import cv2
import numpy as np
import torch
import onnxruntime as ort

from dataset.data import letterbox
from postprocess.decode import decode_hand_bundle

# =====================================================
# 配置
# =====================================================
ONNX_PATH = "weights/best_ubuntu.onnx"
IMG_SIZE = 640
CONF_THRESH = 0.5
STRIDE = 32
REG_MAX = 16
NUM_KPTS = 4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    # =====================================================
    # 1️⃣ ONNX Runtime Session
    # =====================================================
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] \
        if torch.cuda.is_available() else ["CPUExecutionProvider"]

    sess = ort.InferenceSession(ONNX_PATH, providers=providers)

    input_name = sess.get_inputs()[0].name
    output_names = [o.name for o in sess.get_outputs()]

    print("✅ ONNX loaded")
    print("input :", input_name)
    print("outputs:", output_names)

    # =====================================================
    # 2️⃣ Camera
    # =====================================================
    cap = cv2.VideoCapture(0)
    assert cap.isOpened(), "❌ Camera open failed"
    print("📷 Camera started (press q to quit)")

    # =====================================================
    # 3️⃣ Loop
    # =====================================================
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h0, w0 = frame.shape[:2]

        # -------------------------------------------------
        # letterbox
        # -------------------------------------------------
        img_lb, scale, pad = letterbox(frame, IMG_SIZE)

        # -------------------------------------------------
        # preprocess (对齐 PyTorch)
        # -------------------------------------------------
        img = img_lb[:, :, ::-1]                 # BGR → RGB
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))       # HWC → CHW
        img = np.expand_dims(img, 0)             # BCHW

        # -------------------------------------------------
        # ONNX inference
        # -------------------------------------------------
        pred_cls, pred_boxs, pred_kpts = sess.run(
            output_names,
            {input_name: img}
        )

        # -------------------------------------------------
        # numpy → torch（关键！）
        # -------------------------------------------------
        pred_cls = torch.from_numpy(pred_cls)
        pred_boxs = torch.from_numpy(pred_boxs)
        pred_kpts = torch.from_numpy(pred_kpts)

        # -------------------------------------------------
        # decode（完全复用原逻辑）
        # -------------------------------------------------
        result = decode_hand_bundle(
            pred_cls,
            pred_boxs,
            pred_kpts,
            stride=STRIDE,
            conf_thresh=CONF_THRESH,
            reg_max=REG_MAX,
            num_kpts=NUM_KPTS,
            scale=scale,
            pad_left=pad[0],
            pad_top=pad[1],
            orig_w=w0,
            orig_h=h0
        )

        # -------------------------------------------------
        # draw
        # -------------------------------------------------
        if result:
            x1, y1, x2, y2 = result["bbox"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            for x, y, v in result["kpts"]:
                if v > 0.1:
                    cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

            cv2.putText(
                frame,
                f"{result['score']:.2f}",
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        cv2.imshow("ONNX Hand Pose", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
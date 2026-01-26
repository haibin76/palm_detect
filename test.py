import torch
import cv2
import numpy as np
from dataset.data import letterbox
from postprocess.keypoints_decode import decode_keypoints
from models.qm_yolov8 import QMYoloV8
def main():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL_PATH = "weights/best.pt"
    IMG_SIZE = 640
    CONF_THRESH = 0.5

    # === load model ===
    model = QMYoloV8().to(DEVICE)

    # 2️⃣ 加载权重（注意：不是直接 model = torch.load）
    state_dict = torch.load("weights/best.pt", map_location=DEVICE)
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
            _, _, pred_kpts = model(img)

        # --- decode ---
        kpts = decode_keypoints(pred_kpts, 32, CONF_THRESH, IMG_SIZE, scale, pad[0], pad[1], w0, h0)
        # --- draw ---
        for k, (x, y, conf) in enumerate(kpts):
            cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)
            cv2.putText(
                frame,
                f"K{k}:{conf:.2f}",
                (x + 4, y - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1
            )

        cv2.imshow("Keypoints Inference", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
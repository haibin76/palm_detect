import cv2
import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import torch

from dataset.data import letterbox
from postprocess.decode import decode_hand_bundle_np
from postprocess.decode import decode_hand_bundle

PB_PATH = "weights/best_ubuntu.pb"
IMG_SIZE = 640
CONF_THRESH = 0.5

def load_pb(pb_path):
    with tf.gfile.GFile(pb_path, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    graph = tf.Graph()
    with graph.as_default():
        tf.import_graph_def(graph_def, name="")

    return graph

def main():
    # === load pb ===
    graph = load_pb(PB_PATH)

    input_x = graph.get_tensor_by_name("input_images:0")
    out_cls  = graph.get_tensor_by_name("output_cls:0")
    out_box  = graph.get_tensor_by_name("output_box:0")
    out_kpts = graph.get_tensor_by_name("output_kpts:0")

    sess = tf.Session(graph=graph)

    print("✅ PB model loaded")

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

        # --- preprocess (对齐 PyTorch) ---
        img = img_lb[:, :, ::-1]              # BGR → RGB
        img = img.astype(np.float32) / 255.0  # float32
        img = np.expand_dims(img, axis=0)     # NHWC

        # --- inference ---
        cls_v, box_v, kpts_v = sess.run(
            [out_cls, out_box, out_kpts],
            feed_dict={input_x: img}
        )

        result = decode_hand_bundle_np(
            cls_v,
            box_v,
            kpts_v,
            stride=32,
            conf_thresh=0.5,
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

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

'''
        # --- decode（完全复用你的逻辑） ---
        result = decode_hand_bundle_np(
            cls_v,
            box_v,
            kpts_v,
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

        # --- draw ---
        if result:
            x1, y1, x2, y2 = result["bbox"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            for x, y, v in result["kpts"]:
                if v > 0.1:
                    cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
'''

if __name__ == "__main__":
    main()

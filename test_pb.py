import cv2
import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import torch
from tensorflow.python.framework import tensor_util

from dataset.data import letterbox
from postprocess.decode import decode_hand_bundle_np
from postprocess.decode import decode_hand_bundle

PB_PATH = "weights/best.pb"
IMG_SIZE = 640
CONF_THRESH = 0.5
STRIDE = 32
REG_MAX = 16
NUM_KPTS = 4

def dump_first_bn_from_pb():
    print("\n===== LOAD PB =====")
    graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile(PB_PATH, "rb") as f:
        graph_def.ParseFromString(f.read())

    print("Total nodes:", len(graph_def.node))

    # -----------------------------------------
    # 1️⃣ 找第一个 FusedBatchNorm
    # -----------------------------------------
    bn_node = None
    for node in graph_def.node:
        if node.op in ("FusedBatchNorm", "FusedBatchNormV3"):
            bn_node = node
            break

    assert bn_node is not None, "❌ No FusedBatchNorm node found"

    print("\n===== FIRST BN NODE =====")
    print("name :", bn_node.name)
    print("op   :", bn_node.op)
    print("inputs:", bn_node.input)

    # -----------------------------------------
    # 2️⃣ 读取 epsilon
    # -----------------------------------------
    epsilon = bn_node.attr["epsilon"].f
    print("\n[BN epsilon]")
    print("epsilon =", epsilon)

    # -----------------------------------------
    # 3️⃣ 建一个 name → Const node 的映射
    # -----------------------------------------
    const_map = {}
    for node in graph_def.node:
        if node.op == "Const":
            const_map[node.name] = node

    def dump_param(title, tensor_name):
        # PB 里 input 可能带 :0
        base_name = tensor_name.split(":")[0]
        assert base_name in const_map, f"❌ {base_name} not found"

        tensor = const_map[base_name].attr["value"].tensor
        arr = tensor_util.MakeNdarray(tensor)

        print(
            f"{title}: shape={arr.shape}, "
            f"min={arr.min():.6f}, max={arr.max():.6f}, mean={arr.mean():.6f}"
        )
        print(f"{title} first 16:", arr.flatten()[:16])

        return arr

    # -----------------------------------------
    # 4️⃣ 按 TF 规范读取 4 个参数
    # -----------------------------------------
    scale_name = bn_node.input[1]
    beta_name  = bn_node.input[2]
    mean_name  = bn_node.input[3]
    var_name   = bn_node.input[4]

    print("\n===== PB BN params =====")
    gamma = dump_param("gamma(scale)", scale_name)
    beta  = dump_param("beta(offset)", beta_name)
    mean  = dump_param("running_mean", mean_name)
    var   = dump_param("running_var", var_name)

    return {
        "gamma": gamma,
        "beta": beta,
        "mean": mean,
        "var": var,
        "epsilon": epsilon,
        "bn_name": bn_node.name
    }

def dump_pb_first_bn_output(pb_path, input_tensor_name, bn_output_tensor_name, input_shape=(1, 640, 640, 3)):
    """
    pb_path: frozen pb 文件
    input_tensor_name: 例如 'input_images:0'
    bn_output_tensor_name: 例如 'qmcm0:0'
    """
    tf.compat.v1.reset_default_graph()

    # --------------------------------------
    # 1️⃣ load pb
    # --------------------------------------
    with tf.io.gfile.GFile(pb_path, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.compat.v1.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="")

    # --------------------------------------
    # 2️⃣ 拿 tensor
    # --------------------------------------
    x = graph.get_tensor_by_name(input_tensor_name)
    y = graph.get_tensor_by_name(bn_output_tensor_name)

    print("\n===== PB Tensors =====")
    print("Input :", x)
    print("BN out:", y)

    # --------------------------------------
    # 3️⃣ 构造全 1 输入
    # --------------------------------------
    input_data = np.ones(input_shape, dtype=np.float32)

    # --------------------------------------
    # 4️⃣ run
    # --------------------------------------
    with tf.compat.v1.Session(graph=graph) as sess:
        out = sess.run(y, feed_dict={x: input_data})

    # --------------------------------------
    # 5️⃣ 打印（对齐 ONNX）
    # --------------------------------------
    out_flat = out.flatten()

    print("\n===== PB BN Output =====")
    print("shape:", out.shape)
    print("min:", out.min(), "max:", out.max(), "mean:", out.mean())
    print("first 16 values:", out_flat[:16])

    return out

def load_pb(pb_path):
    with tf.gfile.GFile(pb_path, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    graph = tf.Graph()
    with graph.as_default():
        tf.import_graph_def(graph_def, name="")

    return graph

def debug_pb_first_conv():
    PB_PATH = "weights/best.pb"

    # -------------------------------------------------
    # 1️⃣ 构造全 1 输入（NHWC，必须！）
    # -------------------------------------------------
    x = np.ones((1, 640, 640, 3), dtype=np.float32)

    # -------------------------------------------------
    # 2️⃣ 加载 PB
    # -------------------------------------------------
    graph = load_pb(PB_PATH)

    input_x = graph.get_tensor_by_name("input_images:0")

    # -------------------------------------------------
    # 3️⃣ 找到第一个 Conv2D op
    # -------------------------------------------------
    conv_op = None
    for op in graph.get_operations():
        if op.type == "Conv2D":
            conv_op = op
            break

    assert conv_op is not None, "❌ 没找到 Conv2D"

    conv_out = conv_op.outputs[0]   # Conv2D:0

    print("Using Conv op:", conv_op.name)

    # -------------------------------------------------
    # 4️⃣ Session run
    # -------------------------------------------------
    with tf.Session(graph=graph) as sess:
        y = sess.run(conv_out, feed_dict={input_x: x})

    # -------------------------------------------------
    # 5️⃣ 打印（严格按你要求的格式）
    # -------------------------------------------------
    print("===== PB Conv_0 Output =====")
    print("shape:", y.shape)
    print("min:", y.min())
    print("max:", y.max())
    print("mean:", y.mean())
    print("first 16 values:", y.flatten()[:16])

def run_pb_inference_and_dump(
    pb_path,
    input_name="input_images:0",
    output_names=("qmpose.cv3.2/BiasAdd:0", "qmpose.cv2.2/BiasAdd:0", "qmpose.cv4.2/BiasAdd:0"),
    input_shape=(1, 640, 640, 3),
):
    """
    使用 TF1 Session 运行 pb 模型，并打印最终输出结果
    """

    # ===== 1️⃣ 加载 pb =====
    with tf.gfile.GFile(pb_path, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # ===== 2️⃣ 导入 Graph =====
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="")

        input_tensor = graph.get_tensor_by_name(input_name)
        output_tensors = [graph.get_tensor_by_name(n) for n in output_names]

        # ===== 3️⃣ 构造输入（NCHW，全 1）=====
        x = np.ones(input_shape, dtype=np.float32)

        # ===== 4️⃣ 运行推理 =====
        with tf.Session(graph=graph) as sess:
            outputs = sess.run(
                output_tensors,
                feed_dict={input_tensor: x}
            )

    # ===== 5️⃣ 打印结果 =====
    for name, y in zip(output_names, outputs):
        print(f"\n===== PB {name.replace(':0','')} =====")
        print("shape:", y.shape)
        print("min:", float(y.min()))
        print("max:", float(y.max()))
        print("mean:", float(y.mean()))
        print("first 16:", y.flatten()[:16])

    return outputs

def main():
    # === load pb ===
    graph = load_pb(PB_PATH)

    input_x = graph.get_tensor_by_name("input_images:0")
    out_cls  = graph.get_tensor_by_name("qmpose.cv3.2/BiasAdd:0")
    out_box  = graph.get_tensor_by_name("qmpose.cv2.2/BiasAdd:0")
    out_kpts = graph.get_tensor_by_name("qmpose.cv4.2/BiasAdd:0")

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
    run_pb_inference_and_dump(PB_PATH)
    #dump_first_bn_from_pb()
    #dump_pb_first_bn_output(pb_path="weights/best.pb", input_tensor_name="input_images:0", bn_output_tensor_name="Relu:0", input_shape=(1, 640, 640, 3))
    #debug_pb_first_conv()

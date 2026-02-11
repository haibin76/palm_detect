import cv2
import numpy as np
import torch
import onnx
from onnx import helper, TensorProto
from onnx import numpy_helper
import onnxruntime as ort
import onnx.numpy_helper as nh
import torch.nn.functional as F

from dataset.data import letterbox
from postprocess.decode import decode_hand_bundle

# =====================================================
# 配置
# =====================================================
ONNX_PATH = "weights/best.onnx"
IMG_SIZE = 640
CONF_THRESH = 0.5
STRIDE = 32
REG_MAX = 16
NUM_KPTS = 4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def bn_inference_nhwc(
    x,              # shape: [N, H, W, C]
    gamma,          # [C]
    beta,           # [C]
    running_mean,   # [C]
    running_var,    # [C]
    eps=1e-5
):
    assert x.ndim == 4
    C = x.shape[-1]

    gamma = gamma.reshape(1, 1, 1, C)
    beta  = beta.reshape(1, 1, 1, C)
    mean  = running_mean.reshape(1, 1, 1, C)
    var   = running_var.reshape(1, 1, 1, C)

    y = (x - mean) / np.sqrt(var + eps)
    y = y * gamma + beta
    return y

def dump_first_bn_params():
    model = onnx.load(ONNX_PATH)
    graph = model.graph

    print("\n===== FIND FIRST BatchNormalization =====")

    bn_node = None
    for node in graph.node:
        if node.op_type == "BatchNormalization":
            bn_node = node
            break

    assert bn_node is not None, "❌ No BatchNormalization node found"

    print("BN node name:", bn_node.name)
    print("BN inputs :", bn_node.input)
    print("BN outputs:", bn_node.output)

    # -----------------------------------
    # 1️⃣ 读取 epsilon
    # -----------------------------------
    epsilon = None
    for attr in bn_node.attribute:
        if attr.name == "epsilon":
            epsilon = attr.f
    print("\n[BN epsilon]")
    print("epsilon =", epsilon)

    # -----------------------------------
    # 2️⃣ 读取 scale / beta / mean / var
    # -----------------------------------
    init_map = {init.name: init for init in graph.initializer}

    def dump_param(name, tensor_name):
        assert tensor_name in init_map, f"❌ {tensor_name} not found"
        arr = numpy_helper.to_array(init_map[tensor_name])
        print(
            f"{name}: shape={arr.shape}, "
            f"min={arr.min():.6f}, max={arr.max():.6f}, mean={arr.mean():.6f}"
        )
        print(f"{name} first 16:", arr[:16])

    scale_name = bn_node.input[1]
    beta_name  = bn_node.input[2]
    mean_name  = bn_node.input[3]
    var_name   = bn_node.input[4]

    print("\n===== ONNX BN params =====")
    dump_param("gamma(scale)", scale_name)
    dump_param("beta", beta_name)
    dump_param("running_mean", mean_name)
    dump_param("running_var", var_name)

    gamma = numpy_helper.to_array(init_map["qmcm0.bn.weight"])
    beta = numpy_helper.to_array(init_map["qmcm0.bn.bias"])
    mean = numpy_helper.to_array(init_map["qmcm0.bn.running_mean"])
    var = numpy_helper.to_array(init_map["qmcm0.bn.running_var"])

    # BN 节点 epsilon
    eps = 1e-5

    return gamma, beta, mean, var, eps

def dump_onnx_initializers(onnx_path, keyword=None):
    model = onnx.load(onnx_path)
    print("\n===== ONNX Initializers =====")
    for init in model.graph.initializer:
        if keyword is None or keyword in init.name:
            arr = numpy_helper.to_array(init)
            print(
                f"{init.name}: shape={arr.shape}, "
                f"min={arr.min():.6f}, max={arr.max():.6f}, mean={arr.mean():.6f}"
            )

def add_onnx_intermediate_output(src, tensor_name, dst):
    model = onnx.load(src)

    if tensor_name not in [o.name for o in model.graph.output]:
        model.graph.output.append(
            helper.make_tensor_value_info(
                tensor_name,
                onnx.TensorProto.FLOAT,
                None
            )
        )

    onnx.save(model, dst)
    print(f"[OK] add output: {tensor_name}")

def debug_onnx_conv_bn():
    ONNX_PATH = "weights/best.onnx"
    DEBUG_ONNX = "weights/debug_conv_bn.onnx"

    # ====== 你已经确认的节点 ======
    CONV_OUT = "187"   # Conv_0 输出
    BN_OUT   = "188"   # BatchNormalization_1 输出

    # 1️⃣ 打印 Conv / BN 所有权重
    dump_onnx_initializers(ONNX_PATH, keyword="Conv")
    dump_onnx_initializers(ONNX_PATH, keyword="BatchNormalization")

    # 2️⃣ 把 Conv / BN 输出加成 graph.output
    add_onnx_intermediate_output(ONNX_PATH, CONV_OUT, DEBUG_ONNX)
    add_onnx_intermediate_output(DEBUG_ONNX, BN_OUT, DEBUG_ONNX)

    # 3️⃣ 构造全 1 输入（⚠️ ONNX 是 BCHW）
    x = np.ones((1, 3, 640, 640), dtype=np.float32)

    # 4️⃣ 运行
    sess = ort.InferenceSession(DEBUG_ONNX, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    conv_out, bn_out = sess.run([CONV_OUT, BN_OUT], {input_name: x})

    gamma, beta, mean, var, eps = dump_first_bn_params()
    conv_out = np.transpose(conv_out, (0, 2, 3, 1))
    bn_inference_nhwc(conv_out, gamma, beta, mean, var, eps)

    bn_out = np.transpose(bn_out, (0, 2, 3, 1))

    # 5️⃣ 打印统计信息
    print("\n===== ONNX Conv Output =====")
    print("shape:", conv_out.shape)
    print("min:", conv_out.min(), "max:", conv_out.max(), "mean:", conv_out.mean())
    print("first 16 values:", conv_out.flatten()[:16])

    print("\n===== ONNX BN Output =====")
    print("shape:", bn_out.shape)
    print("min:", bn_out.min(), "max:", bn_out.max(), "mean:", bn_out.mean())
    print("first 16 values:", bn_out.flatten()[:16])

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
    debug_onnx_conv_bn()
import torch
import torch.nn as nn
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from tensorflow.python.framework import graph_util
from gen_pb.backbone_ztv2 import *
from models.qm_yolov8 import QMYoloV8
import numpy as np
import onnx
from onnx import numpy_helper

def gen_tf_element():
    # A. 定义输入占位符 (YOLO 标准输入)
    input_x = tf.placeholder(tf.float32, shape=[1, 640, 640, 3], name="input_images")

    # B. 按照 QMYoloV8 的逻辑手动构建图
    # 每一层的 filter_shape: [k, k, in_c, out_c]
    # 注意：这里的命名必须能和后面的赋值逻辑对应上

    #0 self.qmcm0 = QMConv(3, 16, 3, 2, 1, 1, 1)
    qmcm0 = conv_bn_relu(input_x, [3, 3, 3, 16], stride=2, padding='SAME', conv_scope="qmcm0.conv", bn_scope="qmcm0.bn")

    #1 self.qmcm1 = QMConv(16, 32, 3, 2, 1, 1, 1)
    qmcm1 = conv_bn_relu(qmcm0, [3, 3, 16, 32], stride=2, padding='SAME', conv_scope="qmcm1.conv", bn_scope="qmcm1.bn")

    # self.qmcm2 = QMConv(32, 32, 3, 1, 1, 1, 1)
    qmcm2 = conv_bn_relu(qmcm1, [3, 3, 32, 32], stride=1, padding='SAME', conv_scope="qmcm2.conv", bn_scope="qmcm2.bn")

    # self.qmcm3 = QMConv(32, 64, 3, 2, 1, 1, 1)
    qmcm3 = conv_bn_relu(qmcm2, [3, 3, 32, 64], stride=2, padding='SAME', conv_scope="qmcm3.conv", bn_scope="qmcm3.bn")

    # self.qmcm4 = QMConv(64, 64, 3, 1, 1, 1, 1)
    qmcm4 = conv_bn_relu(qmcm3, [3, 3, 64, 64], stride=1, padding='SAME', conv_scope="qmcm4.conv", bn_scope="qmcm4.bn")

    # self.qmcm5 = QMConv(64, 128, 3, 1, 1, 1, 1)
    qmcm5 = conv_bn_relu(qmcm4, [3, 3, 64, 128], stride=1, padding='SAME', conv_scope="qmcm5.conv", bn_scope="qmcm5.bn")

    # self.qmcm6 = QMConv(128, 128, 3, 2, 1, 1, 1)
    qmcm6 = conv_bn_relu(qmcm5, [3, 3, 128, 128], stride=2, padding='SAME', conv_scope="qmcm6.conv", bn_scope="qmcm6.bn")

    # self.qmcm7 = QMConv(128, 128, 3, 1, 1, 1, 1)
    qmcm7 = conv_bn_relu(qmcm6, [3, 3, 128, 128], stride=1, padding='SAME', conv_scope="qmcm7.conv", bn_scope="qmcm7.bn")

    # self.qmcm8 = QMConv(128, 128, 3, 1, 1, 1, 1)
    qmcm8 = conv_bn_relu(qmcm7, [3, 3, 128, 128], stride=1, padding='SAME', conv_scope="qmcm8.conv", bn_scope="qmcm8.bn")

    # self.qmcm9 = QMConv(128, 128, 3, 2, 1, 1, 1)
    qmcm9 = conv_bn_relu(qmcm8, [3, 3, 128, 128], stride=2, padding='SAME', conv_scope="qmcm9.conv", bn_scope="qmcm9.bn")

    # self.qmcm10 = QMConv(128, 128, 3, 1, 1, 1, 1)
    qmcm10 = conv_bn_relu(qmcm9, [3, 3, 128, 128], stride=1, padding='SAME', conv_scope="qmcm10.conv", bn_scope="qmcm10.bn")

    # qmsppf.cv1 = QMConv(128, 64, 1, 1, 0)
    sppf_cv1 = conv_bn_relu(qmcm10, [1, 1, 128, 64], stride=1, padding='SAME', conv_scope="qmsppf.cv1.conv", bn_scope="qmsppf.cv1.bn")

    # qmsppf.conv1 = nn.Conv2d(64, 64, 3, 1, 1, bias=False)
    sppf_conv1 = conv(sppf_cv1, [3, 3, 64, 64], stride=1, padding='SAME', conv_scope="qmsppf.conv1")

    # qmsppf.conv2 = nn.Conv2d(64, 64, 3, 1, 1, bias=False)
    sppf_conv2 = conv(sppf_conv1, [3, 3, 64, 64], stride=1, padding='SAME', conv_scope="qmsppf.conv2")

    # sppf_conv21 = nn.Conv2d(64, 64, 3, 1, 1, bias=False)
    sppf_conv21 = conv(sppf_conv2, [3, 3, 64, 64], stride=1, padding='SAME', conv_scope="qmsppf.conv21")

    # sppf.conv3 = nn.Conv2d(64, 64, 3, 1, 1, bias=False)
    sppf_conv3 = conv(sppf_conv21, [3, 3, 64, 64], stride=1, padding='SAME', conv_scope="qmsppf.conv3")

    # sppf_conv31 = nn.Conv2d(64, 64, 3, 1, 1, bias=False)
    sppf_conv31 = conv(sppf_conv3, [3, 3, 64, 64], stride=1, padding='SAME', conv_scope="qmsppf.conv31")

    # sppf_conv4 = nn.Conv2d(64, 64, 3, 1, 1, bias=False)
    sppf_conv4 = conv(sppf_conv31, [3, 3, 64, 64], stride=1, padding='SAME', conv_scope="qmsppf.conv4")

    # sppf_conv41 = nn.Conv2d(64, 64, 3, 1, 1, bias=False)
    sppf_conv41 = conv(sppf_conv4, [3, 3, 64, 64], stride=1, padding='SAME', conv_scope="qmsppf.conv41")

    # y = [y0, y1, y2, y3]
    # return self.cv2(torch.cat(y, 1))
    sppf_concat = tf.concat([sppf_cv1, sppf_conv21, sppf_conv31, sppf_conv41], axis=-1, name="qmsppf.concat")

    #sppf_cv2 = QMConv(64*4, 128, 1, 1, 0)
    sppf_cv2 = conv_bn_relu(sppf_concat, [1, 1, 256, 128], stride=1, padding='SAME', conv_scope="qmsppf.cv2.conv", bn_scope="qmsppf.cv2.bn")

    #############################################################################################################
    ##########################################QMPOSE.cv3#########################################################
    ###################QMConv(128, 128, 3, 1, 1), QMConv(128, 128, 3, 1, 1), nn.Conv2d(128, 1, 1)
    pose_cv3_0 = conv_bn_relu(sppf_cv2, [3, 3, 128, 128], stride=1, padding='SAME', conv_scope="qmpose.cv3.0.conv", bn_scope="qmpose.cv3.0.bn")
    pose_cv3_1 = conv_bn_relu(pose_cv3_0, [3, 3, 128, 128], stride=1, padding='SAME', conv_scope="qmpose.cv3.1.conv", bn_scope="qmpose.cv3.1.bn")
    pose_cv3_2 = conv_ba(pose_cv3_1, [1, 1, 128, 1], stride=1, padding='SAME', conv_scope="qmpose.cv3.2")

    #############################################################################################################
    ##########################################QMPOSE.cv2#########################################################
    ###################QMConv(128, 64, 3, 1, 1), QMConv(64, 64, 3, 1, 1), nn.Conv2d(64, 64, 1, 1)
    pose_cv2_0 = conv_bn_relu(sppf_cv2, [3, 3, 128, 64], stride=1, padding='SAME', conv_scope="qmpose.cv2.0.conv", bn_scope="qmpose.cv2.0.bn")
    pose_cv2_1 = conv_bn_relu(pose_cv2_0, [3, 3, 64, 64], stride=1, padding='SAME', conv_scope="qmpose.cv2.1.conv", bn_scope="qmpose.cv2.1.bn")
    pose_cv2_2 = conv_ba(pose_cv2_1, [1, 1, 64, 64], stride=1, padding='SAME', conv_scope="qmpose.cv2.2")

    #############################################################################################################
    ##########################################QMPOSE.cv4#########################################################
    ###################QMConv(128, 32, 3, 1, 1), QMConv(32, 32, 3, 1, 1), nn.Conv2d(32, 12, 1, 1)
    pose_cv4_0 = conv_bn_relu(sppf_cv2, [3, 3, 128, 32], stride=1, padding='SAME', conv_scope="qmpose.cv4.0.conv", bn_scope="qmpose.cv4.0.bn")
    pose_cv4_1 = conv_bn_relu(pose_cv4_0, [3, 3, 32, 32], stride=1, padding='SAME', conv_scope="qmpose.cv4.1.conv", bn_scope="qmpose.cv4.1.bn")
    pose_cv4_2 = conv_ba(pose_cv4_1, [1, 1, 32, 12], stride=1, padding='SAME', conv_scope="qmpose.cv4.2")

    # ============================================================
    # 输出节点（⚠️ 关键）
    # ============================================================
    out_cls  = tf.identity(pose_cv3_2, name="output_cls")
    out_box  = tf.identity(pose_cv2_2, name="output_box")
    out_kpts = tf.identity(pose_cv4_2, name="output_kpts")

    return input_x, out_cls, out_box, out_kpts

def pt_2_onnx(pt_file, onnx_file):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = QMYoloV8().to(device)
    #model.load_state_dict(torch.load(pt_file, device, weights_only=True))
    model.load_state_dict(torch.load(pt_file, device))

    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            module.eval()
            print("BatchNorm2d")

    example_input = torch.randn(1, 3, 640, 640).to(device)
    torch.onnx.export(model, example_input, onnx_file,
                      input_names=["input_images"], output_names=["output_cls", "output_box", "output_kpts"],
                      do_constant_folding=False,  opset_version=12, verbose=True, training=torch.onnx.TrainingMode.PRESERVE)
    #现在生成了两个文件，一个是onnx，一个是onnx.data
    #onnx_model = onnx.load("weights/temp.onnx")

    # 保存为一个独立文件（不使用外部数据）
    #onnx.save_model(onnx_model, onnx_file, save_as_external_data=False)
    print(f"✅ 已成功合并为单文件： {onnx_file}")

def load_onnx_weights(onnx_path):
    model = onnx.load(onnx_path)
    weight_map = {}

    for init in model.graph.initializer:
        weight_map[init.name] = numpy_helper.to_array(init)

    return weight_map

#列出 TF 中所有可赋值变量
def get_tf_var_map():
    var_map = {}

    for v in tf.global_variables():
        # 原始 TF 名字: qmcm0.conv/weight
        tf_name = v.name.split(":")[0]

        # ONNX 风格名字: qmcm0.conv.weight
        onnx_like_name = tf_name.replace("/", ".")

        # 两种都存
        #var_map[tf_name] = v
        var_map[onnx_like_name] = v

    return var_map

def onnx_conv_to_tf(w):
    return np.transpose(w, (2, 3, 1, 0))

def assign_onnx_to_tf(sess, onnx_weights):
    tf_vars = get_tf_var_map()

    assign_ops = []
    assigned_tf_names = set()

    print("\n========== Assign ONNX → TF ==========")

    for onnx_name, onnx_value in onnx_weights.items():

        if onnx_name not in tf_vars:
            print(f"[ONNX SKIP] {onnx_name} (no tf var)")
            continue

        tf_var = tf_vars[onnx_name]
        tf_shape = tf_var.shape.as_list()

        # Conv 权值转置 (OIHW -> HWIO)
        if len(onnx_value.shape) == 4:
            onnx_value = onnx_conv_to_tf(onnx_value)

        # shape 校验（非常关键）
        if list(onnx_value.shape) != tf_shape:
            raise ValueError(
                f"[SHAPE ERROR] {onnx_name} "
                f"onnx={onnx_value.shape} tf={tf_shape}"
            )

        print(f"[LOAD] {onnx_name} {onnx_value.shape}")
        assign_ops.append(tf_var.assign(onnx_value))
        assigned_tf_names.add(tf_var.name.split(":")[0])

    sess.run(assign_ops)

    # ============================================================
    # 🔍 检查 TF 中是否有未被赋值的变量
    # ============================================================
    print("\n========== Check TF unassigned vars ==========")

    all_tf_vars = tf.global_variables()
    unassigned = []

    for v in all_tf_vars:
        name = v.name.split(":")[0]
        if name not in assigned_tf_names:
            unassigned.append((name, v.shape.as_list()))

    if len(unassigned) == 0:
        print("✅ All TF variables are assigned from ONNX")
    else:
        print(f"❌ Found {len(unassigned)} TF vars NOT assigned:")
        for name, shape in unassigned:
            print(f"  [MISS] {name} {shape}")

    print("==============================================\n")

def build_tf_var_dict():
    tf_vars = tf.global_variables()
    var_dict = {}
    for v in tf_vars:
        # v.name 示例: qmsppf.cv1.conv/weight:0
        clean_name = v.name.replace(":0", "")
        var_dict[clean_name] = v
    return var_dict


def gen_pb_file(onnx_file, pb_file):
    # 1. 重置默认图（防止多次运行产生冗余节点）
    tf.reset_default_graph()

    # 2. 构建图结构
    input_x, out_cls, out_box, out_kpts = gen_tf_element()

    # 3. 创建 Session 并配置（防止 GPU 占用报错）
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    # 4. 初始化所有变量
    sess.run(tf.global_variables_initializer())

    # 5. 🔥 灌入 ONNX 权重
    print("🚀 Loading ONNX weights...")
    onnx_weights = load_onnx_weights(onnx_file)
    assign_onnx_to_tf(sess, onnx_weights)

    # 6. 🔍 Forward 验证（确认赋值后的数值是否正常）
    dummy = np.zeros([1, 640, 640, 3], np.float32)
    cls_v, box_v, kpts_v = sess.run(
        [out_cls, out_box, out_kpts],
        feed_dict={input_x: dummy}
    )
    print(f"Forward check shapes: {cls_v.shape}, {box_v.shape}, {kpts_v.shape}")

    # 7. ❄️ Freeze PB (核心修复逻辑)
    print("❄️ Freezing graph...")

    # 获取输出节点名，确保不带 :0 后缀
    output_node_names = ["output_cls", "output_box", "output_kpts"]

    # [关键修复 A]: 提取图定义
    input_graph_def = sess.graph.as_graph_def()

    # [关键修复 B]: 将变量转为常量。这会将 Variable + Read 节点合并为单个 Const 节点
    frozen_graph_def = graph_util.convert_variables_to_constants(
        sess,
        input_graph_def,
        output_node_names
    )

    # [关键修复 C]: 彻底移除 ReadVariableOp 和 Identity 等推理不需要的节点
    # 这是解决 "has no attr named value" 的终极方案
    frozen_graph_def = graph_util.remove_training_nodes(frozen_graph_def)

    # 8. 保存最终文件
    with tf.gfile.GFile(pb_file, "wb") as f:
        f.write(frozen_graph_def.SerializeToString())

    # 9. 打印节点确认（调试用）
    const_count = len([n for n in frozen_graph_def.node if n.op == 'Const'])
    print(f"✅ PB generated! Total Const nodes: {const_count}")
    print(f"✅ Final PB saved at: {pb_file}")

    sess.close()
import torch
import torch.nn as nn
import onnx
from onnx_tf.backend import prepare
from models.qm_yolov8 import QMYoloV8
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

def pt2onnx(pt_file, onnx_file):
    # 1. 加载模型
    model = QMYoloV8()
    model.load_state_dict(torch.load(pt_file, map_location="cpu"), strict=False)
    model.eval()

    # 2. 准备数据
    dummy_input = torch.randn(1, 3, 640, 640)

    # 3. 关键：禁用一切新版导出器特性
    # 强制使用旧版的算子追踪逻辑
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            onnx_file,
            export_params=True,
            opset_version=13,  # 必须是13，为了后续转PB
            do_constant_folding=True,
            input_names=['images'],
            output_names=['class_id', 'boxes', 'keypoints'],
            # 核心设置：避免触发 torch.export 的各种 Capture 策略
            training=torch.onnx.TrainingMode.EVAL,
            # 如果还是报错，尝试加入下面这行（仅限特定版本PT）
            # operator_export_type=torch.onnx.OperatorExportTypes.ONNX
        )

    print("ONNX 导出成功！")

def merge_onnx(onnx_file, onnx_output_file):
    #加载时它会自动寻找同目录下的.onnx.data
    model = onnx.load(onnx_file)
    # 保存为一个不带外部数据的文件（如果权重没超过 2GB，这样最稳妥）
    onnx.save_model(model, onnx_output_file, save_as_external_data=False)

def onnx2pb(onnx_file, pb_folder):
    onnx_model = onnx.load(onnx_file)
    tf_rep = prepare(onnx_model)

    # 导出为 SavedModel (包含 .pb 文件)
    tf_rep.export_graph(pb_folder)
    print("PB 模型已生成在 best_pb_model 文件夹中")


def freeze_saved_model(saved_model_path, output_pb_name):
    # 1. 加载模型
    model = tf.saved_model.load(saved_model_path)

    # 2. 获取推理签名 (onnx-tf 导出的模型必须指定签名)
    infer = model.signatures['serving_default']

    # 3. 将推理函数转换为冻结图 (直接处理 ConcreteFunction)
    # 这步会自动将变量（variables）转为常量并嵌入图中
    frozen_func = convert_variables_to_constants_v2(infer)
    frozen_func.graph.as_graph_def()

    # 4. 保存为单一 PB 文件
    # 这一步会生成你想要的那个“独立大PB”
    tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                      logdir=".",
                      name=output_pb_name,
                      as_text=False)

    print(f"✅ 独立的单文件 PB 已生成: {output_pb_name}")

    # 打印输入输出节点名，方便你后续调用
    print(f"输入节点: {[input.name for input in frozen_func.inputs]}")
    print(f"输出节点: {[output.name for output in frozen_func.outputs]}")

if __name__ == "__main__":
    pt2onnx("weights/best.pt", "weights/best.onnx")
    #merge_onnx("weights/best.onnx", "weights/best_single.onnx")
    onnx2pb("weights/best.onnx", "weights/best")

    freeze_saved_model("weights/best", "weights/frozen_model.pb")
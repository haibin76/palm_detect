import torch
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import graph_util
from models.qm_yolov8 import QMYoloV8
from gen_pb.gen_conv import *

def probe_model_structure(model, input_size=(1, 3, 640, 640)):
    """
    不仅打印权重，还打印动态的 Input/Output Shape, Kernel 和 Stride
    """
    print(f"\n{'=' * 120}")
    header = f"{'Index':<5} | {'Layer Name':<30} | {'Input Shape':<18} | {'Output Shape':<18} | {'K':<5} | {'S':<5} | {'Params':<10}"
    print(header)
    print("-" * 120)

    hooks = []
    info = []

    def hook_fn(module, input, output, name, idx):
        # 获取卷积特有属性
        k = getattr(module, 'kernel_size', '-')
        s = getattr(module, 'stride', '-')

        # 记录信息
        info.append({
            'idx': idx,
            'name': name,
            'in': list(input[0].shape),
            'out': list(output.shape) if hasattr(output, 'shape') else "N/A",
            'k': k,
            's': s,
            'params': sum(p.numel() for p in module.parameters())
        })

    # 1. 注册 Hook (只针对最底层的算子)
    idx = 0
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:
            hooks.append(module.register_forward_hook(
                lambda m, i, o, n=name, id=idx: hook_fn(m, i, o, n, id)
            ))
            idx += 1

    # 2. 跑一次前向传播
    model.eval()
    with torch.no_grad():
        try:
            dummy_in = torch.randn(*input_size)
            model(dummy_in)
        except Exception as e:
            print(f"前向传播失败，无法获取Shape: {e}")

    # 3. 打印结果
    for m in info:
        print(
            f"{m['idx']:<5} | {m['name']:<30} | {str(m['in']):<18} | {str(m['out']):<18} | {str(m['k']):<5} | {str(m['s']):<5} | {m['params']:<10}")

    # 4. 移除 Hook
    for h in hooks:
        h.remove()
    print("-" * 120)

def probe_model(pt_path):
    model = QMYoloV8()
    model.load_state_dict(torch.load(pt_path, map_location="cpu"))

    # 暂用一个模拟模型演示
    probe_model_structure(model)

# --- 3. 主函数 ---
def main():
    pt_file = "weights/best.pt"
    onnx_file = "weights/best.onnx"
    pb_file = "weights/best.pb"

    pt_2_onnx(pt_file, onnx_file)

    gen_pb_file(onnx_file, pb_file)
    inspect_pb(pb_file)

if __name__ == "__main__":
    main()
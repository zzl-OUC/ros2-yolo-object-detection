"""
train.py - ros2-yolo-object-detection 训练脚本 (Ultralytics YOLOv8)

用法:
  pip install ultralytics torch  (有 NVIDIA GPU 装 CUDA 版 torch)
  python train.py

模型尺寸选择 (改下面 MODEL 即可, 前缀 yolo8/yolo10/yolo11 都支持):
  yolo11s.pt  推荐: 比 v8 同尺寸精度更高、推理更快 (需较新 ultralytics)
  yolo10s.pt  也可, 精度与 v11 接近
  yolo8s.pt   保守稳定版, 文档/社区最多
  *n/m/l/x 同理: 小数据集用 s 够用, m 冲精度, l/x 易过拟合不推荐
"""
import torch
from ultralytics import YOLO

# 自动用 GPU
DEVICE = "0" if torch.cuda.is_available() else "cpu"

# 默认用 yolo11s (见上方说明)。想对比可改成 yolo8s / yolo10s / yolo11m 重跑。
MODEL = "yolo11s.pt"

if __name__ == "__main__":
    model = YOLO(MODEL)

    # ---- 训练 ----
    results = model.train(
        data="dataset/data.yaml",
        epochs=100,
        imgsz=640,
        batch=16,
        device=DEVICE,
        name="ros2-yolo-s",
        project="runs/train",
        patience=30,          # 验证指标 30 轮不提升则早停 (防过拟合)
        cos_lr=True,          # 余弦学习率
        # ---- 数据增强 (小数据集关键, 防过拟合) ----
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
    )

    # ---- 验证 ----
    metrics = model.val(data="dataset/data.yaml", device=DEVICE)
    print(f"[评估] mAP50-95 = {metrics.box.map:.4f}   mAP50 = {metrics.box.map50:.4f}")

    # ---- 导出 ONNX (便于部署/推理, 可选) ----
    model.export(format="onnx")
    print("[完成] 训练产物在 runs/train/ros2-yolo-s/ , ONNX 已导出")

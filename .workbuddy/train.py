import torch
from ultralytics import YOLO

DEVICE = "0" if torch.cuda.is_available() else "cpu"

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
        patience=30,          # 验证指标
        cos_lr=True,          # 余弦学习率
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
    )

    # ---- 验证 ----
    metrics = model.val(data="dataset/data.yaml", device=DEVICE)
    print(f"[评估] mAP50-95 = {metrics.box.map:.4f}   mAP50 = {metrics.box.map50:.4f}")

    model.export(format="onnx")
    print("[完成] 训练产物在 runs/train/ros2-yolo-s/ , ONNX 已导出")

"""实验一 实例分割训练入口: 在 seg/ 数据集上训练 YOLO11s-seg 并导出 ONNX。"""
import torch
from ultralytics import YOLO

DEVICE = "0" if torch.cuda.is_available() else "cpu"
CFG = dict(
    data="seg/data.yaml", epochs=100, imgsz=640, batch=16, device=DEVICE,
    name="ros2-yolo-seg", project="runs/seg", patience=30, cos_lr=True,
    hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, fliplr=0.5, mosaic=1.0, mixup=0.1,
)


def main():
    net = YOLO("yolo11s-seg.pt")
    net.train(**CFG)
    m = net.val(data=CFG["data"], device=DEVICE)
    print(f"[评估] mask mAP50-95={m.seg.map:.4f}  box mAP50={m.box.map50:.4f}")
    net.export(format="onnx")
    print("[完成] 产物见 runs/seg/ros2-yolo-seg/, ONNX 已导出")


if __name__ == "__main__":
    main()

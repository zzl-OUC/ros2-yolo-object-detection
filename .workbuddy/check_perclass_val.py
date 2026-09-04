#!/usr/bin/env python3
# 拿验证集逐类 AP（workers=0 避免 stdin 多进程崩溃）
from ultralytics import YOLO

m = YOLO("weights/best.pt")
metrics = m.val(data="dataset/data.yaml", verbose=False, workers=0, device=0)
mp = metrics.box.maps  # per-class mAP50-95
names = m.names
print(f"整体 mAP50    = {metrics.box.map50:.3f}")
print(f"整体 mAP50-95 = {metrics.box.map:.3f}")
for i, n in names.items():
    print(f"  class {i} ({n:6s}) mAP50-95 = {mp[i]:.3f}")

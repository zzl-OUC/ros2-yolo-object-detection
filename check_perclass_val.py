"""验证集逐类 AP 评估 (workers=0 规避 stdin 多进程崩溃)。"""
from ultralytics import YOLO

net = YOLO("weights/best.pt")
m = net.val(data="dataset/data.yaml", verbose=False, workers=0, device=0)
per = m.box.maps
print(f"mAP50    = {m.box.map50:.3f}")
print(f"mAP50-95 = {m.box.map:.3f}")
for cid, name in m.names.items():
    print(f"  class {cid} ({name}) mAP50-95 = {per[cid]:.3f}")

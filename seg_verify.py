"""抽查 GrabCut 生成的分割轮廓质量: 在原图上叠加多边形, 存 PNG 供肉眼核对。"""
import glob
import os

import cv2
import numpy as np

SEG = "seg"
OUT = "seg_verify"
N = 8


def main():
    os.makedirs(OUT, exist_ok=True)
    done = 0
    for split in ("train", "val"):
        for img_p in sorted(glob.glob(os.path.join(SEG, "images", split, "**", "*.jpg"), recursive=True)):
            rel = os.path.relpath(img_p, os.path.join(SEG, "images", split))
            lbl_p = os.path.join(SEG, "labels", split, os.path.splitext(rel)[0] + ".txt")
            if not os.path.isfile(lbl_p):
                continue
            img = cv2.imread(img_p)
            h, w = img.shape[:2]
            overlay = img.copy()
            with open(lbl_p) as f:
                for line in f:
                    p = line.split()
                    if len(p) < 7:
                        continue
                    cls = p[0]
                    pts = np.array([[float(p[i]) * w, float(p[i + 1]) * h]
                                    for i in range(1, len(p), 2)], np.int32)
                    cv2.polylines(overlay, [pts], True, (0, 255, 0), 2)
                    m = {"0": "mouse", "1": "phone"}.get(cls, cls)
                    cv2.putText(overlay, m, pts[0], cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            out_p = os.path.join(OUT, rel.replace("\\", "_").replace("/", "_"))
            cv2.imwrite(out_p, overlay)
            done += 1
            if done >= N:
                break
        if done >= N:
            break
    print(f"[完成] 叠加图 {done} 张 -> {OUT}/")


if __name__ == "__main__":
    main()

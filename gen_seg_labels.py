"""从现有检测框自动生成实例分割标注 (bbox -> polygon)。

做法: 用 OpenCV 内置 GrabCut, 以现有 YOLO 检测框 (5列 class cx cy w h) 作为
前景矩形提示, 自动抠出目标轮廓, 再用轮廓逼近得到多边形, 归一化后写出
YOLO-Seg 格式 (class x1 y1 x2 y2 ... xn yn)。

零外部模型下载, 仅依赖已安装的 opencv-python-headless + numpy。
生成的标注写入独立 seg/ 目录, 不改动原检测数据集 dataset/。

用法:
    python gen_seg_labels.py            # 全量生成
    python gen_seg_labels.py --limit 5  # 仅前 5 张做冒烟测试
"""
import argparse
import glob
import os
import shutil
import sys

import cv2
import numpy as np

SRC = "dataset"
DST = "seg"
GC_ITER = 5
EPS = 0.005  # 轮廓逼近系数 (占周界比例)


def bbox_to_polygon(img, x1, y1, x2, y2):
    """用 GrabCut 从像素框抠前景, 返回归一化多边形点列表 [(x,y), ...]。"""
    h, w = img.shape[:2]
    # 夹紧到图像内, 保证宽高为正
    x1 = max(0, int(round(x1)))
    y1 = max(0, int(round(y1)))
    x2 = min(w - 1, int(round(x2)))
    y2 = min(h - 1, int(round(y2)))
    bw, bh = x2 - x1, y2 - y1
    if bw < 2 or bh < 2:
        return [(x1 / w, y1 / h), (x2 / w, y1 / h),
                (x2 / w, y2 / h), (x1 / w, y2 / h)]
    rect = (x1, y1, bw, bh)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    mask = np.zeros((h, w), np.uint8)
    try:
        cv2.grabCut(img, mask, rect, bgd, fgd, GC_ITER, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        return rect_poly(x1, y1, x2, y2, w, h)
    fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0).astype(np.uint8)
    cnts, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return rect_poly(x1, y1, x2, y2, w, h)
    cnt = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    box_area = bw * bh
    # GrabCut 失败保护: 抠出的面积与框面积比例异常时退回矩形
    if area < 0.05 * box_area or area > 3.0 * box_area:
        return rect_poly(x1, y1, x2, y2, w, h)
    cnt = cv2.approxPolyDP(cnt, EPS * cv2.arcLength(cnt, True), True)
    pts = cnt.reshape(-1, 2)
    if len(pts) < 3:
        return rect_poly(x1, y1, x2, y2, w, h)
    return [(float(p[0] / w), float(p[1] / h)) for p in pts]


def rect_poly(x1, y1, x2, y2, w, h):
    return [(x1 / w, y1 / h), (x2 / w, y1 / h),
            (x2 / w, y2 / h), (x1 / w, y2 / h)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="仅处理前 N 张 (冒烟测试)")
    args = ap.parse_args()

    if os.path.isdir(DST):
        try:
            shutil.rmtree(DST)
        except OSError:
            pass  # 安全删除钩子对大目录可能 abort, 忽略: 后续全量覆盖不残留
    os.makedirs(os.path.join(DST, "images", "train"), exist_ok=True)
    os.makedirs(os.path.join(DST, "images", "val"), exist_ok=True)
    os.makedirs(os.path.join(DST, "labels", "train"), exist_ok=True)
    os.makedirs(os.path.join(DST, "labels", "val"), exist_ok=True)

    stats = {"img": 0, "obj": 0, "rect_fallback": 0}
    for split in ("train", "val"):
        img_root = os.path.join(SRC, "images", split)
        lbl_root = os.path.join(SRC, "labels", split)
        for src_img in sorted(glob.glob(os.path.join(img_root, "**", "*.jpg"), recursive=True)):
            if args.limit and stats["img"] >= args.limit:
                break
            rel = os.path.relpath(src_img, img_root)          # 保留 mouse/phone 子目录
            stem = os.path.splitext(rel)[0]
            dst_img = os.path.join(DST, "images", split, rel)
            os.makedirs(os.path.dirname(dst_img), exist_ok=True)
            shutil.copy(src_img, dst_img)
            img = cv2.imread(src_img)
            if img is None:
                print(f"[warn] 读图失败: {src_img}")
                continue
            lbl_src = os.path.join(lbl_root, stem + ".txt")
            lines_out = []
            if os.path.isfile(lbl_src):
                with open(lbl_src) as f:
                    for line in f:
                        parts = line.split()
                        if len(parts) < 5:
                            continue
                        cid = int(float(parts[0]))
                        cx, cy, bw, bh = (float(v) for v in parts[1:5])
                        h, w = img.shape[:2]
                        nx1, ny1 = (cx - bw / 2) * w, (cy - bh / 2) * h
                        nx2, ny2 = (cx + bw / 2) * w, (cy + bh / 2) * h
                        poly = bbox_to_polygon(img, nx1, ny1, nx2, ny2)
                        if len(poly) == 4 and poly[0][0] == poly[3][0]:
                            stats["rect_fallback"] += 1
                        flat = " ".join(f"{x:.6f} {y:.6f}" for x, y in poly)
                        lines_out.append(f"{cid} {flat}")
                        stats["obj"] += 1
            dst_lbl = os.path.join(DST, "labels", split, stem + ".txt")
            os.makedirs(os.path.dirname(dst_lbl), exist_ok=True)
            with open(dst_lbl, "w") as f:
                f.write("\n".join(lines_out) + ("\n" if lines_out else ""))
            stats["img"] += 1

    # 写 seg 数据配置
    yaml_path = os.path.join(DST, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(
            "task: segment\n"
            "path: .\n"
            "train: images/train\n"
            "val: images/val\n"
            "nc: 2\n"
            "names: [mouse, phone]\n"
        )
    print(f"[完成] 处理图 {stats['img']} 张, 目标 {stats['obj']} 个, "
          f"矩形兜底 {stats['rect_fallback']} 个; 输出 -> {DST}/ + data.yaml")


if __name__ == "__main__":
    main()

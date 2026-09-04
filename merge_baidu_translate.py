#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 D:\百度翻译 (Roboflow, cell phone/phone) 合并到 ros2-yolo 数据集:
  - 类别对齐: class 0(cell phone) + class 1(phone) -> 统一为 ros2 的 phone(id=1)
  - 拆分映射: train -> train, valid -> val, test -> train (最大化训练量)
  - 结构对齐: ros2 用 images/{train,val}/{mouse,phone}/, 标签放对应子目录
  - 清洗: 丢弃退化框 (w<0.01 或 h<0.01, 几乎为零的标注)
  - 去重: 目标文件已存在则跳过 (幂等)
  - 名字: 保留 Roboflow 原名 IMG_xxx.rf.xxx (与现有 phone_NN 不冲突, 且可追溯)
"""
import os, shutil, glob, sys

SRC = r"D:/百度翻译"
DST_IMG = r"C:/Users/13907/ros2-yolo-object-detection/dataset/images"
DST_LBL = r"C:/Users/13907/ros2-yolo-object-detection/dataset/labels"

mapping = {"train": "train", "valid": "val", "test": "train"}

stats = {"img_copied": 0, "lbl_written": 0, "boxes_kept": 0,
         "boxes_dropped_degenerate": 0, "skipped_existing": 0,
         "images_no_label_or_vice": 0}

for src_split, dst_split in mapping.items():
    src_img = os.path.join(SRC, src_split, "images")
    src_lbl = os.path.join(SRC, src_split, "labels")
    if not os.path.isdir(src_lbl):
        continue
    tgt_img = os.path.join(DST_IMG, dst_split, "phone")
    tgt_lbl = os.path.join(DST_LBL, dst_split, "phone")
    os.makedirs(tgt_img, exist_ok=True)
    os.makedirs(tgt_lbl, exist_ok=True)

    for lf in sorted(glob.glob(os.path.join(src_lbl, "*.txt"))):
        base = os.path.splitext(os.path.basename(lf))[0]
        img_src = os.path.join(src_img, base + ".jpg")
        if not os.path.exists(img_src):
            stats["images_no_label_or_vice"] += 1
            continue
        # 读+重写: 0/1 -> 1, 过滤退化框
        new_lines = []
        with open(lf, encoding="utf-8") as f:
            for line in f:
                p = line.strip().split()
                if len(p) < 5:
                    continue
                try:
                    x, y, w, h = map(float, p[1:5])
                except ValueError:
                    continue
                if w < 0.01 or h < 0.01:
                    stats["boxes_dropped_degenerate"] += 1
                    continue
                new_lines.append(f"1 {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
                stats["boxes_kept"] += 1
        if not new_lines:
            continue  # 全是退化框, 整张跳过
        ti = os.path.join(tgt_img, base + ".jpg")
        tl = os.path.join(tgt_lbl, base + ".txt")
        if os.path.exists(ti):
            stats["skipped_existing"] += 1
            continue
        shutil.copy2(img_src, ti)
        stats["img_copied"] += 1
        with open(tl, "w", encoding="utf-8") as f:
            f.writelines(new_lines)
        stats["lbl_written"] += 1

print("=== merge 完成 ===")
for k, v in stats.items():
    print(f"  {k}: {v}")

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_accuracy.py - 20 物体识别率测试 (验收要求: 测试 20 个物体, 正确识别率 >= 80%)

对应验收要求:
  1. 测试 20 个物体并统计正确识别率
  2. 保存测试结果 (results.csv + annotated/ 标注图)
  3. 保存典型错误案例 (errors/ 目录, 自动归类识别错的样本)
  4. 顺带输出单帧推理耗时与折算 FPS, 辅助验证 ">=5 FPS" 指标

用法 (Jetson 上):
  1. 拍 20 张测试照片放进 test_images/, 文件名以真实类别开头, 例如:
       mouse_01.jpg  mouse_02.jpg ... phone_01.jpg ...
     (类别名必须与模型类别一致: mouse / phone;
      也可用 --gt ground_truth.csv 指定, 格式: 文件名,期望类别)
  2. python3 test_accuracy.py --model best.engine
  3. 结果输出在 test_results/: summary.txt + results.csv + annotated/ + errors/
"""
import argparse
import csv
import os
import shutil
import time

import cv2

from ultralytics import YOLO

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def parse_args():
    p = argparse.ArgumentParser(description="20 物体识别率测试")
    p.add_argument("--model", default=None, help="模型路径; 默认自动找 best.engine / best.pt")
    p.add_argument("--images", default="test_images", help="测试图片目录")
    p.add_argument("--gt", default=None, help="可选: ground_truth.csv (每行: 文件名,期望类别)")
    p.add_argument("--conf", type=float, default=0.40, help="置信度阈值")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--out", default="test_results", help="结果输出目录")
    return p.parse_args()


def resolve_model(path):
    if path:
        return path
    here = os.path.dirname(os.path.abspath(__file__))
    for cand in [os.path.join(here, "best.engine"), os.path.join(here, "best.pt"),
                 "best.engine", "best.pt"]:
        if os.path.exists(cand):
            return cand
    raise SystemExit("未找到模型文件, 请用 --model 指定 best.pt / best.engine")


def load_ground_truth(path):
    gt = {}
    if path:
        with open(path, encoding="utf-8") as f:
            for row in csv.reader(f):
                if len(row) >= 2 and row[0].strip():
                    gt[row[0].strip()] = row[1].strip()
    return gt


def expected_class(img_path, gt, names):
    if img_path in gt or os.path.basename(img_path) in gt:
        return gt.get(img_path) or gt.get(os.path.basename(img_path))
    stem = os.path.splitext(os.path.basename(img_path))[0]
    guess = stem.split("_")[0]
    # model.names 形如 {0:'mouse', 1:'phone'}: 键是类别 id, 类别名在 values 里
    valid = set(names.values()) if isinstance(names, dict) else set(names)
    return guess if guess in valid else None


def main():
    args = parse_args()
    model = YOLO(resolve_model(args.model))
    try:
        names = model.names
    except AttributeError:
        # 新版 ultralytics 纯 engine/onnx 加载不暴露 names (类别名存于 best.pt 元数据);
        # 同目录的 best.pt 仅用于读取 {id: 类别名} 映射, 推理仍走 engine
        here = os.path.dirname(os.path.abspath(__file__)) or "."
        for cand in (os.path.join(here, "best.pt"), "best.pt"):
            if os.path.exists(cand):
                names = YOLO(cand).names
                break
        else:
            raise SystemExit("无法获取类别名: 请把 best.pt 放到脚本同目录后再跑")

    files = sorted(os.path.join(args.images, f) for f in os.listdir(args.images)
                   if f.lower().endswith(IMG_EXTS))
    if not files:
        raise SystemExit(f"{args.images}/ 里没有测试图片, 请先拍 20 张照片放进去")
    gt = load_ground_truth(args.gt)

    dirs = {k: os.path.join(args.out, k) for k in ("annotated", "errors")}
    for d in dirs.values():
        if os.path.isdir(d):
            shutil.rmtree(d)          # 每次运行清空旧结果, 避免历史残留叠加
        os.makedirs(d, exist_ok=True)

    rows, skipped = [], []
    for img_path in files:
        expected = expected_class(img_path, gt, names)
        if expected is None:
            skipped.append(os.path.basename(img_path))
            continue

        img = cv2.imread(img_path)
        t0 = time.perf_counter()
        result = model.predict(img, imgsz=args.imgsz, conf=args.conf, verbose=False)[0]
        ms = (time.perf_counter() - t0) * 1000

        boxes = result.boxes
        if len(boxes):
            i = int(boxes.conf.argmax())
            pred_id = int(boxes.cls[i])
            pred, conf = names[pred_id], float(boxes.conf[i])
        else:
            pred, conf = "(未检出)", 0.0
        correct = pred == expected

        annotated_path = os.path.join(dirs["annotated"], os.path.basename(img_path))
        cv2.imwrite(annotated_path, result.plot())

        rows.append({"文件名": os.path.basename(img_path), "期望类别": expected,
                     "识别类别": pred, "置信度": f"{conf:.3f}",
                     "推理耗时ms": f"{ms:.0f}", "是否正确": "是" if correct else "否"})
        if not correct:  # 典型错误案例: 识别错/未检出的留档
            shutil.copy(annotated_path, os.path.join(dirs["errors"], os.path.basename(img_path)))
        print(f"[{'OK ' if correct else 'ERR'}] {os.path.basename(img_path):24s} "
              f"期望={expected:8s} 识别={pred:8s} conf={conf:.2f} ({ms:.0f}ms)")

    if skipped:
        print("\n以下文件名无法推断期望类别且不在 ground_truth.csv 中, 已跳过:")
        for f in skipped:
            print("  -", f)

    total = len(rows)
    correct_n = sum(1 for r in rows if r["是否正确"] == "是")
    acc = correct_n / total * 100 if total else 0.0
    avg_ms = sum(float(r["推理耗时ms"]) for r in rows) / total if total else 0.0

    per_class = {}
    for r in rows:
        stat = per_class.setdefault(r["期望类别"], [0, 0])
        stat[0] += 1
        stat[1] += r["是否正确"] == "是"

    lines = [
        f"测试总数: {total}    正确: {correct_n}    识别率: {acc:.1f}%    "
        f"要求 >=80%: {'通过' if acc >= 80 else '未通过'}",
        f"平均单帧推理: {avg_ms:.0f} ms    折算约 {1000 / avg_ms:.1f} FPS" if total else "",
        "分类别: " + "  ".join(f"{k} {v[1]}/{v[0]}" for k, v in sorted(per_class.items())),
        f"错误案例已保存到: {dirs['errors']}",
    ]
    print("\n" + "=" * 60)
    for ln in lines:
        if ln:
            print(ln)

    with open(os.path.join(args.out, "results.csv"), "w", newline="",
              encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with open(os.path.join(args.out, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n结果已保存到 {args.out}/ (results.csv, summary.txt, annotated/, errors/)")


if __name__ == "__main__":
    main()

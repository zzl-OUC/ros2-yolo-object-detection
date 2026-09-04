#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实时检测节点 (实验一: 目标检测与识别) — 部署在 Jetson 上运行

功能对应验收点:
  A. 画面实时叠加类别 / 检测框 / 置信度 (OpenCV 窗口, 左上角显示 FPS)
  B. 通过 ROS2 对外发布识别结果:
       /yolo/detections        Detection2DArray (坐标单位=像素; 未装 vision_msgs 时自动退化为 JSON 文本)
       /yolo/detections/image  标注后的画面 (CompressedImage/jpeg, 供 rqt_image_view 查看或录屏)
  C. 每 5 秒向控制台打印一次平均 FPS, 用于核对 ">=5 FPS" 指标

运行方式 (在 Jetson 上先 source ROS2 环境):
  source /opt/ros/humble/setup.bash
  python3 detect_node.py --model best.engine --source 0
  按 q 退出; 纯 SSH 无桌面环境时追加 --no-display
"""
import argparse
import json
import os
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String

from ultralytics import YOLO

try:
    from vision_msgs.msg import (Detection2D, Detection2DArray,
                                 ObjectHypothesisWithPose)
    HAVE_VISION = True
except ImportError:
    HAVE_VISION = False

TTF = cv2.FONT_HERSHEY_SIMPLEX
PALETTE = [(80, 220, 60), (60, 160, 255), (200, 80, 255), (255, 200, 60),
           (60, 220, 220), (255, 120, 120)]


def build_cli():
    """解析命令行参数。"""
    p = argparse.ArgumentParser(description="YOLO 实时检测 ROS2 节点")
    p.add_argument("--model", default=None,
                   help="权重路径; 缺省时按 脚本目录→当前目录 顺序找 best.engine, 再找 best.pt")
    p.add_argument("--source", default="0",
                   help="视频源: USB 摄像头序号 '0', 视频文件路径, 或 GStreamer 管道字符串")
    p.add_argument("--imgsz", type=int, default=640, help="推理分辨率 (FPS 不够可降到 480)")
    p.add_argument("--conf", type=float, default=0.40, help="置信度阈值")
    p.add_argument("--width", type=int, default=640, help="USB 摄像头采集宽度")
    p.add_argument("--height", type=int, default=480, help="USB 摄像头采集高度")
    p.add_argument("--base-topic", default="yolo/detections")
    p.add_argument("--no-display", action="store_true", help="不弹显示窗口 (SSH 调试用)")
    p.add_argument("--jpeg-quality", type=int, default=80)
    return p.parse_args()


def locate_weights(path):
    """按候选顺序定位权重文件, 找不到就报错退出。"""
    if path:
        return path
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [os.path.join(here, "best.engine"), os.path.join(here, "best.pt"),
                  "best.engine", "best.pt",
                  os.path.join(here, "..", "runs", "detect", "runs", "train",
                               "ros2-yolo-s", "weights", "best.pt")]
    for cand in candidates:
        if os.path.exists(cand):
            return cand
    raise SystemExit("未找到模型文件, 请用 --model 指定 best.pt / best.engine")


def open_video(source, width, height):
    """打开视频源 (CSI 走 GStreamer, USB/文件走默认后端)。"""
    use_gst = "nvarguscamerasrc" in source or "v4l2src" in source  # CSI 摄像头管道
    cap = (cv2.VideoCapture(source, cv2.CAP_GSTREAMER) if use_gst
           else cv2.VideoCapture(int(source) if source.isdigit() else source))
    if not cap.isOpened():
        raise SystemExit(f"无法打开视频源: {source}")
    if source.isdigit():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


class RealtimeDetector(Node):
    """YOLO 实时检测节点: 读帧→推理→发布结果+画面。"""

    def __init__(self, args):
        super().__init__("yolo_detect_node")
        self.args = args
        self.net = YOLO(locate_weights(args.model))
        self.classes = self.net.names
        self.src = open_video(args.source, args.width, args.height)

        self.det_pub = self.create_publisher(
            Detection2DArray if HAVE_VISION else String, args.base_topic, 10)
        self.img_pub = self.create_publisher(CompressedImage,
                                             args.base_topic + "/image", 10)
        if not HAVE_VISION:
            self.get_logger().warn("未安装 vision_msgs, 识别结果将以 JSON 字符串发布")

        self.prev_t = time.monotonic()
        self.fps_rate = 0.0
        self.last_log = 0.0
        self.timer = self.create_timer(0.033, self.tick)
        self.get_logger().info(f"模型: {self.net.ckpt_path if hasattr(self.net, 'ckpt_path') else args.model}, "
                               f"类别: {list(self.classes.values())}")

    def tick(self):
        ok, frame = self.src.read()
        if not ok:
            self.get_logger().error("读取摄像头失败, 节点退出")
            raise SystemExit(1)

        now = time.monotonic()
        dt = now - self.prev_t
        self.prev_t = now
        # 指数滑动平均, 平滑 FPS 显示
        self.fps_rate = 0.9 * self.fps_rate + 0.1 / dt if self.fps_rate > 0 else 1.0 / dt

        result = self.net.predict(frame, imgsz=self.args.imgsz,
                                  conf=self.args.conf, verbose=False)[0]
        boxes = result.boxes
        xyxy = boxes.xyxy.cpu().numpy() if len(boxes) else []
        confs = boxes.conf.cpu().numpy() if len(boxes) else []
        clss = boxes.cls.cpu().numpy().astype(int) if len(boxes) else []
        dets = [(x1, y1, x2, y2, c, f, self.classes[c])
                for (x1, y1, x2, y2), c, f in zip(xyxy, clss, confs)]

        if now - self.last_log > 5.0:
            self.last_log = now
            self.get_logger().info(f"FPS = {self.fps_rate:.1f}, 当前检测到 {len(dets)} 个目标")

        self.emit_detections(dets, frame.shape)
        self.emit_frame(frame, dets)
        if not self.args.no_display:
            self.display(frame, dets)

    def emit_detections(self, dets, shape):
        """发布识别结果: 优先 Detection2DArray, 退化为 JSON 字符串。"""
        if HAVE_VISION:
            msg = Detection2DArray()
            for x1, y1, x2, y2, cls_id, conf, _ in dets:
                d = Detection2D()
                d.bbox.center.position.x = float((x1 + x2) / 2)
                d.bbox.center.position.y = float((y1 + y2) / 2)
                d.bbox.size_x = float(x2 - x1)
                d.bbox.size_y = float(y2 - y1)
                hyp = ObjectHypothesisWithPose()
                hyp.hypothesis.class_id = int(cls_id)
                hyp.hypothesis.score = float(conf)
                d.results.append(hyp)
                msg.detections.append(d)
            self.det_pub.publish(msg)
        else:
            payload = {"frame_size": [shape[1], shape[0]],
                       "detections": [{"name": n, "class_id": int(c), "conf": round(float(f), 4),
                                       "bbox_xyxy": [round(float(v), 1) for v in (x1, y1, x2, y2)]}
                                      for x1, y1, x2, y2, c, f, n in dets]}
            msg = String()
            msg.data = json.dumps(payload, ensure_ascii=False)
            self.det_pub.publish(msg)

    def emit_frame(self, frame, dets):
        """把标注后的画面编码成 jpeg 发布。"""
        annotated = self.render(frame, dets)
        ok, buf = cv2.imencode(".jpg", annotated,
                               [cv2.IMWRITE_JPEG_QUALITY, self.args.jpeg_quality])
        if not ok:
            return
        msg = CompressedImage()
        msg.format = "jpeg"
        # Foxy 的 uint8[] 字段不接受 numpy 数组, 须转 bytes
        msg.data = bytes(buf.tobytes())
        self.img_pub.publish(msg)

    def render(self, frame, dets):
        """在画面上画框+标签+类别名+FPS。"""
        out = frame.copy()
        H, W = out.shape[:2]
        for x1, y1, x2, y2, cls_id, conf, name in dets:
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
            color = PALETTE[cls_id % len(PALETTE)]
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            label = f"{name} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, TTF, 0.6, 1)
            # 标签优先放框上沿, 空间不够放框下沿, 并裁到画面内
            if y1 - th - 10 >= 0:
                bg_tl, bg_br = (x1, y1 - th - 10), (x1 + tw + 6, y1)
                tx, ty = x1 + 3, y1 - 5
            else:
                bg_tl, bg_br = (x1, y2), (x1 + tw + 6, y2 + th + 10)
                tx, ty = x1 + 3, y2 + th + 5
            bg_tl = (max(0, bg_tl[0]), max(0, bg_tl[1]))
            bg_br = (min(W, bg_br[0]), min(H, bg_br[1]))
            cv2.rectangle(out, bg_tl, bg_br, color, -1)
            cv2.putText(out, label, (tx, ty), TTF, 0.6,
                        (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(out, f"FPS: {self.fps_rate:.1f}", (10, 28), TTF, 0.8,
                    (0, 255, 0), 2, cv2.LINE_AA)
        return out

    def display(self, frame, dets):
        """弹出 OpenCV 窗口, 按 q 退出。"""
        cv2.imshow("YOLO detect (q to quit)", self.render(frame, dets))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            raise SystemExit(0)


def main():
    args = build_cli()
    rclpy.init()
    node = RealtimeDetector(args)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.src.release()
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

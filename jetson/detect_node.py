#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
detect_node.py - Jetson 实时目标检测节点 (实验一: 目标检测与识别)

对应验收要求:
  1. 实时显示目标类别、检测框和置信度 (cv2 窗口, 左上角叠加 FPS)
  2. 通过 ROS2 发布识别结果:
       /yolo/detections        Detection2DArray (坐标系为像素; 无 vision_msgs 时自动退化为 JSON 字符串)
       /yolo/detections/image  标注后的画面 (CompressedImage/jpeg, 供 rqt_image_view 查看/录屏)
  3. 每 5 秒向控制台打印一次平均 FPS, 用于验证 ">=5 FPS" 指标

用法 (Jetson 上, 先 source ROS2):
  source /opt/ros/humble/setup.bash
  python3 detect_node.py --model best.engine --source 0
  按 q 退出; SSH 无桌面环境时加 --no-display
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
    HAS_VISION_MSGS = True
except ImportError:
    HAS_VISION_MSGS = False

FONT = cv2.FONT_HERSHEY_SIMPLEX
COLORS = [(80, 220, 60), (60, 160, 255), (200, 80, 255), (255, 200, 60),
          (60, 220, 220), (255, 120, 120)]


def parse_args():
    p = argparse.ArgumentParser(description="YOLO ROS2 实时检测节点")
    p.add_argument("--model", default=None,
                   help="模型路径; 默认自动在脚本目录/当前目录找 best.engine, 其次 best.pt")
    p.add_argument("--source", default="0",
                   help="视频源: USB 摄像头索引 '0', 视频文件路径, 或 GStreamer 管道字符串")
    p.add_argument("--imgsz", type=int, default=640, help="推理分辨率 (FPS 不足时降到 480)")
    p.add_argument("--conf", type=float, default=0.40, help="置信度阈值")
    p.add_argument("--width", type=int, default=640, help="USB 摄像头采集宽")
    p.add_argument("--height", type=int, default=480, help="USB 摄像头采集高")
    p.add_argument("--base-topic", default="yolo/detections")
    p.add_argument("--no-display", action="store_true", help="不弹显示窗口 (SSH 调试用)")
    p.add_argument("--jpeg-quality", type=int, default=80)
    return p.parse_args()


def resolve_model(path):
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


def open_capture(source, width, height):
    use_gst = "nvarguscamerasrc" in source or "v4l2src" in source  # CSI 摄像头管道
    cap = cv2.VideoCapture(source, cv2.CAP_GSTREAMER) if use_gst else cv2.VideoCapture(
        int(source) if source.isdigit() else source)
    if not cap.isOpened():
        raise SystemExit(f"无法打开视频源: {source}")
    if source.isdigit():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


class YoloDetectNode(Node):

    def __init__(self, args):
        super().__init__("yolo_detect_node")
        self.args = args
        self.model = YOLO(resolve_model(args.model))
        self.names = self.model.names
        self.cap = open_capture(args.source, args.width, args.height)

        self.det_pub = self.create_publisher(
            Detection2DArray if HAS_VISION_MSGS else String, args.base_topic, 10)
        self.img_pub = self.create_publisher(CompressedImage,
                                             args.base_topic + "/image", 10)
        if not HAS_VISION_MSGS:
            self.get_logger().warn("未安装 vision_msgs, 识别结果将以 JSON 字符串发布")

        self.prev_t = time.monotonic()
        self.fps = 0.0
        self.last_log = 0.0
        self.timer = self.create_timer(0.033, self.on_frame)
        self.get_logger().info(f"模型: {self.model.ckpt_path if hasattr(self.model, 'ckpt_path') else args.model}, "
                               f"类别: {list(self.names.values())}")

    def on_frame(self):
        ok, frame = self.cap.read()
        if not ok:
            self.get_logger().error("读取摄像头失败, 节点退出")
            raise SystemExit(1)

        now = time.monotonic()
        dt = now - self.prev_t
        self.prev_t = now
        self.fps = 0.9 * self.fps + 0.1 / dt if self.fps > 0 else 1.0 / dt

        result = self.model.predict(frame, imgsz=self.args.imgsz,
                                    conf=self.args.conf, verbose=False)[0]
        boxes = result.boxes
        xyxy = boxes.xyxy.cpu().numpy() if len(boxes) else []
        confs = boxes.conf.cpu().numpy() if len(boxes) else []
        clss = boxes.cls.cpu().numpy().astype(int) if len(boxes) else []
        dets = [(x1, y1, x2, y2, c, f, self.names[c])
                for (x1, y1, x2, y2), c, f in zip(xyxy, clss, confs)]

        if now - self.last_log > 5.0:
            self.last_log = now
            self.get_logger().info(f"FPS = {self.fps:.1f}, 当前检测到 {len(dets)} 个目标")

        self.publish_dets(dets, frame.shape)
        self.publish_image(frame, dets)
        if not self.args.no_display:
            self.show(frame, dets)

    def publish_dets(self, dets, shape):
        if HAS_VISION_MSGS:
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

    def publish_image(self, frame, dets):
        annotated = self.draw(frame, dets)
        ok, buf = cv2.imencode(".jpg", annotated,
                               [cv2.IMWRITE_JPEG_QUALITY, self.args.jpeg_quality])
        if not ok:
            return
        msg = CompressedImage()
        msg.format = "jpeg"
        msg.data = np.frombuffer(buf.tobytes(), np.uint8)
        self.img_pub.publish(msg)

    def draw(self, frame, dets):
        out = frame.copy()
        for x1, y1, x2, y2, cls_id, conf, name in dets:
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
            color = COLORS[cls_id % len(COLORS)]
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            label = f"{name} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, FONT, 0.6, 1)
            cv2.rectangle(out, (x1, y1 - th - 10), (x1 + tw + 6, y1), color, -1)
            cv2.putText(out, label, (x1 + 3, y1 - 5), FONT, 0.6,
                        (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(out, f"FPS: {self.fps:.1f}", (10, 28), FONT, 0.8,
                    (0, 255, 0), 2, cv2.LINE_AA)
        return out

    def show(self, frame, dets):
        cv2.imshow("YOLO detect (q to quit)", self.draw(frame, dets))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            raise SystemExit(0)


def main():
    args = parse_args()
    rclpy.init()
    node = YoloDetectNode(args)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.cap.release()
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

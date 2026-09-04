# ros2-yolo-object-detection

课程实验一「目标检测与识别」：在数据集（鼠标 / 手机两类）上训练 YOLO11s 检测模型，并部署到 **Jetson Orin NX + ROS2 Foxy** 实现实时检测。

## 验收结果

- 识别类别：mouse + phone（2 类）
- 独立测试集识别率：90.0%（36/40）
- Jetson 实时 FPS：33.6 FPS

## 目录结构

```
.
├── dataset/                 # 数据集（images + labels，YOLO 格式）
├── weights/                 # best.pt / best.onnx（.engine 在 Jetson 板端导出）
├── train.py                 # 训练脚本
├── merge_baidu_translate.py # 合并公开手机数据集
├── jetson/                  # Jetson 部署与测试程序
│   ├── detect_node.py       # ROS2 实时检测节点
│   ├── test_accuracy.py     # 识别率测试
│   ├── setup_on_jetson.sh   # 上板一键：装依赖 + 导出 best.engine
│   └── README.md            # 部署运行说明）
├── test_results/            # PC 测试结果（summary + 标注图 + 错误案例）
├── test_results_jetson.csv  # Jetson 实测逐张明细
├── video/                   # 结果录屏 demo_jetson_1/2.mp4
└── 实验报告.md              # 实验报告
```

## 快速开始

- **训练 / PC 测试**：见 `train.py`、`jetson/test_accuracy.py`
- **Jetson 部署**：详见 [`jetson/README.md`](jetson/README.md)

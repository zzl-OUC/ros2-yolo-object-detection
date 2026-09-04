# Jetson 部署与运行说明

实验一「目标检测与识别」在 NVIDIA Jetson Orin NX 上的部署文档。模型基于 YOLO11s 训练，导出为 TensorRT（FP16）引擎，通过 ROS2 节点实现实时检测与识别结果发布。

## 环境规格

| 项目 | 配置 |
|---|---|
| 硬件 | NVIDIA Jetson Orin NX |
| 系统 | Ubuntu 20.04 LTS（JetPack 5） |
| ROS | ROS2 Foxy |
| Python | 3.8（JetPack 预装的 NVIDIA CUDA 构建 PyTorch） |
| 推理后端 | TensorRT（FP16） |

## 目录结构

将以下文件同步至 Jetson 的同一工作目录（例如 `~/yolo_exp/`）：

```
~/yolo_exp/
├── best.pt            # 训练权重（取自 weights/best.pt，已为合并重训模型）
├── detect_node.py     # ROS2 实时检测节点
├── test_accuracy.py   # 识别率测试脚本
├── requirements.txt   # Python 依赖清单
└── setup_on_jetson.sh # 一键环境准备 + TensorRT 引擎导出
```

## 1. 一键部署（推荐）

在 Jetson 上执行单条命令即可完成依赖安装与引擎导出：

```bash
cd ~/yolo_exp
bash setup_on_jetson.sh foxy 5     # 参数: <ROS_DISTRO> <JetPack主版本>; 默认 humble 6
```

脚本依据 JetPack 版本安装对应的 Jetson 版 PyTorch，并导出 `best.engine`（TensorRT FP16）。
若使用 JetPack 6 / Humble 环境，改为 `bash setup_on_jetson.sh humble 6`。

## 2. 手动环境准备

JetPack 已内置 CUDA / cuDNN / TensorRT，仅需补充 Python 包：

```bash
# Jetson 版 PyTorch（JetPack 6 示例；JetPack 5 由系统预装，通常无需重装）
pip install torch torchvision --index-url https://pypi.jetson-ai-lab.dev/jp6/cu126
pip install ultralytics
sudo apt install ros-${ROS_DISTRO}-vision-msgs   # Detection2DArray 消息类型
```

> ROS_DISTRO 取决于 JetPack：JetPack 6 → Humble，JetPack 5 → Foxy/Galactic。
> 每次使用前执行 `source /opt/ros/$ROS_DISTRO/setup.bash` 加载 ROS 环境。

## 3. 导出 TensorRT 引擎

TensorRT 引擎是满足实时性（≥5 FPS）的关键：

```bash
yolo export model=best.pt format=engine half=True device=0
# 生成 best.engine，detect_node.py 自动优先加载
```

yolo11s + TensorRT FP16 在 Orin 系列上可达 30+ FPS；老旧平台（如 Nano）若不足 5 FPS，可在节点启动时追加 `--imgsz 480`。

## 4. 实时检测节点

```bash
source /opt/ros/$ROS_DISTRO/setup.bash
python3 detect_node.py --source 0          # USB 摄像头
```

- 检测窗口实时显示类别 / 边界框 / 置信度，左上角显示 FPS；按 `q` 退出。
- 无桌面环境（SSH）时追加 `--no-display`，仅通过话题发布结果。
- 验证 ROS2 发布：

```bash
ros2 topic list                                  # /yolo/detections, /yolo/detections/image
ros2 topic echo /yolo/detections                 # 查看检测框 / 类别 / 置信度
ros2 run rqt_image_view rqt_image_view           # 订阅 /yolo/detections/image 查看标注画面
```

CSI 摄像头可通过 GStreamer 管道指定 `--source`，例如：
`--source "nvarguscamerasrc sensor-id=0 ! video/x-raw(memory:NVMM),width=1280,height=720 ! nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! appsink"`

常用参数：`--model`（模型路径）、`--conf 0.4`（置信度阈值）、`--imgsz`、`--no-display`。

## 5. 识别率测试

对 20 张 mouse + 20 张 phone（共 40 张，独立于训练集的桌面场景图像）运行识别率测试：

1. 采集图像存入 `test_images/`，文件名以真实类别开头：`mouse_01.jpg`、`phone_03.jpg` …
2. 执行：

```bash
python3 test_accuracy.py --model best.engine
```

3. 终端输出逐样本判定与总识别率；结果写入 `test_results/`：
   - `summary.txt` — 识别率、平均推理时延、折算 FPS、达标判定
   - `results.csv` — 逐张明细（期望类别 / 识别类别 / 置信度 / 时延）
   - `annotated/` — 全部标注图
   - `errors/` — 典型错误案例（漏检 / 误检图自动归档）

## 6. 结果视频录制

录屏应同时呈现检测窗口（含 FPS 角标）与 `ros2 topic echo /yolo/detections` 输出，以一次性覆盖「实时显示 + ROS2 发布 + FPS 达标」三项验收点。
录制方式：PC 端使用 OBS / Win+G；Jetson 端使用 `kazam`，或以外部设备拍摄屏幕。

## 7. 常见问题

| 现象 | 处理 |
|---|---|
| FPS < 5 | 确认已加载 `best.engine`；必要时 `--imgsz 480`；关闭其他占用 GPU 的进程 |
| `vision_msgs` 缺失 | 节点自动降级为 JSON 字符串发布，不影响验收；建议安装对应 ROS 包 |
| imshow 报错 | SSH 会话无桌面，追加 `--no-display` |
| USB 摄像头无法打开 | `ls /dev/video*` 确认设备索引，以 `--source` 指定 |

## 8. PC 端基线（供报告引用）

在 PC（RTX 4060 Laptop，CUDA，`weights/best.pt` 合并重训模型）上对同一 40 张测试图运行 `test_accuracy.py` 基线：

```
测试总数: 40    正确: 36    识别率: 90.0%    要求 >=80%: 通过
平均单帧推理: 108 ms    折算约 9.3 FPS
分类别: mouse 20/20  phone 16/20
错误案例: phone_12 / phone_15 (漏检), phone_14 / phone_18 (误判为 mouse)
```

上板后建议用 `best.engine` 在 Jetson 重跑同一测试集，将此处「PC 基线」替换为「Jetson 实测」填入报告 §5，FPS 取 Jetson 实测值填入 §4。

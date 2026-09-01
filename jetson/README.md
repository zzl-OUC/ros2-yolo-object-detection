# Jetson 部署运行说明 (实验一: 目标检测与识别)

把训练好的模型和 `jetson/` 目录下的两个脚本拷到 Jetson 的同一目录即可,例如 `~/yolo_exp/`:

```
~/yolo_exp/
├── best.pt        # 从 PC 拷贝: weights/best.pt (已是最新"合并重训"模型)
├── detect_node.py # ROS2 实时检测节点 (显示 + 发布识别结果)
├── test_accuracy.py # 识别率测试 (20 mouse + 20 phone = 40 张)
├── requirements.txt # Python 依赖 (ultralytics / numpy)
└── setup_on_jetson.sh # 上板一键: 装依赖 + 导出 best.engine
```

## 0. 最快上板 (推荐)

把本目录 (`best.pt` + `detect_node.py` + `test_accuracy.py` + `requirements.txt` + `setup_on_jetson.sh`)
整体拷到 Jetson,例如 `~/yolo_exp/`,然后在该目录里跑一条命令即可完成"装依赖 + 导出引擎":

```bash
cd ~/yolo_exp
bash setup_on_jetson.sh humble 6      # 参数: [ROS_DISTRO] [JetPack大版本]; 默认 humble 6
```

脚本会自动按 JetPack 版本补装 Jetson 版 PyTorch,并导出 `best.engine` (TensorRT FP16)。
若你的板子是 JetPack 5 / Foxy,改成 `bash setup_on_jetson.sh foxy 5`。
其余实时检测、识别率测试、录屏步骤见下文。

## 1. 环境准备 (Jetson 上执行一次)

JetPack 已自带 CUDA / cuDNN / TensorRT,只缺 Python 包:

```bash
# 安装 Jetson 版 PyTorch (JetPack 6)
pip install torch torchvision --index-url https://pypi.jetson-ai-lab.dev/jp6/cu126
# 安装 ultralytics 与 ROS2 依赖
pip install ultralytics
sudo apt install ros-${ROS_DISTRO}-vision-msgs   # Detection2DArray 消息
```

> ROS_DISTRO 视 JetPack 而定:JetPack 6 → Humble,JetPack 5 → Foxy/Galactic。
> 每次使用前先 `source /opt/ros/$ROS_DISTRO/setup.bash`。

## 2. 导出 TensorRT 模型 (保证 ≥5 FPS 的关键)

```bash
yolo export model=best.pt format=engine half=True device=0
# 生成 best.engine;detect_node.py 会自动优先使用它
```

yolo11s + TensorRT FP16 在 Orin 系列上有 30+ FPS;老款 Nano 若不足 5 FPS,
运行节点时加 `--imgsz 480` 即可。

## 3. 实时检测节点 (对应: 实时显示 + ROS2 发布)

```bash
source /opt/ros/$ROS_DISTRO/setup.bash
python3 detect_node.py --source 0          # USB 摄像头
```

- 屏幕窗口实时显示**类别 / 检测框 / 置信度**,左上角是 FPS,按 `q` 退出
- SSH 无桌面时加 `--no-display`,只看话题
- 另开终端验证 ROS2 发布:

```bash
ros2 topic list                             # /yolo/detections /yolo/detections/image
ros2 topic echo /yolo/detections            # 查看检测框/类别/置信度
ros2 run rqt_image_view rqt_image_view      # 选 /yolo/detections/image 看标注画面
```

CSI 摄像头用 GStreamer 管道作 `--source`,例如:
`--source "nvarguscamerasrc sensor-id=0 ! video/x-raw(memory:NVMM),width=1280,height=720 ! nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! appsink"`

常用参数:`--model`(指定模型)、`--conf 0.4`(置信度阈值)、`--imgsz`、`--no-display`。

## 4. 20 物体识别率测试 (对应: 正确率 ≥80% + 保存结果/错误案例)

1. 拍 20 张 mouse + 20 张 phone (共 40 张, 与训练集不同场景的桌面照, 可含手持/不同背景),
   放进 `test_images/`,文件名以真实类别开头:`mouse_01.jpg`、`phone_03.jpg` …
2. 运行:

   ```bash
   python3 test_accuracy.py --model best.engine
   ```

3. 终端打印每个样本的对错和总识别率,结果保存在 `test_results/`:
   - `summary.txt` — 识别率、平均推理耗时、折算 FPS、是否达标
   - `results.csv` — 逐张明细(期望/识别/置信度/耗时)
   - `annotated/` — 全部标注图
   - `errors/` — **典型错误案例**(识别错或未检出的图自动归档)

## 5. 录制结果视频

录屏同时呈现:检测窗口(有 FPS 角标)+ `ros2 topic echo /yolo/detections` 输出,
一次即可覆盖"实时显示 + ROS2 发布 + FPS 达标"三项验收点。
PC 端可用 OBS / Win+G,Jetson 端可用 `kazam` 或手机拍摄屏幕。

## 6. 常见问题

| 现象 | 处理 |
|---|---|
| FPS < 5 | 确认用的是 `best.engine`;`--imgsz 480`;关掉其他占 GPU 的进程 |
| `vision_msgs` 找不到 | 节点会自动退化为 JSON 字符串发布,不影响验收;建议装上该包 |
| imshow 报错 | SSH 会话没有桌面,加 `--no-display` |
| USB 摄像头打不开 | `ls /dev/video*` 确认索引,`--source` 传对应编号 |

## 7. PC 端基线结果 (供报告引用)

在 PC (RTX 4060 Laptop, CUDA, `weights/best.pt` 合并重训模型) 上对 **40 张** 独立测试图
(20 mouse + 20 phone, 与训练集不同场景的桌面照) 跑 `test_accuracy.py` 的基线:

```
测试总数: 40    正确: 36    识别率: 90.0%    要求 >=80%: 通过
平均单帧推理: 108 ms    折算约 9.3 FPS
分类别: mouse 20/20  phone 16/20
错误案例: phone_12 / phone_14 (未检出), phone_15 / phone_18 (误判为 mouse)
```

> 上板后建议用 `best.engine` 在 Jetson 上再跑一遍同一套 40 张测试图,把这里的
> "PC 基线" 替换为 "Jetson 实测" 填进报告 §5;FPS 改为 Jetson 实测值填 §4。

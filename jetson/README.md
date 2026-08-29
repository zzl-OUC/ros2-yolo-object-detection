# Jetson 部署运行说明 (实验一: 目标检测与识别)

把训练好的模型和 `jetson/` 目录下的两个脚本拷到 Jetson 的同一目录即可,例如 `~/yolo_exp/`:

```
~/yolo_exp/
├── best.pt        # 从 PC 拷贝: runs/detect/runs/train/ros2-yolo-s/weights/best.pt
├── detect_node.py # ROS2 实时检测节点 (显示 + 发布识别结果)
└── test_accuracy.py # 20 物体识别率测试
```

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

1. 拍 20 张训练类别之外的测试照片(每个物体一张,含干扰摆放),放进 `test_images/`,
   文件名以真实类别开头:`mouse_01.jpg`、`phone_03.jpg` …
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

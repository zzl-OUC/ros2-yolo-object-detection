# ros2-yolo-object-detection

![ROS2](https://img.shields.io/badge/ROS2-Foxy-22314E?logo=ros&logoColor=white)
![YOLO](https://img.shields.io/badge/YOLOv11s-Ultralytics-00BFFF)
![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Jetson](https://img.shields.io/badge/Jetson%20Orin%20NX-JetPack%205-76B900?logo=nvidia&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)
![验收](https://img.shields.io/badge/验收-识别率%2090%25%20%7C%20Jetson%2033.6%20FPS-brightgreen)

> 课程实验一「目标检测与识别」：在自建桌面数据集（**鼠标 / 手机** 两类）上训练 YOLO11s 检测模型，并部署到 **Jetson Orin NX + ROS2 Foxy** 实现实时检测。
> 另含单独扩展：**实例分割扩展（YOLO-Seg）**：用 GrabCut 由检测框自动生成像素级伪标注，训练出可同时输出类别框与掩膜的 YOLO11s-seg。

---

## 项目简介

本项目完成一条端到端的「数据采集 → 标注 → 训练 → 导出 → 边缘部署 → 验收」闭环：

- **数据**：桌面场景下自采 384 张图，人工标注鼠标 / 手机两类目标（YOLO txt 格式。
- **模型**：Ultralytics YOLO11s（640×640），100 epoch 训练，导出 ONNX / TensorRT(FP16) 两条推理链路。
- **部署**：在 Jetson Orin NX上用 TensorRT 引擎做实时检测，ROS2 节点发布 `Detection2DArray` 结果。
- **验收**：独立测试集识别率 **90.0%（36/40）**，Jetson 稳态 **33.6 FPS**。

---

## 验收结果对照

| 验收项 | 要求 | 实测结果 | 结论 |
|---|---|---|---|
| 识别类别数 | ≥ 2 类 | mouse + phone 两类  |
| 独立测试集识别率 | ≥ 80% | **90.0%（36/40）** |
| Jetson 实时速度 | ≥ 5 FPS | **33.6 FPS**（稳态 29.7 ms） |
| 保存结果 + 错误案例归档 | 需要 | `test_results_jetson.csv` + `test_results/errors/`  |

**实例分割扩展**额外结果：

| 任务 | 指标 | 结果 |
|---|---|---|
| 实例分割 | 独立测试集识别率（全带掩膜） | **87.5%（35/40）** |
| 实例分割 | 验证集 Mask mAP50 / mAP50-95 | 0.897 / 0.857 |
| 实例分割 | PC 稳态推理 | 53.9 ms ≈ 18.5 FPS |

> 注：分割标注由 GrabCut 自动生成。

---

##  目录结构

```
.
├── dataset/                 # 检测数据集（images + labels，YOLO 格式）
│   ├── data.yaml            # 类别定义：0=mouse, 1=phone
│   ├── images/{train,val}/  # 326 / 58 张
│   └── labels/{train,val}/
├── seg/                     # 分割数据集（labels 为 polygon 伪标注 + data.yaml）
├── weights/                 # best.pt / best.onnx（检测）；best_seg.pt / best_seg.onnx（分割）
├── jetson/                  # Jetson 部署与测试程序
│   ├── detect_node.py       # ROS2 实时检测节点
│   ├── test_accuracy.py     # 识别率测试
│   ├── setup_on_jetson.sh   # 上板一键：装依赖 + 导出 best.engine
│   ├── requirements.txt     # Python 依赖清单
│   └── README.md            # 部署运行说明（详细）
├── train.py                 # 检测训练入口（训 + 验证 + 导 ONNX）
├── train_seg.py             # 分割训练入口（yolo11s-seg）
├── gen_seg_labels.py        # bbox -> polygon 伪标注自动生成（GrabCut）
├── seg_verify.py            # 分割标注叠加抽查图
├── merge_baidu_translate.py # 合并公开手机数据集（补全手机形态）
├── test_images/             # 40 张独立测试图（20 mouse + 20 phone）
├── test_results/            # PC 测试结果（summary + 标注图 + 错误案例）
├── test_results_jetson.csv  # Jetson 实测逐张明细
├── test_results_seg/        # 分割测试全量输出（框 + 掩膜）
├── video/                   # 结果录屏 demo_jetson_1/2.mp4
└── 实验报告.md              # 完整实验报告（含分割扩展实验 §7）
```

---

##  数据集

桌面场景下的物体图像，两类目标：**mouse**、**phone**，YOLO txt 标注，按 `images/{train,val}` + `labels/{train,val}` 组织。

| 项目 | 数量 |
|---|---|
| 图像总数 | 384 张 |
| 标注框（实例）总数 | 441 个 |
| ├ mouse 标注框 | 192 个 |
| └ phone 标注框 | 249 个 |
| 训练集图像 | 326 张（376 框） |
| 验证集图像 | 58 张（65 框） |



---

##  环境依赖

**PC 训练端**
- Python ≥ 3.10
- `torch` + `ultralytics`（YOLO11s）
- `opencv-python-headless` + `numpy`（分割伪标注生成）

**Jetson 部署端**
- NVIDIA Jetson Orin NX，Ubuntu 20.04（JetPack 5），ROS2 Foxy
- 系统预装 CUDA / cuDNN / TensorRT；仅需补充 `ultralytics` 等 Python 包（见 `jetson/requirements.txt`）

---

## 如何运行

### 1. 目标检测（训练 → 导出 → 测试）

```bash
# 训练 + 验证 + 导出 ONNX（产物见 runs/train/ros2-yolo-s/ 与 weights/）
python train.py

# PC 基线识别率测试（对 40 张独立测试图）
python jetson/test_accuracy.py --model weights/best.pt
```

### 2. 实例分割（伪标注 → 训练）

```bash
# 步骤 1：由检测框自动生成分割标注（GrabCut），写入 seg/ 并生成 seg/data.yaml
python gen_seg_labels.py

# 步骤 2：训练 YOLO11s-seg（优先官方 yolo11s-seg.pt；不可达则用本地检测权重迁移初始化）
python train_seg.py
```

### 3. Jetson 部署

详见 **[`jetson/README.md`](jetson/README.md)**（含一键脚本、TensorRT 引擎导出、ROS2 节点运行、识别率测试、录屏规范）。核心三步：

```bash
cd ~/yolo_exp
bash setup_on_jetson.sh foxy 5        # 装依赖 + 导出 best.engine（TensorRT FP16）
source /opt/ros/foxy/setup.bash
python3 detect_node.py --source 0    # USB 摄像头实时检测
python3 test_accuracy.py --model best.engine   # 识别率验收
```

---

## 实例分割扩展（YOLO-Seg）

在检测任务基础上，将输出从「边界框」升级为「像素级掩膜」，训练 **YOLO11s-seg**。

### 方法论：分割标注的自动生成

现有 441 个标注均为检测框（5 列 bbox），不含多边形轮廓。为避免人工重标，采用 **GrabCut 自动生成伪标注**：

1. 以现有检测框作为前景矩形提示，用 OpenCV `grabCut` 抠出目标前景；
2. 取最大连通轮廓，用 `approxPolyDP` 逼近得到多边形；
3. 归一化后写出 YOLO-Seg 格式 `class x1 y1 ... xn yn`。

| 项目 | 数量 |
|---|---|
| 生成分割标注的图像 | 384 张 |
| 生成目标实例 | 441 个 |
| 退回矩形兜底 | 1 个（其余 440 个均为真实轮廓） |

生成脚本 `gen_seg_labels.py`；抽查可视化见 `seg_verify/`（鼠标 / 手机轮廓均贴合目标边缘）。

### 训练配置

- **模型**：YOLO11s-seg（114 层，约 1006 万参数），输入 640×640；
- **初始化**：优先官方 `yolo11s-seg.pt`；不可达时用 `yolo11s-seg_init.pt`（由检测权重 `yolo11s.pt` 迁移，骨干/颈部/检测头 499 层全匹配，仅 mask 原型头随机初始化）；
- **超参**：100 epoch，batch 16，与检测实验一致。

### 验证集结果（58 张 / 65 实例）

| 类别 | Box mAP50 | Box mAP50-95 | Mask mAP50 | Mask mAP50-95 |
|---|---|---|---|---|
| all | 0.924 | 0.892 | **0.897** | **0.857** |
| mouse | 0.967 | 0.961 | 0.967 | 0.961 |
| phone | 0.882 | 0.823 | 0.828 | 0.752 |

### 独立测试集实测（PC, RTX 4060 Laptop）

| 指标 | 结果 |
|---|---|
| 识别率 | **87.5%（35/40）**，且 35 张全部输出掩膜 |
| 稳态推理速度 | **53.9 ms ≈ 18.5 FPS** |
| 失败案例 | 未检出 mouse_04 / phone_08 / phone_12 / phone_14；误判 mouse_05→phone |

掩膜质量示例见 `test_results_seg/`（40 张测试图全量输出，含类别框、置信度与像素级掩膜）。

> 📹 **关于演示视频**：实例分割扩展**暂无实时录屏演示**，目前仅提供上述静态结果图。实时摄像头推理与检测模型共用同一套 `detect_node.py`（启动时切到 `--model weights/best_seg.pt` 即可），上板录屏可复用 `video/` 下的检测演示。

---

## Jetson 实测错误案例分析

独立测试集 40 张（20 mouse + 20 phone，与训练集不同场景的桌面照），在 Jetson 上用 `best.engine` 实测 **90.0%（36/40）**：

- **mouse**：20/20 
- **phone**：16/20（漏检 2、误判 2）
S
典型错误案例（已归档至 `test_results/errors/`）：

| 文件 | 现象 | 归因 |
|---|---|---|
| `phone_12.png` / `phone_15.png` | 未检出 | 最偏分布样本，手机未置于桌面主体位置 |
| `phone_14.png` / `phone_18.png` | 误判为 mouse | 手机与鼠标外观 / 背景混淆 |

> 错误均集中在「最偏分布」的手机样本，属数据分布问题而非模型能力不足（训练桌面分布 mAP50 达 0.961）。过 80% 验收线有余量。

---

##  局限性与后续工作

- **分割标注为伪标注**：Mask mAP 衡量与 GrabCut 轮廓的一致性，非人工精细标注精度；分割检测精度（87.5%）略低于检测模型（90.0%），符合「多任务头 + 伪标注噪声」预期。
- **分割模型未上板**：`best_seg.engine` 尚未在 Jetson 导出，18.5 FPS 为 PC 实测；按检测模型 PC→Jetson 加速经验，上板后预计仍满足 ≥5 FPS。
- **后续可做**：① 人工精修分割标注提升 Mask mAP；② 导出分割 TensorRT 引擎并完成 Jetson 实测；③ 扩充手机偏分布样本消除误检/漏检。

---

##  相关文档

- [`实验报告.md`](实验报告.md) — 完整实验报告（目的、数据集、训练、测试、结论、分割扩展 §7）
- [`jetson/README.md`](jetson/README.md) — Jetson 部署与运行详细说明
- 演示视频：`video/demo_jetson_1.mp4`、`video/demo_jetson_2.mp4`（**目标检测**实时录屏；实例分割扩展暂无视频演示，静态结果见 `test_results_seg/`）

---



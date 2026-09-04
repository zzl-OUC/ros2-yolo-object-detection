#!/usr/bin/env bash
# ============================================================================
# setup_on_jetson.sh — Jetson 上板一键部署 (实验一: 目标检测与识别)
#
# 用法:
#   bash setup_on_jetson.sh [ROS_DISTRO] [JETPACK_MAJOR]
#     例 (JetPack 6 / Humble):  bash setup_on_jetson.sh humble 6
#     例 (JetPack 5 / Foxy):    bash setup_on_jetson.sh foxy  5
#   不传参则默认 humble / 6。
#
# 前置: 已把本目录 (best.pt + 各 .py + requirements.txt) 拷到 Jetson 某目录,
#       例如 ~/yolo_exp/, 然后在该目录里运行本脚本。
#       需要联网 (首次装 pip/apt 包)。
# ============================================================================
set -e

ROS_DIST=${1:-humble}
JP_VER=${2:-6}

echo "==> ROS_DISTRO=$ROS_DIST   JetPack=$JP_VER"

# 1) 导入 ROS2 环境
if [ -f "/opt/ros/$ROS_DIST/setup.bash" ]; then
  source "/opt/ros/$ROS_DIST/setup.bash"
else
  echo "✗ 未找到 /opt/ros/$ROS_DIST/setup.bash, 请先确认 JetPack 自带的 ROS2 是否已装"
  exit 1
fi

# 2) 系统 cv2 (带 GTK, 支持显示) + Python 依赖
echo "==> 安装系统 opencv 与 Python 依赖"
sudo apt-get update
# 精简版镜像可能没带 pip3, 先兜住
if ! command -v pip3 >/dev/null 2>&1; then
  echo "  (未检测到 pip3, 先装 python3-pip)"
  sudo apt-get install -y python3-pip
fi
sudo apt-get install -y python3-opencv
# 用 python3 -m pip 而非裸 pip3: pip 升级装到 ~/.local/bin (不在 PATH),
# 裸 pip3 永远指向 /usr/bin 的旧版 (20.0.2, 认不出 manylinux_2_17 新式 wheel 标签)
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
# pip 装的命令行脚本 (yolo 等) 落在 ~/.local/bin, 默认不在 PATH — 补上
export PATH="$HOME/.local/bin:$PATH"

# 3) vision_msgs (Detection2DArray 消息; 缺了节点会自动退化成 JSON, 不影响验收)
echo "==> 安装 ros-$ROS_DIST-vision-msgs"
sudo apt-get install -y "ros-$ROS_DIST-vision-msgs" || \
  echo "  (vision-msgs 安装失败可忽略, 节点会退化为 JSON 发布)"

# 4) torch: JetPack 通常已预装, 仅当缺失时按版本补装
if python3 -c "import torch" 2>/dev/null; then
  echo "==> torch 已随 JetPack 预装, 跳过"
else
  echo "==> 未检测到 torch, 按 JetPack $JP_VER 安装 Jetson 版 PyTorch"
  if [ "$JP_VER" = "6" ]; then
    python3 -m pip install torch torchvision --index-url https://pypi.jetson-ai-lab.dev/jp6/cu126
  else
    python3 -m pip install torch torchvision --index-url https://pypi.jetson-ai-lab.dev/jp5/cu122
  fi
fi

# 5) 导出 TensorRT 引擎 (>=5 FPS 的关键, 必须在 Jetson 本机做, 绑定硬件/JetPack)
echo "==> 导出 best.engine (TensorRT FP16)"
yolo export model=best.pt format=engine half=True device=0

echo ""
echo "✅ 部署完成. 运行实时节点:"
echo "    source /opt/ros/$ROS_DIST/setup.bash"
echo "    python3 detect_node.py --model best.engine --source 0"
echo "    # SSH 无桌面: 加 --no-display"
echo "识别率测试:  python3 test_accuracy.py --model best.engine"

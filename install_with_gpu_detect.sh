#!/bin/bash
# 安裝依賴並根據平台自動安裝 GPU 相關套件
# 用法： bash install_with_gpu_detect.sh

set -e

pip install -r requirements.txt

# Linux + NVIDIA GPU 自動安裝 CUDA 版 torch
if [[ "$(uname)" == "Linux" ]] && command -v nvidia-smi &>/dev/null; then
  echo "[INFO] 偵測到 NVIDIA GPU，安裝 CUDA 版 torch..."
  pip install torch==2.8.0+cu121 -f https://download.pytorch.org/whl/torch_stable.html
else
  echo "[INFO] 未偵測到 NVIDIA GPU，維持預設安裝 (macOS 會自動支援 MPS)"
fi

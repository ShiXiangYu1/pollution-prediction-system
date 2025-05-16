#!/bin/bash

set -e

# 安装所需依赖
echo "安装应用所需的关键依赖..."
pip install --no-cache-dir jinja2>=3.1.0
pip install --no-cache-dir torch>=2.0.0 --index-url https://download.pytorch.org/whl/cpu || echo "无法安装PyTorch，将使用模拟模块"

echo "启动应用..."
# 启动应用
python -m uvicorn api:app --host 0.0.0.0 --port 8000 
#!/bin/sh

set -e

# 检查并安装关键依赖
echo "检查并安装关键依赖..."
# 检查jinja2是否已安装
if ! python -c "import jinja2" 2>/dev/null; then
  echo "安装jinja2..."
  pip install --no-cache-dir jinja2>=3.1.0
else
  echo "jinja2已安装"
fi

# 检查是否需要安装PyTorch（根据环境变量决定是否使用模拟模块）
if [ "${USE_MOCK_TORCH:-false}" != "true" ]; then
  if ! python -c "import torch" 2>/dev/null; then
    echo "安装PyTorch..."
    pip install --no-cache-dir torch>=2.0.0 --index-url https://download.pytorch.org/whl/cpu
  else
    echo "PyTorch已安装"
  fi
else
  echo "使用模拟PyTorch模块，不安装真实依赖"
fi

# 等待PostgreSQL数据库启动
echo "等待PostgreSQL启动..."
while ! nc -z db 5432; do
  sleep 0.5
done
echo "PostgreSQL已就绪！"

# 等待Redis启动
echo "等待Redis启动..."
while ! nc -z redis 6379; do
  sleep 0.5
done
echo "Redis已就绪！"

# 创建必要的目录
mkdir -p /app/logs /app/data /app/models

# 执行数据库迁移(如果有)
if [ -f "/app/migrations/run_migrations.py" ]; then
  echo "执行数据库迁移..."
  python3 /app/migrations/run_migrations.py
  echo "数据库迁移完成！"
fi

# 初始化应用配置
if [ -f "/app/init_app.py" ]; then
  echo "初始化应用配置..."
  python3 /app/init_app.py
  echo "应用配置初始化完成！"
fi

# 如果不存在模型文件，下载或初始化模型
if [ ! -f "/app/models/prediction_model.pth" ]; then
  echo "初始化预测模型..."
  python3 /app/init_models.py
  echo "模型初始化完成！"
fi

# 执行传入的命令
exec "$@" 
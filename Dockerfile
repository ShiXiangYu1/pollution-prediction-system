FROM python:3.10-slim

WORKDIR /app

# 设置环境变量
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV TZ=Asia/Shanghai
ENV DEBIAN_FRONTEND=noninteractive

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    netcat-openbsd \
    && rm -rf /var/lib/apt/lists/*

# 升级pip
RUN pip install --upgrade pip

# 首先独立安装关键依赖
RUN pip install --no-cache-dir jinja2==3.1.2
RUN echo "Jinja2 installed successfully" && python -c "import jinja2; print(f'Jinja2 version: {jinja2.__version__}')"

# 安装简化版PyTorch(CPU版本)来节省空间
RUN pip install --no-cache-dir torch==2.0.1 --index-url https://download.pytorch.org/whl/cpu
RUN echo "PyTorch installed successfully" && python -c "import torch; print(f'PyTorch version: {torch.__version__}')"

# 复制requirements.txt并安装其他依赖
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# 创建必要的目录
RUN mkdir -p /app/logs /app/data /app/models /app/static /app/templates

# 复制项目文件
COPY . /app/

# 简化应用启动，直接使用CMD
CMD ["python", "-m", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"] 
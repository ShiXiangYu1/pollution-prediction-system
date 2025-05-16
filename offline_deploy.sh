#!/bin/bash
# 电厂污染物排放预测系统 - 离线环境一键部署脚本
# 用于在方天公司离线环境（局域网）中部署系统
# 最后更新：2024-05-10

set -e

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 显示横幅
echo -e "${GREEN}"
echo "========================================================"
echo "    电厂污染物排放预测系统 - 离线环境一键部署脚本      "
echo "========================================================"
echo -e "${NC}"

# 检查Docker和Docker Compose是否已安装
echo -e "${YELLOW}[步骤1] 检查基础环境...${NC}"
if ! command -v docker &> /dev/null; then
    echo -e "${RED}错误: Docker未安装，请先安装Docker${NC}"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo -e "${YELLOW}警告: 未检测到docker-compose命令，尝试使用docker compose...${NC}"
    if ! docker compose version &> /dev/null; then
        echo -e "${RED}错误: Docker Compose未安装或不可用，请安装Docker Compose${NC}"
        exit 1
    else
        echo -e "${GREEN}√ 检测到Docker Compose插件${NC}"
        # 使用别名让后续命令兼容
        alias docker-compose='docker compose'
    fi
else
    echo -e "${GREEN}√ 检测到docker-compose命令${NC}"
fi

echo -e "${GREEN}√ Docker和Docker Compose已安装${NC}"
docker --version
docker-compose --version || docker compose version

# 检查镜像包是否存在
echo -e "${YELLOW}[步骤2] 检查镜像包...${NC}"

IMAGE_PACKAGE="pollution-prediction-images.tar"
if [ ! -f "$IMAGE_PACKAGE" ]; then
    echo -e "${YELLOW}警告: 未找到镜像包 $IMAGE_PACKAGE${NC}"
    echo -e "请确认以下两种方式之一:"
    echo -e "1. 已通过其他方式导入了所需镜像"
    echo -e "2. 当前环境可以通过网络拉取镜像"
    
    read -p "是否继续部署? [Y/n] " continue_deploy
    if [[ $continue_deploy =~ ^[Nn]$ ]]; then
        echo -e "${YELLOW}部署已取消${NC}"
        exit 0
    fi
else
    echo -e "${GREEN}√ 发现镜像包 $IMAGE_PACKAGE${NC}"
    echo -e "${YELLOW}正在导入镜像，这可能需要几分钟时间...${NC}"
    docker load -i $IMAGE_PACKAGE
    echo -e "${GREEN}√ 镜像导入完成${NC}"
fi

# 检查必要文件
echo -e "${YELLOW}[步骤3] 检查配置文件...${NC}"
if [ ! -f "docker-compose.yml" ]; then
    echo -e "${RED}错误: 未找到docker-compose.yml文件${NC}"
    exit 1
fi

# 创建必要的目录
echo -e "${YELLOW}[步骤4] 创建必要目录...${NC}"
mkdir -p data logs models static templates prometheus/
mkdir -p init-scripts/db

# 如果prometheus目录中没有配置文件，则创建一个基本配置
if [ ! -f "prometheus/prometheus.yml" ]; then
    echo -e "${YELLOW}创建基本的Prometheus配置文件...${NC}"
    cat > prometheus/prometheus.yml << EOF
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  - job_name: 'pollution_prediction_app'
    static_configs:
      - targets: ['app:8000']
EOF
    echo -e "${GREEN}√ Prometheus配置文件已创建${NC}"
fi

# 询问是否使用默认端口
echo -e "${YELLOW}[步骤5] 配置服务端口...${NC}"
read -p "使用默认端口配置? [Y/n] " use_default_ports
if [[ $use_default_ports =~ ^[Nn]$ ]]; then
    echo -e "请配置服务端口:"
    read -p "API服务端口 (默认: 8000): " api_port
    read -p "数据库端口 (默认: 5432): " db_port
    read -p "Redis端口 (默认: 6379): " redis_port
    read -p "Prometheus端口 (默认: 9090): " prometheus_port
    read -p "Grafana端口 (默认: 3000): " grafana_port
    
    # 设置默认值
    api_port=${api_port:-8000}
    db_port=${db_port:-5432}
    redis_port=${redis_port:-6379}
    prometheus_port=${prometheus_port:-9090}
    grafana_port=${grafana_port:-3000}
    
    # 使用sed修改端口配置
    sed -i.bak "s/8000:8000/$api_port:8000/g" docker-compose.yml
    sed -i.bak "s/5432:5432/$db_port:5432/g" docker-compose.yml
    sed -i.bak "s/6379:6379/$redis_port:6379/g" docker-compose.yml
    sed -i.bak "s/9090:9090/$prometheus_port:9090/g" docker-compose.yml
    sed -i.bak "s/3000:3000/$grafana_port:3000/g" docker-compose.yml
    
    echo -e "${GREEN}√ 端口配置已更新${NC}"
else
    api_port=8000
    db_port=5432
    redis_port=6379
    prometheus_port=9090
    grafana_port=3000
    echo -e "${GREEN}√ 使用默认端口配置${NC}"
fi

# 启动服务
echo -e "${YELLOW}[步骤6] 启动服务...${NC}"
docker-compose -p pollution-prediction-system up -d

# 检查服务状态
echo -e "${YELLOW}[步骤7] 检查服务状态...${NC}"
sleep 10  # 等待服务启动

if docker-compose -p pollution-prediction-system ps | grep "app" | grep "Up"; then
    echo -e "${GREEN}√ 主应用服务已成功启动${NC}"
else
    echo -e "${RED}! 主应用服务启动异常，请检查日志${NC}"
    docker-compose -p pollution-prediction-system logs app
fi

# 获取本机IP地址
if command -v ip &> /dev/null; then
    ip_addr=$(ip -4 addr show scope global | grep -oP '(?<=inet\s)\d+(\.\d+){3}' | head -n 1)
elif command -v ifconfig &> /dev/null; then
    ip_addr=$(ifconfig | grep -Eo 'inet (addr:)?([0-9]*\.){3}[0-9]*' | grep -Eo '([0-9]*\.){3}[0-9]*' | grep -v '127.0.0.1' | head -n 1)
else
    ip_addr="本机IP"
fi

# 显示访问信息
echo -e "${GREEN}========================================================"
echo -e "电厂污染物排放预测系统已成功部署!"
echo -e ""
echo -e "系统访问地址:"
echo -e "- 本地访问: http://localhost:$api_port"
echo -e "- 局域网访问: http://$ip_addr:$api_port"
echo -e ""
echo -e "其他服务地址:"
echo -e "- API文档: http://$ip_addr:$api_port/docs"
echo -e "- 健康检查: http://$ip_addr:$api_port/health"
echo -e "- Grafana: http://$ip_addr:$grafana_port (用户名:admin, 密码:admin)"
echo -e "========================================================"

# 提示常用命令
echo -e "${YELLOW}常用维护命令:${NC}"
echo -e "查看日志: docker-compose -p pollution-prediction-system logs -f app"
echo -e "停止服务: docker-compose -p pollution-prediction-system down"
echo -e "重启服务: docker-compose -p pollution-prediction-system restart"
echo -e ""
echo -e "${GREEN}部署完成!${NC}"

exit 0 
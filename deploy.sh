#!/bin/bash
# 电厂污染物排放预测系统部署脚本
# 用于在局域网环境中构建和部署系统

set -e

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 显示横幅
echo -e "${GREEN}"
echo "======================================================"
echo "    电厂污染物排放预测与发电数据中心智慧看板部署     "
echo "======================================================"
echo -e "${NC}"

# 检查Docker和Docker Compose是否已安装
echo -e "${YELLOW}[步骤1] 检查基础环境...${NC}"
if ! command -v docker &> /dev/null; then
    echo -e "${RED}错误: Docker未安装，请先安装Docker${NC}"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}错误: Docker Compose未安装，请先安装Docker Compose${NC}"
    exit 1
fi

echo -e "${GREEN}√ Docker和Docker Compose已安装${NC}"

# 显示Docker版本信息
docker --version
docker-compose --version

# 设置环境变量
echo -e "${YELLOW}[步骤2] 配置部署环境...${NC}"

# 询问是否使用默认端口
read -p "使用默认端口(8000)? [Y/n] " use_default_port
if [[ $use_default_port =~ ^[Nn]$ ]]; then
    read -p "请输入API服务端口: " api_port
else
    api_port=8000
fi

# 修改docker-compose.yml中的端口
if [ "$api_port" != "8000" ]; then
    echo "修改API端口为 $api_port..."
    sed -i "s/8000:8000/$api_port:8000/g" docker-compose.yml
fi

echo -e "${GREEN}√ 环境配置完成${NC}"

# 构建和启动服务
echo -e "${YELLOW}[步骤3] 构建和启动服务...${NC}"
docker-compose build --no-cache
docker-compose up -d

# 检查服务是否正常启动
echo -e "${YELLOW}[步骤4] 验证服务状态...${NC}"
if docker-compose ps | grep "api" | grep "Up"; then
    echo -e "${GREEN}√ 服务已成功启动${NC}"
    
    # 获取本机IP地址
    if command -v ip &> /dev/null; then
        ip_addr=$(ip -4 addr show scope global | grep -oP '(?<=inet\s)\d+(\.\d+){3}' | head -n 1)
    elif command -v ifconfig &> /dev/null; then
        ip_addr=$(ifconfig | grep -Eo 'inet (addr:)?([0-9]*\.){3}[0-9]*' | grep -Eo '([0-9]*\.){3}[0-9]*' | grep -v '127.0.0.1' | head -n 1)
    else
        ip_addr="<服务器IP>"
    fi
    
    echo -e "${GREEN}======================================================"
    echo -e "系统已成功部署!"
    echo -e "API文档: http://$ip_addr:$api_port/docs"
    echo -e "健康检查: http://$ip_addr:$api_port/health"
    echo -e "======================================================${NC}"
else
    echo -e "${RED}! 服务启动异常，请检查日志:${NC}"
    docker-compose logs
fi

# 提示查看日志的命令
echo -e "${YELLOW}可随时使用以下命令查看日志:${NC}"
echo -e "docker-compose logs -f api"
echo -e "${YELLOW}可使用以下命令停止服务:${NC}"
echo -e "docker-compose down"

exit 0 
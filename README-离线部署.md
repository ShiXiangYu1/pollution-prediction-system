# 电厂污染物排放预测系统 - 离线部署包

## 概述

本项目是电厂污染物排放预测系统的离线部署包，专为方天公司局域网环境设计，便于向领导及专家评审进行题目2的演示。系统基于Docker容器化技术，实现了服务组件的解耦和可移植性，并解决了服务之间调用的硬编码IP问题。

## 部署包内容

本部署包包含以下内容：

- `docker-compose.yml`: 服务编排配置文件
- `offline_deploy.sh`: Linux环境下的一键部署脚本
- `offline_deploy.bat`: Windows环境下的一键部署脚本
- `电厂污染物排放预测系统离线部署指南.md`: 详细部署说明
- `镜像构建说明.md`: Docker镜像构建与优化说明
- `README-离线部署.md`: 本文件

## 系统架构

系统由以下核心组件组成：

1. **主应用服务 (App)**
   - 基于FastAPI框架
   - 提供REST API接口
   - 集成深度学习模型进行排放预测
   - 支持自然语言查询和语音识别功能

2. **数据库服务 (PostgreSQL)**
   - 存储污染物排放历史数据
   - 保存分析结果和系统配置

3. **缓存服务 (Redis)**
   - 加速数据访问
   - 临时存储分析中间结果

4. **监控服务 (Prometheus)**
   - 收集系统性能指标
   - 监控服务健康状态

5. **仪表板服务 (Grafana)**
   - 可视化系统性能数据
   - 提供自定义监控面板

## 快速开始

### 环境要求

- Docker Engine 19.03+
- Docker Compose 1.27+
- 8GB+ 内存
- 50GB+ 磁盘空间

### 部署步骤

#### Windows环境

1. 解压部署包到任意目录
2. 双击运行 `offline_deploy.bat`
3. 按提示操作完成部署
4. 访问 http://localhost:8000 开始使用系统

#### Linux环境

1. 解压部署包到任意目录
2. 添加执行权限：`chmod +x offline_deploy.sh`
3. 运行部署脚本：`./offline_deploy.sh`
4. 按提示操作完成部署
5. 访问 http://localhost:8000 开始使用系统

## 镜像拉取命令

如果需要单独拉取镜像，可使用以下命令：

```bash
# 拉取主应用镜像
docker pull 294802186/pollution-prediction:v1.1

# 拉取依赖服务镜像
docker pull postgres:14-alpine
docker pull redis:6-alpine
docker pull prom/prometheus:latest
docker pull grafana/grafana:latest
```

## 离线部署说明

对于完全离线环境，建议先在有网络连接的环境中拉取镜像，然后打包为tar文件：

```bash
# 保存所有镜像到一个tar文件
docker save -o pollution-prediction-images.tar \
  294802186/pollution-prediction:v1.1 \
  postgres:14-alpine \
  redis:6-alpine \
  prom/prometheus:latest \
  grafana/grafana:latest
```

将生成的tar文件与部署脚本一起复制到离线环境，然后运行部署脚本即可。

## 系统验证

部署完成后，请访问以下地址验证系统是否正常运行：

1. 系统主页: http://localhost:8000
2. 健康检查: http://localhost:8000/health
3. API文档: http://localhost:8000/docs
4. Grafana监控: http://localhost:3000 (用户名:admin, 密码:admin)

## 常见问题解决

常见问题及解决方法请参考 [电厂污染物排放预测系统离线部署指南.md](./电厂污染物排放预测系统离线部署指南.md#常见问题)。

## 更多信息

如需了解更多信息，请参考以下文档：

- [电厂污染物排放预测系统离线部署指南.md](./电厂污染物排放预测系统离线部署指南.md): 详细部署步骤
- [镜像构建说明.md](./镜像构建说明.md): Docker镜像构建与优化说明
- [电厂污染物排放预测系统综合部署指南.md](./电厂污染物排放预测系统综合部署指南.md): 完整系统部署指南

## 联系方式

如有问题或需要技术支持，请联系：

- 技术支持邮箱: support@powerplant-tech.com
- 技术支持电话: 400-123-4567 
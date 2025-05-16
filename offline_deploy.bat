@echo off
REM 电厂污染物排放预测系统 - 离线环境一键部署脚本（Windows版）
REM 用于在方天公司离线环境（局域网）中部署系统
REM 最后更新：2024-05-10

echo ========================================================
echo     电厂污染物排放预测系统 - 离线环境一键部署脚本      
echo ========================================================
echo.

REM 检查Docker和Docker Compose是否已安装
echo [步骤1] 检查基础环境...
where docker >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo 错误: Docker未安装，请先安装Docker
    exit /b 1
)

docker --version
docker-compose --version 2>nul || docker compose version 2>nul
if %ERRORLEVEL% neq 0 (
    echo 错误: Docker Compose未安装或不可用，请安装Docker Compose
    exit /b 1
)

echo Docker和Docker Compose已安装

REM 检查镜像包是否存在
echo [步骤2] 检查镜像包...

set IMAGE_PACKAGE=pollution-prediction-images.tar
if not exist %IMAGE_PACKAGE% (
    echo 警告: 未找到镜像包 %IMAGE_PACKAGE%
    echo 请确认以下两种方式之一:
    echo 1. 已通过其他方式导入了所需镜像
    echo 2. 当前环境可以通过网络拉取镜像
    
    set /p continue_deploy="是否继续部署? [Y/n] "
    if /i "%continue_deploy%"=="n" (
        echo 部署已取消
        exit /b 0
    )
) else (
    echo 发现镜像包 %IMAGE_PACKAGE%
    echo 正在导入镜像，这可能需要几分钟时间...
    docker load -i %IMAGE_PACKAGE%
    echo 镜像导入完成
)

REM 检查必要文件
echo [步骤3] 检查配置文件...
if not exist docker-compose.yml (
    echo 错误: 未找到docker-compose.yml文件
    exit /b 1
)

REM 创建必要的目录
echo [步骤4] 创建必要目录...
if not exist data mkdir data
if not exist logs mkdir logs
if not exist models mkdir models
if not exist static mkdir static
if not exist templates mkdir templates
if not exist prometheus mkdir prometheus
if not exist init-scripts\db mkdir init-scripts\db

REM 如果prometheus目录中没有配置文件，则创建一个基本配置
if not exist prometheus\prometheus.yml (
    echo 创建基本的Prometheus配置文件...
    (
        echo global:
        echo   scrape_interval: 15s
        echo   evaluation_interval: 15s
        echo.
        echo scrape_configs:
        echo   - job_name: 'prometheus'
        echo     static_configs:
        echo       - targets: ['localhost:9090']
        echo.
        echo   - job_name: 'pollution_prediction_app'
        echo     static_configs:
        echo       - targets: ['app:8000']
    ) > prometheus\prometheus.yml
    echo Prometheus配置文件已创建
)

REM 询问是否使用默认端口
echo [步骤5] 配置服务端口...
set /p use_default_ports="使用默认端口配置? [Y/n] "
if /i "%use_default_ports%"=="n" (
    echo 请配置服务端口:
    set /p api_port="API服务端口 (默认: 8000): "
    set /p db_port="数据库端口 (默认: 5432): "
    set /p redis_port="Redis端口 (默认: 6379): "
    set /p prometheus_port="Prometheus端口 (默认: 9090): "
    set /p grafana_port="Grafana端口 (默认: 3000): "
    
    REM 设置默认值
    if "%api_port%"=="" set api_port=8000
    if "%db_port%"=="" set db_port=5432
    if "%redis_port%"=="" set redis_port=6379
    if "%prometheus_port%"=="" set prometheus_port=9090
    if "%grafana_port%"=="" set grafana_port=3000
    
    REM 创建临时文件进行替换（Windows下的sed替代方案）
    type docker-compose.yml > docker-compose.tmp
    powershell -Command "(Get-Content docker-compose.tmp) -replace '8000:8000', '%api_port%:8000' | Set-Content docker-compose.yml"
    powershell -Command "(Get-Content docker-compose.yml) -replace '5432:5432', '%db_port%:5432' | Set-Content docker-compose.tmp"
    powershell -Command "(Get-Content docker-compose.tmp) -replace '6379:6379', '%redis_port%:6379' | Set-Content docker-compose.yml"
    powershell -Command "(Get-Content docker-compose.yml) -replace '9090:9090', '%prometheus_port%:9090' | Set-Content docker-compose.tmp"
    powershell -Command "(Get-Content docker-compose.tmp) -replace '3000:3000', '%grafana_port%:3000' | Set-Content docker-compose.yml"
    del docker-compose.tmp
    
    echo 端口配置已更新
) else (
    set api_port=8000
    set db_port=5432
    set redis_port=6379
    set prometheus_port=9090
    set grafana_port=3000
    echo 使用默认端口配置
)

REM 启动服务
echo [步骤6] 启动服务...
docker-compose -p pollution-prediction-system up -d

REM 检查服务状态
echo [步骤7] 检查服务状态...
timeout /t 10 /nobreak >nul

docker-compose -p pollution-prediction-system ps | findstr "app" | findstr "Up"
if %ERRORLEVEL% equ 0 (
    echo 主应用服务已成功启动
) else (
    echo 主应用服务启动异常，请检查日志
    docker-compose -p pollution-prediction-system logs app
)

REM 获取本机IP地址
for /f "tokens=2 delims=:" %%a in ('ipconfig ^| findstr /r /c:"IPv4 Address" ^| findstr /v /c:"127.0.0.1"') do (
    set ip_addr=%%a
    set ip_addr=!ip_addr:~1!
    goto :got_ip
)
:got_ip

REM 显示访问信息
echo ========================================================
echo 电厂污染物排放预测系统已成功部署!
echo.
echo 系统访问地址:
echo - 本地访问: http://localhost:%api_port%
if defined ip_addr (
    echo - 局域网访问: http://%ip_addr%:%api_port%
)
echo.
echo 其他服务地址:
echo - API文档: http://localhost:%api_port%/docs
echo - 健康检查: http://localhost:%api_port%/health
echo - Grafana: http://localhost:%grafana_port% (用户名:admin, 密码:admin)
echo ========================================================

REM 提示常用命令
echo 常用维护命令:
echo 查看日志: docker-compose -p pollution-prediction-system logs -f app
echo 停止服务: docker-compose -p pollution-prediction-system down
echo 重启服务: docker-compose -p pollution-prediction-system restart
echo.
echo 部署完成!

pause 
@echo off
setlocal enabledelayedexpansion
REM 电厂污染物排放预测系统部署脚本(Windows)
REM 用于在Windows环境中构建和部署系统

echo ======================================================
echo     电厂污染物排放预测与发电数据中心智慧看板部署     
echo ======================================================
echo.

REM 检查Docker和Docker Compose是否已安装
echo [步骤1] 检查基础环境...
docker --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo 错误: Docker未安装或未启动，请先安装并启动Docker
    pause
    exit /b 1
)

docker-compose --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo 错误: Docker Compose未安装或未启动，请先安装Docker Compose
    pause
    exit /b 1
)

echo Docker和Docker Compose已安装

REM 显示Docker版本信息
docker --version
docker-compose --version

REM 设置环境变量
echo.
echo [步骤2] 配置部署环境...

set /p use_default_port=使用默认端口(8000)? [Y/n] 
if /i "%use_default_port%"=="n" (
    set /p api_port=请输入API服务端口: 
) else (
    set api_port=8000
)

REM 修改docker-compose.yml中的端口(Windows版本使用FINDSTR和临时文件)
if NOT "%api_port%"=="8000" (
    echo 修改API端口为 %api_port%...
    type docker-compose.yml | findstr /V "8000:8000" > docker-compose.tmp
    echo       - "%api_port%:8000" >> docker-compose.tmp
    move /y docker-compose.tmp docker-compose.yml
)

echo 环境配置完成

REM 构建和启动服务
echo.
echo [步骤3] 构建和启动服务...
REM 使用-p参数明确指定项目名称
docker compose -p pollution-prediction-system build --no-cache
docker compose -p pollution-prediction-system up -d

REM 检查服务是否正常启动
echo.
echo [步骤4] 验证服务状态...
docker compose -p pollution-prediction-system ps | findstr "api.*Up" >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo 服务已成功启动
    
    REM 获取本机IP地址
    for /f "tokens=2 delims=:" %%a in ('ipconfig ^| findstr /r /c:"IPv4 Address"') do (
        set ip_addr=%%a
        set ip_addr=!ip_addr:~1!
        goto :break_loop
    )
    :break_loop
    
    echo ======================================================
    echo 系统已成功部署!
    echo API文档: http://localhost:%api_port%/docs
    if defined ip_addr (
        echo 局域网访问: http://!ip_addr!:%api_port%/docs
    )
    echo 健康检查: http://localhost:%api_port%/health
    echo ======================================================
) else (
    echo ! 服务启动异常，请检查日志:
    docker compose -p pollution-prediction-system logs
)

REM 提示查看日志的命令
echo.
echo 可随时使用以下命令查看日志:
echo docker compose -p pollution-prediction-system logs -f api
echo.
echo 可使用以下命令停止服务:
echo docker compose -p pollution-prediction-system down
echo.

pause
endlocal 
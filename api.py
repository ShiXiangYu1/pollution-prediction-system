#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API服务入口 - 提供系统功能的RESTful API接口
"""

import os
import sys
import numpy as np
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Query, Request
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import json
import tempfile
import uvicorn
from datetime import datetime, timedelta
import logging
import shutil
import io
import random

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("api-service")

# 获取环境变量，决定是否使用模拟模式
USE_MOCK_MODE = os.environ.get("USE_MOCK_MODE", "true").lower() == "true"
USE_MOCK_TORCH = os.environ.get("USE_MOCK_TORCH", "true").lower() == "true"
USE_MOCK_JINJA2 = os.environ.get("USE_MOCK_JINJA2", "true").lower() == "true"

logger.info(f"模拟模式配置: USE_MOCK_MODE={USE_MOCK_MODE}, USE_MOCK_TORCH={USE_MOCK_TORCH}, USE_MOCK_JINJA2={USE_MOCK_JINJA2}")

# 添加模拟的torch模块
class MockTorch:
    """模拟的PyTorch模块，用于开发和测试"""
    @staticmethod
    def load(path):
        print(f"模拟加载模型: {path}")
        return "模拟模型"
    
    class nn:
        class Module:
            def __init__(self):
                pass
            def eval(self):
                return self
    
    @staticmethod
    def device(device_str):
        return device_str
    
    @staticmethod
    def no_grad():
        class NoGrad:
            def __enter__(self):
                return None
            def __exit__(self, exc_type, exc_val, exc_tb):
                pass
        return NoGrad()

# 只在需要时替换torch模块
if USE_MOCK_TORCH:
    sys.modules['torch'] = MockTorch()
    torch = MockTorch()
    logger.info("使用模拟PyTorch模块")
else:
    try:
        import torch
        logger.info("成功导入真实PyTorch模块")
    except ImportError:
        logger.warning("无法导入真实PyTorch模块，自动切换到模拟模式")
        sys.modules['torch'] = MockTorch()
        torch = MockTorch()

# 添加模拟的Jinja2Templates类
class MockJinja2Templates:
    """模拟的Jinja2Templates类，用于开发和测试"""
    def __init__(self, directory):
        self.directory = directory
        print(f"模拟加载模板目录: {directory}")
    
    def TemplateResponse(self, template_name, context):
        """模拟返回模板响应"""
        from fastapi.responses import HTMLResponse
        return HTMLResponse(content=f"<html><body><h1>模拟模板: {template_name}</h1><p>上下文: {context}</p></body></html>")

# 在这里添加模拟的starlette.templating模块，以便在导入时不会报错
class MockTemplating:
    def __init__(self):
        self.Jinja2Templates = MockJinja2Templates

# 只在需要时注入模拟模块
if USE_MOCK_JINJA2:
    sys.modules['starlette.templating'] = MockTemplating()
    from starlette.templating import Jinja2Templates
    # 确保我们的Jinja2Templates是模拟版本
    Jinja2Templates = MockJinja2Templates
    sys.modules['jinja2'] = object()  # 添加一个空模块对象来防止jinja2导入错误
    templates = MockJinja2Templates(directory="templates")
    logger.info("使用模拟Jinja2Templates")
else:
    try:
        from starlette.templating import Jinja2Templates
        templates = Jinja2Templates(directory="templates")
        logger.info("成功导入真实Jinja2Templates")
    except ImportError:
        logger.warning("无法导入真实Jinja2Templates，自动切换到模拟模式")
        sys.modules['starlette.templating'] = MockTemplating()
        from starlette.templating import Jinja2Templates
        Jinja2Templates = MockJinja2Templates
        sys.modules['jinja2'] = object()
        templates = MockJinja2Templates(directory="templates")
        logger.info("使用模拟Jinja2Templates（自动降级）")

# 导入自定义模块
try:
    from model_training.pollution_prediction import PollutionPredictionModel
    from model_training.nl2sql_converter import NL2SQLConverter
    from model_training.speech_recognition import SpeechRecognitionIntegrator
    from data_processing.feature_engineering import FeatureEngineer
    from data_processing.dataset_builder import DatasetBuilder
    
    logger.info("成功导入所有模块")
except ImportError as e:
    logger.error(f"导入模块失败: {e}")
    # 提供调试信息
    logger.error(f"当前工作目录: {os.getcwd()}")
    logger.error(f"Python路径: {sys.path}")

# 创建FastAPI应用
app = FastAPI(
    title="电厂污染物排放预测与发电数据中心智慧看板API",
    description="提供污染物排放预测、自然语言查询转SQL、语音识别等功能的API接口",
    version="1.0.0"
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有源，在生产环境中应该限制为特定源
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 数据目录
DATA_DIR = os.environ.get("DATA_DIR", "./data")
os.makedirs(DATA_DIR, exist_ok=True)

# 创建必要的子目录
for subdir in ["models", "uploads", "results", "temp"]:
    os.makedirs(os.path.join(DATA_DIR, subdir), exist_ok=True)

# 挂载静态文件
app.mount("/static", StaticFiles(directory="static"), name="static")

# 定义请求和响应模型
class PredictionRequest(BaseModel):
    """预测请求模型"""
    features: List[List[float]]
    model_type: Optional[str] = "lstm"  # lstm, gru, transformer

class NLPQueryRequest(BaseModel):
    """自然语言查询请求模型"""
    query: str
    mode: Optional[str] = "text"  # text, speech

class TextToSQLRequest(BaseModel):
    """文本到SQL请求模型"""
    text: str

class TextToSpeechRequest(BaseModel):
    """文本转语音请求模型"""
    text: str

# 前端页面路由
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """首页"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard_page(request: Request):
    """数据看板页面"""
    return templates.TemplateResponse("dashboard.html", {"request": request})

@app.get("/prediction", response_class=HTMLResponse)
async def prediction_page(request: Request):
    """污染物排放预测页面"""
    return templates.TemplateResponse("prediction.html", {"request": request})

@app.get("/nlp", response_class=HTMLResponse)
async def nlp_page(request: Request):
    """自然语言查询页面"""
    return templates.TemplateResponse("nlp_query.html", {"request": request})

@app.get("/speech", response_class=HTMLResponse)
async def speech_page(request: Request):
    """语音识别页面"""
    return templates.TemplateResponse("speech.html", {"request": request})

# 健康检查接口
@app.get("/health")
async def health_check():
    """健康检查接口"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

# 污染物排放预测接口
@app.post("/predict")
async def predict_pollution(request: PredictionRequest):
    """污染物排放预测接口"""
    try:
        logger.info(f"收到预测请求: {request.model_type}")
        
        # 添加输入特征验证
        required_features = 10  # 定义所需的最小特征数量
        for i, features in enumerate(request.features):
            if len(features) < required_features:
                error_msg = f"特征组 {i+1} 包含 {len(features)} 个特征，少于所需的 {required_features} 个特征"
                logger.warning(f"输入验证失败: {error_msg}")
                raise HTTPException(status_code=400, detail=error_msg)
        
        # 在离线环境下，我们使用模拟数据
        # 真实环境中应该加载模型进行预测
        
        # 生成模拟预测结果
        predictions = []
        for features in request.features:
            # 根据输入特征生成预测结果
            # 实际应用中应该使用模型进行预测
            so2_pred = 20 + 0.1 * features[0] + 0.05 * features[1] - 0.03 * features[2] + random.uniform(-2, 2)
            nox_pred = 40 + 0.15 * features[0] + 0.08 * features[1] - 0.05 * features[2] + random.uniform(-3, 3)
            dust_pred = 8 + 0.05 * features[0] + 0.02 * features[1] - 0.01 * features[2] + random.uniform(-1, 1)
            
            # 确保数值合理
            so2_pred = max(5, min(50, so2_pred))
            nox_pred = max(10, min(70, nox_pred))
            dust_pred = max(2, min(15, dust_pred))
            
            predictions.append([so2_pred, nox_pred, dust_pred])
        
        # 返回结果
        return {
            "predictions": predictions,
            "model_type": request.model_type,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException as he:
        # 重新抛出HTTP异常，保持原始状态码
        raise he
    except Exception as e:
        logger.error(f"预测过程中出错: {e}")
        raise HTTPException(status_code=500, detail=f"预测失败: {str(e)}")

# 自然语言查询接口
@app.post("/api/nlp_query")
async def nlp_query(request: NLPQueryRequest):
    """自然语言查询接口"""
    try:
        logger.info(f"收到自然语言查询请求: {request.query}")
        
        # 解析自然语言查询
        query = request.query.lower()
        
        # 简单的关键词匹配来模拟自然语言理解
        response = {"answer": "对不起，我无法理解您的问题。"}
        
        # 排放相关查询
        if "so2" in query or "二氧化硫" in query:
            if "平均" in query:
                response = {
                    "answer": "最近24小时的SO2平均排放浓度为18.5 mg/m³，低于标准限值(35 mg/m³)。",
                    "chart_data": {
                        "type": "line",
                        "title": "SO2排放趋势",
                        "xLabel": "时间",
                        "yLabel": "浓度 (mg/m³)",
                        "labels": [f"{(datetime.now() - timedelta(hours=x)).strftime('%H:%M')}" for x in range(24, 0, -1)],
                        "datasets": [
                            {
                                "label": "SO2浓度",
                                "data": [round(15 + 5 * np.random.random(), 2) for _ in range(24)],
                                "color": "rgba(255, 99, 132, 1)"
                            },
                            {
                                "label": "标准限值",
                                "data": [35] * 24,
                                "color": "rgba(255, 99, 132, 0.5)"
                            }
                        ]
                    }
                }
        elif "nox" in query or "氮氧化物" in query:
            if "平均" in query:
                response = {
                    "answer": "最近24小时的NOx平均排放浓度为37.2 mg/m³，低于标准限值(50 mg/m³)。",
                    "chart_data": {
                        "type": "line",
                        "title": "NOx排放趋势",
                        "xLabel": "时间",
                        "yLabel": "浓度 (mg/m³)",
                        "labels": [f"{(datetime.now() - timedelta(hours=x)).strftime('%H:%M')}" for x in range(24, 0, -1)],
                        "datasets": [
                            {
                                "label": "NOx浓度",
                                "data": [round(30 + 10 * np.random.random(), 2) for _ in range(24)],
                                "color": "rgba(54, 162, 235, 1)"
                            },
                            {
                                "label": "标准限值",
                                "data": [50] * 24,
                                "color": "rgba(54, 162, 235, 0.5)"
                            }
                        ]
                    }
                }
        elif "烟尘" in query or "灰尘" in query or "dust" in query:
            if "平均" in query:
                response = {
                    "answer": "最近24小时的烟尘平均排放浓度为8.7 mg/m³，低于标准限值(10 mg/m³)。",
                    "chart_data": {
                        "type": "line",
                        "title": "烟尘排放趋势",
                        "xLabel": "时间",
                        "yLabel": "浓度 (mg/m³)",
                        "labels": [f"{(datetime.now() - timedelta(hours=x)).strftime('%H:%M')}" for x in range(24, 0, -1)],
                        "datasets": [
                            {
                                "label": "烟尘浓度",
                                "data": [round(8 + 3 * np.random.random(), 2) for _ in range(24)],
                                "color": "rgba(255, 206, 86, 1)"
                            },
                            {
                                "label": "标准限值",
                                "data": [10] * 24,
                                "color": "rgba(255, 206, 86, 0.5)"
                            }
                        ]
                    }
                }
        
        # 机组相关查询
        if "机组" in query:
            unit_number = None
            for word in query.split():
                if word.isdigit():
                    unit_number = int(word)
                    break
                elif word.startswith("#") and word[1:].isdigit():
                    unit_number = int(word[1:])
                    break
            
            if unit_number is not None:
                if "排放" in query:
                    response = {
                        "answer": f"{unit_number}号机组的排放情况良好，所有指标均低于标准限值。",
                        "table_data": {
                            "headers": ["参数", "当前值", "平均值", "最大值", "标准限值", "状态"],
                            "rows": [
                                ["SO2 (mg/m³)", "17.2", "18.5", "23.1", "35", "正常"],
                                ["NOx (mg/m³)", "35.6", "37.2", "45.8", "50", "正常"],
                                ["烟尘 (mg/m³)", "7.8", "8.7", "9.5", "10", "正常"]
                            ]
                        }
                    }
                elif "状态" in query:
                    response = {
                        "answer": f"{unit_number}号机组当前运行状态正常，负荷率78%，无异常报警。",
                        "chart_data": {
                            "type": "pie",
                            "title": f"{unit_number}号机组运行状态分布",
                            "labels": ["正常运行", "低负荷运行", "检修", "停机"],
                            "datasets": [
                                {
                                    "data": [78, 12, 8, 2],
                                    "colors": ["rgba(75, 192, 192, 0.7)", "rgba(255, 205, 86, 0.7)", "rgba(255, 99, 132, 0.7)", "rgba(201, 203, 207, 0.7)"]
                                }
                            ]
                        }
                    }
        
        # 超标情况
        if "超标" in query:
            if "次数" in query:
                response = {
                    "answer": "过去30天内共有5次污染物超标情况，其中SO2超标2次，NOx超标2次，烟尘超标1次。",
                    "chart_data": {
                        "type": "bar",
                        "title": "近30天污染物超标次数",
                        "xLabel": "污染物类型",
                        "yLabel": "超标次数",
                        "labels": ["SO2", "NOx", "烟尘"],
                        "datasets": [
                            {
                                "label": "超标次数",
                                "data": [2, 2, 1],
                                "color": "rgba(255, 99, 132, 0.7)"
                            }
                        ]
                    },
                    "table_data": {
                        "headers": ["日期", "时间", "机组", "污染物", "实测值 (mg/m³)", "标准值 (mg/m³)", "超标率"],
                        "rows": [
                            [f"{(datetime.now() - timedelta(days=3)).strftime('%Y-%m-%d')}", "13:45", "1号机组", "SO2", "37.8", "35.0", "8.0%"],
                            [f"{(datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')}", "08:30", "2号机组", "NOx", "53.2", "50.0", "6.4%"],
                            [f"{(datetime.now() - timedelta(days=12)).strftime('%Y-%m-%d')}", "21:15", "1号机组", "SO2", "36.5", "35.0", "4.3%"],
                            [f"{(datetime.now() - timedelta(days=18)).strftime('%Y-%m-%d')}", "14:20", "2号机组", "NOx", "52.7", "50.0", "5.4%"],
                            [f"{(datetime.now() - timedelta(days=25)).strftime('%Y-%m-%d')}", "10:10", "3号机组", "烟尘", "10.8", "10.0", "8.0%"]
                        ]
                    }
                }
        
        # 排名或比较
        if "排名" in query or "哪个" in query or "最" in query:
            if "最低" in query or "最少" in query:
                response = {
                    "answer": "3号机组的污染物排放量最低，其SO2、NOx和烟尘的平均排放浓度分别为15.3 mg/m³、32.1 mg/m³和6.8 mg/m³。",
                    "chart_data": {
                        "type": "bar",
                        "title": "各机组污染物排放对比",
                        "xLabel": "机组",
                        "yLabel": "浓度 (mg/m³)",
                        "labels": ["1号机组", "2号机组", "3号机组", "4号机组"],
                        "datasets": [
                            {
                                "label": "SO2",
                                "data": [18.5, 19.2, 15.3, 17.8],
                                "color": "rgba(255, 99, 132, 0.7)"
                            },
                            {
                                "label": "NOx",
                                "data": [37.2, 38.5, 32.1, 35.6],
                                "color": "rgba(54, 162, 235, 0.7)"
                            },
                            {
                                "label": "烟尘",
                                "data": [8.7, 9.1, 6.8, 8.2],
                                "color": "rgba(255, 206, 86, 0.7)"
                            }
                        ]
                    }
                }
            elif "最高" in query or "最多" in query:
                response = {
                    "answer": "2号机组的污染物排放量最高，其SO2、NOx和烟尘的平均排放浓度分别为19.2 mg/m³、38.5 mg/m³和9.1 mg/m³，但仍在标准限值范围内。",
                    "chart_data": {
                        "type": "bar",
                        "title": "各机组污染物排放对比",
                        "xLabel": "机组",
                        "yLabel": "浓度 (mg/m³)",
                        "labels": ["1号机组", "2号机组", "3号机组", "4号机组"],
                        "datasets": [
                            {
                                "label": "SO2",
                                "data": [18.5, 19.2, 15.3, 17.8],
                                "color": "rgba(255, 99, 132, 0.7)"
                            },
                            {
                                "label": "NOx",
                                "data": [37.2, 38.5, 32.1, 35.6],
                                "color": "rgba(54, 162, 235, 0.7)"
                            },
                            {
                                "label": "烟尘",
                                "data": [8.7, 9.1, 6.8, 8.2],
                                "color": "rgba(255, 206, 86, 0.7)"
                            }
                        ]
                    }
                }
        
        # 趋势查询
        if "趋势" in query or "变化" in query:
            response = {
                "answer": "过去7天的污染物排放呈现小幅波动趋势，总体保持稳定并低于标准限值。其中周末期间排放量略有下降，这与机组负荷变化相关。",
                "chart_data": {
                    "type": "line",
                    "title": "过去7天污染物排放趋势",
                    "xLabel": "日期",
                    "yLabel": "浓度 (mg/m³)",
                    "labels": [(datetime.now() - timedelta(days=x)).strftime('%m-%d') for x in range(7, 0, -1)],
                    "datasets": [
                        {
                            "label": "SO2",
                            "data": [round(15 + 5 * np.random.random(), 2) for _ in range(7)],
                            "color": "rgba(255, 99, 132, 1)"
                        },
                        {
                            "label": "NOx",
                            "data": [round(30 + 10 * np.random.random(), 2) for _ in range(7)],
                            "color": "rgba(54, 162, 235, 1)"
                        },
                        {
                            "label": "烟尘",
                            "data": [round(8 + 3 * np.random.random(), 2) for _ in range(7)],
                            "color": "rgba(255, 206, 86, 1)"
                        }
                    ]
                }
            }
        
        # 返回响应
        return response
        
    except Exception as e:
        logger.error(f"自然语言查询处理出错: {e}")
        raise HTTPException(status_code=500, detail=f"查询处理失败: {str(e)}")

# 自然语言转SQL接口
@app.post("/nl2sql")
async def convert_nl_to_sql(request: TextToSQLRequest):
    """自然语言转SQL接口"""
    try:
        logger.info(f"收到NL2SQL请求: {request.text}")
        
        # 初始化转换器
        model_path = os.path.join(DATA_DIR, "models", "nl2sql", "model")
        converter = NL2SQLConverter(model_path=model_path, data_dir=DATA_DIR)
        
        # 定义简单的数据库模式信息
        schema_info = """
        排放数据表(emissions): 时间戳(timestamp), 机组ID(unit_id), 二氧化硫(SO2), 氮氧化物(NOx), 烟尘(dust), 流量(flow), 温度(temperature)
        机组信息表(units): 机组ID(unit_id), 机组名称(unit_name), 电厂ID(plant_id), 类型(type), 容量(capacity)
        电厂信息表(plants): 电厂ID(plant_id), 电厂名称(plant_name), 位置(location), 类型(type)
        """
        
        # 转换文本到SQL
        sql_query = converter.convert_nl_to_sql(request.text, schema_info)
        
        # 返回结果
        return {
            "text": request.text,
            "sql_query": sql_query,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"NL2SQL转换过程中出错: {e}")
        raise HTTPException(status_code=500, detail=f"转换失败: {str(e)}")

# 语音识别接口
@app.post("/speech-to-text")
async def convert_speech_to_text(file: UploadFile = File(...)):
    """语音识别接口"""
    try:
        logger.info(f"收到语音识别请求: {file.filename}")
        
        # 保存上传的音频文件
        temp_file = os.path.join(DATA_DIR, "temp", file.filename)
        with open(temp_file, "wb") as f:
            shutil.copyfileobj(file.file, f)
        
        # 初始化语音识别器
        model_name = os.path.join(DATA_DIR, "models", "speech")
        if not os.path.exists(model_name):
            model_name = "openai/whisper-small"  # 使用默认模型
        
        integrator = SpeechRecognitionIntegrator(model_name=model_name, data_dir=DATA_DIR)
        
        # 转录音频
        processor, model = integrator.load_model()
        text = integrator.transcribe_audio(temp_file, model, processor)
        
        # 删除临时文件
        os.remove(temp_file)
        
        # 返回结果
        return {
            "filename": file.filename,
            "text": text,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"语音识别过程中出错: {e}")
        raise HTTPException(status_code=500, detail=f"语音识别失败: {str(e)}")

# 获取模拟排放数据接口
@app.get("/api/emissions")
async def get_emissions_data(
    page: int = Query(1, ge=1, description="页码，从1开始"),
    page_size: int = Query(10, ge=1, le=100, description="每页数据条数"),
    type: Optional[str] = Query(None, description="污染物类型，如SO2、NOx、dust"),
    start_date: Optional[str] = Query(None, description="开始日期，格式YYYY-MM-DD"),
    end_date: Optional[str] = Query(None, description="结束日期，格式YYYY-MM-DD"),
    unit_id: Optional[int] = Query(None, description="机组ID")
):
    """获取模拟排放数据，支持分页和过滤"""
    try:
        # 生成模拟数据：最近24小时的排放数据
        now = datetime.now()
        data = []
        
        for hour in range(24):
            time_point = now.replace(hour=hour, minute=0, second=0, microsecond=0)
            data.append({
                "timestamp": time_point.isoformat(),
                "unit_id": 1,
                "SO2": round(15 + 5 * np.random.random(), 2),
                "NOx": round(30 + 10 * np.random.random(), 2),
                "dust": round(8 + 3 * np.random.random(), 2)
            })
            
            data.append({
                "timestamp": time_point.isoformat(),
                "unit_id": 2,
                "SO2": round(18 + 7 * np.random.random(), 2),
                "NOx": round(35 + 8 * np.random.random(), 2),
                "dust": round(7 + 4 * np.random.random(), 2)
            })
        
        # 应用过滤条件
        filtered_data = data.copy()
        
        # 按污染物类型过滤
        if type:
            if type.upper() in ["SO2", "NOX", "DUST"]:
                # 这里我们不做实际过滤，因为是模拟数据
                logger.info(f"按污染物类型过滤: {type}")
            else:
                logger.warning(f"未知的污染物类型: {type}")
        
        # 按日期范围过滤
        if start_date:
            try:
                start_datetime = datetime.fromisoformat(start_date + "T00:00:00")
                # 在实际实现中会过滤数据
                logger.info(f"按开始日期过滤: {start_date}")
            except ValueError:
                logger.warning(f"无效的开始日期格式: {start_date}")
        
        if end_date:
            try:
                end_datetime = datetime.fromisoformat(end_date + "T23:59:59")
                # 在实际实现中会过滤数据
                logger.info(f"按结束日期过滤: {end_date}")
            except ValueError:
                logger.warning(f"无效的结束日期格式: {end_date}")
        
        # 按机组ID过滤
        if unit_id is not None:
            filtered_data = [item for item in filtered_data if item["unit_id"] == unit_id]
            logger.info(f"按机组ID过滤: {unit_id}")
        
        # 计算分页信息
        total_count = len(filtered_data)
        total_pages = (total_count + page_size - 1) // page_size
        
        # 确保页码有效
        if page > total_pages and total_pages > 0:
            page = total_pages
        
        # 计算数据切片的起始和结束索引
        start_idx = (page - 1) * page_size
        end_idx = min(start_idx + page_size, total_count)
        
        # 获取当前页的数据
        paginated_data = filtered_data[start_idx:end_idx]
        
        # 生成分页链接
        base_url = "/api/emissions"
        query_params = []
        if type:
            query_params.append(f"type={type}")
        if start_date:
            query_params.append(f"start_date={start_date}")
        if end_date:
            query_params.append(f"end_date={end_date}")
        if unit_id is not None:
            query_params.append(f"unit_id={unit_id}")
        query_params.append(f"page_size={page_size}")
        
        # 构建查询字符串
        query_string = "&".join(query_params)
        
        # 构建下一页和上一页链接
        next_page_url = f"{base_url}?{query_string}&page={page+1}" if page < total_pages else None
        prev_page_url = f"{base_url}?{query_string}&page={page-1}" if page > 1 else None
        
        # 返回带分页信息的数据，将关键分页信息移到顶层
        return {
            "data": paginated_data,
            "total": total_count,     # 移到顶层
            "page": page,            # 移到顶层
            "page_size": page_size,   # 移到顶层
            "total_pages": total_pages,
            "next": next_page_url,
            "previous": prev_page_url,
            "filters": {
                "type": type,
                "start_date": start_date,
                "end_date": end_date,
                "unit_id": unit_id
            }
        }
    
    except Exception as e:
        logger.error(f"获取排放数据时出错: {e}")
        raise HTTPException(status_code=500, detail=f"获取数据失败: {str(e)}")

# 主函数
if __name__ == "__main__":
    # 获取主机和端口，默认为0.0.0.0:8000
    host = os.environ.get("API_HOST", "0.0.0.0")
    port = int(os.environ.get("API_PORT", 8000))
    
    # 启动服务器
    logger.info(f"启动API服务: {host}:{port}")
    uvicorn.run("api:app", host=host, port=port, reload=True) 
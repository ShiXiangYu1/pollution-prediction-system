#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
错误处理模块 - 提供全局异常处理、错误记录、错误监控和友好提示
"""

import os
import sys
import traceback
import logging
import json
import time
from typing import Dict, Any, Optional, List, Callable, Type, Union
from datetime import datetime
from fastapi import FastAPI, Request, Response, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, ValidationError
import uuid

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("error_logs.log")
    ]
)
logger = logging.getLogger("error-handling")

# 自定义异常类
class APIError(Exception):
    """API错误基类"""
    def __init__(
        self, 
        message: str = "发生了未知错误", 
        status_code: int = 500, 
        error_code: str = "UNKNOWN_ERROR", 
        details: Optional[Dict] = None
    ):
        self.message = message
        self.status_code = status_code
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)

class ValidationAPIError(APIError):
    """验证错误"""
    def __init__(self, message: str = "输入验证失败", details: Optional[Dict] = None):
        super().__init__(
            message=message,
            status_code=400,
            error_code="VALIDATION_ERROR",
            details=details
        )

class AuthenticationAPIError(APIError):
    """认证错误"""
    def __init__(self, message: str = "认证失败", details: Optional[Dict] = None):
        super().__init__(
            message=message,
            status_code=401,
            error_code="AUTHENTICATION_ERROR",
            details=details
        )

class AuthorizationAPIError(APIError):
    """授权错误"""
    def __init__(self, message: str = "权限不足", details: Optional[Dict] = None):
        super().__init__(
            message=message,
            status_code=403,
            error_code="AUTHORIZATION_ERROR",
            details=details
        )

class ResourceNotFoundAPIError(APIError):
    """资源不存在错误"""
    def __init__(self, message: str = "请求的资源不存在", details: Optional[Dict] = None):
        super().__init__(
            message=message,
            status_code=404,
            error_code="RESOURCE_NOT_FOUND",
            details=details
        )

class RateLimitAPIError(APIError):
    """频率限制错误"""
    def __init__(self, message: str = "请求频率超过限制", details: Optional[Dict] = None):
        super().__init__(
            message=message,
            status_code=429,
            error_code="RATE_LIMIT_EXCEEDED",
            details=details
        )

class DatabaseAPIError(APIError):
    """数据库错误"""
    def __init__(self, message: str = "数据库操作失败", details: Optional[Dict] = None):
        super().__init__(
            message=message,
            status_code=500,
            error_code="DATABASE_ERROR",
            details=details
        )

class PredictionModelAPIError(APIError):
    """预测模型错误"""
    def __init__(self, message: str = "预测模型处理失败", details: Optional[Dict] = None):
        super().__init__(
            message=message,
            status_code=500,
            error_code="PREDICTION_MODEL_ERROR",
            details=details
        )

class NLPModelAPIError(APIError):
    """自然语言处理模型错误"""
    def __init__(self, message: str = "自然语言处理失败", details: Optional[Dict] = None):
        super().__init__(
            message=message,
            status_code=500,
            error_code="NLP_MODEL_ERROR",
            details=details
        )

class SpeechRecognitionAPIError(APIError):
    """语音识别错误"""
    def __init__(self, message: str = "语音识别失败", details: Optional[Dict] = None):
        super().__init__(
            message=message,
            status_code=500,
            error_code="SPEECH_RECOGNITION_ERROR",
            details=details
        )

# 错误记录类
class ErrorLogger:
    """错误记录类，负责记录和管理错误日志"""
    
    def __init__(self, log_dir: str = "./logs"):
        """初始化错误记录器"""
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(log_dir, "error_logs.json")
        self.in_memory_logs = []
        self.max_in_memory_logs = 100
    
    def log_error(self, 
                  error: Exception, 
                  request: Optional[Request] = None, 
                  additional_info: Optional[Dict] = None) -> Dict:
        """记录错误"""
        # 生成错误ID
        error_id = str(uuid.uuid4())
        
        # 创建错误日志
        error_log = {
            "error_id": error_id,
            "timestamp": datetime.now().isoformat(),
            "error_type": type(error).__name__,
            "error_message": str(error),
            "traceback": traceback.format_exc()
        }
        
        # 添加请求信息（如果有）
        if request:
            request_info = {
                "method": request.method,
                "url": str(request.url),
                "client_host": request.client.host if request.client else None,
                "headers": dict(request.headers),
            }
            error_log["request"] = request_info
        
        # 添加额外信息（如果有）
        if additional_info:
            error_log["additional_info"] = additional_info
        
        # 添加到内存日志
        self.in_memory_logs.append(error_log)
        if len(self.in_memory_logs) > self.max_in_memory_logs:
            self.in_memory_logs.pop(0)
        
        # 写入日志文件
        try:
            # 如果日志文件存在，加载现有日志
            existing_logs = []
            if os.path.exists(self.log_file):
                with open(self.log_file, "r", encoding="utf-8") as f:
                    existing_logs = json.load(f)
            
            # 添加新日志
            existing_logs.append(error_log)
            
            # 保存日志
            with open(self.log_file, "w", encoding="utf-8") as f:
                json.dump(existing_logs, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"写入错误日志文件时出错: {e}")
        
        # 记录到标准日志
        logger.error(f"错误: {error_log['error_type']} - {error_log['error_message']}")
        logger.debug(f"错误详情: {json.dumps(error_log, ensure_ascii=False)}")
        
        return error_log
    
    def get_recent_errors(self, limit: int = 10) -> List[Dict]:
        """获取最近的错误日志"""
        return self.in_memory_logs[-limit:] if self.in_memory_logs else []
    
    def get_error_by_id(self, error_id: str) -> Optional[Dict]:
        """根据错误ID获取错误日志"""
        for log in self.in_memory_logs:
            if log["error_id"] == error_id:
                return log
        
        # 如果内存中没有找到，尝试从文件中查找
        try:
            if os.path.exists(self.log_file):
                with open(self.log_file, "r", encoding="utf-8") as f:
                    logs = json.load(f)
                    for log in logs:
                        if log["error_id"] == error_id:
                            return log
        except Exception as e:
            logger.error(f"从文件读取错误日志时出错: {e}")
        
        return None
    
    def get_error_stats(self) -> Dict:
        """获取错误统计信息"""
        stats = {
            "total_errors": len(self.in_memory_logs),
            "error_types": {},
            "recent_errors_timestamp": []
        }
        
        for log in self.in_memory_logs:
            # 计算错误类型统计
            error_type = log["error_type"]
            if error_type in stats["error_types"]:
                stats["error_types"][error_type] += 1
            else:
                stats["error_types"][error_type] = 1
            
            # 添加最近错误的时间戳
            stats["recent_errors_timestamp"].append(log["timestamp"])
        
        return stats

# 错误处理器类
class ErrorHandler:
    """错误处理器类，负责统一处理API错误"""
    
    def __init__(self, app: FastAPI, error_logger: Optional[ErrorLogger] = None):
        """初始化错误处理器"""
        self.app = app
        self.error_logger = error_logger or ErrorLogger()
        
        # 注册异常处理器
        self.register_exception_handlers()
    
    def register_exception_handlers(self):
        """注册异常处理器"""
        # 注册自定义API错误处理器
        @self.app.exception_handler(APIError)
        async def api_error_handler(request: Request, exc: APIError) -> JSONResponse:
            return self.handle_api_error(request, exc)
        
        # 注册请求验证错误处理器
        @self.app.exception_handler(RequestValidationError)
        async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
            return self.handle_validation_error(request, exc)
        
        # 注册HTTP异常处理器
        @self.app.exception_handler(HTTPException)
        async def http_error_handler(request: Request, exc: HTTPException) -> JSONResponse:
            return self.handle_http_exception(request, exc)
        
        # 注册未处理异常处理器
        @self.app.exception_handler(Exception)
        async def general_error_handler(request: Request, exc: Exception) -> JSONResponse:
            return self.handle_general_exception(request, exc)
    
    def handle_api_error(self, request: Request, exc: APIError) -> JSONResponse:
        """处理API错误"""
        # 记录错误
        error_log = self.error_logger.log_error(exc, request)
        
        # 创建错误响应
        response_data = {
            "status": "error",
            "message": exc.message,
            "error_code": exc.error_code,
            "error_id": error_log["error_id"]
        }
        
        # 在开发环境中添加详细信息
        if os.environ.get("APP_ENV") == "development":
            response_data["details"] = exc.details
            response_data["traceback"] = error_log["traceback"]
        
        return JSONResponse(
            status_code=exc.status_code,
            content=response_data
        )
    
    def handle_validation_error(self, request: Request, exc: RequestValidationError) -> JSONResponse:
        """处理请求验证错误"""
        # 提取验证错误信息
        error_details = []
        for error in exc.errors():
            loc = " -> ".join(str(loc) for loc in error["loc"])
            error_details.append({
                "location": loc,
                "message": error["msg"],
                "type": error["type"]
            })
        
        # 创建API错误
        api_error = ValidationAPIError(
            message="请求数据验证失败",
            details={"validation_errors": error_details}
        )
        
        # 使用API错误处理器处理
        return self.handle_api_error(request, api_error)
    
    def handle_http_exception(self, request: Request, exc: HTTPException) -> JSONResponse:
        """处理HTTP异常"""
        # 创建API错误
        api_error = APIError(
            message=exc.detail,
            status_code=exc.status_code,
            error_code=f"HTTP_{exc.status_code}",
            details={"headers": exc.headers} if exc.headers else {}
        )
        
        # 使用API错误处理器处理
        return self.handle_api_error(request, api_error)
    
    def handle_general_exception(self, request: Request, exc: Exception) -> JSONResponse:
        """处理未捕获的一般异常"""
        # 记录错误
        error_log = self.error_logger.log_error(exc, request)
        
        # 创建错误响应
        response_data = {
            "status": "error",
            "message": "服务器内部错误",
            "error_code": "INTERNAL_SERVER_ERROR",
            "error_id": error_log["error_id"]
        }
        
        # 在开发环境中添加详细信息
        if os.environ.get("APP_ENV") == "development":
            response_data["details"] = {
                "error_type": type(exc).__name__,
                "error_message": str(exc)
            }
            response_data["traceback"] = error_log["traceback"]
        
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=response_data
        )

# 错误监控接口
class ErrorMonitoringInterface:
    """错误监控接口，用于注册错误监控端点"""
    
    def __init__(self, app: FastAPI, error_logger: ErrorLogger):
        """初始化错误监控接口"""
        self.app = app
        self.error_logger = error_logger
        
        # 注册错误监控端点
        self.register_monitoring_endpoints()
    
    def register_monitoring_endpoints(self):
        """注册错误监控端点"""
        @self.app.get("/api/admin/errors", tags=["admin"])
        async def get_errors(limit: int = 10):
            """获取最近的错误日志"""
            return {
                "errors": self.error_logger.get_recent_errors(limit)
            }
        
        @self.app.get("/api/admin/errors/{error_id}", tags=["admin"])
        async def get_error_details(error_id: str):
            """获取特定错误的详细信息"""
            error = self.error_logger.get_error_by_id(error_id)
            if error:
                return error
            else:
                raise HTTPException(status_code=404, detail=f"错误ID {error_id} 不存在")
        
        @self.app.get("/api/admin/errors/stats", tags=["admin"])
        async def get_error_stats():
            """获取错误统计信息"""
            return self.error_logger.get_error_stats()
        
        @self.app.post("/api/client-side-error", tags=["errors"])
        async def log_client_error(request: Request):
            """记录客户端错误"""
            try:
                # 读取请求体
                body = await request.json()
                
                # 提取错误信息
                error_message = body.get("message", "未知客户端错误")
                error_stack = body.get("stack", "")
                error_url = body.get("url", "")
                error_line = body.get("line", "")
                error_column = body.get("column", "")
                
                # 创建错误对象
                class ClientSideError(Exception):
                    pass
                
                error = ClientSideError(error_message)
                
                # 记录错误
                additional_info = {
                    "source": "client",
                    "url": error_url,
                    "line": error_line,
                    "column": error_column,
                    "stack": error_stack,
                    "user_agent": request.headers.get("user-agent", ""),
                    "timestamp": datetime.now().isoformat()
                }
                
                error_log = self.error_logger.log_error(
                    error, 
                    request, 
                    additional_info=additional_info
                )
                
                return {
                    "status": "success",
                    "message": "客户端错误已记录",
                    "error_id": error_log["error_id"]
                }
            except Exception as e:
                logger.error(f"处理客户端错误时出错: {e}")
                return {
                    "status": "error",
                    "message": "处理客户端错误请求时发生错误"
                }

# 辅助函数：创建用户友好的错误消息
def format_user_friendly_error(error: Exception) -> str:
    """将错误转换为用户友好的错误消息"""
    # 根据错误类型提供特定的友好消息
    if isinstance(error, ValidationAPIError):
        return "输入数据有误，请检查您提供的信息是否完整、格式是否正确"
    elif isinstance(error, AuthenticationAPIError):
        return "身份验证失败，请重新登录或检查您的凭据"
    elif isinstance(error, AuthorizationAPIError):
        return "您没有执行此操作的权限"
    elif isinstance(error, ResourceNotFoundAPIError):
        return "请求的资源不存在，请检查URL或参数是否正确"
    elif isinstance(error, RateLimitAPIError):
        return "请求频率过高，请稍后再试"
    elif isinstance(error, DatabaseAPIError):
        return "数据库操作失败，请稍后再试"
    elif isinstance(error, PredictionModelAPIError):
        return "排放预测模型计算失败，请调整参数或稍后再试"
    elif isinstance(error, NLPModelAPIError):
        return "自然语言处理失败，请尝试重新表述您的查询"
    elif isinstance(error, SpeechRecognitionAPIError):
        return "语音识别失败，请确保录音清晰或使用文本输入"
    else:
        return "系统发生错误，请稍后再试或联系技术支持"

# 示例使用方法
def example_usage():
    """示例：如何使用错误处理模块"""
    from fastapi import FastAPI
    
    app = FastAPI()
    
    # 创建错误日志记录器
    error_logger = ErrorLogger()
    
    # 创建错误处理器
    error_handler = ErrorHandler(app, error_logger)
    
    # 创建错误监控接口
    error_monitor = ErrorMonitoringInterface(app, error_logger)
    
    # 在路由中使用自定义异常
    @app.get("/example/error")
    async def example_error():
        try:
            # 模拟一个错误
            result = 1 / 0
        except Exception as e:
            # 抛出自定义API错误
            raise PredictionModelAPIError(
                message="预测计算过程中发生错误",
                details={"original_error": str(e)}
            )
    
    # 在路由中使用输入验证
    @app.post("/example/validate")
    async def example_validate(data: Dict):
        if "input" not in data:
            raise ValidationAPIError(
                message="缺少必要的输入参数",
                details={"missing_field": "input"}
            )
        return {"result": "valid"} 
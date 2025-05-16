#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
性能优化模块 - 提供缓存、压缩、监控和异步处理机制
"""

import os
import time
import logging
import functools
import asyncio
from typing import Callable, Any, Dict, List, Optional
import aioredis
import orjson
from fastapi import FastAPI, Request, Response
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import ORJSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend
from fastapi_cache.decorator import cache
from prometheus_client import Counter, Histogram, Gauge
import psutil
import datetime
from pydantic import BaseModel

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("performance-optimizations")

# Prometheus指标
REQUEST_COUNT = Counter('app_requests_total', 'Total count of requests by method and endpoint', ['method', 'endpoint'])
REQUEST_LATENCY = Histogram('app_request_latency_seconds', 'Request latency in seconds', ['method', 'endpoint'])
MEMORY_USAGE = Gauge('app_memory_usage_bytes', 'Memory usage in bytes')
CPU_USAGE = Gauge('app_cpu_usage_percent', 'CPU usage in percent')
ACTIVE_REQUESTS = Gauge('app_active_requests', 'Number of active requests')

class PerformanceConfig(BaseModel):
    """性能优化配置模型"""
    enable_cache: bool = True
    cache_ttl: int = 300  # 缓存有效期，单位秒
    cache_prefix: str = "pollution-cache"
    enable_compression: bool = True
    compression_minimum_size: int = 1000  # 最小压缩大小，单位字节
    monitor_system_resources: bool = True
    resource_check_interval: int = 60  # 资源监控间隔，单位秒
    redis_url: str = "redis://localhost"  # Redis连接URL
    redis_encoding: str = "utf8"
    request_timeout: int = 60  # 请求超时时间，单位秒
    max_concurrent_requests: int = 100  # 最大并发请求数

class PerformanceOptimizer:
    """性能优化器类"""
    
    def __init__(self, app: FastAPI, config: Optional[PerformanceConfig] = None):
        """初始化性能优化器"""
        self.app = app
        self.config = config or PerformanceConfig()
        self.redis = None
        self._initialize()
    
    def _initialize(self):
        """初始化性能优化器组件"""
        # 添加ORJSON响应类作为默认响应类
        self.app.default_response_class = ORJSONResponse
        
        # 添加GZip压缩中间件
        if self.config.enable_compression:
            self.app.add_middleware(GZipMiddleware, minimum_size=self.config.compression_minimum_size)
        
        # 添加性能监控中间件
        @self.app.middleware("http")
        async def performance_middleware(request: Request, call_next: Callable) -> Response:
            ACTIVE_REQUESTS.inc()
            
            # 记录请求开始时间
            start_time = time.time()
            
            # 设置请求超时
            try:
                response = await asyncio.wait_for(
                    call_next(request), 
                    timeout=self.config.request_timeout
                )
            except asyncio.TimeoutError:
                ACTIVE_REQUESTS.dec()
                return Response(
                    content=orjson.dumps({"detail": "请求处理超时"}),
                    status_code=504,
                    media_type="application/json"
                )
            except Exception as e:
                ACTIVE_REQUESTS.dec()
                logger.error(f"请求处理异常: {str(e)}")
                return Response(
                    content=orjson.dumps({"detail": "服务器内部错误"}),
                    status_code=500,
                    media_type="application/json"
                )
            finally:
                ACTIVE_REQUESTS.dec()
            
            # 计算请求处理时间
            duration = time.time() - start_time
            
            # 记录Prometheus指标
            endpoint = request.url.path
            REQUEST_COUNT.labels(request.method, endpoint).inc()
            REQUEST_LATENCY.labels(request.method, endpoint).observe(duration)
            
            # 添加性能指标到响应头
            response.headers["X-Process-Time"] = str(duration)
            
            return response
    
    async def setup_cache(self):
        """设置缓存"""
        if not self.config.enable_cache:
            logger.info("缓存功能已禁用")
            return
        
        try:
            self.redis = await aioredis.from_url(
                self.config.redis_url,
                encoding=self.config.redis_encoding
            )
            FastAPICache.init(
                RedisBackend(self.redis), 
                prefix=self.config.cache_prefix
            )
            logger.info("缓存初始化成功")
        except Exception as e:
            logger.error(f"缓存初始化失败: {str(e)}")
    
    async def start_resource_monitoring(self):
        """启动资源监控"""
        if not self.config.monitor_system_resources:
            logger.info("系统资源监控已禁用")
            return
        
        logger.info("启动系统资源监控")
        
        while True:
            # 记录内存使用情况
            memory_info = psutil.virtual_memory()
            MEMORY_USAGE.set(memory_info.used)
            
            # 记录CPU使用情况
            CPU_USAGE.set(psutil.cpu_percent())
            
            # 按配置的间隔等待
            await asyncio.sleep(self.config.resource_check_interval)
    
    async def cleanup(self):
        """清理资源"""
        if self.redis:
            await self.redis.close()
            logger.info("Redis连接已关闭")

# 创建自定义缓存装饰器
def timed_cache(seconds: int = 60, key_prefix: str = ""):
    """时间缓存装饰器，可在无Redis情况下使用的简单内存缓存"""
    cache_data = {}
    
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # 创建缓存键
            cache_key = f"{key_prefix}:{func.__name__}:{str(args)}:{str(kwargs)}"
            
            # 检查缓存
            if cache_key in cache_data:
                entry_time, result = cache_data[cache_key]
                # 检查是否过期
                if time.time() - entry_time < seconds:
                    return result
            
            # 执行原始函数
            result = await func(*args, **kwargs)
            
            # 更新缓存
            cache_data[cache_key] = (time.time(), result)
            
            # 清理过期缓存
            current_time = time.time()
            expired_keys = [
                k for k, (t, _) in cache_data.items() if current_time - t >= seconds
            ]
            for k in expired_keys:
                del cache_data[k]
                
            return result
        return wrapper
    return decorator

# 批处理请求处理器
class BatchRequestProcessor:
    """批处理请求处理器，将多个请求合并为一个批处理以提高性能"""
    
    def __init__(self, max_batch_size: int = 10, max_wait_time: float = 0.1):
        """初始化批处理处理器"""
        self.max_batch_size = max_batch_size  # 最大批处理大小
        self.max_wait_time = max_wait_time    # 最大等待时间(秒)
        self.batches = {}                     # 批处理队列
        self.lock = asyncio.Lock()            # 锁，用于同步访问批处理队列
    
    async def process(self, batch_key: str, item: Any, processor_func: Callable):
        """处理单个请求，将其加入批处理队列"""
        async with self.lock:
            # 如果批处理键不存在，初始化新批处理
            if batch_key not in self.batches:
                self.batches[batch_key] = {
                    "items": [],
                    "futures": [],
                    "start_time": time.time(),
                    "processing": False
                }
            
            # 创建Future对象，用于返回处理结果
            future = asyncio.Future()
            
            # 将请求项和Future添加到批处理中
            self.batches[batch_key]["items"].append(item)
            self.batches[batch_key]["futures"].append(future)
            
            # 检查是否达到最大批处理大小
            if len(self.batches[batch_key]["items"]) >= self.max_batch_size:
                batch = self.batches.pop(batch_key)
                
                # 异步处理批次
                asyncio.create_task(self._process_batch(batch, processor_func))
            
            # 检查是否有达到等待时间的批次
            current_time = time.time()
            expired_batches = []
            for key, batch in self.batches.items():
                if not batch["processing"] and (current_time - batch["start_time"]) >= self.max_wait_time:
                    expired_batches.append(key)
            
            # 处理过期批次
            for key in expired_batches:
                batch = self.batches.pop(key)
                batch["processing"] = True
                asyncio.create_task(self._process_batch(batch, processor_func))
        
        # 等待处理结果
        return await future
    
    async def _process_batch(self, batch: Dict, processor_func: Callable):
        """处理批次"""
        items = batch["items"]
        futures = batch["futures"]
        
        try:
            # 调用批处理函数处理所有项
            results = await processor_func(items)
            
            # 设置每个Future的结果
            for i, future in enumerate(futures):
                if not future.done():
                    future.set_result(results[i])
        except Exception as e:
            # 如果发生错误，将异常设置到所有Future
            for future in futures:
                if not future.done():
                    future.set_exception(e)
    
    def get_stats(self) -> Dict:
        """获取批处理处理器的统计信息"""
        stats = {
            "active_batches": len(self.batches),
            "pending_requests": sum(len(batch["items"]) for batch in self.batches.values()),
            "batch_details": {
                key: {
                    "items_count": len(batch["items"]),
                    "wait_time": time.time() - batch["start_time"],
                    "processing": batch["processing"]
                }
                for key, batch in self.batches.items()
            }
        }
        return stats

# 系统状态监控器
class SystemMonitor:
    """系统状态监控器，收集系统性能指标"""
    
    def __init__(self, app: FastAPI, check_interval: int = 60):
        """初始化系统监控器"""
        self.app = app
        self.check_interval = check_interval
        self.start_time = datetime.datetime.now()
        self.stats = {
            "request_count": 0,
            "error_count": 0,
            "avg_response_time": 0,
            "peak_memory_usage": 0,
            "peak_cpu_usage": 0,
            "last_check_time": None
        }
    
    def track_request(self, response_time: float, is_error: bool = False):
        """跟踪请求"""
        self.stats["request_count"] += 1
        if is_error:
            self.stats["error_count"] += 1
        
        # 更新平均响应时间
        current_avg = self.stats["avg_response_time"]
        count = self.stats["request_count"]
        self.stats["avg_response_time"] = (current_avg * (count - 1) + response_time) / count
    
    async def start_monitoring(self):
        """启动监控"""
        logger.info("启动系统监控")
        
        while True:
            # 获取当前系统状态
            memory_usage = psutil.virtual_memory().percent
            cpu_usage = psutil.cpu_percent()
            
            # 更新峰值
            self.stats["peak_memory_usage"] = max(self.stats["peak_memory_usage"], memory_usage)
            self.stats["peak_cpu_usage"] = max(self.stats["peak_cpu_usage"], cpu_usage)
            self.stats["last_check_time"] = datetime.datetime.now().isoformat()
            
            # 记录当前状态
            logger.debug(f"系统状态: CPU {cpu_usage}%, 内存 {memory_usage}%")
            
            # 等待下一次检查
            await asyncio.sleep(self.check_interval)
    
    def get_status(self) -> Dict:
        """获取系统状态"""
        # 计算系统正常运行时间
        uptime = datetime.datetime.now() - self.start_time
        uptime_str = str(uptime).split('.')[0]  # 移除微秒部分
        
        # 获取最新系统指标
        current_memory = psutil.virtual_memory()
        current_cpu = psutil.cpu_percent(interval=0.1)
        
        return {
            "version": "1.0.0",
            "uptime": uptime_str,
            "start_time": self.start_time.isoformat(),
            "system": {
                "cpu_usage": current_cpu,
                "memory_usage": current_memory.percent,
                "memory_available": f"{current_memory.available / (1024 * 1024):.2f} MB",
                "peak_cpu_usage": self.stats["peak_cpu_usage"],
                "peak_memory_usage": self.stats["peak_memory_usage"]
            },
            "requests": {
                "total": self.stats["request_count"],
                "errors": self.stats["error_count"],
                "avg_response_time": f"{self.stats['avg_response_time'] * 1000:.2f} ms"
            },
            "last_check_time": self.stats["last_check_time"]
        }

# 测试缓存功能的示例函数
@cache(expire=60)
async def example_cached_function(x: int):
    """一个缓存函数示例"""
    await asyncio.sleep(1)  # 模拟耗时操作
    return {"result": x * x} 
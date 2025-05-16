#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
电厂污染物排放预测系统 - 性能测试脚本
"""

import requests
import json
import time
import os
import sys
import threading
import queue
import random
from datetime import datetime, timedelta
import logging
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import psutil
import numpy as np
from tqdm import tqdm

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("性能测试结果.log")
    ]
)

logger = logging.getLogger(__name__)

# API基础URL，默认为本地地址，可通过命令行参数修改
BASE_URL = "http://localhost:8000"

# 测试结果数据
performance_results = {
    "api_response_times": {},
    "api_error_rates": {},
    "system_metrics": {},
    "concurrent_users": [],
    "test_duration": 0
}

class LoadGenerator:
    """负载生成器类，用于模拟并发用户"""
    
    def __init__(self, api_endpoint, payload=None, method="GET", 
                 num_threads=10, request_interval=1.0, duration=60):
        """
        初始化负载生成器
        
        参数:
            api_endpoint: API端点URL
            payload: POST请求的数据
            method: 请求方法 (GET/POST)
            num_threads: 线程数量，模拟的并发用户数
            request_interval: 请求间隔时间 (秒)
            duration: 测试持续时间 (秒)
        """
        self.api_endpoint = api_endpoint
        self.payload = payload
        self.method = method
        self.num_threads = num_threads
        self.request_interval = request_interval
        self.duration = duration
        
        # 结果队列
        self.results_queue = queue.Queue()
        
        # 停止标志
        self.stop_flag = threading.Event()
        
        # 测试结果
        self.response_times = []
        self.error_count = 0
        self.success_count = 0
        self.total_requests = 0
    
    def worker(self, worker_id):
        """工作线程，执行API请求"""
        logger.debug(f"工作线程 {worker_id} 已启动")
        
        while not self.stop_flag.is_set():
            try:
                # 发送请求
                start_time = time.time()
                
                if self.method == "GET":
                    response = requests.get(self.api_endpoint, timeout=10)
                else:  # POST
                    response = requests.post(self.api_endpoint, json=self.payload, timeout=10)
                
                end_time = time.time()
                response_time = end_time - start_time
                
                # 请求结果
                result = {
                    "worker_id": worker_id,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
                    "response_time": response_time,
                    "status_code": response.status_code,
                    "success": response.status_code < 400,
                    "content_length": len(response.content) if response.content else 0
                }
                
                # 放入结果队列
                self.results_queue.put(result)
                
                # 等待一段时间
                if self.request_interval > 0:
                    time.sleep(self.request_interval)
            
            except Exception as e:
                # 请求错误
                error_result = {
                    "worker_id": worker_id,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
                    "response_time": 0,
                    "status_code": -1,
                    "success": False,
                    "error": str(e)
                }
                self.results_queue.put(error_result)
                
                logger.debug(f"工作线程 {worker_id} 请求错误: {str(e)}")
                
                # 等待一段时间
                if self.request_interval > 0:
                    time.sleep(self.request_interval)
        
        logger.debug(f"工作线程 {worker_id} 已停止")
    
    def result_collector(self):
        """结果收集线程"""
        logger.debug("结果收集线程已启动")
        
        while not self.stop_flag.is_set() or not self.results_queue.empty():
            try:
                # 从队列获取结果
                result = self.results_queue.get(timeout=1)
                
                # 处理结果
                self.total_requests += 1
                
                if result.get("success", False):
                    self.success_count += 1
                    self.response_times.append(result["response_time"])
                else:
                    self.error_count += 1
                
                self.results_queue.task_done()
            
            except queue.Empty:
                # 队列为空，继续等待
                pass
            except Exception as e:
                logger.error(f"结果收集线程错误: {str(e)}")
        
        logger.debug("结果收集线程已停止")
    
    def run(self):
        """运行负载测试"""
        logger.info(f"开始负载测试: 并发用户={self.num_threads}, 持续时间={self.duration}秒")
        
        # 创建并启动工作线程
        threads = []
        for i in range(self.num_threads):
            t = threading.Thread(target=self.worker, args=(i,))
            t.daemon = True
            threads.append(t)
            t.start()
        
        # 创建并启动结果收集线程
        collector_thread = threading.Thread(target=self.result_collector)
        collector_thread.daemon = True
        collector_thread.start()
        
        # 等待指定的持续时间
        start_time = time.time()
        try:
            # 使用tqdm显示进度条
            for _ in tqdm(range(self.duration), desc=f"负载测试中 ({self.num_threads}个并发用户)"):
                time.sleep(1)
        except KeyboardInterrupt:
            logger.warning("测试被用户中断")
        
        end_time = time.time()
        actual_duration = end_time - start_time
        
        # 设置停止标志
        self.stop_flag.set()
        
        # 等待工作线程结束
        for t in threads:
            t.join(timeout=2)
        
        # 等待结果收集线程结束
        collector_thread.join(timeout=2)
        
        # 计算结果
        if self.response_times:
            avg_response_time = sum(self.response_times) / len(self.response_times)
            min_response_time = min(self.response_times)
            max_response_time = max(self.response_times)
            p95_response_time = np.percentile(self.response_times, 95)
            p99_response_time = np.percentile(self.response_times, 99)
        else:
            avg_response_time = min_response_time = max_response_time = p95_response_time = p99_response_time = 0
        
        error_rate = (self.error_count / self.total_requests) * 100 if self.total_requests > 0 else 0
        throughput = self.total_requests / actual_duration if actual_duration > 0 else 0
        
        # 返回测试结果
        result = {
            "concurrent_users": self.num_threads,
            "duration": actual_duration,
            "total_requests": self.total_requests,
            "successful_requests": self.success_count,
            "failed_requests": self.error_count,
            "avg_response_time": avg_response_time,
            "min_response_time": min_response_time,
            "max_response_time": max_response_time,
            "p95_response_time": p95_response_time,
            "p99_response_time": p99_response_time,
            "error_rate": error_rate,
            "throughput": throughput,
            "response_times": self.response_times
        }
        
        logger.info(f"负载测试完成: 总请求数={self.total_requests}, 成功={self.success_count}, 失败={self.error_count}")
        logger.info(f"平均响应时间: {avg_response_time:.4f}秒, 最小: {min_response_time:.4f}秒, 最大: {max_response_time:.4f}秒")
        logger.info(f"P95响应时间: {p95_response_time:.4f}秒, P99响应时间: {p99_response_time:.4f}秒")
        logger.info(f"错误率: {error_rate:.2f}%, 吞吐量: {throughput:.2f}请求/秒")
        
        return result


def test_api_performance(api_endpoint, payload=None, method="GET", users_list=[1, 5, 10, 25, 50], 
                         duration=30, request_interval=1.0):
    """
    测试API在不同并发用户数下的性能
    
    参数:
        api_endpoint: API端点URL
        payload: POST请求的数据
        method: 请求方法 (GET/POST)
        users_list: 并发用户数列表
        duration: 每个测试的持续时间 (秒)
        request_interval: 请求间隔时间 (秒)
    
    返回:
        测试结果字典
    """
    logger.info(f"开始API性能测试: {api_endpoint}")
    
    results = []
    
    # 对每个并发用户数进行测试
    for num_users in users_list:
        # 创建负载生成器
        load_gen = LoadGenerator(
            api_endpoint=api_endpoint,
            payload=payload,
            method=method,
            num_threads=num_users,
            request_interval=request_interval,
            duration=duration
        )
        
        # 运行测试
        result = load_gen.run()
        results.append(result)
        
        # 等待一段时间，让系统恢复
        time.sleep(5)
    
    # 提取结果数据
    concurrent_users = [r["concurrent_users"] for r in results]
    avg_response_times = [r["avg_response_time"] for r in results]
    p95_response_times = [r["p95_response_time"] for r in results]
    error_rates = [r["error_rate"] for r in results]
    throughputs = [r["throughput"] for r in results]
    
    # 保存到全局结果
    api_name = api_endpoint.split('/')[-1] if '/' in api_endpoint else api_endpoint
    performance_results["api_response_times"][api_name] = {
        "concurrent_users": concurrent_users,
        "avg_response_times": avg_response_times,
        "p95_response_times": p95_response_times
    }
    
    performance_results["api_error_rates"][api_name] = {
        "concurrent_users": concurrent_users,
        "error_rates": error_rates
    }
    
    performance_results["concurrent_users"] = users_list
    
    # 绘制结果图表
    plt.figure(figsize=(12, 8))
    
    # 响应时间图表
    plt.subplot(2, 2, 1)
    plt.plot(concurrent_users, avg_response_times, 'o-', label='平均响应时间')
    plt.plot(concurrent_users, p95_response_times, 's-', label='P95响应时间')
    plt.xlabel('并发用户数')
    plt.ylabel('响应时间 (秒)')
    plt.title(f'API响应时间 - {api_name}')
    plt.grid(True)
    plt.legend()
    
    # 错误率图表
    plt.subplot(2, 2, 2)
    plt.plot(concurrent_users, error_rates, 'o-', color='red')
    plt.xlabel('并发用户数')
    plt.ylabel('错误率 (%)')
    plt.title(f'API错误率 - {api_name}')
    plt.grid(True)
    
    # 吞吐量图表
    plt.subplot(2, 2, 3)
    plt.plot(concurrent_users, throughputs, 'o-', color='green')
    plt.xlabel('并发用户数')
    plt.ylabel('吞吐量 (请求/秒)')
    plt.title(f'API吞吐量 - {api_name}')
    plt.grid(True)
    
    # 保存图表
    plt.tight_layout()
    plt.savefig(f"性能测试_{api_name}.png")
    logger.info(f"性能测试图表已保存为 性能测试_{api_name}.png")
    
    return results


def monitor_system_resources(duration=60, interval=1.0):
    """
    监控系统资源使用情况
    
    参数:
        duration: 监控持续时间 (秒)
        interval: 监控间隔时间 (秒)
    
    返回:
        监控结果字典
    """
    logger.info(f"开始系统资源监控: 持续时间={duration}秒, 间隔={interval}秒")
    
    # 结果数据
    timestamps = []
    cpu_usage = []
    memory_usage = []
    disk_io_read = []
    disk_io_write = []
    net_io_sent = []
    net_io_recv = []
    
    # 初始IO计数
    disk_io_start = psutil.disk_io_counters()
    net_io_start = psutil.net_io_counters()
    
    # 监控循环
    start_time = time.time()
    try:
        for _ in tqdm(range(int(duration / interval)), desc="系统资源监控中"):
            # 记录时间戳
            current_time = time.time()
            timestamps.append(current_time - start_time)
            
            # CPU使用率
            cpu_percent = psutil.cpu_percent(interval=None)
            cpu_usage.append(cpu_percent)
            
            # 内存使用率
            memory_info = psutil.virtual_memory()
            memory_percent = memory_info.percent
            memory_usage.append(memory_percent)
            
            # 磁盘IO
            disk_io_current = psutil.disk_io_counters()
            disk_io_read.append(disk_io_current.read_bytes - disk_io_start.read_bytes)
            disk_io_write.append(disk_io_current.write_bytes - disk_io_start.write_bytes)
            
            # 网络IO
            net_io_current = psutil.net_io_counters()
            net_io_sent.append(net_io_current.bytes_sent - net_io_start.bytes_sent)
            net_io_recv.append(net_io_current.bytes_recv - net_io_start.bytes_recv)
            
            # 等待一段时间
            time.sleep(interval)
    
    except KeyboardInterrupt:
        logger.warning("系统资源监控被用户中断")
    
    end_time = time.time()
    actual_duration = end_time - start_time
    
    # 转换IO增量为速率
    disk_read_rate = [io / interval for io in disk_io_read[1:]]
    disk_write_rate = [io / interval for io in disk_io_write[1:]]
    net_sent_rate = [io / interval for io in net_io_sent[1:]]
    net_recv_rate = [io / interval for io in net_io_recv[1:]]
    
    # 计算平均值
    avg_cpu = sum(cpu_usage) / len(cpu_usage) if cpu_usage else 0
    avg_memory = sum(memory_usage) / len(memory_usage) if memory_usage else 0
    avg_disk_read = sum(disk_read_rate) / len(disk_read_rate) if disk_read_rate else 0
    avg_disk_write = sum(disk_write_rate) / len(disk_write_rate) if disk_write_rate else 0
    avg_net_sent = sum(net_sent_rate) / len(net_sent_rate) if net_sent_rate else 0
    avg_net_recv = sum(net_recv_rate) / len(net_recv_rate) if net_recv_rate else 0
    
    # 结果字典
    result = {
        "duration": actual_duration,
        "timestamps": timestamps,
        "cpu_usage": cpu_usage,
        "memory_usage": memory_usage,
        "disk_io_read": disk_io_read,
        "disk_io_write": disk_io_write,
        "net_io_sent": net_io_sent,
        "net_io_recv": net_io_recv,
        "avg_cpu": avg_cpu,
        "avg_memory": avg_memory,
        "avg_disk_read": avg_disk_read,
        "avg_disk_write": avg_disk_write,
        "avg_net_sent": avg_net_sent,
        "avg_net_recv": avg_net_recv
    }
    
    # 保存到全局结果
    performance_results["system_metrics"] = result
    
    # 绘制资源使用图表
    plt.figure(figsize=(12, 10))
    
    # CPU使用率
    plt.subplot(3, 2, 1)
    plt.plot(timestamps, cpu_usage)
    plt.xlabel('时间 (秒)')
    plt.ylabel('CPU使用率 (%)')
    plt.title(f'CPU使用率 (平均: {avg_cpu:.2f}%)')
    plt.grid(True)
    
    # 内存使用率
    plt.subplot(3, 2, 2)
    plt.plot(timestamps, memory_usage)
    plt.xlabel('时间 (秒)')
    plt.ylabel('内存使用率 (%)')
    plt.title(f'内存使用率 (平均: {avg_memory:.2f}%)')
    plt.grid(True)
    
    # 磁盘读取速率
    plt.subplot(3, 2, 3)
    plt.plot(timestamps[1:], disk_read_rate)
    plt.xlabel('时间 (秒)')
    plt.ylabel('磁盘读取 (字节/秒)')
    plt.title(f'磁盘读取速率 (平均: {avg_disk_read:.2f} B/s)')
    plt.grid(True)
    
    # 磁盘写入速率
    plt.subplot(3, 2, 4)
    plt.plot(timestamps[1:], disk_write_rate)
    plt.xlabel('时间 (秒)')
    plt.ylabel('磁盘写入 (字节/秒)')
    plt.title(f'磁盘写入速率 (平均: {avg_disk_write:.2f} B/s)')
    plt.grid(True)
    
    # 网络发送速率
    plt.subplot(3, 2, 5)
    plt.plot(timestamps[1:], net_sent_rate)
    plt.xlabel('时间 (秒)')
    plt.ylabel('网络发送 (字节/秒)')
    plt.title(f'网络发送速率 (平均: {avg_net_sent:.2f} B/s)')
    plt.grid(True)
    
    # 网络接收速率
    plt.subplot(3, 2, 6)
    plt.plot(timestamps[1:], net_recv_rate)
    plt.xlabel('时间 (秒)')
    plt.ylabel('网络接收 (字节/秒)')
    plt.title(f'网络接收速率 (平均: {avg_net_recv:.2f} B/s)')
    plt.grid(True)
    
    # 保存图表
    plt.tight_layout()
    plt.savefig("系统资源使用情况.png")
    logger.info("系统资源使用情况图表已保存为 系统资源使用情况.png")
    
    logger.info(f"系统资源监控完成: CPU平均使用率={avg_cpu:.2f}%, 内存平均使用率={avg_memory:.2f}%")
    
    return result


def generate_performance_report():
    """生成性能测试报告"""
    logger.info("正在生成性能测试报告...")
    
    # 创建报告数据
    report = {
        "测试时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "API基础URL": BASE_URL,
        "测试持续时间": performance_results.get("test_duration", 0),
        "并发用户数": performance_results.get("concurrent_users", [])
    }
    
    # API响应时间数据
    for api_name, data in performance_results.get("api_response_times", {}).items():
        report[f"{api_name} API响应时间"] = {
            "并发用户数": data.get("concurrent_users", []),
            "平均响应时间": data.get("avg_response_times", []),
            "P95响应时间": data.get("p95_response_times", [])
        }
    
    # API错误率数据
    for api_name, data in performance_results.get("api_error_rates", {}).items():
        report[f"{api_name} API错误率"] = {
            "并发用户数": data.get("concurrent_users", []),
            "错误率": data.get("error_rates", [])
        }
    
    # 系统资源数据
    system_metrics = performance_results.get("system_metrics", {})
    if system_metrics:
        report["系统资源使用情况"] = {
            "平均CPU使用率": system_metrics.get("avg_cpu", 0),
            "平均内存使用率": system_metrics.get("avg_memory", 0),
            "平均磁盘读取速率": system_metrics.get("avg_disk_read", 0),
            "平均磁盘写入速率": system_metrics.get("avg_disk_write", 0),
            "平均网络发送速率": system_metrics.get("avg_net_sent", 0),
            "平均网络接收速率": system_metrics.get("avg_net_recv", 0)
        }
    
    # 打印报告
    logger.info("====== 性能测试报告 ======")
    for key, value in report.items():
        if isinstance(value, dict):
            logger.info(f"{key}:")
            for sub_key, sub_value in value.items():
                logger.info(f"  {sub_key}: {sub_value}")
        else:
            logger.info(f"{key}: {value}")
    
    # 保存JSON格式的报告
    try:
        with open("性能测试报告.json", "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=4)
        logger.info("性能测试报告已保存为 性能测试报告.json")
    except Exception as e:
        logger.warning(f"保存性能测试报告时出错: {str(e)}")
    
    # 生成综合图表
    try:
        plt.figure(figsize=(12, 10))
        
        # 所有API的响应时间对比
        plt.subplot(2, 1, 1)
        for api_name, data in performance_results.get("api_response_times", {}).items():
            plt.plot(data.get("concurrent_users", []), data.get("avg_response_times", []), 'o-', label=f'{api_name} API')
        
        plt.xlabel('并发用户数')
        plt.ylabel('平均响应时间 (秒)')
        plt.title('API响应时间对比')
        plt.grid(True)
        plt.legend()
        
        # 所有API的错误率对比
        plt.subplot(2, 1, 2)
        for api_name, data in performance_results.get("api_error_rates", {}).items():
            plt.plot(data.get("concurrent_users", []), data.get("error_rates", []), 'o-', label=f'{api_name} API')
        
        plt.xlabel('并发用户数')
        plt.ylabel('错误率 (%)')
        plt.title('API错误率对比')
        plt.grid(True)
        plt.legend()
        
        # 保存图表
        plt.tight_layout()
        plt.savefig("性能测试综合报告.png")
        logger.info("性能测试综合报告图表已保存为 性能测试综合报告.png")
    except Exception as e:
        logger.warning(f"生成综合图表时出错: {str(e)}")
    
    return report


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="电厂污染物排放预测系统性能测试")
    parser.add_argument("--url", default="http://localhost:8000", help="API基础URL")
    parser.add_argument("--users", default="1,5,10,25,50", help="并发用户数列表，用逗号分隔")
    parser.add_argument("--duration", type=int, default=30, help="每个测试的持续时间 (秒)")
    parser.add_argument("--interval", type=float, default=1.0, help="请求间隔时间 (秒)")
    parser.add_argument("--monitor-duration", type=int, default=60, help="系统资源监控持续时间 (秒)")
    args = parser.parse_args()
    
    global BASE_URL
    BASE_URL = args.url
    
    # 解析并发用户数列表
    users_list = [int(u) for u in args.users.split(",")]
    
    logger.info("====== 开始性能测试 ======")
    logger.info(f"API基础URL: {BASE_URL}")
    logger.info(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"并发用户数列表: {users_list}")
    logger.info(f"每个测试持续时间: {args.duration}秒")
    logger.info(f"请求间隔时间: {args.interval}秒")
    
    # 准备示例数据
    features = [
        75.0,  # 负荷
        25.0,  # 风速
        60.0,  # 湿度
        3.5,   # 氧含量
        120.0, # 烟气温度
        datetime.now().hour,
        datetime.now().day,
        datetime.now().month,
        datetime.now().year,
        datetime.now().weekday(),
        20.0,  # 上次SO2排放值
        40.0,  # 上次NOx排放值
        8.0    # 上次烟尘排放值
    ]
    
    prediction_payload = {
        "features": [features],
        "model_type": "lstm"
    }
    
    nlp_query_payload = {
        "query": "最近24小时的SO2平均排放浓度是多少?",
        "mode": "text"
    }
    
    # 记录开始时间
    start_time = time.time()
    
    # 测试健康检查API
    test_api_performance(
        api_endpoint=f"{BASE_URL}/health",
        method="GET",
        users_list=users_list,
        duration=args.duration,
        request_interval=args.interval
    )
    
    # 测试排放数据API
    test_api_performance(
        api_endpoint=f"{BASE_URL}/api/emissions",
        method="GET",
        users_list=users_list,
        duration=args.duration,
        request_interval=args.interval
    )
    
    # 测试排放预测API
    test_api_performance(
        api_endpoint=f"{BASE_URL}/predict",
        payload=prediction_payload,
        method="POST",
        users_list=users_list,
        duration=args.duration,
        request_interval=args.interval
    )
    
    # 测试自然语言查询API
    test_api_performance(
        api_endpoint=f"{BASE_URL}/api/nlp_query",
        payload=nlp_query_payload,
        method="POST",
        users_list=users_list,
        duration=args.duration,
        request_interval=args.interval
    )
    
    # 记录测试持续时间
    end_time = time.time()
    performance_results["test_duration"] = end_time - start_time
    
    # 监控系统资源
    logger.info("开始监控系统资源使用情况...")
    monitor_system_resources(duration=args.monitor_duration, interval=1.0)
    
    # 生成测试报告
    generate_performance_report()
    
    logger.info("====== 性能测试完成 ======")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 
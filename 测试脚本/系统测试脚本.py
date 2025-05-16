#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
电厂污染物排放预测系统 - 系统测试脚本
此脚本执行端到端的系统测试，验证各组件是否能够协同工作
"""

import os
import sys
import json
import time
import logging
import argparse
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("系统测试结果.log")
    ]
)

logger = logging.getLogger(__name__)

# 全局变量
BASE_URL = "http://localhost:8000"
FRONTEND_URL = "http://localhost:3000"
TEST_RESULTS = {
    "总测试数": 0,
    "通过测试数": 0,
    "失败测试数": 0,
    "通过率": 0.0,
    "系统集成测试": {},
    "端到端流程测试": {},
    "数据流测试": {},
    "异常恢复测试": {}
}

# 装饰器：运行测试并记录结果
def run_test(test_func):
    """测试装饰器，用于运行测试并记录结果"""
    def wrapper(*args, **kwargs):
        global TEST_RESULTS
        TEST_RESULTS["总测试数"] += 1
        
        test_name = test_func.__name__
        logger.info(f"执行测试: {test_name}")
        
        start_time = time.time()
        try:
            result = test_func(*args, **kwargs)
            elapsed_time = time.time() - start_time
            
            if result["success"]:
                TEST_RESULTS["通过测试数"] += 1
                logger.info(f"测试通过: {test_name}, 耗时: {elapsed_time:.2f}秒")
            else:
                TEST_RESULTS["失败测试数"] += 1
                logger.error(f"测试失败: {test_name}, 原因: {result.get('message', '未知原因')}, 耗时: {elapsed_time:.2f}秒")
            
            # 保存详细测试结果
            category = None
            if "系统集成" in test_name:
                category = "系统集成测试"
            elif "端到端" in test_name:
                category = "端到端流程测试"
            elif "数据流" in test_name:
                category = "数据流测试"
            elif "异常恢复" in test_name:
                category = "异常恢复测试"
            
            if category:
                if category not in TEST_RESULTS:
                    TEST_RESULTS[category] = {}
                
                TEST_RESULTS[category][test_name] = {
                    "成功": result["success"],
                    "耗时": f"{elapsed_time:.2f}秒",
                    "消息": result.get("message", ""),
                    "详情": result.get("details", {})
                }
            
            return result
        except Exception as e:
            elapsed_time = time.time() - start_time
            TEST_RESULTS["失败测试数"] += 1
            error_msg = f"测试异常: {str(e)}"
            logger.exception(error_msg)
            return {"success": False, "message": error_msg, "details": {}}
    
    return wrapper

# 辅助函数：检查系统服务是否在线
def check_system_online(base_url=BASE_URL):
    """检查系统的基础服务是否在线"""
    try:
        # 检查后端API
        api_response = requests.get(f"{base_url}/api/health", timeout=5)
        api_online = api_response.status_code == 200
        
        # 检查前端
        frontend_online = False
        try:
            frontend_response = requests.get(FRONTEND_URL, timeout=5)
            frontend_online = frontend_response.status_code == 200
        except:
            pass
        
        # 连接是否正常
        connection_ok = api_online and frontend_online
        
        return {
            "api_online": api_online,
            "frontend_online": frontend_online,
            "connection_ok": connection_ok
        }
    except Exception as e:
        logger.error(f"检查系统在线状态失败: {str(e)}")
        return {
            "api_online": False,
            "frontend_online": False,
            "connection_ok": False
        }

# 辅助函数：设置WebDriver
def setup_webdriver():
    """设置Selenium WebDriver"""
    try:
        chrome_options = Options()
        chrome_options.add_argument("--headless")  # 无头模式
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        
        driver = webdriver.Chrome(options=chrome_options)
        driver.set_window_size(1366, 768)
        return driver
    except Exception as e:
        logger.error(f"设置WebDriver失败: {str(e)}")
        return None

# 系统集成测试
@run_test
def test_系统集成_前后端连接():
    """测试前端和后端的连接是否正常"""
    try:
        driver = setup_webdriver()
        if not driver:
            return {"success": False, "message": "无法初始化WebDriver"}
        
        # 访问前端首页
        driver.get(FRONTEND_URL)
        
        # 等待页面加载
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
        
        # 检查是否有API错误提示
        error_elements = driver.find_elements(By.XPATH, "//*[contains(text(), 'API') and contains(text(), 'error')]")
        
        # 截图
        screenshot_path = "系统集成测试_前后端连接.png"
        driver.save_screenshot(screenshot_path)
        
        # 检查页面标题是否正确
        title = driver.title
        has_correct_title = "排放预测" in title or "电厂" in title
        
        driver.quit()
        
        success = len(error_elements) == 0 and has_correct_title
        return {
            "success": success,
            "message": "前后端连接正常" if success else "前后端连接异常",
            "details": {
                "页面标题": title,
                "API错误数": len(error_elements),
                "截图路径": screenshot_path
            }
        }
    except Exception as e:
        return {"success": False, "message": f"测试前后端连接异常: {str(e)}"}

@run_test
def test_系统集成_数据库连接():
    """测试系统与数据库的连接是否正常"""
    try:
        # 使用API端点检查数据库连接
        response = requests.get(f"{BASE_URL}/api/database/status", timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            db_status = data.get("status", "unknown")
            
            success = db_status == "connected"
            return {
                "success": success,
                "message": f"数据库连接{'' if success else '不'}正常",
                "details": data
            }
        else:
            return {
                "success": False,
                "message": f"数据库状态检查失败，HTTP状态码: {response.status_code}",
                "details": {"response": response.text}
            }
    except Exception as e:
        return {"success": False, "message": f"测试数据库连接异常: {str(e)}"}

@run_test
def test_系统集成_模型加载():
    """测试预测模型是否正常加载"""
    try:
        # 检查模型状态API
        response = requests.get(f"{BASE_URL}/api/model/status", timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            models = data.get("models", {})
            
            # 检查是否有模型加载
            models_loaded = len(models) > 0
            all_models_ready = all(model.get("status") == "ready" for model in models.values())
            
            success = models_loaded and all_models_ready
            return {
                "success": success,
                "message": "所有模型已正常加载" if success else "部分或全部模型未正常加载",
                "details": {
                    "已加载模型数": len(models),
                    "模型详情": models
                }
            }
        else:
            return {
                "success": False,
                "message": f"模型状态检查失败，HTTP状态码: {response.status_code}",
                "details": {"response": response.text}
            }
    except Exception as e:
        return {"success": False, "message": f"测试模型加载异常: {str(e)}"}

# 端到端流程测试
@run_test
def test_端到端_数据监控流程():
    """测试从数据采集到监控展示的完整流程"""
    try:
        driver = setup_webdriver()
        if not driver:
            return {"success": False, "message": "无法初始化WebDriver"}
        
        # 访问数据监控页面
        driver.get(f"{FRONTEND_URL}/dashboard")
        
        # 等待数据图表加载
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, ".chart-container"))
        )
        
        # 检查是否有数据显示
        chart_elements = driver.find_elements(By.CSS_SELECTOR, ".chart-container")
        has_charts = len(chart_elements) > 0
        
        # 检查数据刷新功能
        refresh_button = None
        try:
            refresh_button = WebDriverWait(driver, 5).until(
                EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), '刷新') or contains(@title, '刷新')]"))
            )
        except:
            pass
        
        if refresh_button:
            refresh_button.click()
            time.sleep(2)  # 等待数据刷新
        
        # 截图
        screenshot_path = "端到端测试_数据监控流程.png"
        driver.save_screenshot(screenshot_path)
        
        driver.quit()
        
        success = has_charts
        return {
            "success": success,
            "message": "数据监控流程正常" if success else "数据监控流程异常",
            "details": {
                "图表元素数": len(chart_elements),
                "刷新按钮存在": refresh_button is not None,
                "截图路径": screenshot_path
            }
        }
    except Exception as e:
        return {"success": False, "message": f"测试数据监控流程异常: {str(e)}"}

@run_test
def test_端到端_预测流程():
    """测试数据预测的完整流程"""
    try:
        # 准备预测请求数据
        prediction_data = {
            "plant_id": "plant_001",
            "data_values": [
                {"timestamp": (datetime.now() - timedelta(hours=i)).isoformat(), 
                 "temperature": 80 + i, 
                 "pressure": 120 - i*0.5, 
                 "oxygen_level": 10 + i*0.2}
                for i in range(24)
            ],
            "prediction_horizon": 12,
            "model_type": "lstm"
        }
        
        # 发送预测请求
        response = requests.post(
            f"{BASE_URL}/api/predict",
            json=prediction_data,
            timeout=15
        )
        
        if response.status_code == 200:
            pred_data = response.json()
            
            # 检查预测结果
            has_predictions = "predictions" in pred_data
            predictions_valid = False
            
            if has_predictions:
                predictions = pred_data["predictions"]
                predictions_valid = len(predictions) > 0 and all(isinstance(p, (int, float)) for p in predictions)
            
            success = has_predictions and predictions_valid
            return {
                "success": success,
                "message": "预测流程正常" if success else "预测流程异常",
                "details": {
                    "预测点数": len(pred_data.get("predictions", [])),
                    "接口响应": pred_data
                }
            }
        else:
            return {
                "success": False,
                "message": f"预测请求失败，HTTP状态码: {response.status_code}",
                "details": {"response": response.text}
            }
    except Exception as e:
        return {"success": False, "message": f"测试预测流程异常: {str(e)}"}

@run_test
def test_端到端_自然语言查询流程():
    """测试自然语言查询的完整流程"""
    try:
        # 准备查询数据
        query_data = {
            "query": "上周SO2排放量是多少？",
            "plant_id": "plant_001"
        }
        
        # 发送查询请求
        response = requests.post(
            f"{BASE_URL}/api/nlp_query",
            json=query_data,
            timeout=10
        )
        
        if response.status_code == 200:
            query_result = response.json()
            
            # 检查查询结果
            has_answer = "answer" in query_result
            answer_valid = False
            
            if has_answer:
                answer = query_result["answer"]
                answer_valid = isinstance(answer, str) and len(answer) > 0
            
            success = has_answer and answer_valid
            return {
                "success": success,
                "message": "自然语言查询流程正常" if success else "自然语言查询流程异常",
                "details": {
                    "查询": query_data["query"],
                    "回答": query_result.get("answer", ""),
                    "接口响应": query_result
                }
            }
        else:
            return {
                "success": False,
                "message": f"自然语言查询请求失败，HTTP状态码: {response.status_code}",
                "details": {"response": response.text}
            }
    except Exception as e:
        return {"success": False, "message": f"测试自然语言查询流程异常: {str(e)}"}

# 数据流测试
@run_test
def test_数据流_实时数据处理():
    """测试实时数据的接收与处理流程"""
    try:
        # 准备实时数据
        real_time_data = {
            "plant_id": "plant_001",
            "timestamp": datetime.now().isoformat(),
            "metrics": {
                "so2": 25.7,
                "nox": 40.2,
                "co2": 350.5,
                "particulate_matter": 12.3,
                "temperature": 85.6,
                "pressure": 118.9,
                "oxygen_level": 10.8
            }
        }
        
        # 发送实时数据
        response = requests.post(
            f"{BASE_URL}/api/data/real_time",
            json=real_time_data,
            timeout=5
        )
        
        # 检查响应
        if response.status_code == 200:
            result = response.json()
            data_processed = result.get("success", False)
            
            # 验证数据是否已处理
            time.sleep(1)  # 等待数据处理
            verify_response = requests.get(
                f"{BASE_URL}/api/data/latest?plant_id={real_time_data['plant_id']}",
                timeout=5
            )
            
            data_verified = False
            if verify_response.status_code == 200:
                latest_data = verify_response.json()
                if "data" in latest_data:
                    # 找到我们刚发送的数据点
                    found = False
                    for point in latest_data["data"]:
                        # 检查时间戳是否接近
                        if "timestamp" in point:
                            time_diff = abs(datetime.fromisoformat(point["timestamp"].replace("Z", "+00:00")) - 
                                         datetime.fromisoformat(real_time_data["timestamp"]))
                            if time_diff.total_seconds() < 60:
                                found = True
                                break
                    
                    data_verified = found
            
            success = data_processed and data_verified
            return {
                "success": success,
                "message": "实时数据处理流程正常" if success else "实时数据处理流程异常",
                "details": {
                    "数据已处理": data_processed,
                    "数据已验证": data_verified,
                    "处理响应": result,
                    "验证响应": latest_data if verify_response.status_code == 200 else "验证请求失败"
                }
            }
        else:
            return {
                "success": False,
                "message": f"实时数据处理请求失败，HTTP状态码: {response.status_code}",
                "details": {"response": response.text}
            }
    except Exception as e:
        return {"success": False, "message": f"测试实时数据处理异常: {str(e)}"}

@run_test
def test_数据流_报表生成():
    """测试报表生成流程"""
    try:
        # 准备报表生成请求
        report_request = {
            "plant_id": "plant_001",
            "start_date": (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d"),
            "end_date": datetime.now().strftime("%Y-%m-%d"),
            "report_type": "emissions_summary"
        }
        
        # 发送报表生成请求
        response = requests.post(
            f"{BASE_URL}/api/reports/generate",
            json=report_request,
            timeout=15
        )
        
        if response.status_code == 200:
            result = response.json()
            report_id = result.get("report_id")
            
            # 检查报表是否生成
            report_status = None
            if report_id:
                # 等待报表生成
                max_retries = 5
                for i in range(max_retries):
                    status_response = requests.get(
                        f"{BASE_URL}/api/reports/status/{report_id}",
                        timeout=5
                    )
                    
                    if status_response.status_code == 200:
                        status_data = status_response.json()
                        report_status = status_data.get("status")
                        
                        if report_status == "completed":
                            break
                        
                        time.sleep(2)  # 等待报表生成
            
            # 下载报表
            report_url = None
            if report_status == "completed":
                download_response = requests.get(
                    f"{BASE_URL}/api/reports/download/{report_id}",
                    timeout=10
                )
                
                if download_response.status_code == 200:
                    # 保存报表
                    report_url = f"report_{report_id}.pdf"
                    with open(report_url, "wb") as f:
                        f.write(download_response.content)
            
            success = report_status == "completed" and report_url is not None
            return {
                "success": success,
                "message": "报表生成流程正常" if success else "报表生成流程异常",
                "details": {
                    "报表ID": report_id,
                    "报表状态": report_status,
                    "报表URL": report_url,
                    "生成响应": result
                }
            }
        else:
            return {
                "success": False,
                "message": f"报表生成请求失败，HTTP状态码: {response.status_code}",
                "details": {"response": response.text}
            }
    except Exception as e:
        return {"success": False, "message": f"测试报表生成流程异常: {str(e)}"}

# 异常恢复测试
@run_test
def test_异常恢复_输入异常处理():
    """测试系统对异常输入的处理能力"""
    try:
        # 准备异常输入数据
        invalid_data = {
            "plant_id": "plant_001",
            "data_values": [
                {"timestamp": "invalid_timestamp", 
                 "temperature": "not_a_number", 
                 "pressure": None, 
                 "oxygen_level": "高"}
            ],
            "prediction_horizon": -5,
            "model_type": "unknown_model"
        }
        
        # 发送异常输入请求
        response = requests.post(
            f"{BASE_URL}/api/predict",
            json=invalid_data,
            timeout=10
        )
        
        # 检查响应是否为错误码而非服务器崩溃
        is_4xx_response = 400 <= response.status_code < 500
        
        # 检查错误消息是否明确
        has_clear_error = False
        if is_4xx_response and response.headers.get('content-type') == 'application/json':
            error_data = response.json()
            has_clear_error = "error" in error_data and len(error_data["error"]) > 0
        
        success = is_4xx_response and has_clear_error
        return {
            "success": success,
            "message": "系统正确处理了异常输入" if success else "系统未能正确处理异常输入",
            "details": {
                "HTTP状态码": response.status_code,
                "错误信息": response.json() if is_4xx_response and response.headers.get('content-type') == 'application/json' else response.text,
                "响应头": dict(response.headers)
            }
        }
    except Exception as e:
        return {"success": False, "message": f"测试异常输入处理异常: {str(e)}"}

@run_test
def test_异常恢复_高并发处理():
    """测试系统在高并发情况下的稳定性"""
    try:
        # 模拟高并发请求
        num_requests = 20
        concurrent_workers = 10
        
        # 准备请求数据
        request_data = {
            "plant_id": "plant_001",
            "start_date": (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d"),
            "end_date": datetime.now().strftime("%Y-%m-%d")
        }
        
        # 发送并发请求函数
        def send_request():
            try:
                response = requests.get(
                    f"{BASE_URL}/api/emissions",
                    params=request_data,
                    timeout=10
                )
                return {
                    "status_code": response.status_code,
                    "response_time": response.elapsed.total_seconds(),
                    "success": 200 <= response.status_code < 300
                }
            except Exception as e:
                return {
                    "status_code": 0,
                    "response_time": 0,
                    "success": False,
                    "error": str(e)
                }
        
        # 并发执行请求
        results = []
        with ThreadPoolExecutor(max_workers=concurrent_workers) as executor:
            futures = [executor.submit(send_request) for _ in range(num_requests)]
            results = [future.result() for future in futures]
        
        # 分析结果
        success_count = sum(1 for r in results if r["success"])
        failure_count = num_requests - success_count
        
        avg_response_time = 0
        if success_count > 0:
            avg_response_time = sum(r["response_time"] for r in results if r["success"]) / success_count
        
        # 判断测试结果
        success_rate = success_count / num_requests
        success = success_rate >= 0.9  # 成功率至少90%
        
        return {
            "success": success,
            "message": f"高并发处理正常，成功率: {success_rate*100:.1f}%" if success else f"高并发处理异常，成功率过低: {success_rate*100:.1f}%",
            "details": {
                "请求总数": num_requests,
                "成功请求数": success_count,
                "失败请求数": failure_count,
                "成功率": f"{success_rate*100:.1f}%",
                "平均响应时间": f"{avg_response_time*1000:.2f}ms",
                "请求详情": results
            }
        }
    except Exception as e:
        return {"success": False, "message": f"测试高并发处理异常: {str(e)}"}

def generate_report():
    """生成测试报告"""
    # 计算通过率
    if TEST_RESULTS["总测试数"] > 0:
        TEST_RESULTS["通过率"] = (TEST_RESULTS["通过测试数"] / TEST_RESULTS["总测试数"]) * 100
    
    # 创建报告文本
    report = f"""
========================================================
            电厂污染物排放预测系统 系统测试报告
========================================================
测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

测试摘要:
- 总测试数: {TEST_RESULTS["总测试数"]}
- 通过测试: {TEST_RESULTS["通过测试数"]}
- 失败测试: {TEST_RESULTS["失败测试数"]}
- 通过率: {TEST_RESULTS["通过率"]:.1f}%

系统集成测试结果:
"""
    
    # 添加系统集成测试结果
    for test_name, result in TEST_RESULTS.get("系统集成测试", {}).items():
        status = "✅ 通过" if result["成功"] else "❌ 失败"
        report += f"- {test_name}: {status}, 耗时: {result['耗时']}\n  {result['消息']}\n"
    
    # 添加端到端流程测试结果
    report += "\n端到端流程测试结果:\n"
    for test_name, result in TEST_RESULTS.get("端到端流程测试", {}).items():
        status = "✅ 通过" if result["成功"] else "❌ 失败"
        report += f"- {test_name}: {status}, 耗时: {result['耗时']}\n  {result['消息']}\n"
    
    # 添加数据流测试结果
    report += "\n数据流测试结果:\n"
    for test_name, result in TEST_RESULTS.get("数据流测试", {}).items():
        status = "✅ 通过" if result["成功"] else "❌ 失败"
        report += f"- {test_name}: {status}, 耗时: {result['耗时']}\n  {result['消息']}\n"
    
    # 添加异常恢复测试结果
    report += "\n异常恢复测试结果:\n"
    for test_name, result in TEST_RESULTS.get("异常恢复测试", {}).items():
        status = "✅ 通过" if result["成功"] else "❌ 失败"
        report += f"- {test_name}: {status}, 耗时: {result['耗时']}\n  {result['消息']}\n"
    
    # 添加结束信息
    report += "\n========================================================\n"
    report += "注意: 详细的测试数据和截图可以在日志和测试结果目录中找到\n"
    report += "========================================================\n"
    
    # 保存报告
    with open("系统测试报告.txt", "w", encoding="utf-8") as f:
        f.write(report)
    
    # 保存测试结果JSON
    with open("系统测试报告.json", "w", encoding="utf-8") as f:
        json.dump(TEST_RESULTS, f, ensure_ascii=False, indent=2)
    
    # 生成测试结果图表
    plt.figure(figsize=(10, 6))
    
    # 按类别统计成功和失败的测试
    categories = ["系统集成测试", "端到端流程测试", "数据流测试", "异常恢复测试"]
    success_counts = []
    failure_counts = []
    
    for category in categories:
        if category in TEST_RESULTS:
            success_count = sum(1 for result in TEST_RESULTS[category].values() if result["成功"])
            failure_count = len(TEST_RESULTS[category]) - success_count
            success_counts.append(success_count)
            failure_counts.append(failure_count)
        else:
            success_counts.append(0)
            failure_counts.append(0)
    
    # 创建堆叠条形图
    bar_width = 0.5
    indices = range(len(categories))
    
    plt.bar(indices, success_counts, bar_width, label='通过', color='#4CAF50')
    plt.bar(indices, failure_counts, bar_width, bottom=success_counts, label='失败', color='#F44336')
    
    plt.xlabel('测试类别')
    plt.ylabel('测试数量')
    plt.title('系统测试结果统计')
    plt.xticks(indices, categories)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("系统测试结果.png")
    
    # 生成饼图显示总体通过率
    plt.figure(figsize=(8, 8))
    labels = ['通过', '失败']
    sizes = [TEST_RESULTS["通过测试数"], TEST_RESULTS["失败测试数"]]
    colors = ['#4CAF50', '#F44336']
    explode = (0.1, 0)  # 突出显示通过的部分
    
    plt.pie(sizes, explode=explode, labels=labels, colors=colors,
            autopct='%1.1f%%', shadow=True, startangle=90)
    plt.axis('equal')
    plt.title('测试通过率')
    
    plt.tight_layout()
    plt.savefig("系统测试通过率.png")
    
    logger.info(f"系统测试报告已生成: 系统测试报告.txt")
    
    return report

def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="电厂污染物排放预测系统 - 系统测试脚本")
    parser.add_argument("--url", help="API基础URL", default="http://localhost:8000")
    parser.add_argument("--frontend", help="前端URL", default="http://localhost:3000")
    args = parser.parse_args()
    
    global BASE_URL, FRONTEND_URL
    BASE_URL = args.url
    FRONTEND_URL = args.frontend
    
    logger.info(f"开始执行系统测试，后端API: {BASE_URL}, 前端: {FRONTEND_URL}")
    
    # 检查系统是否在线
    system_status = check_system_online()
    if not system_status["api_online"]:
        logger.error("系统API不在线，无法执行测试")
        return
    
    if not system_status["frontend_online"]:
        logger.warning("系统前端不在线，某些测试可能会失败")
    
    # 执行系统集成测试
    test_系统集成_前后端连接()
    test_系统集成_数据库连接()
    test_系统集成_模型加载()
    
    # 执行端到端流程测试
    test_端到端_数据监控流程()
    test_端到端_预测流程()
    test_端到端_自然语言查询流程()
    
    # 执行数据流测试
    test_数据流_实时数据处理()
    test_数据流_报表生成()
    
    # 执行异常恢复测试
    test_异常恢复_输入异常处理()
    test_异常恢复_高并发处理()
    
    # 生成报告
    report = generate_report()
    
    # 输出报告
    print("\n" + report)

if __name__ == "__main__":
    main() 
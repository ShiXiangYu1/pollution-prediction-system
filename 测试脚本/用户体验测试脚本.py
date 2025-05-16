#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
电厂污染物排放预测系统 - 用户体验测试脚本
"""

import requests
import json
import time
import os
import sys
from datetime import datetime, timedelta
import logging
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import random
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import traceback

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("用户体验测试结果.log")
    ]
)

logger = logging.getLogger(__name__)

# 基础URL，默认为本地地址，可通过命令行参数修改
BASE_URL = "http://localhost:8000"

# 测试结果统计
test_results = {
    "total": 0,
    "passed": 0,
    "failed": 0,
    "ux_metrics": {
        "avg_page_load_time": 0,
        "avg_response_time": 0,
        "error_message_clarity": 0,
        "navigation_steps": {},
        "form_completion_time": {}
    }
}

def setup_driver():
    """设置Selenium WebDriver"""
    # 使用Chrome浏览器，也可以更换为其他浏览器
    options = webdriver.ChromeOptions()
    # options.add_argument('--headless')  # 无头模式，不显示浏览器窗口
    driver = webdriver.Chrome(options=options)
    driver.implicitly_wait(10)  # 隐式等待时间
    return driver

def run_test(test_func):
    """测试装饰器，用于统计测试结果"""
    def wrapper(*args, **kwargs):
        test_name = test_func.__name__
        test_results["total"] += 1
        logger.info(f"正在执行测试: {test_name}")
        
        try:
            result, metrics = test_func(*args, **kwargs)
            if result:
                test_results["passed"] += 1
                logger.info(f"✅ 测试通过: {test_name}")
            else:
                test_results["failed"] += 1
                logger.error(f"❌ 测试失败: {test_name}")
            
            # 记录指标
            if metrics:
                for key, value in metrics.items():
                    if isinstance(test_results["ux_metrics"].get(key, {}), dict):
                        test_results["ux_metrics"][key][test_name] = value
                    else:
                        # 累加并计算平均值
                        current_value = test_results["ux_metrics"].get(key, 0)
                        current_count = test_results.get(f"{key}_count", 0)
                        new_count = current_count + 1
                        new_value = (current_value * current_count + value) / new_count
                        test_results["ux_metrics"][key] = new_value
                        test_results[f"{key}_count"] = new_count
            
            return result, metrics
        except Exception as e:
            test_results["failed"] += 1
            logger.error(f"❌ 测试异常: {test_name} - {str(e)}")
            traceback.print_exc()
            return False, {}
    
    return wrapper

@run_test
def test_home_page_load(driver):
    """测试首页加载性能和响应时间"""
    logger.info("测试首页加载性能...")
    
    metrics = {}
    
    # 记录加载开始时间
    start_time = time.time()
    
    # 访问首页
    driver.get(f"{BASE_URL}/")
    
    # 等待页面主要元素加载完成
    try:
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, "so2-avg"))
        )
        load_time = time.time() - start_time
        logger.info(f"首页加载时间: {load_time:.2f}秒")
        metrics["page_load_time"] = load_time
        
        # 检查是否有主要元素
        elements_found = True
        required_elements = ["so2-avg", "nox-avg", "dust-avg"]
        for element_id in required_elements:
            if not driver.find_element(By.ID, element_id):
                elements_found = False
                logger.warning(f"未找到页面元素: {element_id}")
        
        # 记录页面尺寸
        scroll_height = driver.execute_script("return document.body.scrollHeight")
        viewport_height = driver.execute_script("return window.innerHeight")
        metrics["scroll_ratio"] = scroll_height / viewport_height
        
        if metrics["scroll_ratio"] > 3:
            logger.warning(f"页面太长，可能需要过多滚动: 滚动比例={metrics['scroll_ratio']:.2f}")
        
        # 检查页面响应式设计
        driver.set_window_size(1920, 1080)  # 桌面尺寸
        time.sleep(1)
        desktop_layout = driver.execute_script("return document.body.clientWidth")
        
        driver.set_window_size(375, 812)  # 移动设备尺寸
        time.sleep(1)
        mobile_layout = driver.execute_script("return document.body.clientWidth")
        
        metrics["responsive_design"] = True if mobile_layout < desktop_layout else False
        
        # 恢复窗口大小
        driver.maximize_window()
        
        return elements_found and load_time < 5, metrics  # 加载时间应小于5秒
    
    except TimeoutException:
        logger.error("首页加载超时")
        return False, {"page_load_time": 10}  # 超时时间

@run_test
def test_navigation_flow(driver):
    """测试导航流程的流畅度"""
    logger.info("测试导航流流畅度...")
    
    metrics = {"navigation_steps": {}}
    
    # 访问首页
    driver.get(f"{BASE_URL}/")
    time.sleep(1)
    
    # 定义要测试的导航链接
    navigation_items = [
        {"name": "数据看板", "link_text": "数据看板", "expected_url": "/dashboard"},
        {"name": "排放预测", "link_text": "排放预测", "expected_url": "/prediction"},
        {"name": "自然语言查询", "link_text": "自然语言查询", "expected_url": "/nlp"},
        {"name": "语音识别", "link_text": "语音识别", "expected_url": "/speech"}
    ]
    
    success = True
    for item in navigation_items:
        try:
            # 查找并点击导航链接
            start_time = time.time()
            link = driver.find_element(By.LINK_TEXT, item["link_text"])
            link.click()
            
            # 等待页面加载
            WebDriverWait(driver, 10).until(
                EC.url_contains(item["expected_url"])
            )
            
            # 记录导航时间
            nav_time = time.time() - start_time
            logger.info(f"导航到{item['name']}页面耗时: {nav_time:.2f}秒")
            metrics["navigation_steps"][item["name"]] = nav_time
            
            if nav_time > 3:
                logger.warning(f"导航到{item['name']}页面耗时过长: {nav_time:.2f}秒")
                success = False
            
            # 确认URL正确
            current_url = driver.current_url
            if item["expected_url"] not in current_url:
                logger.error(f"导航错误: 期望URL包含 {item['expected_url']}，实际为 {current_url}")
                success = False
            
            # 返回首页测试下一个导航项
            driver.get(f"{BASE_URL}/")
            time.sleep(1)
            
        except Exception as e:
            logger.error(f"导航到{item['name']}页面失败: {str(e)}")
            success = False
    
    return success, metrics

@run_test
def test_dashboard_interaction(driver):
    """测试数据看板页面的交互性能"""
    logger.info("测试数据看板页面交互...")
    
    metrics = {}
    
    # 访问数据看板页面
    driver.get(f"{BASE_URL}/dashboard")
    
    try:
        # 等待主要图表加载
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, "emissions-chart"))
        )
        
        # 测试筛选功能
        filter_elements = {
            "date_range": "date-range-selector",
            "pollutant_type": "pollutant-selector",
            "unit_id": "unit-selector"
        }
        
        interaction_times = []
        for name, element_id in filter_elements.items():
            try:
                # 尝试查找并操作筛选控件
                element = driver.find_element(By.ID, element_id)
                start_time = time.time()
                element.click()
                time.sleep(0.5)  # 等待下拉列表显示
                
                # 选择一个选项
                if name == "date_range":
                    option = driver.find_element(By.XPATH, f"//select[@id='{element_id}']/option[2]")
                else:
                    option = driver.find_element(By.XPATH, f"//select[@id='{element_id}']/option[1]")
                
                option.click()
                
                # 应用筛选
                apply_button = driver.find_element(By.ID, "apply-filter-btn")
                if apply_button:
                    apply_button.click()
                
                # 等待数据重新加载
                time.sleep(2)
                
                interaction_time = time.time() - start_time
                interaction_times.append(interaction_time)
                logger.info(f"筛选控件 {name} 交互时间: {interaction_time:.2f}秒")
                
            except Exception as e:
                logger.warning(f"未找到或无法操作筛选控件 {name}: {str(e)}")
        
        if interaction_times:
            metrics["avg_interaction_time"] = sum(interaction_times) / len(interaction_times)
        
        # 测试图表交互
        try:
            chart = driver.find_element(By.ID, "emissions-chart")
            
            # 尝试悬停在图表上
            action = webdriver.ActionChains(driver)
            action.move_to_element(chart)
            action.move_by_offset(50, 0)  # 移动鼠标位置
            start_time = time.time()
            action.perform()
            
            # 等待工具提示显示
            time.sleep(1)
            
            # 检查是否有工具提示
            tooltip = driver.find_element(By.CLASS_NAME, "tooltip")
            
            if tooltip:
                tooltip_time = time.time() - start_time
                logger.info(f"图表工具提示显示时间: {tooltip_time:.2f}秒")
                metrics["tooltip_response_time"] = tooltip_time
                
        except Exception as e:
            logger.warning(f"图表交互测试失败: {str(e)}")
        
        return True, metrics
        
    except TimeoutException:
        logger.error("数据看板页面加载超时")
        return False, {}

@run_test
def test_prediction_form(driver):
    """测试排放预测表单填写和提交"""
    logger.info("测试排放预测表单...")
    
    metrics = {}
    
    # 访问排放预测页面
    driver.get(f"{BASE_URL}/prediction")
    
    try:
        # 等待表单加载
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, "prediction-form"))
        )
        
        # 填写表单
        form_fields = {
            "feature-load": "80.5",
            "feature-temp": "25.3",
            "feature-humidity": "62.5",
            "feature-wind": "3.8"
        }
        
        start_time = time.time()
        for field_id, value in form_fields.items():
            try:
                field = driver.find_element(By.ID, field_id)
                field.clear()
                field.send_keys(value)
            except Exception as e:
                logger.warning(f"未找到或无法填写表单字段 {field_id}: {str(e)}")
        
        # 选择模型
        try:
            model_selector = driver.find_element(By.ID, "model-type")
            model_selector.click()
            model_option = driver.find_element(By.XPATH, "//select[@id='model-type']/option[text()='LSTM']")
            model_option.click()
        except Exception as e:
            logger.warning(f"模型选择失败: {str(e)}")
        
        # 提交表单
        try:
            submit_button = driver.find_element(By.ID, "predict-btn")
            submit_button.click()
            
            # 等待结果显示
            WebDriverWait(driver, 15).until(
                EC.presence_of_element_located((By.ID, "prediction-results"))
            )
            
            form_time = time.time() - start_time
            logger.info(f"预测表单填写和提交时间: {form_time:.2f}秒")
            metrics["form_completion_time"] = form_time
            
            # 检查结果是否包含预期元素
            results = driver.find_element(By.ID, "prediction-results")
            if "SO2" in results.text and "NOx" in results.text:
                logger.info("预测结果包含预期内容")
                return True, metrics
            else:
                logger.warning(f"预测结果内容不完整: {results.text}")
                return False, metrics
            
        except Exception as e:
            logger.error(f"表单提交或结果加载失败: {str(e)}")
            return False, metrics
        
    except TimeoutException:
        logger.error("排放预测页面加载超时")
        return False, {}

@run_test
def test_nlp_query(driver):
    """测试自然语言查询功能"""
    logger.info("测试自然语言查询功能...")
    
    metrics = {}
    
    # 访问自然语言查询页面
    driver.get(f"{BASE_URL}/nlp")
    
    try:
        # 等待查询输入框加载
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, "query-input"))
        )
        
        # 输入查询
        query_input = driver.find_element(By.ID, "query-input")
        query_input.clear()
        
        # 测试不同的查询
        test_queries = [
            "昨天的SO2平均排放量是多少?",
            "上周NOx排放超标了几次?",
            "本月烟尘排放趋势如何?"
        ]
        
        response_times = []
        for query in test_queries:
            query_input.clear()
            query_input.send_keys(query)
            
            # 提交查询
            start_time = time.time()
            submit_button = driver.find_element(By.ID, "query-btn")
            submit_button.click()
            
            # 等待结果显示
            try:
                WebDriverWait(driver, 20).until(
                    EC.presence_of_element_located((By.ID, "query-result"))
                )
                
                response_time = time.time() - start_time
                response_times.append(response_time)
                logger.info(f"查询 '{query}' 响应时间: {response_time:.2f}秒")
                
                # 检查结果是否有内容
                result = driver.find_element(By.ID, "query-result")
                if len(result.text) > 20:  # 简单判断结果是否有实质内容
                    logger.info(f"查询 '{query}' 返回有效结果")
                else:
                    logger.warning(f"查询 '{query}' 结果可能不完整: {result.text}")
                
                # 检查图表是否显示
                try:
                    chart = driver.find_element(By.ID, "result-chart")
                    if chart:
                        logger.info(f"查询 '{query}' 显示了图表")
                except:
                    logger.info(f"查询 '{query}' 未显示图表，可能是查询类型不需要图表")
                
                # 给系统一些恢复时间
                time.sleep(2)
                
            except TimeoutException:
                logger.error(f"查询 '{query}' 响应超时")
                response_times.append(20)  # 超时时间
        
        if response_times:
            metrics["avg_query_response_time"] = sum(response_times) / len(response_times)
            metrics["max_query_response_time"] = max(response_times)
        
        # 测试错误处理
        query_input.clear()
        query_input.send_keys("这是一个无效的查询，系统应该给出友好的错误提示")
        submit_button.click()
        
        try:
            # 等待错误提示
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CLASS_NAME, "error-message"))
            )
            
            error_message = driver.find_element(By.CLASS_NAME, "error-message")
            metrics["error_message_clarity"] = 1 if len(error_message.text) > 15 else 0
            logger.info(f"错误提示: {error_message.text}")
            
        except:
            logger.warning("系统未显示错误提示或错误提示格式不符合预期")
            metrics["error_message_clarity"] = 0
        
        return metrics["avg_query_response_time"] < 10, metrics  # 平均响应时间应小于10秒
        
    except Exception as e:
        logger.error(f"自然语言查询测试失败: {str(e)}")
        return False, {}

@run_test
def test_speech_interface(driver):
    """测试语音识别界面（不测试实际语音识别功能）"""
    logger.info("测试语音识别界面...")
    
    metrics = {}
    
    # 访问语音识别页面
    driver.get(f"{BASE_URL}/speech")
    
    try:
        # 等待录音按钮加载
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, "record-btn"))
        )
        
        # 检查界面元素
        required_elements = [
            "record-btn",
            "recognized-text",
            "execute-btn"
        ]
        
        elements_found = True
        for element_id in required_elements:
            try:
                driver.find_element(By.ID, element_id)
            except NoSuchElementException:
                logger.warning(f"未找到语音识别界面元素: {element_id}")
                elements_found = False
        
        # 检查按钮状态和交互指引
        record_btn = driver.find_element(By.ID, "record-btn")
        if "disabled" in record_btn.get_attribute("class"):
            logger.warning("录音按钮初始状态为禁用，这可能会让用户困惑")
        
        # 查找用户指引
        instructions = driver.find_elements(By.CLASS_NAME, "instruction")
        if instructions:
            logger.info(f"找到 {len(instructions)} 条用户指引")
            metrics["has_user_instructions"] = 1
        else:
            logger.warning("未找到明显的用户指引")
            metrics["has_user_instructions"] = 0
        
        # 模拟点击录音按钮
        start_time = time.time()
        record_btn.click()
        
        # 等待录音状态变化
        time.sleep(1)
        
        # 检查视觉反馈
        if "recording" in record_btn.get_attribute("class"):
            logger.info("录音按钮提供了明显的视觉反馈")
            metrics["visual_feedback"] = 1
        else:
            logger.warning("录音按钮未提供明显的视觉反馈")
            metrics["visual_feedback"] = 0
        
        return elements_found, metrics
        
    except Exception as e:
        logger.error(f"语音识别界面测试失败: {str(e)}")
        return False, {}

def evaluate_results():
    """评估测试结果并给出用户体验评分"""
    # 加载时间评分 (满分5分)
    if "avg_page_load_time" in test_results["ux_metrics"] and test_results["ux_metrics"]["avg_page_load_time"] > 0:
        load_time = test_results["ux_metrics"]["avg_page_load_time"]
        if load_time < 1:
            load_score = 5
        elif load_time < 2:
            load_score = 4
        elif load_time < 3:
            load_score = 3
        elif load_time < 5:
            load_score = 2
        else:
            load_score = 1
    else:
        load_score = 0
    
    # 响应时间评分 (满分5分)
    if "avg_response_time" in test_results["ux_metrics"] and test_results["ux_metrics"]["avg_response_time"] > 0:
        response_time = test_results["ux_metrics"]["avg_response_time"]
        if response_time < 2:
            response_score = 5
        elif response_time < 5:
            response_score = 4
        elif response_time < 8:
            response_score = 3
        elif response_time < 12:
            response_score = 2
        else:
            response_score = 1
    else:
        response_score = 0
    
    # 导航流畅度评分 (满分5分)
    if "navigation_steps" in test_results["ux_metrics"] and test_results["ux_metrics"]["navigation_steps"]:
        nav_times = list(test_results["ux_metrics"]["navigation_steps"].values())
        avg_nav_time = sum(nav_times) / len(nav_times)
        
        if avg_nav_time < 1:
            nav_score = 5
        elif avg_nav_time < 2:
            nav_score = 4
        elif avg_nav_time < 3:
            nav_score = 3
        elif avg_nav_time < 4:
            nav_score = 2
        else:
            nav_score = 1
    else:
        nav_score = 0
    
    # 错误提示清晰度评分 (满分5分)
    if "error_message_clarity" in test_results["ux_metrics"]:
        error_score = test_results["ux_metrics"]["error_message_clarity"] * 5
    else:
        error_score = 0
    
    # 表单完成时间评分 (满分5分)
    if "form_completion_time" in test_results["ux_metrics"] and test_results["ux_metrics"]["form_completion_time"] > 0:
        form_time = test_results["ux_metrics"]["form_completion_time"]
        if form_time < 10:
            form_score = 5
        elif form_time < 15:
            form_score = 4
        elif form_time < 20:
            form_score = 3
        elif form_time < 30:
            form_score = 2
        else:
            form_score = 1
    else:
        form_score = 0
    
    # 计算总分 (满分25分，转换为百分比)
    max_score = 20  # 最大可能分数（如果所有指标都有值）
    actual_max = 0  # 实际最大可能分数（根据有值的指标）
    
    total_score = 0
    if load_score > 0:
        total_score += load_score
        actual_max += 5
    
    if response_score > 0:
        total_score += response_score
        actual_max += 5
    
    if nav_score > 0:
        total_score += nav_score
        actual_max += 5
    
    if error_score > 0:
        total_score += error_score
        actual_max += 5
    
    if form_score > 0:
        total_score += form_score
        actual_max += 5
    
    # 如果没有足够的指标，无法给出评分
    if actual_max < 10:
        final_score = "N/A"
        grade = "无法评分"
    else:
        # 计算百分比分数
        percentage = (total_score / actual_max) * 100
        final_score = f"{percentage:.1f}%"
        
        # 确定等级
        if percentage >= 90:
            grade = "优秀"
        elif percentage >= 80:
            grade = "良好"
        elif percentage >= 70:
            grade = "中等"
        elif percentage >= 60:
            grade = "及格"
        else:
            grade = "不及格"
    
    # 返回评分结果
    return {
        "load_score": load_score,
        "response_score": response_score,
        "nav_score": nav_score,
        "error_score": error_score,
        "form_score": form_score,
        "total_score": total_score,
        "max_score": actual_max,
        "final_score": final_score,
        "grade": grade
    }

def generate_ux_report(score_results):
    """生成用户体验测试报告"""
    report = f"""
====== 电厂污染物排放预测系统用户体验测试报告 ======
测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
测试环境: {BASE_URL}

测试结果概要:
- 总测试数: {test_results['total']}
- 通过测试: {test_results['passed']}
- 失败测试: {test_results['failed']}
- 通过率: {(test_results['passed'] / test_results['total'] * 100) if test_results['total'] > 0 else 0:.1f}%

用户体验评分:
- 页面加载时间: {score_results['load_score']}/5分
- API响应时间: {score_results['response_score']}/5分
- 导航流畅度: {score_results['nav_score']}/5分
- 错误提示清晰度: {score_results['error_score']}/5分
- 表单操作体验: {score_results['form_score']}/5分

总体评分: {score_results['total_score']}/{score_results['max_score']}分 ({score_results['final_score']})
用户体验等级: {score_results['grade']}

详细测试指标:
"""
    
    # 添加详细指标
    for key, value in test_results["ux_metrics"].items():
        if isinstance(value, dict):
            report += f"\n{key}:\n"
            for sub_key, sub_value in value.items():
                report += f"  - {sub_key}: {sub_value:.2f}\n"
        else:
            report += f"- {key}: {value:.2f}\n"
    
    report += f"\n{'='*50}\n"
    
    # 写入文件
    with open("用户体验测试报告.txt", "w", encoding="utf-8") as f:
        f.write(report)
    
    # 生成图表
    try:
        # 页面加载时间分析
        if "navigation_steps" in test_results["ux_metrics"] and test_results["ux_metrics"]["navigation_steps"]:
            plt.figure(figsize=(10, 6))
            pages = list(test_results["ux_metrics"]["navigation_steps"].keys())
            times = list(test_results["ux_metrics"]["navigation_steps"].values())
            
            plt.bar(pages, times, color='skyblue')
            plt.title('各页面导航时间对比')
            plt.xlabel('页面')
            plt.ylabel('导航时间 (秒)')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            # 保存图表
            plt.tight_layout()
            plt.savefig("用户体验测试_导航时间.png")
        
        # 表单操作时间分析
        if "form_completion_time" in test_results["ux_metrics"] and test_results["ux_metrics"]["form_completion_time"] > 0:
            plt.figure(figsize=(8, 6))
            
            # 创建环形图
            score_data = [
                score_results['load_score'],
                score_results['response_score'],
                score_results['nav_score'],
                score_results['error_score'],
                score_results['form_score']
            ]
            
            labels = ['页面加载', 'API响应', '导航流畅度', '错误提示', '表单操作']
            colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99','#c2c2f0']
            
            plt.pie(score_data, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            # 画一个白色的圆在饼图中间，这样就变成了环形图
            centre_circle = plt.Circle((0,0),0.70,fc='white')
            fig = plt.gcf()
            fig.gca().add_artist(centre_circle)
            
            plt.title('用户体验各方面评分占比')
            
            # 保存图表
            plt.tight_layout()
            plt.savefig("用户体验测试_评分占比.png")
    
    except Exception as e:
        logger.error(f"生成图表失败: {str(e)}")
    
    logger.info(f"用户体验测试报告已生成: 用户体验测试报告.txt")
    
    return report

def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="电厂污染物排放预测系统 - 用户体验测试脚本")
    parser.add_argument("--url", help="API基础URL", default="http://localhost:8000")
    args = parser.parse_args()
    
    global BASE_URL
    BASE_URL = args.url
    
    logger.info(f"开始用户体验测试，目标系统: {BASE_URL}")
    
    # 设置WebDriver
    driver = setup_driver()
    driver.maximize_window()
    
    try:
        # 执行测试
        test_home_page_load(driver)
        test_navigation_flow(driver)
        test_dashboard_interaction(driver)
        test_prediction_form(driver)
        test_nlp_query(driver)
        test_speech_interface(driver)
        
        # 评估结果
        score_results = evaluate_results()
        
        # 生成报告
        report = generate_ux_report(score_results)
        print("\n" + report)
        
    except Exception as e:
        logger.error(f"测试过程中出现异常: {str(e)}")
        traceback.print_exc()
    finally:
        # 关闭浏览器
        driver.quit()

if __name__ == "__main__":
    main() 
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
电厂污染物排放预测系统 - 功能测试脚本
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

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("功能测试结果.log")
    ]
)

logger = logging.getLogger(__name__)

# API基础URL，默认为本地地址，可通过命令行参数修改
BASE_URL = "http://localhost:8000"

# 测试结果统计
test_results = {
    "total": 0,
    "passed": 0,
    "failed": 0,
    "skipped": 0
}

def run_test(test_func):
    """测试装饰器，用于统计测试结果"""
    def wrapper(*args, **kwargs):
        test_name = test_func.__name__
        test_results["total"] += 1
        logger.info(f"正在执行测试: {test_name}")
        
        try:
            result = test_func(*args, **kwargs)
            if result:
                test_results["passed"] += 1
                logger.info(f"✅ 测试通过: {test_name}")
            else:
                test_results["failed"] += 1
                logger.error(f"❌ 测试失败: {test_name}")
            return result
        except Exception as e:
            test_results["failed"] += 1
            logger.error(f"❌ 测试异常: {test_name} - {str(e)}")
            return False
    
    return wrapper

@run_test
def test_health():
    """测试健康检查接口"""
    logger.info("测试健康检查接口...")
    
    response = requests.get(f"{BASE_URL}/health", timeout=10)
    
    if response.status_code == 200:
        logger.info(f"健康检查接口返回: {response.json()}")
        return True
    else:
        logger.error(f"健康检查接口异常: HTTP {response.status_code}")
        return False

@run_test
def test_emissions_api():
    """测试排放数据API"""
    logger.info("测试排放数据API...")
    
    response = requests.get(f"{BASE_URL}/api/emissions", timeout=10)
    
    if response.status_code == 200:
        data = response.json()
        if "data" in data and len(data["data"]) > 0:
            logger.info(f"排放数据API返回了 {len(data['data'])} 条数据")
            return True
        else:
            logger.warning(f"排放数据API返回数据可能有问题: {data}")
            return False
    else:
        logger.error(f"排放数据API异常: HTTP {response.status_code}")
        return False

@run_test
def test_emissions_api_with_filters():
    """测试带过滤条件的排放数据API"""
    logger.info("测试带过滤条件的排放数据API...")
    
    # 获取当前日期和前7天的日期
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
    
    params = {
        "start_date": start_date,
        "end_date": end_date,
        "type": "SO2",
        "page": 1,
        "page_size": 10
    }
    
    response = requests.get(f"{BASE_URL}/api/emissions", params=params, timeout=10)
    
    if response.status_code == 200:
        data = response.json()
        if "data" in data:
            logger.info(f"带过滤条件的排放数据API返回了 {len(data['data'])} 条数据")
            
            # 检查分页是否正确
            if "total" in data and "page" in data and "page_size" in data:
                logger.info(f"分页信息: 总记录数={data['total']}, 当前页={data['page']}, 每页记录数={data['page_size']}")
                return True
            else:
                logger.warning("分页信息不完整")
                return False
        else:
            logger.warning(f"排放数据API返回数据可能有问题: {data}")
            return False
    else:
        logger.error(f"排放数据API异常: HTTP {response.status_code}")
        return False

@run_test
def test_prediction_api_lstm():
    """测试使用LSTM模型的排放预测API"""
    logger.info("测试使用LSTM模型的排放预测API...")
    
    # 准备测试数据
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
    
    request_data = {
        "features": [features],
        "model_type": "lstm"
    }
    
    response = requests.post(f"{BASE_URL}/predict", json=request_data, timeout=15)
    
    if response.status_code == 200:
        data = response.json()
        if "predictions" in data and len(data["predictions"]) > 0:
            logger.info(f"LSTM模型预测结果: {data['predictions'][0]}")
            return True
        else:
            logger.warning(f"预测API返回数据可能有问题: {data}")
            return False
    else:
        logger.error(f"预测API异常: HTTP {response.status_code}")
        return False

@run_test
def test_prediction_api_gru():
    """测试使用GRU模型的排放预测API"""
    logger.info("测试使用GRU模型的排放预测API...")
    
    # 准备测试数据（与LSTM相同）
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
    
    request_data = {
        "features": [features],
        "model_type": "gru"
    }
    
    response = requests.post(f"{BASE_URL}/predict", json=request_data, timeout=15)
    
    if response.status_code == 200:
        data = response.json()
        if "predictions" in data and len(data["predictions"]) > 0:
            logger.info(f"GRU模型预测结果: {data['predictions'][0]}")
            return True
        else:
            logger.warning(f"预测API返回数据可能有问题: {data}")
            return False
    else:
        logger.error(f"预测API异常: HTTP {response.status_code}")
        return False

@run_test
def test_batch_prediction():
    """测试批量预测API"""
    logger.info("测试批量预测API...")
    
    # 准备多组测试数据
    base_features = [
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
    
    # 创建5组略有不同的特征数据
    features_list = []
    for i in range(5):
        features = base_features.copy()
        # 添加一些随机变化
        features[0] += random.uniform(-10, 10)  # 负荷变化
        features[1] += random.uniform(-5, 5)    # 风速变化
        features[2] += random.uniform(-10, 10)  # 湿度变化
        features_list.append(features)
    
    request_data = {
        "features": features_list,
        "model_type": "lstm"
    }
    
    response = requests.post(f"{BASE_URL}/predict", json=request_data, timeout=20)
    
    if response.status_code == 200:
        data = response.json()
        if "predictions" in data and len(data["predictions"]) == len(features_list):
            logger.info(f"批量预测成功，返回了 {len(data['predictions'])} 组预测结果")
            return True
        else:
            logger.warning(f"批量预测API返回数据可能有问题: {data}")
            return False
    else:
        logger.error(f"批量预测API异常: HTTP {response.status_code}")
        return False

@run_test
def test_invalid_prediction_input():
    """测试无效预测输入的处理"""
    logger.info("测试无效预测输入的处理...")
    
    # 准备无效测试数据（特征数量不足）
    features = [75.0, 25.0, 60.0]  # 只有3个特征
    
    request_data = {
        "features": [features],
        "model_type": "lstm"
    }
    
    response = requests.post(f"{BASE_URL}/predict", json=request_data, timeout=10)
    
    # 预期应该返回错误，而不是服务器崩溃
    if response.status_code == 400:
        logger.info(f"服务器正确拒绝了无效输入: {response.json()}")
        return True
    elif response.status_code == 200:
        logger.warning("服务器接受了无效输入，这可能是一个问题")
        return False
    else:
        logger.error(f"服务器返回了意外的状态码: {response.status_code}")
        return False

@run_test
def test_nlp_query_api():
    """测试自然语言查询API"""
    logger.info("测试自然语言查询API...")
    
    queries = [
        "最近24小时的SO2平均排放浓度是多少?",
        "1号机组昨天的NOx排放量趋势如何?",
        "本周烟尘排放最高的是哪一天?"
    ]
    
    all_passed = True
    
    for query in queries:
        logger.info(f"测试查询: '{query}'")
        
        request_data = {
            "query": query,
            "mode": "text"
        }
        
        response = requests.post(f"{BASE_URL}/api/nlp_query", json=request_data, timeout=20)
        
        if response.status_code == 200:
            data = response.json()
            if "answer" in data:
                logger.info(f"查询返回了答案: {data['answer']}")
                
                # 检查是否有数据可视化
                if "chart_data" in data:
                    logger.info("查询返回了图表数据")
                
                if "sql" in data:
                    logger.info(f"生成的SQL: {data['sql']}")
            else:
                logger.warning(f"查询API返回数据可能有问题: {data}")
                all_passed = False
        else:
            logger.error(f"查询API异常: HTTP {response.status_code}")
            all_passed = False
    
    return all_passed

@run_test
def test_frontend_pages():
    """测试前端页面是否可访问"""
    logger.info("测试前端页面是否可访问...")
    
    pages = [
        ("首页", "/"),
        ("数据看板", "/dashboard"),
        ("排放预测", "/prediction"),
        ("自然语言查询", "/nlp"),
        ("语音识别", "/speech")
    ]
    
    all_passed = True
    
    for name, path in pages:
        logger.info(f"测试页面: {name}")
        
        response = requests.get(f"{BASE_URL}{path}", timeout=10)
        
        if response.status_code == 200:
            logger.info(f"{name}页面可以访问")
            
            # 简单检查页面内容
            content = response.text.lower()
            if "html" in content and "<body" in content:
                logger.info(f"{name}页面内容看起来是有效的HTML")
            else:
                logger.warning(f"{name}页面内容可能不是有效的HTML")
                all_passed = False
        else:
            logger.error(f"{name}页面无法访问: HTTP {response.status_code}")
            all_passed = False
    
    return all_passed

def generate_test_report():
    """生成测试报告"""
    logger.info("正在生成测试报告...")
    
    # 计算通过率
    if test_results["total"] > 0:
        pass_rate = (test_results["passed"] / test_results["total"]) * 100
    else:
        pass_rate = 0
    
    # 生成测试报告
    report = {
        "测试时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "API基础URL": BASE_URL,
        "总测试数": test_results["total"],
        "通过测试数": test_results["passed"],
        "失败测试数": test_results["failed"],
        "跳过测试数": test_results["skipped"],
        "通过率": f"{pass_rate:.2f}%"
    }
    
    # 打印报告
    logger.info("====== 功能测试报告 ======")
    for key, value in report.items():
        logger.info(f"{key}: {value}")
    
    # 生成可视化报告
    try:
        result_data = {
            "类别": ["通过", "失败", "跳过"],
            "数量": [test_results["passed"], test_results["failed"], test_results["skipped"]]
        }
        df = pd.DataFrame(result_data)
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(df["类别"], df["数量"], color=["green", "red", "gray"])
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f"{height}", ha="center", va="bottom")
        
        plt.title("功能测试结果")
        plt.xlabel("测试结果")
        plt.ylabel("测试数量")
        plt.tight_layout()
        
        # 保存图表
        plt.savefig("功能测试结果.png")
        logger.info("测试结果图表已保存为 功能测试结果.png")
    except Exception as e:
        logger.warning(f"生成图表时出错: {str(e)}")
    
    # 保存JSON格式的报告
    try:
        with open("功能测试报告.json", "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=4)
        logger.info("测试报告已保存为 功能测试报告.json")
    except Exception as e:
        logger.warning(f"保存测试报告时出错: {str(e)}")
    
    return report

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="电厂污染物排放预测系统功能测试")
    parser.add_argument("--url", default="http://localhost:8000", help="API基础URL")
    args = parser.parse_args()
    
    global BASE_URL
    BASE_URL = args.url
    
    logger.info("====== 开始功能测试 ======")
    logger.info(f"API基础URL: {BASE_URL}")
    logger.info(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 运行所有测试
    test_health()
    test_emissions_api()
    test_emissions_api_with_filters()
    test_prediction_api_lstm()
    test_prediction_api_gru()
    test_batch_prediction()
    test_invalid_prediction_input()
    test_nlp_query_api()
    test_frontend_pages()
    
    # 生成测试报告
    report = generate_test_report()
    
    logger.info("====== 功能测试完成 ======")
    
    # 如果有测试失败，返回非零退出码
    return 0 if test_results["failed"] == 0 else 1

if __name__ == "__main__":
    sys.exit(main()) 
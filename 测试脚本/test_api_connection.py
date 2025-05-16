#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API连接测试脚本 - 测试API接口与前端页面的连接
"""

import requests
import json
import time
import os
from datetime import datetime

# API基础URL
BASE_URL = "http://localhost:8000"

def test_health():
    """测试健康检查接口"""
    print("\n测试健康检查接口...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            print(f"✅ 健康检查接口正常: {response.json()}")
            return True
        else:
            print(f"❌ 健康检查接口异常: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 健康检查接口异常: {str(e)}")
        return False

def test_emissions_api():
    """测试排放数据API"""
    print("\n测试排放数据API...")
    try:
        response = requests.get(f"{BASE_URL}/api/emissions")
        if response.status_code == 200:
            data = response.json()
            if "data" in data and len(data["data"]) > 0:
                print(f"✅ 排放数据API正常，返回了 {len(data['data'])} 条数据")
                return True
            else:
                print(f"⚠️ 排放数据API返回数据可能有问题: {data}")
                return False
        else:
            print(f"❌ 排放数据API异常: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 排放数据API异常: {str(e)}")
        return False

def test_prediction_api():
    """测试排放预测API"""
    print("\n测试排放预测API...")
    try:
        # 准备测试数据
        features = [75.0, 25.0, 60.0, 3.5, 120.0, datetime.now().hour, 
                  datetime.now().day, datetime.now().month, 
                  datetime.now().year, datetime.now().weekday(),
                  20.0, 40.0, 8.0]
        
        request_data = {
            "features": [features],
            "model_type": "lstm"
        }
        
        response = requests.post(f"{BASE_URL}/predict", json=request_data)
        if response.status_code == 200:
            data = response.json()
            if "predictions" in data and len(data["predictions"]) > 0:
                print(f"✅ 排放预测API正常，返回了预测结果: {data['predictions'][0]}")
                return True
            else:
                print(f"⚠️ 排放预测API返回数据可能有问题: {data}")
                return False
        else:
            print(f"❌ 排放预测API异常: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 排放预测API异常: {str(e)}")
        return False

def test_nlp_query_api():
    """测试自然语言查询API"""
    print("\n测试自然语言查询API...")
    try:
        request_data = {
            "query": "最近24小时的SO2平均排放浓度是多少?",
            "mode": "text"
        }
        
        response = requests.post(f"{BASE_URL}/api/nlp_query", json=request_data)
        if response.status_code == 200:
            data = response.json()
            if "answer" in data:
                print(f"✅ 自然语言查询API正常，返回了答案: {data['answer']}")
                return True
            else:
                print(f"⚠️ 自然语言查询API返回数据可能有问题: {data}")
                return False
        else:
            print(f"❌ 自然语言查询API异常: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 自然语言查询API异常: {str(e)}")
        return False

def test_frontend_pages():
    """测试前端页面是否可访问"""
    print("\n测试前端页面是否可访问...")
    pages = [
        ("首页", "/"),
        ("数据看板", "/dashboard"),
        ("排放预测", "/prediction"),
        ("自然语言查询", "/nlp"),
        ("语音识别", "/speech")
    ]
    
    success = True
    for name, path in pages:
        try:
            response = requests.get(f"{BASE_URL}{path}")
            if response.status_code == 200:
                print(f"✅ {name}页面可以访问")
            else:
                print(f"❌ {name}页面无法访问: HTTP {response.status_code}")
                success = False
        except Exception as e:
            print(f"❌ {name}页面异常: {str(e)}")
            success = False
    
    return success

def run_all_tests():
    """运行所有测试"""
    print("====== 开始测试API接口与前端页面的连接 ======")
    print(f"API基础URL: {BASE_URL}")
    print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*50)
    
    # 先测试健康检查，如果失败则可能是API服务未启动
    if not test_health():
        print("\n⚠️ 健康检查失败，请确保API服务已启动")
        return False
    
    # 测试其他API
    emissions_ok = test_emissions_api()
    prediction_ok = test_prediction_api()
    nlp_query_ok = test_nlp_query_api()
    
    # 测试前端页面
    frontend_ok = test_frontend_pages()
    
    # 总结结果
    print("\n====== 测试结果汇总 ======")
    print(f"健康检查API: {'✅ 正常' if True else '❌ 异常'}")
    print(f"排放数据API: {'✅ 正常' if emissions_ok else '❌ 异常'}")
    print(f"排放预测API: {'✅ 正常' if prediction_ok else '❌ 异常'}")
    print(f"自然语言查询API: {'✅ 正常' if nlp_query_ok else '❌ 异常'}")
    print(f"前端页面访问: {'✅ 正常' if frontend_ok else '❌ 异常'}")
    
    # 总体结果
    all_ok = emissions_ok and prediction_ok and nlp_query_ok and frontend_ok
    print("\n总体结果: " + ("✅ 所有测试通过" if all_ok else "❌ 部分测试失败"))
    print("="*50)
    
    return all_ok

if __name__ == "__main__":
    run_all_tests() 
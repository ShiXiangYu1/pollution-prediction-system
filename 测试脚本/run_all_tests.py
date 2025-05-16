#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
电厂污染物排放预测系统 - 综合测试执行脚本
"""

import os
import sys
import subprocess
import json
import time
import logging
import argparse
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import shutil

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("综合测试结果.log")
    ]
)

logger = logging.getLogger(__name__)

# 测试结果目录
RESULTS_DIR = "测试结果"

# 测试类型
TEST_TYPES = {
    "功能测试": {
        "script": "功能测试脚本.py",
        "description": "验证系统各功能模块是否正常工作",
        "result_files": ["功能测试结果.log", "功能测试报告.json"]
    },
    "性能测试": {
        "script": "性能测试脚本.py",
        "description": "评估系统在不同负载下的响应性能",
        "result_files": ["性能测试结果.log", "性能测试报告.json"]
    },
    "用户体验测试": {
        "script": "用户体验测试脚本.py",
        "description": "评估系统的用户交互体验",
        "result_files": ["用户体验测试结果.log", "用户体验测试报告.txt", "用户体验测试_导航时间.png", "用户体验测试_评分占比.png"]
    },
    "API连接测试": {
        "script": "test_api_connection.py",
        "description": "验证API接口与前端的连接",
        "result_files": []
    }
}

def create_results_directory():
    """创建测试结果目录，如果已存在，则清空"""
    if os.path.exists(RESULTS_DIR):
        # 清空目录
        for filename in os.listdir(RESULTS_DIR):
            file_path = os.path.join(RESULTS_DIR, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                logger.error(f"清空测试结果目录时出错: {e}")
    else:
        # 创建目录
        os.makedirs(RESULTS_DIR)
        
    # 为每种测试创建子目录
    for test_type in TEST_TYPES:
        os.makedirs(os.path.join(RESULTS_DIR, test_type), exist_ok=True)

def run_test(test_type, base_url="http://localhost:8000"):
    """运行指定类型的测试"""
    logger.info(f"开始执行 {test_type}...")
    
    test_info = TEST_TYPES[test_type]
    script = test_info["script"]
    
    if not os.path.exists(os.path.join("测试脚本", script)):
        logger.error(f"测试脚本不存在: {script}")
        return False, 0
    
    # 构建命令
    cmd = [sys.executable, os.path.join("测试脚本", script)]
    
    # 添加URL参数
    if "--url" in subprocess.check_output([sys.executable, os.path.join("测试脚本", script), "--help"]).decode():
        cmd.extend(["--url", base_url])
    
    # 运行测试脚本
    start_time = time.time()
    try:
        logger.info(f"执行命令: {' '.join(cmd)}")
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate()
        
        # 记录输出
        with open(os.path.join(RESULTS_DIR, test_type, "stdout.log"), "w", encoding="utf-8") as f:
            f.write(stdout)
        
        if stderr:
            with open(os.path.join(RESULTS_DIR, test_type, "stderr.log"), "w", encoding="utf-8") as f:
                f.write(stderr)
        
        elapsed_time = time.time() - start_time
        success = process.returncode == 0
        
        # 复制测试结果文件到结果目录
        for result_file in test_info["result_files"]:
            if os.path.exists(result_file):
                shutil.copy2(result_file, os.path.join(RESULTS_DIR, test_type))
        
        if success:
            logger.info(f"{test_type} 执行成功，耗时: {elapsed_time:.2f}秒")
        else:
            logger.error(f"{test_type} 执行失败，耗时: {elapsed_time:.2f}秒")
        
        return success, elapsed_time
    
    except Exception as e:
        elapsed_time = time.time() - start_time
        logger.error(f"{test_type} 执行出错: {str(e)}")
        return False, elapsed_time

def parse_test_results():
    """解析各种测试的结果文件，提取关键指标"""
    results = {}
    
    # 解析功能测试结果
    try:
        func_report_path = os.path.join(RESULTS_DIR, "功能测试", "功能测试报告.json")
        if os.path.exists(func_report_path):
            with open(func_report_path, "r", encoding="utf-8") as f:
                func_data = json.load(f)
            
            results["功能测试"] = {
                "通过率": f"{func_data.get('通过率', 0):.1f}%",
                "总测试数": func_data.get("总测试数", 0),
                "通过测试数": func_data.get("通过测试数", 0),
                "失败测试数": func_data.get("失败测试数", 0)
            }
        else:
            logger.warning("功能测试报告文件不存在")
    except Exception as e:
        logger.error(f"解析功能测试结果出错: {str(e)}")
    
    # 解析性能测试结果
    try:
        perf_report_path = os.path.join(RESULTS_DIR, "性能测试", "性能测试报告.json")
        if os.path.exists(perf_report_path):
            with open(perf_report_path, "r", encoding="utf-8") as f:
                perf_data = json.load(f)
            
            results["性能测试"] = {
                "平均响应时间": f"{perf_data.get('平均响应时间', 0):.2f}ms",
                "最大响应时间": f"{perf_data.get('最大响应时间', 0):.2f}ms",
                "最大每秒请求数": perf_data.get("最大每秒请求数", 0),
                "CPU使用率": f"{perf_data.get('CPU使用率', 0):.1f}%",
                "内存使用率": f"{perf_data.get('内存使用率', 0):.1f}%"
            }
        else:
            logger.warning("性能测试报告文件不存在")
    except Exception as e:
        logger.error(f"解析性能测试结果出错: {str(e)}")
    
    # 解析用户体验测试结果
    try:
        ux_report_path = os.path.join(RESULTS_DIR, "用户体验测试", "用户体验测试报告.txt")
        if os.path.exists(ux_report_path):
            with open(ux_report_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            # 提取最终评分
            import re
            score_match = re.search(r"总体评分: (\d+)/(\d+)分 \((.+)\)", content)
            grade_match = re.search(r"用户体验等级: (.+)", content)
            
            if score_match and grade_match:
                results["用户体验测试"] = {
                    "总评分": f"{score_match.group(1)}/{score_match.group(2)}分",
                    "评分百分比": score_match.group(3),
                    "评级": grade_match.group(1)
                }
            else:
                logger.warning("未能从用户体验测试报告中提取评分信息")
        else:
            logger.warning("用户体验测试报告文件不存在")
    except Exception as e:
        logger.error(f"解析用户体验测试结果出错: {str(e)}")
    
    return results

def generate_summary_report(test_results, execution_results):
    """生成测试摘要报告"""
    # 创建报告
    report = f"""
========================================================
            电厂污染物排放预测系统 综合测试报告
========================================================
测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

测试执行摘要:
"""
    
    # 添加执行结果
    total_time = 0
    all_success = True
    for test_type, result in execution_results.items():
        success, time_taken = result
        total_time += time_taken
        all_success = all_success and success
        status = "✅ 成功" if success else "❌ 失败"
        report += f"- {test_type}: {status}, 耗时: {time_taken:.2f}秒\n"
    
    report += f"\n总测试耗时: {total_time:.2f}秒\n"
    report += f"总体测试状态: {'✅ 全部通过' if all_success else '❌ 部分测试失败'}\n\n"
    
    # 添加各测试类型的结果
    report += "各测试类型结果:\n"
    
    # 功能测试结果
    if "功能测试" in test_results:
        report += "\n--- 功能测试 ---\n"
        for key, value in test_results["功能测试"].items():
            report += f"{key}: {value}\n"
    
    # 性能测试结果
    if "性能测试" in test_results:
        report += "\n--- 性能测试 ---\n"
        for key, value in test_results["性能测试"].items():
            report += f"{key}: {value}\n"
    
    # 用户体验测试结果
    if "用户体验测试" in test_results:
        report += "\n--- 用户体验测试 ---\n"
        for key, value in test_results["用户体验测试"].items():
            report += f"{key}: {value}\n"
    
    # 添加结束信息
    report += "\n========================================================\n"
    report += "测试详细记录可查看测试结果目录中的各测试报告\n"
    report += "========================================================\n"
    
    # 保存报告
    with open(os.path.join(RESULTS_DIR, "综合测试报告.txt"), "w", encoding="utf-8") as f:
        f.write(report)
    
    logger.info(f"综合测试报告已生成: {os.path.join(RESULTS_DIR, '综合测试报告.txt')}")
    
    # 生成图表展示测试结果
    try:
        # 创建测试结果概览图
        plt.figure(figsize=(10, 6))
        
        # 准备数据
        test_types = list(execution_results.keys())
        execution_times = [result[1] for result in execution_results.values()]
        success_status = [result[0] for result in execution_results.values()]
        
        # 创建柱状图
        bars = plt.bar(test_types, execution_times, color=['green' if status else 'red' for status in success_status])
        
        # 添加数据标签
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                     f"{height:.1f}s", ha='center', va='bottom')
        
        plt.title('各测试类型执行时间对比')
        plt.xlabel('测试类型')
        plt.ylabel('执行时间 (秒)')
        plt.tight_layout()
        
        # 保存图表
        plt.savefig(os.path.join(RESULTS_DIR, "测试时间对比.png"))
        
        # 如果有功能测试结果，创建通过率饼图
        if "功能测试" in test_results:
            plt.figure(figsize=(8, 8))
            
            # 提取数据
            passed = test_results["功能测试"]["通过测试数"]
            failed = test_results["功能测试"]["失败测试数"]
            
            # 创建饼图
            labels = ['通过', '失败']
            sizes = [passed, failed]
            colors = ['#4CAF50', '#F44336']
            explode = (0.1, 0)  # 突出显示通过的部分
            
            plt.pie(sizes, explode=explode, labels=labels, colors=colors,
                    autopct='%1.1f%%', shadow=True, startangle=90)
            plt.axis('equal')  # 保证饼图是圆形
            
            plt.title('功能测试通过率')
            plt.tight_layout()
            
            # 保存图表
            plt.savefig(os.path.join(RESULTS_DIR, "功能测试通过率.png"))
    
    except Exception as e:
        logger.error(f"生成图表失败: {str(e)}")
    
    return report

def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="电厂污染物排放预测系统 - 综合测试执行脚本")
    parser.add_argument("--url", help="API基础URL", default="http://localhost:8000")
    parser.add_argument("--skip", help="跳过的测试类型，多个类型用逗号分隔", default="")
    args = parser.parse_args()
    
    base_url = args.url
    skip_tests = args.skip.split(",") if args.skip else []
    
    logger.info(f"开始执行综合测试，目标系统: {base_url}")
    
    # 创建结果目录
    create_results_directory()
    
    # 执行各类测试
    execution_results = {}
    
    for test_type in TEST_TYPES:
        if test_type in skip_tests:
            logger.info(f"跳过 {test_type}")
            continue
        
        success, time_taken = run_test(test_type, base_url)
        execution_results[test_type] = (success, time_taken)
    
    # 解析测试结果
    test_results = parse_test_results()
    
    # 生成摘要报告
    report = generate_summary_report(test_results, execution_results)
    
    # 输出报告
    print("\n" + report)

if __name__ == "__main__":
    main() 
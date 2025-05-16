#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NL2SQL转换模型模块 - 实现自然语言到SQL的转换
"""

import os
import torch
import numpy as np
import pandas as pd
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    BitsAndBytesConfig
)
from peft import PeftModel
import json
import re
from tqdm import tqdm
import time
from datetime import datetime
import sqlite3

try:
    from .mock_models import MockNL2SQLModel, MockTokenizer as MockNL2SQLTokenizer
except ImportError:
    # 这些是备用的内联模拟实现
    class MockNL2SQLModel:
        def to(self, device): return self
        def eval(self): return self
        def generate(self, **kwargs): return [torch.tensor([1, 2, 3, 4, 5])]
    
    class MockNL2SQLTokenizer:
        def __init__(self): self.eos_token_id = 0
        def __call__(self, text, **kwargs): 
            return {"input_ids": torch.tensor([[1, 2, 3]]), "attention_mask": torch.tensor([[1, 1, 1]])}
        def decode(self, tokens, **kwargs): return "SELECT * FROM emissions LIMIT 10"

class NL2SQLConverter:
    """
    自然语言到SQL转换类
    """
    
    def __init__(self, model_path=None, data_dir='./data'):
        """
        初始化自然语言到SQL转换类
        
        参数:
            model_path: 模型路径，默认为None（使用基础模型）
            data_dir: 数据目录
        """
        self.data_dir = data_dir
        self.model_path = model_path
        
        # 创建模型目录（如果不存在）
        os.makedirs(os.path.join(data_dir, 'models', 'nl2sql'), exist_ok=True)
        
        # 检查是否有GPU可用
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")
        
        # 加载模型和分词器
        self.model, self.tokenizer = self.load_model()
    
    def load_model(self):
        """
        加载模型
        
        返回:
            model: 模型
            tokenizer: 分词器
        """
        print("加载NL2SQL模型...")
        
        # 修改为默认不使用离线模式
        offline_mode = False
        
        # 如果提供了模型路径，则加载微调后的模型
        if self.model_path and os.path.exists(self.model_path):
            print(f"加载微调后的模型: {self.model_path}")
            
            try:
                # 配置量化参数
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                
                # 加载基础模型
                base_model_name = "deepseek-ai/deepseek-llm-7b-base"
                base_model = AutoModelForCausalLM.from_pretrained(
                    base_model_name,
                    quantization_config=quantization_config,
                    device_map="auto",
                    trust_remote_code=True
                )
                
                # 加载LoRA权重
                model = PeftModel.from_pretrained(base_model, self.model_path)
                
                # 加载分词器
                tokenizer = AutoTokenizer.from_pretrained(self.model_path)
                return model, tokenizer
            except Exception as e:
                print(f"加载微调模型失败: {e}")
                print("尝试加载基础模型...")
                offline_mode = True
        
        # 尝试加载基础模型
        if not offline_mode:
            try:
                # 使用更小的模型以便于实际运行
                model_name = "gpt2"  # 替换为较小的模型，方便测试
                
                # 加载模型和分词器
                model = AutoModelForCausalLM.from_pretrained(model_name)
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                
                if tokenizer.pad_token_id is None:
                    # 为没有pad_token的分词器设置pad_token
                    tokenizer.pad_token = tokenizer.eos_token
                    model.config.pad_token_id = tokenizer.eos_token_id
                
                print(f"成功加载基础模型: {model_name}")
                return model, tokenizer
            except Exception as e:
                print(f"加载基础模型失败: {e}")
                print("回退到离线模式...")
                offline_mode = True
        
        # 如果处于离线模式，创建简单的模拟实现
        if offline_mode:
            print("使用离线模式的模拟实现")
            # 创建一个简单的模拟模型和分词器
            model = MockNL2SQLModel()
            tokenizer = MockNL2SQLTokenizer()
        
        print("NL2SQL模型加载完成")
        return model, tokenizer
    
    def convert_nl_to_sql(self, nl_query, schema_info, max_length=512, temperature=0.3):
        """
        将自然语言查询转换为SQL查询
        
        参数:
            nl_query: 自然语言查询
            schema_info: 数据库模式信息
            max_length: 最大生成长度
            temperature: 温度参数
            
        返回:
            str: SQL查询
        """
        # 构建提示
        prompt = f"根据以下数据库模式，将自然语言查询转换为SQL查询。\n\n数据库模式:\n{schema_info}\n\n自然语言查询: {nl_query}\n\nSQL查询:"
        
        # 编码提示文本
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        # 检查是否为离线模式的模拟返回值（字典）
        if isinstance(inputs, dict) and not hasattr(inputs, 'to'):
            # 手动将字典中的tensor移动到设备
            for key in inputs:
                if isinstance(inputs[key], torch.Tensor):
                    inputs[key] = inputs[key].to(self.device)
        else:
            # 如果是正常的BatchEncoding对象，则直接使用to方法
            inputs = inputs.to(self.device)
        
        # 生成SQL查询
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                top_p=0.9,
                top_k=50,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # 解码输出
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 提取SQL查询部分
        sql_query = generated_text.split("SQL查询:", 1)[-1].strip()
        
        return sql_query
    
    def extract_sql_from_response(self, response):
        """
        从响应中提取SQL查询
        
        参数:
            response: 模型响应
            
        返回:
            str: SQL查询
        """
        # 尝试使用正则表达式提取SQL查询
        sql_pattern = r"```sql\s*(.*?)\s*```"
        matches = re.findall(sql_pattern, response, re.DOTALL)
        
        if matches:
            return matches[0].strip()
        
        # 如果没有找到SQL代码块，尝试查找以SELECT开头的语句
        select_pattern = r"(SELECT\s+.*?;)"
        matches = re.findall(select_pattern, response, re.DOTALL | re.IGNORECASE)
        
        if matches:
            return matches[0].strip()
        
        # 如果仍然没有找到，返回整个响应
        return response.strip()
    
    def validate_sql(self, sql_query, db_path=None):
        """
        验证SQL查询
        
        参数:
            sql_query: SQL查询
            db_path: 数据库路径
            
        返回:
            bool: 是否有效
            str: 错误信息（如果有）
        """
        # 如果没有提供数据库路径，则使用内存数据库
        if db_path is None:
            conn = sqlite3.connect(":memory:")
        else:
            conn = sqlite3.connect(db_path)
        
        cursor = conn.cursor()
        
        try:
            # 尝试解析SQL查询
            cursor.execute(f"EXPLAIN {sql_query}")
            conn.close()
            return True, ""
        except sqlite3.Error as e:
            conn.close()
            return False, str(e)
    
    def fix_sql_errors(self, sql_query, error_message, schema_info):
        """
        修复SQL查询中的错误
        
        参数:
            sql_query: SQL查询
            error_message: 错误信息
            schema_info: 数据库模式信息
            
        返回:
            str: 修复后的SQL查询
        """
        # 构建提示
        prompt = f"""
        以下SQL查询存在错误，请修复它。

        数据库模式:
        {schema_info}

        SQL查询:
        {sql_query}

        错误信息:
        {error_message}

        修复后的SQL查询:
        """
        
        # 编码提示文本
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        # 检查是否为离线模式的模拟返回值（字典）
        if isinstance(inputs, dict) and not hasattr(inputs, 'to'):
            # 手动将字典中的tensor移动到设备
            for key in inputs:
                if isinstance(inputs[key], torch.Tensor):
                    inputs[key] = inputs[key].to(self.device)
        else:
            # 如果是正常的BatchEncoding对象，则直接使用to方法
            inputs = inputs.to(self.device)
        
        # 生成修复后的SQL查询
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=512,
                temperature=0.3,
                top_p=0.9,
                top_k=50,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # 解码输出
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 提取修复后的SQL查询部分
        fixed_sql = generated_text.split("修复后的SQL查询:", 1)[-1].strip()
        
        return fixed_sql
    
    def evaluate_on_test_set(self, test_data, schema_info, output_file=None):
        """
        在测试集上评估模型
        
        参数:
            test_data: 测试数据列表
            schema_info: 数据库模式信息
            output_file: 输出文件路径
            
        返回:
            dict: 评估结果
        """
        print("在测试集上评估NL2SQL模型...")
        
        results = []
        
        for i, item in enumerate(tqdm(test_data)):
            nl_query = item.get('natural_language', '')
            reference_sql = item.get('sql_query', '')
            
            # 转换自然语言查询为SQL查询
            predicted_sql = self.convert_nl_to_sql(nl_query, schema_info)
            
            # 验证SQL查询
            is_valid, error_message = self.validate_sql(predicted_sql)
            
            # 如果SQL查询无效，尝试修复
            if not is_valid:
                fixed_sql = self.fix_sql_errors(predicted_sql, error_message, schema_info)
                is_valid, error_message = self.validate_sql(fixed_sql)
                
                if is_valid:
                    predicted_sql = fixed_sql
            
            # 添加到结果
            results.append({
                'natural_language': nl_query,
                'reference_sql': reference_sql,
                'predicted_sql': predicted_sql,
                'is_valid': is_valid,
                'error_message': error_message if not is_valid else ""
            })
        
        # 计算有效SQL查询的比例
        valid_count = sum(1 for result in results if result['is_valid'])
        valid_ratio = valid_count / len(results) if results else 0
        
        print(f"评估完成，有效SQL查询比例: {valid_ratio:.2f}")
        
        # 保存评估结果
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'results': results,
                    'metrics': {
                        'valid_ratio': valid_ratio
                    }
                }, f, ensure_ascii=False, indent=2)
            print(f"评估结果已保存到: {output_file}")
        
        return {
            'results': results,
            'metrics': {
                'valid_ratio': valid_ratio
            }
        }
    
    def create_nl2sql_pipeline(self, schema_info):
        """
        创建NL2SQL流水线
        
        参数:
            schema_info: 数据库模式信息
            
        返回:
            function: NL2SQL转换函数
        """
        def nl2sql_pipeline(nl_query):
            # 转换自然语言查询为SQL查询
            sql_query = self.convert_nl_to_sql(nl_query, schema_info)
            
            # 验证SQL查询
            is_valid, error_message = self.validate_sql(sql_query)
            
            # 如果SQL查询无效，尝试修复
            if not is_valid:
                fixed_sql = self.fix_sql_errors(sql_query, error_message, schema_info)
                is_valid, error_message = self.validate_sql(fixed_sql)
                
                if is_valid:
                    sql_query = fixed_sql
            
            return {
                'natural_language': nl_query,
                'sql_query': sql_query,
                'is_valid': is_valid,
                'error_message': error_message if not is_valid else ""
            }
        
        return nl2sql_pipeline
    
    def execute_sql(self, sql_query, db_path):
        """
        执行SQL查询
        
        参数:
            sql_query: SQL查询
            db_path: 数据库路径
            
        返回:
            list: 查询结果
            list: 列名
        """
        # 连接数据库
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        try:
            # 执行SQL查询
            cursor.execute(sql_query)
            
            # 获取列名
            column_names = [description[0] for description in cursor.description]
            
            # 获取结果
            rows = cursor.fetchall()
            
            # 转换为字典列表
            results = []
            for row in rows:
                results.append(dict(row))
            
            conn.close()
            return results, column_names
        except sqlite3.Error as e:
            conn.close()
            return [], []

# 测试代码
if __name__ == "__main__":
    # 确保测试数据库存在
    db_path = './data/test_pollution_db.sqlite'
    
    if not os.path.exists(db_path):
        print(f"测试数据库不存在: {db_path}")
        print("创建测试数据库...")
        try:
            from create_test_db import main as create_db
            create_db()
        except Exception as e:
            print(f"创建测试数据库失败: {e}")
            print("请先运行 python app_code/model_training/create_test_db.py 创建测试数据库")
    
    # 创建NL2SQL转换器
    converter = NL2SQLConverter()
    
    # 设置输入和输出目录
    data_dir = "./data"
    models_dir = "./data/models/nl2sql"
    
    # 定义数据库模式信息
    schema_info = """
    地区表(TS_AREA): 地区编号(AREA_CODE), 地区名称(AREA_NAME), 父地区编号(PARENT_CODE)
    电厂表(TB_FACTORY): 电厂编号(FAC_ID), 地区编号(AREA_CODE), 电厂名称(FAC_NAME), 归属集团(GROUP_ID), 地址(FAC_ADDR), 电厂坐标(X_MAP, Y_MAP), 电厂简称(FAC_ALIAS), 电厂是否启用(ACTIVE_FLAG)
    机组表(TB_STEAMER): 机组编号(STEAMER_ID), 机组名称(STEAMER_NAME), 机组所属电厂编号(FAC_ID), 机组负荷(RATING_FH), 机组是否启用(ACTIVE_FLAG), 机组类型(STEAMER_TYPE)
    电厂测点信息表(tb_rtu_channel): 测点所属电厂编号(FAC_ID), 测点编号(CHANNEL_NUM), 测点名称(CHANNEL_NAME), 测点工程值KKS(KKS_WORK), 测点是否启用(ACTIVE_FLAG)
    测点采集数据表(td_hisdata): 工程值KKS(KKS_WORK), 时间(RECTIME), 采集值(VALUE)
    二氧化硫小时数据表(td_dtl_2025): 时间(RECTIME), 所属机组编号(STEAMER_ID), SO2二氧化硫排放浓度(VENT_SO2_CHK), SO2二氧化硫排放量(VENT_SO2_T), 脱硫效率(FGD_EFCY), 发电量(FD_KWH), 考核时间(CHK_TIME), 停运时间(STOP_TIME), 停运原因(STOP_CAUSE), SO2二氧化硫超标倍数(OVER_MULTIPLE), SO2二氧化硫超标时长(TIME_2TO30, TIME_30TO50)
    氮氧化物小时数据表(td_dtx_2025): 时间(RECTIME), 所属机组编号(STEAMER_ID), NOx氮氧化物排放浓度(VENT_NOX_CHK), NOx氮氧化物排放量(VENT_NOX_T), 脱硝效率(SCR_EFCY), 发电量(FD_KWH), 考核时间(CHK_TIME), 停运时间(STOP_TIME), 停运原因(STOP_CAUSE), NOx氮氧化物超标倍数(OVER_MULTIPLE), NOx氮氧化物超标时长(TIME_2TO30, TIME_30TO50)
    烟尘小时数据表(td_dcc_2025): 时间(RECTIME), 所属机组编号(STEAMER_ID), 烟尘排放浓度(VENT_SOOT_CHK), 烟尘排放量(VENT_SOOT_T), 发电量(FD_KWH), 考核时间(CHK_TIME), 停运时间(STOP_TIME), 停运原因(STOP_CAUSE), 烟尘超标倍数(OVER_MULTIPLE), 烟尘超标时长(TIME_2TO30, TIME_30TO50)
    """
    
    # 测试一些常见的自然语言查询
    test_queries = [
        "当前哪些机组实时污染物排放超标？",
        "过去24小时有哪些机组发生污染物小时浓度超标？",
        "江苏省燃煤机组SO2排放情况统计",
        "南京电厂的机组列表"
    ]
    
    # 测试每个查询
    for nl_query in test_queries:
        print("\n" + "="*80)
        print(f"自然语言查询: {nl_query}")
        
        # 转换为SQL查询
        sql_query = converter.convert_nl_to_sql(nl_query, schema_info)
        print(f"生成的SQL查询: \n{sql_query}")
        
        # 验证SQL查询
        is_valid, error_message = converter.validate_sql(sql_query, db_path)
        print(f"SQL查询有效: {is_valid}")
        
        if not is_valid:
            print(f"错误信息: {error_message}")
            
            # 尝试修复SQL查询
            fixed_sql = converter.fix_sql_errors(sql_query, error_message, schema_info)
            print(f"修复后的SQL查询: \n{fixed_sql}")
            
            # 验证修复后的SQL查询
            is_valid, error_message = converter.validate_sql(fixed_sql, db_path)
            print(f"修复后的SQL查询有效: {is_valid}")
            
            if is_valid:
                sql_query = fixed_sql
            else:
                print(f"错误信息: {error_message}")
        
        # 如果SQL查询有效，尝试执行
        if is_valid:
            print("执行SQL查询...")
            try:
                results, column_names = converter.execute_sql(sql_query, db_path)
                
                if results:
                    # 打印前5行结果
                    print(f"查询结果预览 (共{len(results)}行):")
                    print("列名:", column_names)
                    for i, row in enumerate(results[:5]):
                        print(f"行 {i+1}:", dict(row))
                    
                    if len(results) > 5:
                        print(f"... 共{len(results)}行 ...")
                else:
                    print("查询没有返回结果")
            except Exception as e:
                print(f"执行查询时出错: {e}")
    
    print("\n所有测试查询完成")

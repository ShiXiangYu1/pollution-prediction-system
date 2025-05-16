#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
模拟模型类 - 用于离线模式
"""

import torch
import numpy as np

class MockPollutionModel:
    """
    模拟污染物排放预测模型，用于离线模式
    """
    
    def __init__(self):
        """初始化模拟模型"""
        pass
    
    def to(self, device):
        """模拟设备转移"""
        return self
    
    def eval(self):
        """模拟评估模式"""
        return self
    
    def __call__(self, inputs):
        """
        模拟预测
        
        参数:
            inputs: 输入特征
            
        返回:
            torch.Tensor: 模拟预测结果
        """
        # 获取输入的形状
        if isinstance(inputs, dict):
            # 对于传入的词典类型（例如来自Tokenizer的输出）
            batch_size = 1
        elif isinstance(inputs, torch.Tensor):
            # 对于张量类型
            batch_size = inputs.shape[0]
        elif isinstance(inputs, np.ndarray):
            # 对于numpy数组
            batch_size = inputs.shape[0]
        else:
            # 其他类型，默认批次大小为1
            batch_size = 1
        
        # 随机生成预测值，模拟三个污染物的排放量
        # SO2: 10-30 mg/m³
        # NOx: 20-50 mg/m³
        # 烟尘: 5-15 mg/m³
        mock_predictions = torch.tensor(
            np.random.uniform(
                low=[10.0, 20.0, 5.0],
                high=[30.0, 50.0, 15.0],
                size=(batch_size, 3)
            ),
            dtype=torch.float32
        )
        
        return mock_predictions

class MockNL2SQLModel:
    """离线模式下的模拟NL2SQL模型"""
    
    def to(self, device):
        return self
    
    def eval(self):
        return self
    
    def generate(self, **kwargs):
        # 返回一个简单的模拟输出
        return [torch.tensor([1, 2, 3, 4, 5])]

class MockTokenizer:
    """离线模式下的模拟分词器"""
    
    def __init__(self):
        self.eos_token_id = 0
    
    def __call__(self, text, **kwargs):
        # 返回一个简单的模拟输入
        return {"input_ids": torch.tensor([[1, 2, 3]]), "attention_mask": torch.tensor([[1, 1, 1]])}
    
    def decode(self, tokens, **kwargs):
        # 为不同的查询返回预定义的SQL
        common_queries = {
            "昨天1号机组的NOx排放量是多少": "SELECT AVG(NOx) FROM emissions WHERE unit_id = 1 AND timestamp >= date('now', '-1 day') AND timestamp < date('now')",
            "最近一周2号机组的SO2排放趋势如何": "SELECT DATE(timestamp) as date, AVG(SO2) as avg_so2 FROM emissions WHERE unit_id = 2 AND timestamp >= date('now', '-7 days') GROUP BY DATE(timestamp) ORDER BY date",
            "哪个机组的烟尘排放最高": "SELECT unit_id, MAX(dust) as max_dust FROM emissions GROUP BY unit_id ORDER BY max_dust DESC LIMIT 1",
            "今天上午所有机组的平均污染物排放水平是多少": "SELECT AVG(SO2) as avg_so2, AVG(NOx) as avg_nox, AVG(dust) as avg_dust FROM emissions WHERE timestamp >= datetime('now', 'start of day') AND timestamp < datetime('now', 'start of day', '+12 hours')"
        }
        
        # 默认返回一个简单的查询
        return "SQL查询: " + common_queries.get("哪个机组的烟尘排放最高", "SELECT * FROM emissions LIMIT 10") 
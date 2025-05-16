#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
污染物排放预测模型优化器 - 提供模型参数调优、特征工程和模型更新机制
"""

import os
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from datetime import datetime
import math
from sklearn.model_selection import KFold
import joblib
from tqdm import tqdm

from model_training.pollution_prediction import LSTMModel, GRUModel, TransformerModel

class PositionalEncoding(nn.Module):
    """位置编码模块，用于Transformer模型"""
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class EnhancedTransformerModel(nn.Module):
    """增强版Transformer模型"""
    def __init__(self, input_dim, d_model, nhead, num_layers, dim_feedforward, output_dim, dropout=0.1):
        super(EnhancedTransformerModel, self).__init__()
        self.input_linear = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout, 
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # 添加额外的全连接层
        self.fc1 = nn.Linear(d_model, d_model//2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(d_model//2, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.input_linear(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        # 使用序列中的所有时间步而不仅是最后一个
        x = x.mean(dim=1)  # 全局平均池化
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class ModelOptimizer:
    """模型优化器类"""
    def __init__(self, base_model, data_dir='./data'):
        """
        初始化模型优化器
        
        参数:
            base_model: 基础模型实例
            data_dir: 数据目录
        """
        self.base_model = base_model
        self.data_dir = data_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 创建输出目录
        os.makedirs(os.path.join(data_dir, 'optimized_models'), exist_ok=True)

    #----------------------------------------
    # 特征工程优化
    #----------------------------------------
    def enhanced_time_features(self, df):
        """增强的时间特征提取"""
        # 确保时间戳列存在
        if 'timestamp' not in df.columns:
            print("警告: 数据中缺少timestamp列，跳过时间特征生成")
            return df
        
        print("生成增强时间特征...")
        
        # 转换时间戳为datetime类型
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # 基本时间特征
        df['hour'] = df['timestamp'].dt.hour
        df['day'] = df['timestamp'].dt.day
        df['month'] = df['timestamp'].dt.month
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
        
        # 周期性时间特征编码
        df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24)
        df['day_sin'] = np.sin(2 * np.pi * df['day']/31)
        df['day_cos'] = np.cos(2 * np.pi * df['day']/31)
        df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
        df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week']/7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week']/7)
        
        # 季节特征
        df['season'] = df['month'].apply(lambda x: 1 if 3 <= x <= 5 else 2 if 6 <= x <= 8 else 3 if 9 <= x <= 11 else 4)
        
        return df
    
    def environment_feature_interactions(self, df):
        """环境特征之间的交互项"""
        print("生成环境特征交互项...")
        
        # 温度与湿度交互
        if 'temperature' in df.columns and 'humidity' in df.columns:
            df['temp_humid_interaction'] = df['temperature'] * df['humidity']
            
            # 温度湿度组合特征 - 不适感指数(DI)
            df['discomfort_index'] = 0.81 * df['temperature'] + 0.01 * df['humidity'] * (0.99 * df['temperature'] - 14.3) + 46.3
        
        # 温度与风速交互 - 风寒指数
        if 'temperature' in df.columns and 'wind_speed' in df.columns:
            df['wind_chill'] = 13.12 + 0.6215 * df['temperature'] - 11.37 * df['wind_speed']**0.16 + 0.3965 * df['temperature'] * df['wind_speed']**0.16
        
        # 气压变化率
        if 'pressure' in df.columns:
            df['pressure_change'] = df['pressure'].diff()
            # 填充NaN
            df['pressure_change'] = df['pressure_change'].fillna(0)
        
        return df
    
    def load_emission_features(self, df):
        """发电负荷与排放关系特征"""
        print("生成负荷-排放关系特征...")
        
        if 'load' in df.columns:
            # 负荷梯度 - 表示负荷变化率
            df['load_gradient'] = df['load'].diff()
            df['load_gradient'] = df['load_gradient'].fillna(0)
            
            # 负荷高低状态 - 分档位
            df['load_level'] = pd.qcut(df['load'].rank(method='first'), q=4, labels=[0, 1, 2, 3])
            
            # 负荷二次项 - 捕捉非线性关系
            df['load_squared'] = df['load'] ** 2
            
            # 负荷指数平滑
            alpha = 0.3
            df['load_ema'] = df['load'].ewm(alpha=alpha).mean()
        
        return df
    
    def historical_emission_features(self, df, target_cols, lookback_windows=[1, 3, 6, 12, 24, 48]):
        """增强的历史排放特征"""
        print("生成历史排放特征...")
        
        for col in target_cols:
            # 历史排放滑动窗口特征
            for window in lookback_windows:
                # 平均值
                df[f'{col}_avg_{window}h'] = df[col].rolling(window=window).mean().shift(1)
                # 最大值
                df[f'{col}_max_{window}h'] = df[col].rolling(window=window).max().shift(1)
                # 最小值
                df[f'{col}_min_{window}h'] = df[col].rolling(window=window).min().shift(1)
                # 标准差 - 表示波动性
                df[f'{col}_std_{window}h'] = df[col].rolling(window=window).std().shift(1)
                
            # 24小时同期排放 - 捕捉日周期性
            df[f'{col}_24h_ago'] = df[col].shift(24)
            
            # 排放变化趋势
            df[f'{col}_trend_1h'] = df[col].diff()
            df[f'{col}_trend_3h'] = df[col] - df[col].shift(3)
            
            # 超标状态特征
            if f'{col}_limit' in df.columns:
                df[f'{col}_over_limit'] = (df[col] > df[f'{col}_limit']).astype(int)
                # 历史超标次数
                df[f'{col}_over_limit_24h'] = df[f'{col}_over_limit'].rolling(window=24).sum().shift(1)
        
        # 填充缺失值
        df = df.fillna(method='bfill').fillna(method='ffill')
        return df
    
    def apply_all_feature_engineering(self, df, target_cols):
        """应用所有特征工程方法"""
        print("开始应用所有特征工程优化...")
        
        # 应用时间特征
        df = self.enhanced_time_features(df)
        
        # 应用环境特征交互
        df = self.environment_feature_interactions(df)
        
        # 应用负荷特征
        df = self.load_emission_features(df)
        
        # 应用历史排放特征
        df = self.historical_emission_features(df, target_cols)
        
        print(f"特征工程完成，特征数量: {df.shape[1]}")
        return df
    
    #----------------------------------------
    # 参数调优
    #----------------------------------------
    def _train_quick_eval(self, model, train_loader, val_loader, criterion, optimizer, epochs=5):
        """模型快速训练和评估"""
        model.to(self.device)
        
        for epoch in range(epochs):
            # 训练模式
            model.train()
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # 前向传播
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                # 反向传播和优化
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        # 评估模式
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = model(inputs)
                val_loss += criterion(outputs, targets).item()
        
        val_loss /= len(val_loader)
        return val_loss
    
    def hyper_parameter_tuning_lstm(self, train_data, input_dim, output_dim):
        """LSTM模型超参数调优"""
        print("开始LSTM模型超参数调优...")
        
        # 参数搜索范围
        param_grid = {
            'hidden_dim': [64, 128, 256],
            'num_layers': [2, 3],
            'dropout': [0.1, 0.2, 0.3],
            'learning_rate': [0.0001, 0.0005, 0.001],
            'batch_size': [16, 32, 64]
        }
        
        best_val_loss = float('inf')
        best_params = {}
        
        # 记录所有参数组合结果
        tuning_results = []
        
        # 使用网格搜索寻找最佳参数组合
        for hidden_dim in param_grid['hidden_dim']:
            for num_layers in param_grid['num_layers']:
                for dropout in param_grid['dropout']:
                    for lr in param_grid['learning_rate']:
                        for batch_size in param_grid['batch_size']:
                            print(f"测试参数: hidden_dim={hidden_dim}, num_layers={num_layers}, "
                                  f"dropout={dropout}, lr={lr}, batch_size={batch_size}")
                            
                            # 更新批大小
                            train_loader = DataLoader(
                                train_data['dataset'], batch_size=batch_size, shuffle=True
                            )
                            val_loader = DataLoader(
                                train_data['val_dataset'], batch_size=batch_size
                            )
                            
                            # 训练LSTM模型
                            model = LSTMModel(input_dim, hidden_dim, num_layers, output_dim, dropout)
                            model = model.to(self.device)
                            
                            # 定义损失函数和优化器
                            criterion = nn.MSELoss()
                            optimizer = optim.Adam(model.parameters(), lr=lr)
                            
                            # 训练5轮进行快速评估
                            val_loss = self._train_quick_eval(
                                model, train_loader, val_loader, criterion, optimizer, epochs=5
                            )
                            
                            print(f"验证损失: {val_loss:.6f}")
                            
                            # 记录结果
                            tuning_results.append({
                                'hidden_dim': hidden_dim,
                                'num_layers': num_layers,
                                'dropout': dropout,
                                'learning_rate': lr,
                                'batch_size': batch_size,
                                'val_loss': val_loss
                            })
                            
                            if val_loss < best_val_loss:
                                best_val_loss = val_loss
                                best_params = {
                                    'hidden_dim': hidden_dim,
                                    'num_layers': num_layers,
                                    'dropout': dropout,
                                    'learning_rate': lr,
                                    'batch_size': batch_size
                                }
        
        print(f"最佳参数: {best_params}")
        print(f"最佳验证损失: {best_val_loss:.6f}")
        
        # 保存调优结果
        results_path = os.path.join(self.data_dir, 'optimized_models', 'lstm_tuning_results.json')
        with open(results_path, 'w') as f:
            json.dump({
                'best_params': best_params,
                'best_val_loss': best_val_loss,
                'all_results': tuning_results
            }, f, indent=2)
        
        return best_params
    
    def optimize_transformer_model(self, train_data, input_dim, output_dim):
        """优化Transformer模型结构"""
        print("开始优化Transformer模型...")
        
        # 优化的参数
        d_model = 128  # 增加到128维度
        nhead = 8      # 增加到8个头
        num_layers = 3  # 增加到3层
        dim_feedforward = 256  # 增加前馈网络维度
        dropout = 0.1
        
        # 创建增强的Transformer模型
        model = EnhancedTransformerModel(
            input_dim, d_model, nhead, num_layers, dim_feedforward, output_dim, dropout
        )
        
        print("增强型Transformer模型创建完成")
        return model
    
    def ensemble_models(self, models, inputs, weights=None):
        """
        集成多个模型的预测结果
        
        参数:
            models: 模型列表
            inputs: 输入特征
            weights: 各模型权重，默认平均权重
        
        返回:
            预测结果
        """
        if weights is None:
            weights = [1.0/len(models)] * len(models)
        
        # 确保权重归一化
        weights = np.array(weights) / sum(weights)
        
        # 获取各模型预测
        predictions = []
        for i, model in enumerate(models):
            model.eval()
            with torch.no_grad():
                pred = model(inputs).cpu().numpy()
                predictions.append(pred)
        
        # 加权平均
        ensemble_pred = np.zeros_like(predictions[0])
        for i, pred in enumerate(predictions):
            ensemble_pred += weights[i] * pred
        
        return torch.tensor(ensemble_pred, device=self.device)
    
    #----------------------------------------
    # 模型更新机制
    #----------------------------------------
    def incremental_learning(self, model, new_data, old_model_path, learning_rate=0.0005, epochs=10):
        """
        增量学习 - 使用新数据更新现有模型
        
        参数:
            model: 当前模型
            new_data: 新数据集
            old_model_path: 旧模型路径，用于获取配置
            learning_rate: 学习率
            epochs: 训练轮数
        
        返回:
            更新后的模型
        """
        print("开始增量学习...")
        
        # 加载旧模型的元数据
        metadata_path = old_model_path.replace('.pth', '_metadata.json')
        if not os.path.exists(metadata_path):
            metadata_path = os.path.join(os.path.dirname(old_model_path), 'metadata.json')
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # 准备新数据
        new_train_data = self.base_model.prepare_data_for_training(
            new_data, metadata['target_cols'], test_size=0.2
        )
        
        # 设置较小的学习率进行微调
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        # 使用较少轮数进行微调
        model.train()
        for epoch in range(epochs):
            train_loss = 0.0
            for inputs, targets in new_train_data['train_loader']:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # 前向传播
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                # 反向传播和优化
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(new_train_data['train_loader'])
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_train_loss:.6f}")
            
            # 在验证集上评估
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, targets in new_train_data['val_loader']:
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    outputs = model(inputs)
                    val_loss += criterion(outputs, targets).item()
            
            val_loss /= len(new_train_data['val_loader'])
            print(f"Epoch {epoch+1}/{epochs}, Val Loss: {val_loss:.6f}")
        
        return model
    
    def model_performance_monitor(self, model, validation_data, performance_threshold=0.2, 
                                 metrics_history_path=None):
        """
        模型性能监控，自动触发更新
        
        参数:
            model: 当前模型
            validation_data: 验证数据
            performance_threshold: 性能下降阈值，默认20%
            metrics_history_path: 性能指标历史记录路径
        
        返回:
            是否需要更新模型
        """
        print("开始模型性能监控...")
        
        # 评估当前模型性能
        current_metrics = self.base_model.evaluate_model(model, validation_data)
        
        # 加载历史性能指标
        if metrics_history_path and os.path.exists(metrics_history_path):
            with open(metrics_history_path, 'r') as f:
                metrics_history = json.load(f)
            
            # 获取最佳性能
            best_rmse = min([m['rmse'] for m in metrics_history])
            
            # 计算性能下降百分比
            performance_decline = (current_metrics['rmse'] - best_rmse) / best_rmse
            
            # 判断是否需要更新
            need_update = performance_decline > performance_threshold
            
            # 记录当前性能指标
            metrics_history.append({
                'timestamp': datetime.now().isoformat(),
                'rmse': current_metrics['rmse'],
                'mae': current_metrics['mae'],
                'r2': current_metrics['r2'],
                'performance_decline': performance_decline,
                'need_update': need_update
            })
            
            # 保存更新后的指标历史
            with open(metrics_history_path, 'w') as f:
                json.dump(metrics_history, f, indent=2)
            
            print(f"性能下降: {performance_decline*100:.2f}%, 是否需要更新: {need_update}")
            return need_update, performance_decline
        else:
            # 如果没有历史记录，创建初始记录
            metrics_history = [{
                'timestamp': datetime.now().isoformat(),
                'rmse': current_metrics['rmse'],
                'mae': current_metrics['mae'],
                'r2': current_metrics['r2'],
                'performance_decline': 0,
                'need_update': False
            }]
            
            if metrics_history_path:
                os.makedirs(os.path.dirname(metrics_history_path), exist_ok=True)
                with open(metrics_history_path, 'w') as f:
                    json.dump(metrics_history, f, indent=2)
            
            print("创建初始性能记录")
            return False, 0
    
    def model_ab_testing(self, model_a, model_b, test_data, test_period=7, 
                        metrics=['rmse', 'mae', 'r2'], 
                        weights={'rmse': 0.6, 'mae': 0.3, 'r2': 0.1}):
        """
        模型A/B测试框架
        
        参数:
            model_a: 现有模型
            model_b: 新模型
            test_data: 测试数据
            test_period: 测试周期（天）
            metrics: 评价指标
            weights: 各指标权重
        
        返回:
            是否采用新模型
        """
        print("开始A/B测试...")
        
        # 初始化测试结果
        test_results = {
            'model_a': {'daily_metrics': [], 'avg_metrics': {}},
            'model_b': {'daily_metrics': [], 'avg_metrics': {}}
        }
        
        # 按天分割测试数据
        for day in range(test_period):
            # 在实际场景中，这里应该基于时间戳分割数据
            day_data = test_data  # 这里简化处理，实际应按时间筛选
            
            # 评估两个模型
            metrics_a = self.base_model.evaluate_model(model_a, day_data)
            metrics_b = self.base_model.evaluate_model(model_b, day_data)
            
            # 记录每天的指标
            test_results['model_a']['daily_metrics'].append(metrics_a)
            test_results['model_b']['daily_metrics'].append(metrics_b)
            
            print(f"Day {day+1} metrics:")
            print(f"  Model A - RMSE: {metrics_a['rmse']:.4f}, MAE: {metrics_a['mae']:.4f}, R2: {metrics_a['r2']:.4f}")
            print(f"  Model B - RMSE: {metrics_b['rmse']:.4f}, MAE: {metrics_b['mae']:.4f}, R2: {metrics_b['r2']:.4f}")
        
        # 计算平均指标
        for model_key in ['model_a', 'model_b']:
            for metric in metrics:
                values = [day_metric[metric] for day_metric in test_results[model_key]['daily_metrics']]
                test_results[model_key]['avg_metrics'][metric] = sum(values) / len(values)
        
        # 计算加权得分 (注意对RMSE和MAE是越小越好，对R2是越大越好)
        score_a = test_results['model_a']['avg_metrics']['r2'] * weights['r2'] - \
                  test_results['model_a']['avg_metrics']['rmse'] * weights['rmse'] - \
                  test_results['model_a']['avg_metrics']['mae'] * weights['mae']
                  
        score_b = test_results['model_b']['avg_metrics']['r2'] * weights['r2'] - \
                  test_results['model_b']['avg_metrics']['rmse'] * weights['rmse'] - \
                  test_results['model_b']['avg_metrics']['mae'] * weights['mae']
        
        # 判断是否采用新模型
        use_new_model = score_b > score_a
        
        print(f"A/B测试结果:")
        print(f"  模型A得分: {score_a:.6f}")
        print(f"  模型B得分: {score_b:.6f}")
        print(f"  是否采用新模型: {use_new_model}")
        
        # 返回比较结果
        return {
            'use_new_model': use_new_model,
            'score_a': score_a,
            'score_b': score_b,
            'improvement': (score_b - score_a) / abs(score_a) if score_a != 0 else float('inf'),
            'test_results': test_results
        }
    
    def automated_model_update_pipeline(self, current_model_path, new_data_path, output_dir):
        """
        自动化模型更新流程
        
        参数:
            current_model_path: 当前模型路径
            new_data_path: 新数据路径
            output_dir: 输出目录
        
        返回:
            是否更新成功
        """
        print(f"开始自动化模型更新流程...")
        print(f"当前模型: {current_model_path}")
        print(f"新数据路径: {new_data_path}")
        print(f"输出目录: {output_dir}")
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 步骤1: 加载当前模型
        current_model, metadata = self.base_model.load_model(current_model_path)
        if current_model is None:
            print("加载当前模型失败")
            return False
        
        # 步骤2: 加载新数据
        new_data = self.base_model.load_dataset(new_data_path)
        if new_data is None:
            print("加载新数据失败")
            return False
        
        # 步骤3: 增强特征工程
        enhanced_data = dict(new_data)  # 复制数据
        if 'features' in enhanced_data:
            enhanced_data['features'] = self.apply_all_feature_engineering(
                enhanced_data['features'], metadata['target_cols']
            )
        
        # 步骤4: 准备数据
        validation_data = self.base_model.prepare_data_for_training(
            enhanced_data, metadata['target_cols'], test_size=0.3
        )
        
        # 步骤5: 监控模型性能
        metrics_history_path = os.path.join(os.path.dirname(current_model_path), 'metrics_history.json')
        need_update, decline = self.model_performance_monitor(
            current_model, validation_data, metrics_history_path=metrics_history_path
        )
        
        if not need_update:
            print("当前模型性能良好，无需更新")
            return False
        
        # 步骤6: 增量学习现有模型
        print("模型性能下降，开始模型更新...")
        updated_model = self.incremental_learning(
            current_model, enhanced_data, current_model_path, learning_rate=0.0002, epochs=15
        )
        
        # 步骤7: 使用新架构训练模型
        model_type = metadata.get('model_type', 'lstm')
        input_dim = metadata.get('input_dim', validation_data['X_train'].shape[1])
        output_dim = metadata.get('output_dim', validation_data['y_train'].shape[1])
        
        if model_type == 'transformer':
            new_arch_model = self.optimize_transformer_model(validation_data, input_dim, output_dim)
            
            # 训练新架构模型
            self.base_model.train_transformer_model(
                validation_data, input_dim, d_model=128, nhead=8, 
                num_layers=3, dim_feedforward=256, output_dim=output_dim
            )
        elif model_type == 'lstm':
            # 获取最佳参数
            best_params = self.hyper_parameter_tuning_lstm(validation_data, input_dim, output_dim)
            
            # 使用最佳参数训练模型
            new_arch_model, _ = self.base_model.train_lstm_model(
                validation_data, input_dim, 
                hidden_dim=best_params['hidden_dim'],
                num_layers=best_params['num_layers'],
                output_dim=output_dim,
                learning_rate=best_params['learning_rate']
            )
        else:
            print(f"不支持的模型类型: {model_type}")
            new_arch_model = None
        
        # 步骤8: A/B测试
        if new_arch_model is not None:
            ab_results = self.model_ab_testing(
                updated_model, new_arch_model, validation_data
            )
            
            # 选择最佳模型
            final_model = new_arch_model if ab_results['use_new_model'] else updated_model
            model_name = "new_architecture" if ab_results['use_new_model'] else "incremental_updated"
        else:
            final_model = updated_model
            model_name = "incremental_updated"
        
        # 步骤9: 保存更新后的模型
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"{model_type}_{model_name}_{timestamp}.pth"
        model_path = os.path.join(output_dir, model_filename)
        
        # 更新元数据
        updated_metadata = dict(metadata)
        updated_metadata['update_timestamp'] = timestamp
        updated_metadata['update_type'] = model_name
        updated_metadata['previous_model'] = current_model_path
        
        # 保存模型
        self.base_model.save_model(
            final_model, validation_data, {}, 
            self.base_model.evaluate_model(final_model, validation_data),
            model_path
        )
        
        print(f"模型更新完成，新模型已保存到: {model_path}")
        return True

# 测试运行
if __name__ == "__main__":
    print("模型优化器测试") 
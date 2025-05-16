#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
污染物排放预测模型优化脚本
此脚本执行端到端的模型优化流程，包括特征工程、参数调优和模型更新
"""

import os
import sys
import argparse
import json
import torch
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

# 导入项目模块
from model_training.pollution_prediction import (
    load_dataset, LSTMModel, GRUModel, TransformerModel, TimeSeriesDataset
)
from model_training.model_optimizer import ModelOptimizer

class ModelOptimizationRunner:
    """污染物排放预测模型优化执行器"""
    
    def __init__(self, data_dir='./data', model_dir='./models', output_dir='./optimized_models'):
        """
        初始化模型优化执行器
        
        参数:
            data_dir: 数据目录
            model_dir: 模型目录
            output_dir: 输出目录
        """
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.output_dir = output_dir
        
        # 确保目录存在
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        
        # 检测设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")
        
        # 创建基础模型实例用于加载数据和模型
        from model_training.pollution_prediction import PollutionPredictionModel
        self.base_model = PollutionPredictionModel()
        
        # 创建模型优化器
        self.optimizer = ModelOptimizer(self.base_model, data_dir=data_dir)
    
    def load_training_data(self, data_path):
        """加载训练数据"""
        print(f"加载训练数据: {data_path}")
        try:
            data = load_dataset(data_path)
            print(f"数据加载成功，样本数: {len(data['features'])}")
            return data
        except Exception as e:
            print(f"数据加载失败: {str(e)}")
            return None
    
    def optimize_lstm_model(self, data, target_cols):
        """优化LSTM模型"""
        print("开始LSTM模型优化流程...")
        
        # 1. 应用特征工程
        print("步骤1: 应用特征工程...")
        enhanced_data = dict(data)
        enhanced_data['features'] = self.optimizer.apply_all_feature_engineering(
            data['features'].copy(), target_cols
        )
        
        # 2. 准备训练数据
        print("步骤2: 准备训练数据...")
        train_data = self.base_model.prepare_data_for_training(
            enhanced_data, target_cols, test_size=0.2
        )
        
        # 3. 参数调优
        print("步骤3: 执行参数调优...")
        input_dim = train_data['X_train'].shape[1]
        output_dim = train_data['y_train'].shape[1]
        
        best_params = self.optimizer.hyper_parameter_tuning_lstm(
            train_data, input_dim, output_dim
        )
        
        # 4. 使用最佳参数训练模型
        print("步骤4: 使用最佳参数训练最终模型...")
        
        # 4.1 创建LSTM模型
        model = LSTMModel(
            input_dim=input_dim,
            hidden_dim=best_params['hidden_dim'],
            num_layers=best_params['num_layers'],
            output_dim=output_dim,
            dropout=best_params['dropout']
        )
        model = model.to(self.device)
        
        # 4.2 训练模型
        import torch.optim as optim
        import torch.nn as nn
        from torch.utils.data import DataLoader
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=best_params['learning_rate'])
        
        train_loader = DataLoader(
            train_data['dataset'], 
            batch_size=best_params['batch_size'], 
            shuffle=True
        )
        val_loader = DataLoader(
            train_data['val_dataset'], 
            batch_size=best_params['batch_size']
        )
        
        # 训练参数
        num_epochs = 30
        patience = 5
        best_val_loss = float('inf')
        early_stop_counter = 0
        best_model_state = None
        
        # 记录训练历史
        history = {
            'train_loss': [],
            'val_loss': [],
            'best_epoch': 0
        }
        
        # 训练循环
        print(f"开始训练，总轮数: {num_epochs}")
        for epoch in range(num_epochs):
            # 训练模式
            model.train()
            train_loss = 0.0
            
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # 前向传播
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                # 反向传播和优化
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            
            # 验证模式
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    outputs = model(inputs)
                    val_loss += criterion(outputs, targets).item()
            
            avg_val_loss = val_loss / len(val_loader)
            
            # 记录历史
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            
            print(f"Epoch {epoch+1}/{num_epochs}, "
                  f"Train Loss: {avg_train_loss:.6f}, "
                  f"Val Loss: {avg_val_loss:.6f}")
            
            # 早停检查
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = model.state_dict().copy()
                history['best_epoch'] = epoch
                early_stop_counter = 0
            else:
                early_stop_counter += 1
                if early_stop_counter >= patience:
                    print(f"早停触发，最佳轮数: {history['best_epoch']+1}")
                    break
        
        # 加载最佳模型状态
        model.load_state_dict(best_model_state)
        
        # 5. 评估最终模型
        print("步骤5: 评估最终模型...")
        test_metrics = self.base_model.evaluate_model(model, train_data)
        print(f"测试结果 - RMSE: {test_metrics['rmse']:.4f}, "
              f"MAE: {test_metrics['mae']:.4f}, R2: {test_metrics['r2']:.4f}")
        
        # 6. 保存模型
        print("步骤6: 保存优化后的模型...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"lstm_optimized_{timestamp}.pth"
        model_path = os.path.join(self.output_dir, model_filename)
        
        # 保存模型状态
        torch.save(model.state_dict(), model_path)
        
        # 保存模型元数据
        metadata = {
            'model_type': 'lstm',
            'input_dim': input_dim,
            'hidden_dim': best_params['hidden_dim'],
            'num_layers': best_params['num_layers'],
            'output_dim': output_dim,
            'dropout': best_params['dropout'],
            'timestamp': timestamp,
            'target_cols': target_cols,
            'metrics': test_metrics,
            'parameters': best_params,
            'history': history
        }
        
        metadata_path = os.path.join(self.output_dir, f"lstm_optimized_{timestamp}_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # 7. 绘制训练历史图表
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='训练损失')
        plt.plot(history['val_loss'], label='验证损失')
        plt.axvline(x=history['best_epoch'], color='r', linestyle='--', label='最佳模型')
        plt.xlabel('轮数')
        plt.ylabel('损失')
        plt.title('训练历史')
        plt.legend()
        
        # 保存图表
        chart_path = os.path.join(self.output_dir, f"lstm_optimized_{timestamp}_history.png")
        plt.savefig(chart_path)
        
        print(f"优化完成，模型已保存到: {model_path}")
        print(f"元数据已保存到: {metadata_path}")
        print(f"训练历史图表已保存到: {chart_path}")
        
        return {
            'model': model,
            'model_path': model_path,
            'metadata': metadata,
            'metrics': test_metrics
        }
    
    def optimize_transformer_model(self, data, target_cols):
        """优化Transformer模型"""
        print("开始Transformer模型优化流程...")
        
        # 1. 应用特征工程
        print("步骤1: 应用特征工程...")
        enhanced_data = dict(data)
        enhanced_data['features'] = self.optimizer.apply_all_feature_engineering(
            data['features'].copy(), target_cols
        )
        
        # 2. 准备训练数据
        print("步骤2: 准备训练数据...")
        train_data = self.base_model.prepare_data_for_training(
            enhanced_data, target_cols, test_size=0.2
        )
        
        # 3. 创建增强的Transformer模型
        print("步骤3: 创建增强的Transformer模型...")
        input_dim = train_data['X_train'].shape[1]
        output_dim = train_data['y_train'].shape[1]
        
        model = self.optimizer.optimize_transformer_model(
            train_data, input_dim, output_dim
        )
        model = model.to(self.device)
        
        # 4. 训练模型
        print("步骤4: 训练Transformer模型...")
        import torch.optim as optim
        import torch.nn as nn
        from torch.utils.data import DataLoader
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.0005)
        
        train_loader = DataLoader(
            train_data['dataset'], 
            batch_size=32, 
            shuffle=True
        )
        val_loader = DataLoader(
            train_data['val_dataset'], 
            batch_size=32
        )
        
        # 训练参数
        num_epochs = 30
        patience = 5
        best_val_loss = float('inf')
        early_stop_counter = 0
        best_model_state = None
        
        # 记录训练历史
        history = {
            'train_loss': [],
            'val_loss': [],
            'best_epoch': 0
        }
        
        # 训练循环
        print(f"开始训练，总轮数: {num_epochs}")
        for epoch in range(num_epochs):
            # 训练模式
            model.train()
            train_loss = 0.0
            
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # 前向传播
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                # 反向传播和优化
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            
            # 验证模式
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    outputs = model(inputs)
                    val_loss += criterion(outputs, targets).item()
            
            avg_val_loss = val_loss / len(val_loader)
            
            # 记录历史
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            
            print(f"Epoch {epoch+1}/{num_epochs}, "
                  f"Train Loss: {avg_train_loss:.6f}, "
                  f"Val Loss: {avg_val_loss:.6f}")
            
            # 早停检查
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = model.state_dict().copy()
                history['best_epoch'] = epoch
                early_stop_counter = 0
            else:
                early_stop_counter += 1
                if early_stop_counter >= patience:
                    print(f"早停触发，最佳轮数: {history['best_epoch']+1}")
                    break
        
        # 加载最佳模型状态
        model.load_state_dict(best_model_state)
        
        # 5. 评估最终模型
        print("步骤5: 评估最终模型...")
        test_metrics = self.base_model.evaluate_model(model, train_data)
        print(f"测试结果 - RMSE: {test_metrics['rmse']:.4f}, "
              f"MAE: {test_metrics['mae']:.4f}, R2: {test_metrics['r2']:.4f}")
        
        # 6. 保存模型
        print("步骤6: 保存优化后的模型...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"transformer_optimized_{timestamp}.pth"
        model_path = os.path.join(self.output_dir, model_filename)
        
        # 保存模型状态
        torch.save(model.state_dict(), model_path)
        
        # 保存模型元数据
        metadata = {
            'model_type': 'enhanced_transformer',
            'input_dim': input_dim,
            'd_model': 128,
            'nhead': 8,
            'num_layers': 3,
            'dim_feedforward': 256,
            'output_dim': output_dim,
            'dropout': 0.1,
            'timestamp': timestamp,
            'target_cols': target_cols,
            'metrics': test_metrics,
            'history': history
        }
        
        metadata_path = os.path.join(self.output_dir, f"transformer_optimized_{timestamp}_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # 7. 绘制训练历史图表
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='训练损失')
        plt.plot(history['val_loss'], label='验证损失')
        plt.axvline(x=history['best_epoch'], color='r', linestyle='--', label='最佳模型')
        plt.xlabel('轮数')
        plt.ylabel('损失')
        plt.title('训练历史')
        plt.legend()
        
        # 保存图表
        chart_path = os.path.join(self.output_dir, f"transformer_optimized_{timestamp}_history.png")
        plt.savefig(chart_path)
        
        print(f"优化完成，模型已保存到: {model_path}")
        print(f"元数据已保存到: {metadata_path}")
        print(f"训练历史图表已保存到: {chart_path}")
        
        return {
            'model': model,
            'model_path': model_path,
            'metadata': metadata,
            'metrics': test_metrics
        }
    
    def create_model_ensemble(self, models_dict):
        """创建模型集成"""
        print("创建模型集成...")
        
        models = []
        model_weights = []
        
        # 根据性能指标计算权重
        total_r2 = 0
        model_r2_list = []
        
        for model_name, model_info in models_dict.items():
            model = model_info['model']
            metrics = model_info['metrics']
            r2 = metrics['r2']
            
            models.append(model)
            model_r2_list.append(r2)
            total_r2 += r2
        
        # 如果所有R2都为负，则使用均等权重
        if total_r2 <= 0:
            model_weights = [1.0/len(models)] * len(models)
        else:
            model_weights = [r2/total_r2 for r2 in model_r2_list]
        
        print(f"模型集成权重: {model_weights}")
        
        # 保存集成元数据
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ensemble_metadata = {
            'model_type': 'ensemble',
            'base_models': [m['model_path'] for m in models_dict.values()],
            'weights': model_weights,
            'timestamp': timestamp
        }
        
        metadata_path = os.path.join(self.output_dir, f"ensemble_{timestamp}_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(ensemble_metadata, f, indent=2)
        
        print(f"集成模型元数据已保存到: {metadata_path}")
        
        return {
            'models': models,
            'weights': model_weights,
            'metadata': ensemble_metadata,
            'metadata_path': metadata_path
        }
    
    def update_existing_model(self, model_path, new_data_path):
        """更新现有模型"""
        print(f"更新现有模型: {model_path}")
        print(f"使用新数据: {new_data_path}")
        
        # 执行自动化更新流程
        update_result = self.optimizer.automated_model_update_pipeline(
            model_path, new_data_path, self.output_dir
        )
        
        return update_result

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='污染物排放预测模型优化脚本')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='数据目录路径')
    parser.add_argument('--model_dir', type=str, default='./models',
                        help='模型目录路径')
    parser.add_argument('--output_dir', type=str, default='./optimized_models',
                        help='输出目录路径')
    parser.add_argument('--data_path', type=str, required=True,
                        help='训练数据路径')
    parser.add_argument('--optimize_lstm', action='store_true',
                        help='优化LSTM模型')
    parser.add_argument('--optimize_transformer', action='store_true',
                        help='优化Transformer模型')
    parser.add_argument('--create_ensemble', action='store_true',
                        help='创建模型集成')
    parser.add_argument('--update_model', type=str, default='',
                        help='要更新的模型路径')
    parser.add_argument('--target_cols', type=str, default='so2,nox,dust',
                        help='目标列，用逗号分隔')
    
    args = parser.parse_args()
    
    # 解析目标列
    target_cols = args.target_cols.split(',')
    
    # 创建优化执行器
    runner = ModelOptimizationRunner(
        data_dir=args.data_dir,
        model_dir=args.model_dir,
        output_dir=args.output_dir
    )
    
    # 加载数据
    data = runner.load_training_data(args.data_path)
    if data is None:
        print("数据加载失败，退出")
        return
    
    optimized_models = {}
    
    # 优化LSTM模型
    if args.optimize_lstm:
        lstm_result = runner.optimize_lstm_model(data, target_cols)
        optimized_models['lstm'] = lstm_result
    
    # 优化Transformer模型
    if args.optimize_transformer:
        transformer_result = runner.optimize_transformer_model(data, target_cols)
        optimized_models['transformer'] = transformer_result
    
    # 创建模型集成
    if args.create_ensemble and len(optimized_models) > 1:
        ensemble = runner.create_model_ensemble(optimized_models)
    
    # 更新现有模型
    if args.update_model:
        runner.update_existing_model(args.update_model, args.data_path)

if __name__ == "__main__":
    main() 
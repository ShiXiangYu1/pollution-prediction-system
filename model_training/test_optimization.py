#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
污染物排放预测模型优化效果测试脚本
用于对比优化前后的模型性能，验证优化方案的有效性
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime

# 导入项目模块
from model_training.pollution_prediction import (
    load_dataset, LSTMModel, TransformerModel, PollutionPredictionModel
)
from model_training.model_optimizer import ModelOptimizer, EnhancedTransformerModel

class OptimizationTester:
    """模型优化效果测试器"""
    
    def __init__(self, test_data_path, original_model_path=None, optimized_model_path=None, 
                output_dir='./test_results'):
        """
        初始化测试器
        
        参数:
            test_data_path: 测试数据路径
            original_model_path: 原始模型路径
            optimized_model_path: 优化后模型路径
            output_dir: 输出目录
        """
        self.test_data_path = test_data_path
        self.original_model_path = original_model_path
        self.optimized_model_path = optimized_model_path
        self.output_dir = output_dir
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 检测设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")
        
        # 创建基础模型实例
        self.base_model = PollutionPredictionModel()
        
        # 创建模型优化器实例
        self.optimizer = ModelOptimizer(self.base_model)
    
    def load_test_data(self):
        """加载测试数据"""
        print(f"加载测试数据: {self.test_data_path}")
        try:
            data = load_dataset(self.test_data_path)
            print(f"测试数据加载成功，样本数: {len(data['features'])}")
            return data
        except Exception as e:
            print(f"测试数据加载失败: {str(e)}")
            return None
    
    def load_model_and_metadata(self, model_path):
        """加载模型和元数据"""
        if not model_path:
            return None, None
        
        print(f"加载模型: {model_path}")
        
        # 尝试加载元数据
        metadata_path = model_path.replace('.pth', '_metadata.json')
        if not os.path.exists(metadata_path):
            # 尝试在同目录下查找元数据文件
            dir_path = os.path.dirname(model_path)
            base_name = os.path.basename(model_path).split('.')[0]
            for file in os.listdir(dir_path):
                if file.startswith(base_name) and file.endswith('_metadata.json'):
                    metadata_path = os.path.join(dir_path, file)
                    break
        
        # 如果还是找不到元数据
        if not os.path.exists(metadata_path):
            print(f"未找到元数据文件，使用默认配置")
            metadata = {
                'model_type': 'lstm', 
                'input_dim': 10,
                'hidden_dim': 64,
                'num_layers': 2,
                'output_dim': 3,
                'dropout': 0.1,
                'target_cols': ['so2', 'nox', 'dust']
            }
        else:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                print(f"元数据加载成功: {metadata['model_type']} 模型")
        
        # 根据模型类型创建模型实例
        model_type = metadata.get('model_type', 'lstm')
        input_dim = metadata.get('input_dim', 10)
        output_dim = metadata.get('output_dim', 3)
        
        if model_type == 'lstm':
            hidden_dim = metadata.get('hidden_dim', 64)
            num_layers = metadata.get('num_layers', 2)
            dropout = metadata.get('dropout', 0.1)
            
            model = LSTMModel(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                output_dim=output_dim,
                dropout=dropout
            )
        elif model_type == 'transformer':
            d_model = metadata.get('d_model', 64)
            nhead = metadata.get('nhead', 2)
            num_layers = metadata.get('num_layers', 2)
            dim_feedforward = metadata.get('dim_feedforward', 128)
            dropout = metadata.get('dropout', 0.1)
            
            model = TransformerModel(
                input_dim=input_dim,
                d_model=d_model,
                nhead=nhead,
                num_layers=num_layers,
                dim_feedforward=dim_feedforward,
                output_dim=output_dim,
                dropout=dropout
            )
        elif model_type == 'enhanced_transformer':
            d_model = metadata.get('d_model', 128)
            nhead = metadata.get('nhead', 8)
            num_layers = metadata.get('num_layers', 3)
            dim_feedforward = metadata.get('dim_feedforward', 256)
            dropout = metadata.get('dropout', 0.1)
            
            model = EnhancedTransformerModel(
                input_dim=input_dim,
                d_model=d_model,
                nhead=nhead,
                num_layers=num_layers,
                dim_feedforward=dim_feedforward,
                output_dim=output_dim,
                dropout=dropout
            )
        else:
            print(f"不支持的模型类型: {model_type}")
            return None, metadata
        
        # 加载模型权重
        try:
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            model = model.to(self.device)
            model.eval()
            print(f"模型加载成功")
            return model, metadata
        except Exception as e:
            print(f"模型加载失败: {str(e)}")
            return None, metadata
    
    def prepare_test_data(self, data, metadata):
        """准备测试数据"""
        print("准备测试数据...")
        
        # 获取目标列
        target_cols = metadata.get('target_cols', ['so2', 'nox', 'dust'])
        
        # 应用特征工程（对于优化后的模型）
        if 'optimized' in metadata.get('model_type', ''):
            enhanced_data = dict(data)
            enhanced_data['features'] = self.optimizer.apply_all_feature_engineering(
                data['features'].copy(), target_cols
            )
            prepared_data = self.base_model.prepare_data_for_training(
                enhanced_data, target_cols, test_size=0
            )
        else:
            prepared_data = self.base_model.prepare_data_for_training(
                data, target_cols, test_size=0
            )
        
        return prepared_data
    
    def evaluate_model(self, model, test_data):
        """评估模型性能"""
        if model is None:
            return None
        
        # 设置为评估模式
        model.eval()
        
        # 提取数据
        X_test = test_data['X_test']
        y_test = test_data['y_test']
        
        # 转换为tensor
        X_tensor = torch.tensor(X_test.values, dtype=torch.float32)
        y_tensor = torch.tensor(y_test.values, dtype=torch.float32)
        
        # 预测
        with torch.no_grad():
            X_tensor = X_tensor.to(self.device)
            y_pred = model(X_tensor).cpu().numpy()
            
        # 计算评估指标
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        metrics = {
            'rmse': float(rmse),
            'mae': float(mae),
            'r2': float(r2)
        }
        
        return {
            'metrics': metrics,
            'predictions': y_pred,
            'actual': y_test.values
        }
    
    def run_comparison_test(self):
        """运行对比测试"""
        # 1. 加载测试数据
        test_data = self.load_test_data()
        if test_data is None:
            print("测试失败：无法加载测试数据")
            return False
        
        # 2. 加载原始模型和优化后的模型
        original_model, original_metadata = self.load_model_and_metadata(self.original_model_path)
        optimized_model, optimized_metadata = self.load_model_and_metadata(self.optimized_model_path)
        
        if original_model is None and optimized_model is None:
            print("测试失败：无法加载任何模型")
            return False
        
        results = {}
        
        # 3. 准备原始模型的测试数据并评估
        if original_model is not None:
            original_test_data = self.prepare_test_data(test_data, original_metadata)
            original_results = self.evaluate_model(original_model, original_test_data)
            results['original'] = {
                'model': original_model,
                'metadata': original_metadata,
                'results': original_results
            }
            print(f"原始模型评估完成 - RMSE: {original_results['metrics']['rmse']:.4f}, "
                  f"MAE: {original_results['metrics']['mae']:.4f}, R2: {original_results['metrics']['r2']:.4f}")
        
        # 4. 准备优化后模型的测试数据并评估
        if optimized_model is not None:
            optimized_test_data = self.prepare_test_data(test_data, optimized_metadata)
            optimized_results = self.evaluate_model(optimized_model, optimized_test_data)
            results['optimized'] = {
                'model': optimized_model,
                'metadata': optimized_metadata,
                'results': optimized_results
            }
            print(f"优化模型评估完成 - RMSE: {optimized_results['metrics']['rmse']:.4f}, "
                  f"MAE: {optimized_results['metrics']['mae']:.4f}, R2: {optimized_results['metrics']['r2']:.4f}")
        
        # 5. 对比模型性能
        if 'original' in results and 'optimized' in results:
            # 计算性能提升百分比
            original_metrics = results['original']['results']['metrics']
            optimized_metrics = results['optimized']['results']['metrics']
            
            rmse_improvement = (original_metrics['rmse'] - optimized_metrics['rmse']) / original_metrics['rmse'] * 100
            mae_improvement = (original_metrics['mae'] - optimized_metrics['mae']) / original_metrics['mae'] * 100
            r2_improvement = (optimized_metrics['r2'] - original_metrics['r2']) / abs(original_metrics['r2']) * 100 if original_metrics['r2'] != 0 else float('inf')
            
            print("\n性能对比：")
            print(f"RMSE 提升: {rmse_improvement:.2f}%")
            print(f"MAE 提升: {mae_improvement:.2f}%")
            print(f"R2 提升: {r2_improvement:.2f}%")
            
            # 生成对比图表
            self.generate_comparison_charts(results)
            
            # 保存对比结果
            self.save_comparison_results(results, {
                'rmse_improvement': float(rmse_improvement),
                'mae_improvement': float(mae_improvement),
                'r2_improvement': float(r2_improvement)
            })
        
        return results
    
    def generate_comparison_charts(self, results):
        """生成对比图表"""
        print("生成对比图表...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 确保结果包含所需数据
        if not all(key in results for key in ['original', 'optimized']):
            available_models = list(results.keys())
            print(f"无法生成对比图表，可用模型: {available_models}")
            return
        
        # 提取预测结果和实际值
        original_pred = results['original']['results']['predictions']
        optimized_pred = results['optimized']['results']['predictions']
        actual = results['original']['results']['actual']  # 假设两种模型使用相同的测试集
        
        # 确定预测目标的数量
        n_targets = original_pred.shape[1]
        target_names = results['original']['metadata'].get('target_cols', [f'target_{i}' for i in range(n_targets)])
        
        # 为每个目标创建散点图和时间序列图
        plt.figure(figsize=(20, 5 * n_targets))
        
        for i in range(n_targets):
            # 散点图
            plt.subplot(n_targets, 2, i*2+1)
            plt.scatter(actual[:, i], original_pred[:, i], alpha=0.5, label='原始模型')
            plt.scatter(actual[:, i], optimized_pred[:, i], alpha=0.5, label='优化模型')
            plt.plot([actual[:, i].min(), actual[:, i].max()], [actual[:, i].min(), actual[:, i].max()], 'k--')
            plt.xlabel('实际值')
            plt.ylabel('预测值')
            plt.title(f'{target_names[i]} 预测散点图')
            plt.legend()
            
            # 时间序列图（取前200个样本以便可视化）
            n_samples = min(200, len(actual))
            plt.subplot(n_targets, 2, i*2+2)
            plt.plot(actual[:n_samples, i], label='实际值')
            plt.plot(original_pred[:n_samples, i], label='原始模型预测')
            plt.plot(optimized_pred[:n_samples, i], label='优化模型预测')
            plt.xlabel('样本')
            plt.ylabel('值')
            plt.title(f'{target_names[i]} 预测时间序列')
            plt.legend()
        
        plt.tight_layout()
        
        # 保存图表
        chart_path = os.path.join(self.output_dir, f"model_comparison_{timestamp}.png")
        plt.savefig(chart_path)
        plt.close()
        
        print(f"对比图表已保存到: {chart_path}")
        
        # 生成误差分布图
        plt.figure(figsize=(15, 5))
        
        for i in range(n_targets):
            plt.subplot(1, n_targets, i+1)
            original_errors = actual[:, i] - original_pred[:, i]
            optimized_errors = actual[:, i] - optimized_pred[:, i]
            
            plt.hist(original_errors, bins=30, alpha=0.5, label='原始模型')
            plt.hist(optimized_errors, bins=30, alpha=0.5, label='优化模型')
            plt.xlabel('预测误差')
            plt.ylabel('频率')
            plt.title(f'{target_names[i]} 预测误差分布')
            plt.legend()
        
        plt.tight_layout()
        
        # 保存误差分布图
        error_chart_path = os.path.join(self.output_dir, f"error_distribution_{timestamp}.png")
        plt.savefig(error_chart_path)
        plt.close()
        
        print(f"误差分布图已保存到: {error_chart_path}")
    
    def save_comparison_results(self, results, improvements):
        """保存对比结果"""
        print("保存对比结果...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 准备结果数据
        comparison_data = {
            'timestamp': timestamp,
            'improvements': improvements
        }
        
        # 添加原始模型信息
        if 'original' in results:
            comparison_data['original'] = {
                'model_path': self.original_model_path,
                'model_type': results['original']['metadata'].get('model_type', 'unknown'),
                'metrics': results['original']['results']['metrics']
            }
        
        # 添加优化模型信息
        if 'optimized' in results:
            comparison_data['optimized'] = {
                'model_path': self.optimized_model_path,
                'model_type': results['optimized']['metadata'].get('model_type', 'unknown'),
                'metrics': results['optimized']['results']['metrics']
            }
        
        # 保存为JSON文件
        results_path = os.path.join(self.output_dir, f"comparison_results_{timestamp}.json")
        with open(results_path, 'w') as f:
            json.dump(comparison_data, f, indent=2)
        
        print(f"对比结果已保存到: {results_path}")
        
        return results_path

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='污染物排放预测模型优化效果测试脚本')
    parser.add_argument('--test_data', type=str, required=True,
                        help='测试数据路径')
    parser.add_argument('--original_model', type=str,
                        help='原始模型路径')
    parser.add_argument('--optimized_model', type=str,
                        help='优化后模型路径')
    parser.add_argument('--output_dir', type=str, default='./test_results',
                        help='输出目录路径')
    
    args = parser.parse_args()
    
    # 创建测试器
    tester = OptimizationTester(
        test_data_path=args.test_data,
        original_model_path=args.original_model,
        optimized_model_path=args.optimized_model,
        output_dir=args.output_dir
    )
    
    # 运行对比测试
    tester.run_comparison_test()

if __name__ == "__main__":
    main() 
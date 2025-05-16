#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
污染物排放预测模型模块 - 实现时序预测模型
"""

import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
from tqdm import tqdm
import json
import time
from datetime import datetime

# 深度学习库
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset

class TimeSeriesDataset(Dataset):
    """
    时序数据集类
    """
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class LSTMModel(nn.Module):
    """
    LSTM模型类
    """
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

class GRUModel(nn.Module):
    """
    GRU模型类
    """
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout=0.2):
        super(GRUModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        return out

class TransformerModel(nn.Module):
    """
    Transformer模型类
    """
    def __init__(self, input_dim, d_model, nhead, num_layers, dim_feedforward, output_dim, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.input_linear = nn.Linear(input_dim, d_model)
        
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, 
                                                   dim_feedforward=dim_feedforward, 
                                                   dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        self.output_linear = nn.Linear(d_model, output_dim)
    
    def forward(self, x):
        x = self.input_linear(x)
        x = self.transformer_encoder(x)
        x = self.output_linear(x[:, -1, :])
        return x

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

class PollutionPredictionModel:
    """
    污染物排放预测模型类
    """
    
    def __init__(self, data_dir='./data'):
        """
        初始化污染物排放预测模型类
        
        参数:
            data_dir: 数据目录
        """
        self.data_dir = data_dir
        
        # 创建模型目录（如果不存在）
        os.makedirs(os.path.join(data_dir, 'models', 'prediction'), exist_ok=True)
        
        # 检查是否有GPU可用
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")
    
    def load_dataset(self, dataset_path):
        """
        加载数据集
        
        参数:
            dataset_path: 数据集路径
            
        返回:
            dict: 包含特征和目标的字典
        """
        print(f"加载数据集: {dataset_path}")
        
        try:
            # 根据文件扩展名选择加载方法
            ext = os.path.splitext(dataset_path)[1].lower()
            
            if ext == '.joblib':
                data = joblib.load(dataset_path)
                print(f"成功加载数据集，包含以下键: {list(data.keys())}")
                return data
            else:
                print(f"不支持的文件格式: {ext}")
                return None
        except Exception as e:
            print(f"加载数据集失败: {e}")
            return None
    
    def prepare_data_for_training(self, data, target_cols, test_size=0.2, val_size=0.1, scale=True):
        """
        准备训练数据
        
        参数:
            data: 数据字典
            target_cols: 目标列列表
            test_size: 测试集比例
            val_size: 验证集比例
            scale: 是否进行缩放
            
        返回:
            dict: 包含训练集、验证集和测试集的字典
        """
        print("准备训练数据...")
        
        # 提取特征和目标
        X = data.get('features')
        y = data.get('targets')
        
        if X is None or y is None:
            print("数据中缺少特征或目标")
            return None
        
        # 确保X和y是numpy数组
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.DataFrame):
            y = y.values
        
        # 数据缩放
        if scale:
            # 特征缩放
            X_scaler = StandardScaler()
            X = X_scaler.fit_transform(X)
            
            # 目标缩放
            y_scaler = StandardScaler()
            y = y_scaler.fit_transform(y)
            
            # 保存缩放器
            scalers = {
                'X_scaler': X_scaler,
                'y_scaler': y_scaler
            }
        else:
            scalers = None
        
        # 划分训练集、验证集和测试集
        # 首先划分训练集和测试集
        train_size = 1 - test_size
        train_samples = int(len(X) * train_size)
        
        X_train = X[:train_samples]
        y_train = y[:train_samples]
        X_test = X[train_samples:]
        y_test = y[train_samples:]
        
        # 从训练集中划分验证集
        if val_size > 0:
            val_samples = int(len(X_train) * (val_size / train_size))
            X_val = X_train[-val_samples:]
            y_val = y_train[-val_samples:]
            X_train = X_train[:-val_samples]
            y_train = y_train[:-val_samples]
        else:
            X_val = None
            y_val = None
        
        # 转换为PyTorch张量
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train)
        
        if X_val is not None:
            X_val_tensor = torch.FloatTensor(X_val)
            y_val_tensor = torch.FloatTensor(y_val)
        else:
            X_val_tensor = None
            y_val_tensor = None
        
        X_test_tensor = torch.FloatTensor(X_test)
        y_test_tensor = torch.FloatTensor(y_test)
        
        # 创建数据加载器
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        
        if X_val is not None:
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        else:
            val_loader = None
        
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        # 创建结果字典
        result = {
            'train_loader': train_loader,
            'val_loader': val_loader,
            'test_loader': test_loader,
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test,
            'X_train_tensor': X_train_tensor,
            'y_train_tensor': y_train_tensor,
            'X_val_tensor': X_val_tensor,
            'y_val_tensor': y_val_tensor,
            'X_test_tensor': X_test_tensor,
            'y_test_tensor': y_test_tensor,
            'scalers': scalers,
            'target_cols': target_cols
        }
        
        print(f"训练数据准备完成，训练集: {len(X_train)}条，验证集: {len(X_val) if X_val is not None else 0}条，测试集: {len(X_test)}条")
        return result
    
    def train_lstm_model(self, train_data, input_dim, hidden_dim=64, num_layers=2, output_dim=3, 
                        num_epochs=50, learning_rate=0.001, patience=10):
        """
        训练LSTM模型
        
        参数:
            train_data: 训练数据字典
            input_dim: 输入维度
            hidden_dim: 隐藏层维度
            num_layers: LSTM层数
            output_dim: 输出维度
            num_epochs: 训练轮数
            learning_rate: 学习率
            patience: 早停耐心值
            
        返回:
            model: 训练后的模型
            history: 训练历史
        """
        print("开始训练LSTM模型...")
        
        # 提取数据加载器
        train_loader = train_data['train_loader']
        val_loader = train_data['val_loader']
        
        # 创建模型
        model = LSTMModel(input_dim, hidden_dim, num_layers, output_dim)
        model = model.to(self.device)
        
        # 定义损失函数和优化器
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # 训练历史
        history = {
            'train_loss': [],
            'val_loss': []
        }
        
        # 早停
        best_val_loss = float('inf')
        counter = 0
        
        # 训练循环
        for epoch in range(num_epochs):
            # 训练模式
            model.train()
            train_loss = 0.0
            
            for inputs, targets in train_loader:
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
            
            # 计算平均训练损失
            train_loss /= len(train_loader)
            history['train_loss'].append(train_loss)
            
            # 验证
            if val_loader is not None:
                model.eval()
                val_loss = 0.0
                
                with torch.no_grad():
                    for inputs, targets in val_loader:
                        inputs = inputs.to(self.device)
                        targets = targets.to(self.device)
                        
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                        
                        val_loss += loss.item()
                
                # 计算平均验证损失
                val_loss /= len(val_loader)
                history['val_loss'].append(val_loss)
                
                # 早停检查
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    counter = 0
                    # 保存最佳模型
                    best_model = model.state_dict().copy()
                else:
                    counter += 1
                
                print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
                
                if counter >= patience:
                    print(f"早停触发，在{epoch+1}轮停止训练")
                    # 恢复最佳模型
                    model.load_state_dict(best_model)
                    break
            else:
                print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}")
        
        print("LSTM模型训练完成")
        return model, history
    
    def train_gru_model(self, train_data, input_dim, hidden_dim=64, num_layers=2, output_dim=3, 
                       num_epochs=50, learning_rate=0.001, patience=10):
        """
        训练GRU模型
        
        参数:
            train_data: 训练数据字典
            input_dim: 输入维度
            hidden_dim: 隐藏层维度
            num_layers: GRU层数
            output_dim: 输出维度
            num_epochs: 训练轮数
            learning_rate: 学习率
            patience: 早停耐心值
            
        返回:
            model: 训练后的模型
            history: 训练历史
        """
        print("开始训练GRU模型...")
        
        # 提取数据加载器
        train_loader = train_data['train_loader']
        val_loader = train_data['val_loader']
        
        # 创建模型
        model = GRUModel(input_dim, hidden_dim, num_layers, output_dim)
        model = model.to(self.device)
        
        # 定义损失函数和优化器
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # 训练历史
        history = {
            'train_loss': [],
            'val_loss': []
        }
        
        # 早停
        best_val_loss = float('inf')
        counter = 0
        
        # 训练循环
        for epoch in range(num_epochs):
            # 训练模式
            model.train()
            train_loss = 0.0
            
            for inputs, targets in train_loader:
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
            
            # 计算平均训练损失
            train_loss /= len(train_loader)
            history['train_loss'].append(train_loss)
            
            # 验证
            if val_loader is not None:
                model.eval()
                val_loss = 0.0
                
                with torch.no_grad():
                    for inputs, targets in val_loader:
                        inputs = inputs.to(self.device)
                        targets = targets.to(self.device)
                        
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                        
                        val_loss += loss.item()
                
                # 计算平均验证损失
                val_loss /= len(val_loader)
                history['val_loss'].append(val_loss)
                
                # 早停检查
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    counter = 0
                    # 保存最佳模型
                    best_model = model.state_dict().copy()
                else:
                    counter += 1
                
                print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
                
                if counter >= patience:
                    print(f"早停触发，在{epoch+1}轮停止训练")
                    # 恢复最佳模型
                    model.load_state_dict(best_model)
                    break
            else:
                print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}")
        
        print("GRU模型训练完成")
        return model, history
    
    def train_transformer_model(self, train_data, input_dim, d_model=64, nhead=4, num_layers=2, 
                               dim_feedforward=128, output_dim=3, num_epochs=50, 
                               learning_rate=0.001, patience=10):
        """
        训练Transformer模型
        
        参数:
            train_data: 训练数据字典
            input_dim: 输入维度
            d_model: 模型维度
            nhead: 注意力头数
            num_layers: Transformer层数
            dim_feedforward: 前馈网络维度
            output_dim: 输出维度
            num_epochs: 训练轮数
            learning_rate: 学习率
            patience: 早停耐心值
            
        返回:
            model: 训练后的模型
            history: 训练历史
        """
        print("开始训练Transformer模型...")
        
        # 提取数据加载器
        train_loader = train_data['train_loader']
        val_loader = train_data['val_loader']
        
        # 创建模型
        model = TransformerModel(input_dim, d_model, nhead, num_layers, dim_feedforward, output_dim)
        model = model.to(self.device)
        
        # 定义损失函数和优化器
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # 训练历史
        history = {
            'train_loss': [],
            'val_loss': []
        }
        
        # 早停
        best_val_loss = float('inf')
        counter = 0
        
        # 训练循环
        for epoch in range(num_epochs):
            # 训练模式
            model.train()
            train_loss = 0.0
            
            for inputs, targets in train_loader:
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
            
            # 计算平均训练损失
            train_loss /= len(train_loader)
            history['train_loss'].append(train_loss)
            
            # 验证
            if val_loader is not None:
                model.eval()
                val_loss = 0.0
                
                with torch.no_grad():
                    for inputs, targets in val_loader:
                        inputs = inputs.to(self.device)
                        targets = targets.to(self.device)
                        
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                        
                        val_loss += loss.item()
                
                # 计算平均验证损失
                val_loss /= len(val_loader)
                history['val_loss'].append(val_loss)
                
                # 早停检查
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    counter = 0
                    # 保存最佳模型
                    best_model = model.state_dict().copy()
                else:
                    counter += 1
                
                print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
                
                if counter >= patience:
                    print(f"早停触发，在{epoch+1}轮停止训练")
                    # 恢复最佳模型
                    model.load_state_dict(best_model)
                    break
            else:
                print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}")
        
        print("Transformer模型训练完成")
        return model, history
    
    def evaluate_model(self, model, test_data, scalers=None):
        """
        评估模型
        
        参数:
            model: 模型
            test_data: 测试数据字典
            scalers: 缩放器字典
            
        返回:
            dict: 评估结果
        """
        print("评估模型...")
        
        # 提取测试数据
        test_loader = test_data['test_loader']
        target_cols = test_data['target_cols']
        
        # 评估模式
        model.eval()
        
        # 存储预测和真实值
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to(self.device)
                
                # 前向传播
                outputs = model(inputs)
                
                # 转换为CPU并转换为numpy数组
                predictions = outputs.cpu().numpy()
                targets = targets.cpu().numpy()
                
                all_predictions.append(predictions)
                all_targets.append(targets)
        
        # 合并批次
        all_predictions = np.vstack(all_predictions)
        all_targets = np.vstack(all_targets)
        
        # 如果使用了缩放器，则反向转换
        if scalers is not None:
            y_scaler = scalers.get('y_scaler')
            if y_scaler is not None:
                all_predictions = y_scaler.inverse_transform(all_predictions)
                all_targets = y_scaler.inverse_transform(all_targets)
        
        # 计算评估指标
        mse = mean_squared_error(all_targets, all_predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(all_targets, all_predictions)
        r2 = r2_score(all_targets, all_predictions)
        
        # 计算每个目标的指标
        metrics_by_target = {}
        for i, col in enumerate(target_cols):
            target_mse = mean_squared_error(all_targets[:, i], all_predictions[:, i])
            target_rmse = np.sqrt(target_mse)
            target_mae = mean_absolute_error(all_targets[:, i], all_predictions[:, i])
            target_r2 = r2_score(all_targets[:, i], all_predictions[:, i])
            
            metrics_by_target[col] = {
                'mse': target_mse,
                'rmse': target_rmse,
                'mae': target_mae,
                'r2': target_r2
            }
        
        # 创建结果字典
        results = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'metrics_by_target': metrics_by_target,
            'predictions': all_predictions,
            'targets': all_targets
        }
        
        print(f"模型评估完成，MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
        return results
    
    def save_model(self, model, train_data, history, results, model_path):
        """
        保存模型
        
        参数:
            model: 模型
            train_data: 训练数据字典
            history: 训练历史
            results: 评估结果
            model_path: 模型保存路径
            
        返回:
            bool: 是否保存成功
        """
        print(f"保存模型到: {model_path}")
        
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            # 保存模型状态
            torch.save(model.state_dict(), model_path)
            
            # 保存模型元数据
            metadata_path = os.path.splitext(model_path)[0] + '_metadata.joblib'
            
            # 提取需要保存的数据
            metadata = {
                'target_cols': train_data['target_cols'],
                'scalers': train_data['scalers'],
                'history': history,
                'results': {
                    'mse': results['mse'],
                    'rmse': results['rmse'],
                    'mae': results['mae'],
                    'r2': results['r2'],
                    'metrics_by_target': results['metrics_by_target']
                },
                'model_type': model.__class__.__name__,
                'model_params': {
                    'input_dim': model.lstm.input_size if hasattr(model, 'lstm') else 
                                (model.gru.input_size if hasattr(model, 'gru') else 
                                 model.input_linear.in_features),
                    'hidden_dim': model.hidden_dim if hasattr(model, 'hidden_dim') else None,
                    'num_layers': model.num_layers if hasattr(model, 'num_layers') else None,
                    'output_dim': model.fc.out_features if hasattr(model, 'fc') else model.output_linear.out_features
                },
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            joblib.dump(metadata, metadata_path)
            
            print(f"模型和元数据保存成功")
            return True
        except Exception as e:
            print(f"保存模型失败: {e}")
            return False
    
    def load_model(self, model_path):
        """
        加载模型
        
        参数:
            model_path: 模型路径
            
        返回:
            model: 模型
            metadata: 元数据
        """
        print(f"加载模型: {model_path}")
        
        # 检查模型文件是否存在
        if os.path.exists(model_path):
            try:
                # 加载模型文件
                checkpoint = torch.load(model_path, map_location=self.device)
                
                # 提取元数据
                metadata = checkpoint.get('metadata', {})
                
                # 创建模型
                model_type = metadata.get('model_type', 'lstm')
                input_dim = metadata.get('input_dim', 10)
                output_dim = metadata.get('output_dim', 3)
                
                if model_type == 'lstm':
                    hidden_dim = metadata.get('hidden_dim', 64)
                    num_layers = metadata.get('num_layers', 2)
                    dropout = metadata.get('dropout', 0.2)
                    
                    model = LSTMModel(input_dim, hidden_dim, num_layers, output_dim, dropout)
                elif model_type == 'gru':
                    hidden_dim = metadata.get('hidden_dim', 64)
                    num_layers = metadata.get('num_layers', 2)
                    dropout = metadata.get('dropout', 0.2)
                    
                    model = GRUModel(input_dim, hidden_dim, num_layers, output_dim, dropout)
                elif model_type == 'transformer':
                    d_model = metadata.get('d_model', 64)
                    nhead = metadata.get('nhead', 4)
                    num_layers = metadata.get('num_layers', 2)
                    dim_feedforward = metadata.get('dim_feedforward', 128)
                    dropout = metadata.get('dropout', 0.1)
                    
                    model = TransformerModel(input_dim, d_model, nhead, num_layers, dim_feedforward, output_dim, dropout)
                else:
                    print(f"不支持的模型类型: {model_type}")
                    return None, None
                
                # 加载模型参数
                model.load_state_dict(checkpoint['model_state_dict'])
                model.to(self.device)
                model.eval()
                
                print(f"成功加载模型: {model_type}")
                return model, metadata
            except Exception as e:
                print(f"加载模型失败: {e}")
                print("使用模拟模型")
                return self._create_mock_model(), self._create_mock_metadata()
        else:
            print(f"模型文件不存在: {model_path}")
            print("使用模拟模型")
            return self._create_mock_model(), self._create_mock_metadata()
    
    def _create_mock_model(self):
        """
        创建模拟模型用于离线模式
        
        返回:
            MockPollutionModel: 模拟模型
        """
        try:
            from .mock_models import MockPollutionModel
            return MockPollutionModel()
        except ImportError:
            print("警告: 无法导入MockPollutionModel，使用内联模拟实现")
            # 使用非常简单的内联实现
            class SimpleMockModel:
                def to(self, device): return self
                def eval(self): return self
                def __call__(self, x): 
                    return torch.tensor(np.random.random((1 if not hasattr(x, "shape") else x.shape[0], 3)), dtype=torch.float32)
            return SimpleMockModel()
    
    def _create_mock_metadata(self):
        """
        创建模拟元数据用于离线模式
        
        返回:
            dict: 模拟元数据
        """
        return {
            'model_type': 'mock',
            'input_dim': 10,
            'hidden_dim': 64,
            'output_dim': 3,
            'num_layers': 2,
            'feature_cols': ['load', 'temperature', 'humidity', 'wind_speed', 
                             'hour', 'day', 'month', 'year', 'dayofweek', 'SO2_lag1'],
            'target_cols': ['SO2', 'NOx', 'dust'],
            'window_size': 24,
            'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'scaler_info': None
        }
    
    def predict(self, model, inputs, metadata=None):
        """
        使用模型进行预测
        
        参数:
            model: 模型
            inputs: 输入特征
            metadata: 模型元数据
            
        返回:
            numpy.ndarray: 预测结果
        """
        # 确保模型处于评估模式
        model.eval()
        
        # 如果输入不是张量，则转换为张量
        if not isinstance(inputs, torch.Tensor):
            # 如果有缩放器，则应用缩放
            if metadata and 'scalers' in metadata and metadata['scalers']:
                X_scaler = metadata['scalers'].get('X_scaler')
                if X_scaler:
                    inputs = X_scaler.transform(inputs)
            
            inputs = torch.FloatTensor(inputs).to(self.device)
        
        # 进行预测
        with torch.no_grad():
            outputs = model(inputs)
        
        # 转换为numpy数组
        predictions = outputs.cpu().numpy()
        
        # 如果有缩放器，则反向转换
        if metadata and 'scalers' in metadata and metadata['scalers']:
            y_scaler = metadata['scalers'].get('y_scaler')
            if y_scaler:
                predictions = y_scaler.inverse_transform(predictions)
        
        return predictions
    
    def plot_training_history(self, history, output_path=None):
        """
        绘制训练历史
        
        参数:
            history: 训练历史
            output_path: 输出路径
            
        返回:
            matplotlib.figure.Figure: 图形对象
        """
        plt.figure(figsize=(10, 6))
        plt.plot(history['train_loss'], label='训练损失')
        
        if 'val_loss' in history and history['val_loss']:
            plt.plot(history['val_loss'], label='验证损失')
        
        plt.title('模型训练历史')
        plt.xlabel('轮次')
        plt.ylabel('损失')
        plt.legend()
        plt.grid(True)
        
        if output_path:
            plt.savefig(output_path)
            print(f"训练历史图已保存到: {output_path}")
        
        return plt.gcf()
    
    def plot_predictions(self, results, target_cols, output_path=None):
        """
        绘制预测结果
        
        参数:
            results: 评估结果
            target_cols: 目标列列表
            output_path: 输出路径
            
        返回:
            matplotlib.figure.Figure: 图形对象
        """
        predictions = results['predictions']
        targets = results['targets']
        
        n_targets = len(target_cols)
        fig, axes = plt.subplots(n_targets, 1, figsize=(12, 4 * n_targets))
        
        if n_targets == 1:
            axes = [axes]
        
        for i, col in enumerate(target_cols):
            ax = axes[i]
            
            # 绘制真实值和预测值
            ax.plot(targets[:, i], label='真实值', color='blue')
            ax.plot(predictions[:, i], label='预测值', color='red')
            
            ax.set_title(f'{col} 预测结果')
            ax.set_xlabel('样本')
            ax.set_ylabel('值')
            ax.legend()
            ax.grid(True)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path)
            print(f"预测结果图已保存到: {output_path}")
        
        return fig
    
    def train_and_evaluate_all_models(self, dataset_path, output_dir, target_cols=None):
        """
        训练和评估所有模型
        
        参数:
            dataset_path: 数据集路径
            output_dir: 输出目录
            target_cols: 目标列列表
            
        返回:
            dict: 包含所有模型结果的字典
        """
        print(f"训练和评估所有模型，数据集: {dataset_path}")
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 加载数据集
        data = self.load_dataset(dataset_path)
        if data is None:
            return None
        
        # 如果未提供目标列，则使用数据中的目标列
        if target_cols is None:
            if 'target_cols' in data:
                target_cols = data['target_cols']
            else:
                print("未提供目标列，且数据中不包含目标列信息")
                return None
        
        # 准备训练数据
        train_data = self.prepare_data_for_training(data, target_cols)
        if train_data is None:
            return None
        
        # 获取输入维度和输出维度
        input_dim = train_data['X_train'].shape[1]
        output_dim = train_data['y_train'].shape[1]
        
        # 训练LSTM模型
        lstm_model, lstm_history = self.train_lstm_model(
            train_data, 
            input_dim=input_dim, 
            output_dim=output_dim
        )
        
        # 评估LSTM模型
        lstm_results = self.evaluate_model(lstm_model, train_data, train_data['scalers'])
        
        # 保存LSTM模型
        lstm_model_path = os.path.join(output_dir, 'lstm_model.pth')
        self.save_model(lstm_model, train_data, lstm_history, lstm_results, lstm_model_path)
        
        # 绘制LSTM训练历史
        lstm_history_path = os.path.join(output_dir, 'lstm_history.png')
        self.plot_training_history(lstm_history, lstm_history_path)
        
        # 绘制LSTM预测结果
        lstm_predictions_path = os.path.join(output_dir, 'lstm_predictions.png')
        self.plot_predictions(lstm_results, target_cols, lstm_predictions_path)
        
        # 训练GRU模型
        gru_model, gru_history = self.train_gru_model(
            train_data, 
            input_dim=input_dim, 
            output_dim=output_dim
        )
        
        # 评估GRU模型
        gru_results = self.evaluate_model(gru_model, train_data, train_data['scalers'])
        
        # 保存GRU模型
        gru_model_path = os.path.join(output_dir, 'gru_model.pth')
        self.save_model(gru_model, train_data, gru_history, gru_results, gru_model_path)
        
        # 绘制GRU训练历史
        gru_history_path = os.path.join(output_dir, 'gru_history.png')
        self.plot_training_history(gru_history, gru_history_path)
        
        # 绘制GRU预测结果
        gru_predictions_path = os.path.join(output_dir, 'gru_predictions.png')
        self.plot_predictions(gru_results, target_cols, gru_predictions_path)
        
        # 训练Transformer模型
        transformer_model, transformer_history = self.train_transformer_model(
            train_data, 
            input_dim=input_dim, 
            output_dim=output_dim
        )
        
        # 评估Transformer模型
        transformer_results = self.evaluate_model(transformer_model, train_data, train_data['scalers'])
        
        # 保存Transformer模型
        transformer_model_path = os.path.join(output_dir, 'transformer_model.pth')
        self.save_model(transformer_model, train_data, transformer_history, transformer_results, transformer_model_path)
        
        # 绘制Transformer训练历史
        transformer_history_path = os.path.join(output_dir, 'transformer_history.png')
        self.plot_training_history(transformer_history, transformer_history_path)
        
        # 绘制Transformer预测结果
        transformer_predictions_path = os.path.join(output_dir, 'transformer_predictions.png')
        self.plot_predictions(transformer_results, target_cols, transformer_predictions_path)
        
        # 比较所有模型
        models_comparison = {
            'lstm': {
                'mse': lstm_results['mse'],
                'rmse': lstm_results['rmse'],
                'mae': lstm_results['mae'],
                'r2': lstm_results['r2']
            },
            'gru': {
                'mse': gru_results['mse'],
                'rmse': gru_results['rmse'],
                'mae': gru_results['mae'],
                'r2': gru_results['r2']
            },
            'transformer': {
                'mse': transformer_results['mse'],
                'rmse': transformer_results['rmse'],
                'mae': transformer_results['mae'],
                'r2': transformer_results['r2']
            }
        }
        
        # 保存模型比较结果
        comparison_path = os.path.join(output_dir, 'models_comparison.json')
        with open(comparison_path, 'w', encoding='utf-8') as f:
            json.dump(models_comparison, f, ensure_ascii=False, indent=2)
        
        print(f"所有模型训练和评估完成，结果已保存到: {output_dir}")
        
        # 创建结果字典
        results = {
            'lstm': {
                'model': lstm_model,
                'history': lstm_history,
                'results': lstm_results,
                'model_path': lstm_model_path
            },
            'gru': {
                'model': gru_model,
                'history': gru_history,
                'results': gru_results,
                'model_path': gru_model_path
            },
            'transformer': {
                'model': transformer_model,
                'history': transformer_history,
                'results': transformer_results,
                'model_path': transformer_model_path
            },
            'comparison': models_comparison
        }
        
        return results

# 测试代码
if __name__ == "__main__":
    # 创建污染物排放预测模型
    predictor = PollutionPredictionModel()
    
    # 设置输入和输出目录
    data_dir = "./data"
    models_dir = "./data/models/prediction"
    
    # 加载数据集
    dataset_path = os.path.join(data_dir, "datasets", "pollution_prediction_dataset.joblib")
    
    # 定义目标列
    target_cols = ['VENT_SO2_CHK', 'VENT_NOX_CHK', 'VENT_SOOT_CHK']
    
    # 训练和评估所有模型
    if os.path.exists(dataset_path):
        results = predictor.train_and_evaluate_all_models(
            dataset_path,
            models_dir,
            target_cols
        )

# 如果直接运行此脚本
if __name__ == "__main__":
    # 模型测试代码
    model = PollutionPredictionModel(data_dir='./data')
    
    # 测试加载模型
    model_path = os.path.join('./data', 'models', 'prediction', 'pollution_model.pth')
    loaded_model, metadata = model.load_model(model_path)
    
    if loaded_model is not None:
        print("模型加载成功")
        print(f"元数据: {metadata}")
    else:
        print("模型加载失败")

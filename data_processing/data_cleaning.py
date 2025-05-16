#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据清洗模块 - 处理时序数据和污染物排放数据
"""

import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class DataCleaner:
    """
    数据清洗类，用于处理时序数据和污染物排放数据
    """
    
    def __init__(self, data_dir='./data'):
        """
        初始化数据清洗类
        
        参数:
            data_dir: 数据目录
        """
        self.data_dir = data_dir
        # 创建数据目录（如果不存在）
        os.makedirs(os.path.join(data_dir, 'processed'), exist_ok=True)
        os.makedirs(os.path.join(data_dir, 'raw'), exist_ok=True)
        
    def load_time_series_data(self, file_path):
        """
        加载时序数据文件
        
        参数:
            file_path: 时序数据文件路径
            
        返回:
            DataFrame: 处理后的时序数据
        """
        print(f"加载时序数据: {file_path}")
        
        try:
            # 读取时序数据，第一列为时间
            df = pd.read_csv(file_path, sep='\t', header=0, index_col=0)
            
            # 转换索引为datetime格式
            df.index = pd.to_datetime(df.index, format='%Y/%m/%d %H:%M')
            
            print(f"成功加载时序数据，形状: {df.shape}")
            return df
        except Exception as e:
            print(f"加载时序数据失败: {e}")
            return None
    
    def load_emission_data(self, file_path):
        """
        加载污染物排放数据
        
        参数:
            file_path: 污染物排放数据文件路径
            
        返回:
            DataFrame: 处理后的污染物排放数据
        """
        print(f"加载污染物排放数据: {file_path}")
        
        try:
            # 读取CSV文件，假设第一行为字段名，第二行为字段描述
            df = pd.read_csv(file_path, header=0, skiprows=[1])
            
            # 转换时间列为datetime格式
            if 'RECTIME' in df.columns:
                df['RECTIME'] = pd.to_datetime(df['RECTIME'])
                
            print(f"成功加载污染物排放数据，形状: {df.shape}")
            return df
        except Exception as e:
            print(f"加载污染物排放数据失败: {e}")
            return None
    
    def clean_time_series_data(self, df):
        """
        清洗时序数据
        
        参数:
            df: 时序数据DataFrame
            
        返回:
            DataFrame: 清洗后的时序数据
        """
        if df is None or df.empty:
            print("无数据需要清洗")
            return None
        
        print("开始清洗时序数据...")
        
        # 复制数据，避免修改原始数据
        cleaned_df = df.copy()
        
        # 1. 处理缺失值（'-'表示缺失）
        cleaned_df = cleaned_df.replace('-', np.nan)
        
        # 2. 转换数据类型为float
        for col in cleaned_df.columns:
            try:
                cleaned_df[col] = cleaned_df[col].astype(float)
            except:
                print(f"列 {col} 无法转换为float类型，保持原始类型")
        
        # 3. 统计缺失值情况
        missing_stats = cleaned_df.isna().sum()
        missing_percent = (missing_stats / len(cleaned_df)) * 100
        
        print("缺失值统计:")
        for col, count in missing_stats.items():
            if count > 0:
                print(f"  {col}: {count} 缺失值 ({missing_percent[col]:.2f}%)")
        
        # 4. 处理缺失值
        # 对于缺失比例低于30%的列，使用前向填充和后向填充
        for col in cleaned_df.columns:
            if missing_percent[col] < 30:
                cleaned_df[col] = cleaned_df[col].fillna(method='ffill').fillna(method='bfill')
        
        # 5. 处理异常值（使用3倍标准差法）
        for col in cleaned_df.columns:
            if cleaned_df[col].dtype == float:
                # 计算均值和标准差
                mean = cleaned_df[col].mean()
                std = cleaned_df[col].std()
                
                # 定义上下限
                lower_bound = mean - 3 * std
                upper_bound = mean + 3 * std
                
                # 将超出范围的值替换为上下限
                outliers = ((cleaned_df[col] < lower_bound) | (cleaned_df[col] > upper_bound))
                outlier_count = outliers.sum()
                
                if outlier_count > 0:
                    print(f"  列 {col} 中发现 {outlier_count} 个异常值")
                    cleaned_df.loc[cleaned_df[col] < lower_bound, col] = lower_bound
                    cleaned_df.loc[cleaned_df[col] > upper_bound, col] = upper_bound
        
        # 6. 重采样为小时数据（如果原始数据为分钟级）
        # 检查索引是否为DatetimeIndex类型，避免RangeIndex没有freq属性的错误
        if isinstance(cleaned_df.index, pd.DatetimeIndex):
            # 检查索引是否有频率属性，如果没有频率则尝试推断
            if not hasattr(cleaned_df.index, 'freq') or cleaned_df.index.freq is None:
                try:
                    # 尝试推断频率
                    cleaned_df = cleaned_df.asfreq('H')
                except ValueError:
                    print("无法推断数据频率，跳过重采样步骤")
            elif cleaned_df.index.freq != 'H':
                print("将数据重采样为小时级别")
                cleaned_df = cleaned_df.resample('H').mean()
        else:
            print("索引不是DatetimeIndex类型，跳过重采样步骤")
        
        print(f"时序数据清洗完成，形状: {cleaned_df.shape}")
        return cleaned_df
    
    def clean_emission_data(self, df):
        """
        清洗污染物排放数据
        
        参数:
            df: 污染物排放数据DataFrame
            
        返回:
            DataFrame: 清洗后的污染物排放数据
        """
        if df is None or df.empty:
            print("无数据需要清洗")
            return None
        
        print("开始清洗污染物排放数据...")
        
        # 复制数据，避免修改原始数据
        cleaned_df = df.copy()
        
        # 1. 处理缺失值
        missing_stats = cleaned_df.isna().sum()
        missing_percent = (missing_stats / len(cleaned_df)) * 100
        
        print("缺失值统计:")
        for col, count in missing_stats.items():
            if count > 0:
                print(f"  {col}: {count} 缺失值 ({missing_percent[col]:.2f}%)")
        
        # 2. 对数值列进行处理
        numeric_cols = cleaned_df.select_dtypes(include=['int64', 'float64']).columns
        
        for col in numeric_cols:
            # 使用中位数填充缺失值
            if missing_percent[col] > 0:
                median_value = cleaned_df[col].median()
                cleaned_df[col] = cleaned_df[col].fillna(median_value)
            
            # 处理异常值（使用3倍标准差法）
            mean = cleaned_df[col].mean()
            std = cleaned_df[col].std()
            
            lower_bound = mean - 3 * std
            upper_bound = mean + 3 * std
            
            outliers = ((cleaned_df[col] < lower_bound) | (cleaned_df[col] > upper_bound))
            outlier_count = outliers.sum()
            
            if outlier_count > 0:
                print(f"  列 {col} 中发现 {outlier_count} 个异常值")
                cleaned_df.loc[cleaned_df[col] < lower_bound, col] = lower_bound
                cleaned_df.loc[cleaned_df[col] > upper_bound, col] = upper_bound
        
        # 3. 对分类列进行处理
        categorical_cols = cleaned_df.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            # 使用众数填充缺失值
            if missing_percent[col] > 0:
                mode_value = cleaned_df[col].mode()[0]
                cleaned_df[col] = cleaned_df[col].fillna(mode_value)
        
        print(f"污染物排放数据清洗完成，形状: {cleaned_df.shape}")
        return cleaned_df
    
    def normalize_data(self, df, method='standard'):
        """
        数据标准化/归一化
        
        参数:
            df: 数据DataFrame
            method: 标准化方法，'standard'或'minmax'
            
        返回:
            DataFrame: 标准化后的数据
            object: 标准化器（用于后续转换）
        """
        if df is None or df.empty:
            print("无数据需要标准化")
            return None, None
        
        print(f"使用{method}方法进行数据标准化...")
        
        # 复制数据，避免修改原始数据
        normalized_df = df.copy()
        
        # 选择数值列
        numeric_cols = normalized_df.select_dtypes(include=['int64', 'float64']).columns
        
        # 创建标准化器
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        else:
            print(f"不支持的标准化方法: {method}")
            return normalized_df, None
        
        # 对数值列进行标准化
        if len(numeric_cols) > 0:
            normalized_values = scaler.fit_transform(normalized_df[numeric_cols])
            normalized_df[numeric_cols] = normalized_values
            
            print(f"已标准化 {len(numeric_cols)} 列数据")
        else:
            print("没有找到数值列，无需标准化")
        
        return normalized_df, scaler
    
    def add_time_features(self, df):
        """
        添加时间特征
        
        参数:
            df: 带有时间索引的DataFrame
            
        返回:
            DataFrame: 添加时间特征后的数据
        """
        if df is None or df.empty:
            print("无数据需要添加时间特征")
            return None
        
        print("添加时间特征...")
        
        # 复制数据，避免修改原始数据
        enhanced_df = df.copy()
        
        # 确保索引是datetime类型
        if not isinstance(enhanced_df.index, pd.DatetimeIndex):
            print("索引不是datetime类型，无法添加时间特征")
            return enhanced_df
        
        # 添加时间特征
        enhanced_df['hour'] = enhanced_df.index.hour
        enhanced_df['day'] = enhanced_df.index.day
        enhanced_df['month'] = enhanced_df.index.month
        enhanced_df['year'] = enhanced_df.index.year
        enhanced_df['dayofweek'] = enhanced_df.index.dayofweek
        enhanced_df['quarter'] = enhanced_df.index.quarter
        enhanced_df['is_weekend'] = enhanced_df.index.dayofweek >= 5
        
        # 添加季节特征
        enhanced_df['season'] = enhanced_df['month'].apply(lambda x: 
                                                         1 if x in [3, 4, 5] else  # 春季
                                                         2 if x in [6, 7, 8] else  # 夏季
                                                         3 if x in [9, 10, 11] else  # 秋季
                                                         4)  # 冬季
        
        # 添加小时周期特征
        enhanced_df['hour_sin'] = np.sin(2 * np.pi * enhanced_df['hour'] / 24)
        enhanced_df['hour_cos'] = np.cos(2 * np.pi * enhanced_df['hour'] / 24)
        
        # 添加日期周期特征
        days_in_month = enhanced_df.index.daysinmonth
        enhanced_df['day_sin'] = np.sin(2 * np.pi * enhanced_df['day'] / days_in_month)
        enhanced_df['day_cos'] = np.cos(2 * np.pi * enhanced_df['day'] / days_in_month)
        
        print(f"已添加时间特征，新形状: {enhanced_df.shape}")
        return enhanced_df
    
    def visualize_data(self, df, output_dir, prefix=''):
        """
        数据可视化
        
        参数:
            df: 数据DataFrame
            output_dir: 输出目录
            prefix: 文件名前缀
        """
        if df is None or df.empty:
            print("无数据需要可视化")
            return
        
        print("生成数据可视化...")
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. 绘制时间序列图（最多显示10个列）
        if isinstance(df.index, pd.DatetimeIndex):
            plt.figure(figsize=(15, 8))
            
            # 选择最多10个数值列
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns[:10]
            
            for col in numeric_cols:
                plt.plot(df.index, df[col], label=col)
            
            plt.title('时间序列数据')
            plt.xlabel('时间')
            plt.ylabel('值')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{prefix}time_series.png'))
            plt.close()
        
        # 2. 绘制相关性热力图
        numeric_df = df.select_dtypes(include=['int64', 'float64'])
        
        if numeric_df.shape[1] > 1:
            plt.figure(figsize=(12, 10))
            corr_matrix = numeric_df.corr()
            sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', linewidths=0.5)
            plt.title('特征相关性热力图')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{prefix}correlation.png'))
            plt.close()
        
        # 3. 绘制箱线图（最多显示10个列）
        if numeric_df.shape[1] > 0:
            plt.figure(figsize=(15, 8))
            numeric_df.iloc[:, :10].boxplot()
            plt.title('数值特征箱线图')
            plt.xticks(rotation=90)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{prefix}boxplot.png'))
            plt.close()
        
        # 4. 绘制直方图（最多显示9个列）
        if numeric_df.shape[1] > 0:
            cols = min(9, numeric_df.shape[1])
            rows = (cols + 2) // 3
            
            plt.figure(figsize=(15, rows * 4))
            
            for i, col in enumerate(numeric_df.columns[:cols]):
                plt.subplot(rows, 3, i + 1)
                sns.histplot(numeric_df[col], kde=True)
                plt.title(col)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{prefix}histogram.png'))
            plt.close()
        
        print(f"数据可视化已保存到 {output_dir}")
    
    def save_processed_data(self, df, output_path):
        """
        保存处理后的数据
        
        参数:
            df: 数据DataFrame
            output_path: 输出文件路径
        """
        if df is None or df.empty:
            print("无数据需要保存")
            return
        
        print(f"保存处理后的数据到: {output_path}")
        
        # 确保输出目录存在
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存为CSV文件
        df.to_csv(output_path)
        
        print(f"数据已保存，形状: {df.shape}")
    
    def process_all_time_series_data(self, input_dir, output_dir):
        """
        处理所有时序数据文件
        
        参数:
            input_dir: 输入目录
            output_dir: 输出目录
        """
        print(f"处理目录 {input_dir} 中的所有时序数据文件...")
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 获取所有txt文件
        file_pattern = os.path.join(input_dir, "*.txt")
        files = glob.glob(file_pattern)
        
        if not files:
            print(f"在 {input_dir} 中没有找到时序数据文件")
            return
        
        print(f"找到 {len(files)} 个时序数据文件")
        
        # 处理每个文件
        for file_path in files:
            file_name = os.path.basename(file_path)
            print(f"处理文件: {file_name}")
            
            # 加载数据
            df = self.load_time_series_data(file_path)
            
            if df is not None:
                # 清洗数据
                cleaned_df = self.clean_time_series_data(df)
                
                # 添加时间特征
                enhanced_df = self.add_time_features(cleaned_df)
                
                # 标准化数据
                normalized_df, _ = self.normalize_data(enhanced_df)
                
                # 可视化数据
                vis_output_dir = os.path.join(output_dir, 'visualizations')
                self.visualize_data(normalized_df, vis_output_dir, prefix=f"{file_name.split('.')[0]}_")
                
                # 保存处理后的数据
                output_path = os.path.join(output_dir, f"processed_{file_name.split('.')[0]}.csv")
                self.save_processed_data(normalized_df, output_path)
        
        print("所有时序数据文件处理完成")
    
    def process_all_emission_data(self, input_dir, output_dir):
        """
        处理所有污染物排放数据文件
        
        参数:
            input_dir: 输入目录
            output_dir: 输出目录
        """
        print(f"处理目录 {input_dir} 中的所有污染物排放数据文件...")
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 获取所有csv文件
        file_pattern = os.path.join(input_dir, "*.csv")
        files = glob.glob(file_pattern)
        
        if not files:
            print(f"在 {input_dir} 中没有找到污染物排放数据文件")
            return
        
        print(f"找到 {len(files)} 个污染物排放数据文件")
        
        # 处理每个文件
        for file_path in files:
            file_name = os.path.basename(file_path)
            print(f"处理文件: {file_name}")
            
            # 加载数据
            df = self.load_emission_data(file_path)
            
            if df is not None:
                # 清洗数据
                cleaned_df = self.clean_emission_data(df)
                
                # 标准化数据
                normalized_df, _ = self.normalize_data(cleaned_df)
                
                # 可视化数据
                vis_output_dir = os.path.join(output_dir, 'visualizations')
                self.visualize_data(normalized_df, vis_output_dir, prefix=f"{file_name.split('.')[0]}_")
                
                # 保存处理后的数据
                output_path = os.path.join(output_dir, f"processed_{file_name}")
                self.save_processed_data(normalized_df, output_path)
        
        print("所有污染物排放数据文件处理完成")

# 测试代码
if __name__ == "__main__":
    # 创建数据清洗器
    cleaner = DataCleaner()
    
    # 设置输入和输出目录
    input_dir = "./data/raw"
    output_dir = "./data/processed"
    
    print("=== 开始数据清洗处理 ===")
    
    # 处理时序数据示例文件
    time_series_file = os.path.join(input_dir, "time_series/sample_time_series.txt")
    if os.path.exists(time_series_file):
        print(f"正在处理时序数据文件: {time_series_file}")
        # 加载数据
        df = cleaner.load_time_series_data(time_series_file)
        
        if df is not None:
            # 清洗数据
            cleaned_df = cleaner.clean_time_series_data(df)
            
            # 添加时间特征
            enhanced_df = cleaner.add_time_features(cleaned_df)
            
            # 标准化数据
            normalized_df, _ = cleaner.normalize_data(enhanced_df)
            
            # 可视化数据
            vis_output_dir = os.path.join(output_dir, 'visualizations')
            os.makedirs(vis_output_dir, exist_ok=True)
            cleaner.visualize_data(normalized_df, vis_output_dir, prefix="time_series_")
            
            # 保存处理后的数据
            output_path = os.path.join(output_dir, "processed_time_series.csv")
            cleaner.save_processed_data(normalized_df, output_path)
            print(f"时序数据处理完成，已保存到: {output_path}")
    else:
        print(f"时序数据文件不存在: {time_series_file}")
    
    # 处理污染物排放小时表
    emission_file = os.path.join(input_dir, "emission/sample_emission.csv")
    if os.path.exists(emission_file):
        print(f"正在处理排放数据文件: {emission_file}")
        # 加载数据
        df = cleaner.load_emission_data(emission_file)
        
        if df is not None:
            # 清洗数据
            cleaned_df = cleaner.clean_emission_data(df)
            
            # 标准化数据
            normalized_df, _ = cleaner.normalize_data(cleaned_df)
            
            # 可视化数据
            vis_output_dir = os.path.join(output_dir, 'visualizations')
            os.makedirs(vis_output_dir, exist_ok=True)
            cleaner.visualize_data(normalized_df, vis_output_dir, prefix="emission_")
            
            # 保存处理后的数据
            output_path = os.path.join(output_dir, "processed_emission.csv")
            cleaner.save_processed_data(normalized_df, output_path)
            print(f"排放数据处理完成，已保存到: {output_path}")
    else:
        print(f"排放数据文件不存在: {emission_file}")
    
    print("=== 数据清洗处理完成 ===")

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
特征工程模块 - 提取时序数据特征和构建预测模型输入
"""

import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime, timedelta
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression
import joblib

class FeatureEngineering:
    """
    特征工程类，用于提取时序数据特征和构建预测模型输入
    """
    
    def __init__(self, data_dir='./data'):
        """
        初始化特征工程类
        
        参数:
            data_dir: 数据目录
        """
        self.data_dir = data_dir
        # 创建特征目录（如果不存在）
        os.makedirs(os.path.join(data_dir, 'features'), exist_ok=True)
    
    def create_time_features(self, df, datetime_col):
        """
        从时间戳列创建时间特征
        
        参数:
            df: 数据DataFrame
            datetime_col: 时间戳列名
            
        返回:
            DataFrame: 包含时间特征的数据
        """
        print(f"从列 {datetime_col} 创建时间特征...")
        
        if df is None or df.empty:
            print("无数据需要创建时间特征")
            return df
        
        # 复制数据，避免修改原始数据
        feature_df = df.copy()
        
        # 确保datetime_col是datetime类型
        if datetime_col in feature_df.columns:
            if not pd.api.types.is_datetime64_any_dtype(feature_df[datetime_col]):
                try:
                    feature_df[datetime_col] = pd.to_datetime(feature_df[datetime_col])
                except Exception as e:
                    print(f"转换时间戳列失败: {e}")
                    return feature_df
        else:
            print(f"列 {datetime_col} 不存在")
            return feature_df
        
        # 提取时间特征
        feature_df['hour'] = feature_df[datetime_col].dt.hour
        feature_df['dayofweek'] = feature_df[datetime_col].dt.dayofweek
        feature_df['month'] = feature_df[datetime_col].dt.month
        feature_df['quarter'] = feature_df[datetime_col].dt.quarter
        feature_df['year'] = feature_df[datetime_col].dt.year
        feature_df['dayofyear'] = feature_df[datetime_col].dt.dayofyear
        feature_df['is_month_start'] = feature_df[datetime_col].dt.is_month_start
        feature_df['is_month_end'] = feature_df[datetime_col].dt.is_month_end
        feature_df['is_weekend'] = feature_df['dayofweek'].isin([5, 6]).astype(int)
        
        # 创建周期性特征（使用正弦和余弦变换）
        # 小时周期性特征（24小时周期）
        hours_in_day = 24
        feature_df['hour_sin'] = np.sin(2 * np.pi * feature_df['hour'] / hours_in_day)
        feature_df['hour_cos'] = np.cos(2 * np.pi * feature_df['hour'] / hours_in_day)
        
        # 星期周期性特征（7天周期）
        days_in_week = 7
        feature_df['dayofweek_sin'] = np.sin(2 * np.pi * feature_df['dayofweek'] / days_in_week)
        feature_df['dayofweek_cos'] = np.cos(2 * np.pi * feature_df['dayofweek'] / days_in_week)
        
        # 月份周期性特征（12个月周期）
        months_in_year = 12
        feature_df['month_sin'] = np.sin(2 * np.pi * feature_df['month'] / months_in_year)
        feature_df['month_cos'] = np.cos(2 * np.pi * feature_df['month'] / months_in_year)
        
        print(f"时间特征创建完成，新特征数量: {len(feature_df.columns) - len(df.columns)}")
        return feature_df
    
    def create_sliding_window_features(self, df, target_cols, window_sizes=[24, 48, 72], forecast_horizon=24):
        """
        创建滑动窗口特征
        
        参数:
            df: 时序数据DataFrame
            target_cols: 目标列列表
            window_sizes: 窗口大小列表（小时）
            forecast_horizon: 预测时间范围（小时）
            
        返回:
            DataFrame: 包含滑动窗口特征的数据
        """
        print("创建滑动窗口特征...")
        
        if df is None or df.empty:
            print("无数据需要创建特征")
            return None
        
        # 确保索引是datetime类型
        if not isinstance(df.index, pd.DatetimeIndex):
            print("索引不是datetime类型，无法创建时间窗口特征")
            return df
        
        # 复制数据，避免修改原始数据
        feature_df = df.copy()
        
        # 选择数值列
        numeric_cols = feature_df.select_dtypes(include=['int64', 'float64']).columns
        
        # 如果没有指定目标列，使用所有数值列
        if not target_cols:
            target_cols = numeric_cols.tolist()
        
        # 过滤掉不存在的目标列
        existing_target_cols = [col for col in target_cols if col in feature_df.columns]
        if len(existing_target_cols) < len(target_cols):
            missing_cols = set(target_cols) - set(existing_target_cols)
            print(f"警告：以下目标列不存在: {missing_cols}")
        
        if not existing_target_cols:
            print("没有有效的目标列，无法创建特征")
            return feature_df
        
        print(f"为 {len(existing_target_cols)} 个目标列创建滑动窗口特征")
        
        # 检查数据点数量是否足够
        min_required_points = max(window_sizes) + forecast_horizon
        if len(feature_df) < min_required_points:
            print(f"警告：数据点数量({len(feature_df)})不足以创建最大窗口大小({max(window_sizes)})和预测范围({forecast_horizon})的特征")
            # 调整窗口大小和预测范围
            max_possible_window = min(window_sizes)
            while max_possible_window > 1 and (max_possible_window + forecast_horizon) > len(feature_df):
                max_possible_window = max_possible_window // 2
            
            if max_possible_window < 1:
                print("数据点太少，无法创建任何窗口特征")
                # 仍然尝试创建目标列
                for col in existing_target_cols:
                    for h in range(1, min(forecast_horizon, len(feature_df) - 1) + 1):
                        feature_df[f'{col}_target_{h}h'] = feature_df[col].shift(-h)
                
                # 删除包含NaN的行
                feature_df = feature_df.dropna()
                print(f"仅创建了目标列，新形状: {feature_df.shape}")
                return feature_df
                
            print(f"已调整最大窗口大小为: {max_possible_window}")
            window_sizes = [w for w in window_sizes if w <= max_possible_window]
            if not window_sizes:
                window_sizes = [max_possible_window]
            
            forecast_horizon = min(forecast_horizon, len(feature_df) - max_possible_window)
            print(f"已调整预测范围为: {forecast_horizon}")
        
        # 检查时间序列是否连续(间隔一致)
        time_diffs = feature_df.index.to_series().diff().dropna()
        if time_diffs.nunique() > 1:
            print("警告：时间序列不连续，可能影响滑动窗口特征的准确性")
            # 尝试重新采样到一致的频率
            try:
                # 检测最常见的频率
                most_common_diff = time_diffs.value_counts().idxmax()
                freq_str = None
                
                # 转换时间差为pandas频率字符串
                if most_common_diff == pd.Timedelta(hours=1):
                    freq_str = 'H'
                elif most_common_diff == pd.Timedelta(days=1):
                    freq_str = 'D'
                else:
                    # 以秒为单位的频率
                    freq_str = f'{int(most_common_diff.total_seconds())}S'
                
                print(f"尝试使用频率 '{freq_str}' 重新采样")
                # 创建完整的时间索引
                full_idx = pd.date_range(start=feature_df.index.min(), 
                                         end=feature_df.index.max(),
                                         freq=freq_str)
                
                # 重新索引数据
                feature_df = feature_df.reindex(full_idx)
                
                # 对缺失值进行插值
                for col in existing_target_cols:
                    feature_df[col] = feature_df[col].interpolate(method='time')
                
                print(f"重新采样后的数据形状: {feature_df.shape}")
            except Exception as e:
                print(f"重新采样失败: {e}")
        
        # 为每个目标列创建滑动窗口特征
        for col in existing_target_cols:
            # 为每个窗口大小创建特征
            for window in window_sizes:
                try:
                    # 创建滑动窗口统计特征
                    feature_df[f'{col}_mean_{window}h'] = feature_df[col].rolling(window=window, min_periods=max(1, window//2)).mean()
                    feature_df[f'{col}_std_{window}h'] = feature_df[col].rolling(window=window, min_periods=max(1, window//2)).std()
                    feature_df[f'{col}_min_{window}h'] = feature_df[col].rolling(window=window, min_periods=max(1, window//2)).min()
                    feature_df[f'{col}_max_{window}h'] = feature_df[col].rolling(window=window, min_periods=max(1, window//2)).max()
                    
                    # 创建差分特征
                    feature_df[f'{col}_diff_{window}h'] = feature_df[col].diff(periods=min(window, len(feature_df)-1))
                    
                    # 创建变化率特征 (处理零值和无穷大)
                    pct = feature_df[col].pct_change(periods=min(window, len(feature_df)-1))
                    feature_df[f'{col}_pct_change_{window}h'] = pct.replace([np.inf, -np.inf], np.nan)
                except Exception as e:
                    print(f"为列 {col} 创建窗口 {window} 的特征时出错: {e}")
        
        # 创建目标变量（未来n小时的值）
        for col in existing_target_cols:
            for h in range(1, min(forecast_horizon, len(feature_df)-1) + 1):
                feature_df[f'{col}_target_{h}h'] = feature_df[col].shift(-h)
        
        # 删除包含NaN的行，但仍保留一些有用的数据
        if len(feature_df.dropna()) < 5 and len(feature_df) > 5:
            # 如果清除NaN后数据太少，只删除目标变量为NaN的行
            target_cols_with_horizon = [f'{col}_target_{h}h' for col in existing_target_cols 
                                        for h in range(1, min(forecast_horizon, len(feature_df)-1) + 1)]
            feature_df = feature_df.dropna(subset=target_cols_with_horizon)
            # 对其他特征列进行填充
            feature_df = feature_df.fillna(feature_df.mean())
            print("保留部分含NaN值的行，仅删除目标为NaN的行")
        else:
            # 否则，删除所有含NaN的行
            old_len = len(feature_df)
            feature_df = feature_df.dropna()
            if len(feature_df) < old_len:
                print(f"删除了 {old_len - len(feature_df)} 行含NaN值的数据")
        
        print(f"滑动窗口特征创建完成，新形状: {feature_df.shape}")
        return feature_df
    
    def extract_statistical_features(self, df, group_by='day'):
        """
        提取统计特征
        
        参数:
            df: 时序数据DataFrame
            group_by: 分组方式，'day'、'week'或'month'
            
        返回:
            DataFrame: 包含统计特征的数据
        """
        print(f"按{group_by}提取统计特征...")
        
        if df is None or df.empty:
            print("无数据需要提取特征")
            return None
        
        # 确保索引是datetime类型
        if not isinstance(df.index, pd.DatetimeIndex):
            print("索引不是datetime类型，无法提取时间统计特征")
            return df
        
        # 复制数据，避免修改原始数据
        feature_df = df.copy()
        
        # 选择数值列
        numeric_cols = feature_df.select_dtypes(include=['int64', 'float64']).columns
        
        # 根据分组方式设置分组键
        if group_by == 'day':
            feature_df['group_key'] = feature_df.index.date
        elif group_by == 'week':
            feature_df['group_key'] = feature_df.index.isocalendar().week
        elif group_by == 'month':
            feature_df['group_key'] = feature_df.index.month
        else:
            print(f"不支持的分组方式: {group_by}")
            return feature_df
        
        # 创建统计特征
        stats_df = pd.DataFrame()
        
        # 对每个分组计算统计特征
        for name, group in feature_df.groupby('group_key'):
            row = {'group_key': name}
            
            # 计算每列的统计特征
            for col in numeric_cols:
                row[f'{col}_mean'] = group[col].mean()
                row[f'{col}_std'] = group[col].std()
                row[f'{col}_min'] = group[col].min()
                row[f'{col}_max'] = group[col].max()
                row[f'{col}_range'] = group[col].max() - group[col].min()
                row[f'{col}_median'] = group[col].median()
                row[f'{col}_skew'] = group[col].skew()
                row[f'{col}_kurt'] = group[col].kurtosis()
                
                # 计算分位数
                row[f'{col}_q25'] = group[col].quantile(0.25)
                row[f'{col}_q75'] = group[col].quantile(0.75)
                row[f'{col}_iqr'] = row[f'{col}_q75'] - row[f'{col}_q25']
            
            # 添加到结果DataFrame
            stats_df = pd.concat([stats_df, pd.DataFrame([row])], ignore_index=True)
        
        print(f"统计特征提取完成，形状: {stats_df.shape}")
        return stats_df
    
    def extract_frequency_features(self, df, cols=None):
        """
        提取频域特征（使用FFT）
        
        参数:
            df: 时序数据DataFrame
            cols: 需要提取频域特征的列，默认为None（所有数值列）
            
        返回:
            DataFrame: 包含频域特征的数据
        """
        print("提取频域特征...")
        
        if df is None or df.empty:
            print("无数据需要提取频域特征")
            return None
        
        # 复制数据，避免修改原始数据
        feature_df = df.copy()
        
        # 选择数值列
        numeric_cols = feature_df.select_dtypes(include=['int64', 'float64']).columns
        
        # 如果没有指定列，使用所有数值列
        if cols is None:
            cols = numeric_cols
        
        # 确保所有指定的列都存在
        cols = [col for col in cols if col in feature_df.columns]
        
        if not cols:
            print("没有找到有效的列来提取频域特征")
            return feature_df
        
        # 对每列提取频域特征
        for col in cols:
            # 应用快速傅里叶变换
            fft_values = np.fft.fft(feature_df[col].values)
            fft_abs = np.abs(fft_values)
            
            # 提取主要频率成分
            n = len(fft_abs) // 2
            top_k = 5  # 提取前5个主要频率成分
            
            # 获取前k个主要频率的索引
            top_indices = np.argsort(fft_abs[1:n])[-top_k:]
            
            # 添加频域特征
            for i, idx in enumerate(top_indices):
                feature_df[f'{col}_freq_{i+1}_idx'] = idx
                feature_df[f'{col}_freq_{i+1}_mag'] = fft_abs[idx]
        
        print(f"频域特征提取完成，新形状: {feature_df.shape}")
        return feature_df
    
    def reduce_dimensions(self, df, method='pca', n_components=0.95):
        """
        降维处理
        
        参数:
            df: 特征DataFrame
            method: 降维方法，'pca'或'select_k_best'
            n_components: PCA保留的方差比例或SelectKBest选择的特征数量
            
        返回:
            DataFrame: 降维后的数据
            object: 降维器（用于后续转换）
        """
        print(f"使用{method}方法进行降维...")
        
        if df is None or df.empty:
            print("无数据需要降维")
            return None, None
        
        # 复制数据，避免修改原始数据
        reduced_df = df.copy()
        
        # 选择数值列
        numeric_cols = reduced_df.select_dtypes(include=['int64', 'float64']).columns
        
        if len(numeric_cols) <= 1:
            print("数值列不足，无需降维")
            return reduced_df, None
        
        # 分离特征和目标变量（如果存在）
        target_cols = [col for col in numeric_cols if 'target' in col]
        feature_cols = [col for col in numeric_cols if col not in target_cols]
        
        if not feature_cols:
            print("没有找到特征列，无法降维")
            return reduced_df, None
        
        # 准备特征数据
        X = reduced_df[feature_cols].values
        
        # 降维处理
        if method == 'pca':
            # 使用PCA降维
            reducer = PCA(n_components=n_components)
            X_reduced = reducer.fit_transform(X)
            
            # 创建降维后的DataFrame
            pca_cols = [f'PC{i+1}' for i in range(X_reduced.shape[1])]
            pca_df = pd.DataFrame(X_reduced, columns=pca_cols, index=reduced_df.index)
            
            # 合并降维结果和目标变量
            if target_cols:
                pca_df = pd.concat([pca_df, reduced_df[target_cols]], axis=1)
            
            print(f"PCA降维完成，从{len(feature_cols)}个特征降至{len(pca_cols)}个主成分")
            print(f"解释方差比例: {reducer.explained_variance_ratio_}")
            
            return pca_df, reducer
            
        elif method == 'select_k_best':
            # 确保有目标变量
            if not target_cols:
                print("没有找到目标变量，无法使用SelectKBest")
                return reduced_df, None
            
            # 使用第一个目标变量
            y = reduced_df[target_cols[0]].values
            
            # 使用SelectKBest选择特征
            reducer = SelectKBest(f_regression, k=n_components)
            X_reduced = reducer.fit_transform(X, y)
            
            # 获取选择的特征
            selected_indices = reducer.get_support(indices=True)
            selected_cols = [feature_cols[i] for i in selected_indices]
            
            # 创建选择特征后的DataFrame
            selected_df = reduced_df[selected_cols + target_cols].copy()
            
            print(f"特征选择完成，从{len(feature_cols)}个特征选择了{len(selected_cols)}个特征")
            
            return selected_df, reducer
        
        else:
            print(f"不支持的降维方法: {method}")
            return reduced_df, None
    
    def merge_external_data(self, time_series_df, external_df, date_col='date'):
        """
        合并外部数据
        
        参数:
            time_series_df: 时序数据DataFrame
            external_df: 外部数据DataFrame
            date_col: 外部数据中的日期列名
            
        返回:
            DataFrame: 合并后的数据
        """
        print("合并外部数据...")
        
        if time_series_df is None or time_series_df.empty:
            print("无时序数据需要合并")
            return None
        
        if external_df is None or external_df.empty:
            print("无外部数据需要合并")
            return time_series_df
        
        # 复制数据，避免修改原始数据
        merged_df = time_series_df.copy()
        
        # 确保索引是datetime类型
        if not isinstance(merged_df.index, pd.DatetimeIndex):
            print("时序数据索引不是datetime类型，无法合并外部数据")
            return merged_df
        
        # 确保外部数据的日期列是datetime类型
        if date_col in external_df.columns:
            if not pd.api.types.is_datetime64_any_dtype(external_df[date_col]):
                external_df[date_col] = pd.to_datetime(external_df[date_col])
        else:
            print(f"外部数据中没有找到日期列 {date_col}")
            return merged_df
        
        # 设置外部数据的索引为日期
        external_df = external_df.set_index(date_col)
        
        # 对齐日期索引
        merged_df['date'] = merged_df.index.date
        
        # 合并数据
        merged_df = pd.merge(
            merged_df,
            external_df,
            left_on='date',
            right_index=True,
            how='left'
        )
        
        # 删除临时日期列
        merged_df = merged_df.drop('date', axis=1)
        
        print(f"外部数据合并完成，新形状: {merged_df.shape}")
        return merged_df
    
    def prepare_model_input(self, df, target_cols, feature_cols=None, test_size=0.2, random_state=42):
        """
        准备模型输入数据
        
        参数:
            df: 特征数据DataFrame
            target_cols: 目标列列表
            feature_cols: 特征列列表，默认为None（使用除目标列外的所有列）
            test_size: 测试集比例，默认为0.2
            random_state: 随机种子，默认为42
            
        返回:
            dict: 包含训练集和测试集的字典
        """
        print("准备模型输入数据...")
        
        if df is None or df.empty:
            print("输入数据为空，无法准备模型输入")
            return {"X": None, "y": None, "X_train": None, "X_test": None, "y_train": None, "y_test": None}
        
        # 复制数据，避免修改原始数据
        data = df.copy()
        
        # 检查目标列是否存在
        existing_target_cols = [col for col in target_cols if col in data.columns]
        if not existing_target_cols:
            print("没有找到有效的目标列")
            return {"X": data, "y": None, "X_train": None, "X_test": None, "y_train": None, "y_test": None}
        
        if len(existing_target_cols) < len(target_cols):
            missing_cols = set(target_cols) - set(existing_target_cols)
            print(f"警告：以下目标列不存在: {missing_cols}")
            print(f"将使用以下有效的目标列: {existing_target_cols}")
        
        # 准备目标变量
        if len(existing_target_cols) == 1:
            # 单目标
            y = data[existing_target_cols[0]]
        else:
            # 多目标
            y = data[existing_target_cols]
        
        # 准备特征
        if feature_cols is None:
            # 使用除目标列外的所有列作为特征
            feature_cols = [col for col in data.columns if col not in existing_target_cols]
        else:
            # 检查指定的特征列是否存在
            existing_feature_cols = [col for col in feature_cols if col in data.columns]
            if len(existing_feature_cols) < len(feature_cols):
                missing_cols = set(feature_cols) - set(existing_feature_cols)
                print(f"警告：以下特征列不存在: {missing_cols}")
                feature_cols = existing_feature_cols
        
        if not feature_cols:
            print("没有有效的特征列，无法准备模型输入")
            return {"X": data, "y": y, "X_train": None, "X_test": None, "y_train": None, "y_test": None}
        
        X = data[feature_cols]
        
        # 处理缺失值
        if X.isnull().any().any():
            print("特征中存在缺失值，使用均值填充")
            X = X.fillna(X.mean())
        
        if isinstance(y, pd.DataFrame) and y.isnull().any().any():
            print("目标变量中存在缺失值，删除相应的行")
            # 获取非NaN的索引
            valid_idx = y.dropna().index
            X = X.loc[valid_idx]
            y = y.loc[valid_idx]
        elif isinstance(y, pd.Series) and y.isnull().any():
            print("目标变量中存在缺失值，删除相应的行")
            # 获取非NaN的索引
            valid_idx = y.dropna().index
            X = X.loc[valid_idx]
            y = y.loc[valid_idx]
        
        # 检查数据是否足够进行训练/测试划分
        if len(X) < 5:
            print("数据点太少，无法进行训练/测试划分")
            return {"X": X, "y": y, "X_train": X, "X_test": None, "y_train": y, "y_test": None}
        
        try:
            # 划分训练集和测试集
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            
            print(f"模型输入准备完成，特征数量: {X.shape[1]}, 训练样本数: {len(X_train)}, 测试样本数: {len(X_test)}")
            
            return {
                "X": X, 
                "y": y, 
                "X_train": X_train, 
                "X_test": X_test, 
                "y_train": y_train, 
                "y_test": y_test,
                "feature_cols": feature_cols,
                "target_cols": existing_target_cols
            }
        except Exception as e:
            print(f"划分训练集和测试集时出错: {e}")
            return {"X": X, "y": y, "X_train": None, "X_test": None, "y_train": None, "y_test": None, 
                   "feature_cols": feature_cols, "target_cols": existing_target_cols}
    
    def save_features(self, df, output_path):
        """
        保存特征数据
        
        参数:
            df: 特征DataFrame
            output_path: 输出文件路径
        """
        if df is None or df.empty:
            print("无特征数据需要保存")
            return
        
        print(f"保存特征数据到: {output_path}")
        
        # 确保输出目录存在
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存为CSV文件
        df.to_csv(output_path)
        
        print(f"特征数据已保存，形状: {df.shape}")
    
    def save_model_input(self, model_input, output_path):
        """
        保存模型输入数据
        
        参数:
            model_input: 模型输入数据字典
            output_path: 输出文件路径
        """
        if model_input is None:
            print("无模型输入数据需要保存")
            return
        
        print(f"保存模型输入数据到: {output_path}")
        
        # 确保输出目录存在
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存为joblib文件
        joblib.dump(model_input, output_path)
        
        print("模型输入数据已保存")
    
    def process_time_series_features(self, input_path, output_dir, target_cols=None):
        """
        处理时序数据特征
        
        参数:
            input_path: 输入文件路径
            output_dir: 输出目录
            target_cols: 目标列列表，默认为None
        """
        print(f"处理时序数据特征: {input_path}")
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 加载处理后的时序数据
        try:
            df = pd.read_csv(input_path, index_col=0, parse_dates=True)
            print(f"加载数据成功，形状: {df.shape}")
        except Exception as e:
            print(f"加载数据失败: {e}")
            return
        
        # 如果没有指定目标列，使用所有污染物相关列
        if target_cols is None:
            # 假设污染物相关列包含特定关键词
            pollution_keywords = ['SO2', 'NOx', 'SOOT', '二氧化硫', '氮氧化物', '烟尘']
            target_cols = []
            
            for col in df.columns:
                if any(keyword in col.upper() for keyword in pollution_keywords):
                    target_cols.append(col)
            
            if not target_cols:
                print("未找到污染物相关列，使用所有数值列作为目标")
                target_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        # 1. 创建滑动窗口特征
        window_df = self.create_sliding_window_features(df, target_cols)
        
        # 保存滑动窗口特征
        window_output_path = os.path.join(output_dir, "window_features.csv")
        self.save_features(window_df, window_output_path)
        
        # 2. 提取统计特征
        stats_df = self.extract_statistical_features(df, group_by='day')
        
        # 保存统计特征
        stats_output_path = os.path.join(output_dir, "statistical_features.csv")
        self.save_features(stats_df, stats_output_path)
        
        # 3. 提取频域特征
        freq_df = self.extract_frequency_features(df, cols=target_cols)
        
        # 保存频域特征
        freq_output_path = os.path.join(output_dir, "frequency_features.csv")
        self.save_features(freq_df, freq_output_path)
        
        # 4. 降维处理
        pca_df, pca = self.reduce_dimensions(window_df, method='pca')
        
        # 保存降维结果
        pca_output_path = os.path.join(output_dir, "pca_features.csv")
        self.save_features(pca_df, pca_output_path)
        
        # 保存PCA模型
        pca_model_path = os.path.join(output_dir, "pca_model.joblib")
        joblib.dump(pca, pca_model_path)
        
        # 5. 准备模型输入数据
        # 使用滑动窗口特征作为模型输入
        model_input = self.prepare_model_input(window_df, target_cols=[f'{col}_target_24h' for col in target_cols])
        
        # 保存模型输入数据
        model_input_path = os.path.join(output_dir, "model_input.joblib")
        self.save_model_input(model_input, model_input_path)
        
        print("时序数据特征处理完成")

# 测试代码
if __name__ == "__main__":
    # 创建特征工程器
    feature_eng = FeatureEngineering()
    
    # 设置输入和输出目录
    input_dir = "./data/processed"
    output_dir = "./data/features"
    
    # 处理时序数据特征
    time_series_file = os.path.join(input_dir, "processed_time_series.csv")
    if os.path.exists(time_series_file):
        feature_eng.process_time_series_features(time_series_file, output_dir)

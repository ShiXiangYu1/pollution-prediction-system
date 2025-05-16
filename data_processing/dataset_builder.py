#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据集构建模块 - 整合所有数据并构建训练集和测试集
"""

import pandas as pd
import numpy as np
import os
import glob
import joblib
from datetime import datetime, timedelta
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

class DatasetBuilder:
    """
    数据集构建类，用于整合所有数据并构建训练集和测试集
    """
    
    def __init__(self, data_dir='./data'):
        """
        初始化数据集构建类
        
        参数:
            data_dir: 数据目录
        """
        self.data_dir = data_dir
        # 创建数据集目录（如果不存在）
        os.makedirs(os.path.join(data_dir, 'datasets'), exist_ok=True)
    
    def load_processed_data(self, file_path):
        """
        加载处理后的数据
        
        参数:
            file_path: 数据文件路径
            
        返回:
            DataFrame: 处理后的数据
        """
        print(f"加载处理后的数据: {file_path}")
        
        try:
            # 根据文件扩展名选择加载方法
            ext = os.path.splitext(file_path)[1].lower()
            
            if ext == '.csv':
                df = pd.read_csv(file_path)
                
                # 尝试将日期列转换为datetime类型
                date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
                for col in date_cols:
                    try:
                        df[col] = pd.to_datetime(df[col])
                    except:
                        pass
                
            elif ext == '.joblib':
                df = joblib.load(file_path)
            else:
                print(f"不支持的文件格式: {ext}")
                return None
            
            print(f"成功加载数据，形状: {df.shape if isinstance(df, pd.DataFrame) else '非DataFrame对象'}")
            return df
        except Exception as e:
            print(f"加载数据失败: {e}")
            return None
    
    def merge_time_series_and_external_data(self, time_series_df, external_df, date_col='date'):
        """
        合并时序数据和外部数据
        
        参数:
            time_series_df: 时序数据DataFrame
            external_df: 外部数据DataFrame
            date_col: 外部数据中的日期列名
            
        返回:
            DataFrame: 合并后的数据
        """
        print("合并时序数据和外部数据...")
        
        if time_series_df is None or time_series_df.empty:
            print("无时序数据需要合并")
            return None
        
        if external_df is None or external_df.empty:
            print("无外部数据需要合并")
            return time_series_df
        
        # 复制数据，避免修改原始数据
        merged_df = time_series_df.copy()
        
        # 确保时序数据的索引是datetime类型
        if not isinstance(merged_df.index, pd.DatetimeIndex):
            print("时序数据索引不是datetime类型，尝试转换...")
            try:
                # 尝试将索引转换为datetime
                merged_df.index = pd.to_datetime(merged_df.index)
                # 修复：成功转换索引后，创建merge_date列
                merged_df['merge_date'] = merged_df.index.date
                print("索引成功转换为datetime类型，并创建了merge_date列")
            except Exception as e:
                print(f"无法将时序数据索引转换为datetime类型: {e}，使用日期列进行合并")
                # 如果无法转换索引，尝试使用日期列
                date_cols = [col for col in merged_df.columns if 'date' in col.lower() or 'time' in col.lower()]
                if date_cols:
                    merged_df['merge_date'] = pd.to_datetime(merged_df[date_cols[0]]).dt.date
                    print(f"使用列 {date_cols[0]} 创建了merge_date列")
                else:
                    print("无法找到合适的日期列进行合并")
                    return merged_df
        else:
            # 如果索引是datetime类型，创建日期列用于合并
            merged_df['merge_date'] = merged_df.index.date
        
        # 确保外部数据的日期列是datetime类型
        if date_col in external_df.columns:
            if not pd.api.types.is_datetime64_any_dtype(external_df[date_col]):
                external_df[date_col] = pd.to_datetime(external_df[date_col])
            
            # 创建日期列用于合并
            external_df['merge_date'] = external_df[date_col].dt.date
            print("已为外部数据创建merge_date列")
        else:
            print(f"外部数据中没有找到日期列 {date_col}")
            return merged_df
        
        # 打印检查合并前的列是否存在
        print(f"检查合并前的列: 'merge_date' 是否在merged_df中: {'merge_date' in merged_df.columns}")
        print(f"检查合并前的列: 'merge_date' 是否在external_df中: {'merge_date' in external_df.columns}")
        
        # 合并数据
        merged_df = pd.merge(
            merged_df,
            external_df.drop(date_col, axis=1),
            on='merge_date',
            how='left'
        )
        
        # 删除临时日期列
        merged_df = merged_df.drop('merge_date', axis=1)
        
        print(f"数据合并完成，新形状: {merged_df.shape}")
        return merged_df
    
    def build_prediction_dataset(self, time_series_data, external_data, target_cols, window_sizes=[24, 48, 72], forecast_horizon=24, output_path=None):
        """
        构建预测数据集
        
        参数:
            time_series_data: 时序数据
            external_data: 外部数据
            target_cols: 目标列列表
            window_sizes: 窗口大小列表（小时）
            forecast_horizon: 预测时间范围（小时）
            output_path: 输出文件路径，默认为None
            
        返回:
            dict: 包含训练集和测试集的字典
        """
        print("构建预测数据集...")
        
        if time_series_data is None:
            print("无时序数据，无法构建预测数据集")
            return None
        
        # 检查目标列是否存在于时序数据中
        missing_cols = [col for col in target_cols if col not in time_series_data.columns]
        if missing_cols:
            print(f"警告：以下目标列在时序数据中不存在: {missing_cols}")
            # 过滤掉不存在的目标列
            target_cols = [col for col in target_cols if col in time_series_data.columns]
            if not target_cols:
                print("错误：没有有效的目标列可用于构建预测数据集")
                return None
            print(f"将使用以下有效的目标列: {target_cols}")
        
        # 合并时序数据和外部数据
        if external_data is not None:
            merged_data = self.merge_time_series_and_external_data(time_series_data, external_data)
        else:
            merged_data = time_series_data.copy()
        
        if merged_data is None or merged_data.empty:
            print("合并后的数据为空，无法构建预测数据集")
            return None
            
        # 确保合并后的数据索引是datetime类型
        if not isinstance(merged_data.index, pd.DatetimeIndex):
            print("警告：合并后的数据索引不是datetime类型，尝试转换...")
            try:
                # 查找可能的日期列
                date_cols = [col for col in merged_data.columns if 'date' in col.lower() or 'time' in col.lower() or 'rec' in col.lower()]
                
                if date_cols:
                    # 使用第一个发现的日期列作为索引
                    print(f"使用列 '{date_cols[0]}' 作为索引")
                    merged_data[date_cols[0]] = pd.to_datetime(merged_data[date_cols[0]])
                    merged_data = merged_data.set_index(date_cols[0])
                else:
                    # 如果没有找到日期列，尝试将当前索引转换为datetime
                    print("尝试将当前索引转换为datetime")
                    merged_data.index = pd.to_datetime(merged_data.index)
                print("成功将索引转换为datetime类型")
            except Exception as e:
                print(f"无法将索引转换为datetime类型: {e}")
                # 创建一个人工时间索引
                print("创建人工时间索引")
                start_date = pd.Timestamp('2024-01-01')
                merged_data.index = pd.date_range(start=start_date, periods=len(merged_data), freq='H')
                print("已创建人工时间索引")
        
        # 打印最终索引类型进行确认
        print(f"最终数据索引类型: {type(merged_data.index)}")
        print(f"索引示例: {merged_data.index[:5]}")
        
        # 再次检查目标列是否存在于合并后的数据中
        missing_cols = [col for col in target_cols if col not in merged_data.columns]
        if missing_cols:
            print(f"警告：合并后以下目标列不存在: {missing_cols}")
            # 过滤掉不存在的目标列
            target_cols = [col for col in target_cols if col in merged_data.columns]
            if not target_cols:
                print("错误：合并后没有有效的目标列")
                return None
            print(f"将使用以下有效的目标列: {target_cols}")
            
        # 打印目标列的前几行数据，确保有数据
        print("目标列数据示例:")
        for col in target_cols:
            print(f"{col}:\n{merged_data[col].head()}")
        
        try:
            # 创建滑动窗口特征
            from feature_engineering import FeatureEngineering
            feature_eng = FeatureEngineering()
            
            print("创建滑动窗口特征...")
            window_df = feature_eng.create_sliding_window_features(merged_data, target_cols, window_sizes, forecast_horizon)
            
            if window_df is None or window_df.empty:
                print("无法创建滑动窗口特征，返回原始合并数据")
                model_input = {"X": merged_data, "y": None, "targets": target_cols}
            else:
                print("准备模型输入数据...")
                # 创建目标列名
                target_cols_with_horizon = [f'{col}_target_{forecast_horizon}h' for col in target_cols]
                
                # 检查目标列是否存在
                missing_target_cols = [col for col in target_cols_with_horizon if col not in window_df.columns]
                if missing_target_cols:
                    print(f"警告：以下目标列在窗口数据中不存在: {missing_target_cols}")
                    # 过滤掉不存在的目标列
                    target_cols_with_horizon = [col for col in target_cols_with_horizon if col in window_df.columns]
                    
                if not target_cols_with_horizon:
                    print("警告：没有有效的目标列用于模型输入，返回窗口特征数据")
                    model_input = {"X": window_df, "y": None, "targets": target_cols}
                else:
                    print(f"使用以下目标列: {target_cols_with_horizon}")
                    model_input = feature_eng.prepare_model_input(window_df, target_cols_with_horizon)
        except Exception as e:
            print(f"创建特征时出错: {e}")
            print("返回原始合并数据")
            model_input = {"X": merged_data, "y": None, "targets": target_cols}
        
        # 保存数据集
        if output_path is not None:
            # 确保输出目录存在
            output_dir = os.path.dirname(output_path)
            os.makedirs(output_dir, exist_ok=True)
            
            # 保存为joblib文件
            joblib.dump(model_input, output_path)
            print(f"预测数据集已保存到: {output_path}")
        
        print("预测数据集构建完成")
        return model_input
    
    def build_nl2sql_dataset(self, schema_info, example_queries, output_path=None):
        """
        构建NL2SQL数据集
        
        参数:
            schema_info: 数据库模式信息
            example_queries: 示例查询列表
            output_path: 输出文件路径，默认为None
            
        返回:
            DataFrame: NL2SQL数据集
        """
        print("构建NL2SQL数据集...")
        
        # 创建结果DataFrame
        nl2sql_data = pd.DataFrame(columns=['natural_language', 'sql_query', 'db_schema'])
        
        # 添加示例查询
        for example in example_queries:
            nl = example.get('natural_language', '')
            sql = example.get('sql_query', '')
            
            if nl and sql:
                nl2sql_data = pd.concat([nl2sql_data, pd.DataFrame({
                    'natural_language': [nl],
                    'sql_query': [sql],
                    'db_schema': [schema_info]
                })], ignore_index=True)
        
        print(f"NL2SQL数据集构建完成，共{len(nl2sql_data)}条记录")
        
        # 保存数据集
        if output_path is not None:
            # 确保输出目录存在
            output_dir = os.path.dirname(output_path)
            os.makedirs(output_dir, exist_ok=True)
            
            # 保存为CSV文件
            nl2sql_data.to_csv(output_path, index=False)
            print(f"NL2SQL数据集已保存到: {output_path}")
        
        return nl2sql_data
    
    def build_speech_recognition_dataset(self, text_examples, output_path=None):
        """
        构建语音识别数据集
        
        参数:
            text_examples: 文本示例列表
            output_path: 输出文件路径，默认为None
            
        返回:
            DataFrame: 语音识别数据集
        """
        print("构建语音识别数据集...")
        
        # 创建结果DataFrame
        speech_data = pd.DataFrame(columns=['text', 'domain'])
        
        # 添加文本示例
        for example in text_examples:
            text = example.get('text', '')
            domain = example.get('domain', 'general')
            
            if text:
                speech_data = pd.concat([speech_data, pd.DataFrame({
                    'text': [text],
                    'domain': [domain]
                })], ignore_index=True)
        
        print(f"语音识别数据集构建完成，共{len(speech_data)}条记录")
        
        # 保存数据集
        if output_path is not None:
            # 确保输出目录存在
            output_dir = os.path.dirname(output_path)
            os.makedirs(output_dir, exist_ok=True)
            
            # 保存为CSV文件
            speech_data.to_csv(output_path, index=False)
            print(f"语音识别数据集已保存到: {output_path}")
        
        return speech_data
    
    def build_all_datasets(self, processed_dir, external_dir, output_dir):
        """
        构建所有数据集
        
        参数:
            processed_dir: 处理后的数据目录
            external_dir: 外部数据目录
            output_dir: 输出目录
            
        返回:
            dict: 包含所有数据集的字典
        """
        print("构建所有数据集...")
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 加载处理后的时序数据
        time_series_file = os.path.join(processed_dir, "processed_time_series.csv")
        time_series_data = self.load_processed_data(time_series_file)
        
        # 加载处理后的污染物排放数据
        emission_file = os.path.join(processed_dir, "processed_emission.csv")
        emission_data = self.load_processed_data(emission_file)
        
        # 打印emission_data的列名，用于调试
        print("\n=== emission_data的列名 ===")
        if emission_data is not None and isinstance(emission_data, pd.DataFrame):
            print(emission_data.columns.tolist())
        else:
            print("emission_data不是有效的DataFrame")
        print("===========================\n")
        
        # 加载合并后的外部数据
        external_file = os.path.join(external_dir, "merged_external_data.csv")
        external_data = self.load_processed_data(external_file)
        
        # 1. 构建污染物排放预测数据集
        # 根据实际列名调整目标列
        target_cols = []
        if emission_data is not None and isinstance(emission_data, pd.DataFrame):
            # 检查数据中是否存在SO2、NOX、烟尘相关列
            possible_cols = [col for col in emission_data.columns if 'SO2' in col.upper() or 'NOX' in col.upper() or 'SOOT' in col.upper()]
            if possible_cols:
                target_cols = possible_cols
                print(f"从数据中找到的目标列: {target_cols}")
            else:
                # 如果没有找到相关列，则使用排放数据中的所有数值列作为目标
                numeric_cols = emission_data.select_dtypes(include=[np.number]).columns.tolist()
                if numeric_cols:
                    # 选择前三个数值列作为目标（如果存在）
                    target_cols = numeric_cols[:min(3, len(numeric_cols))]
                    print(f"未找到SO2/NOX/SOOT相关列，使用前几个数值列作为目标: {target_cols}")
                else:
                    print("未找到任何可用作目标的数值列")
        
        if not target_cols:
            # 如果还是没有找到合适的列，使用默认目标列
            target_cols = ['VENT_SO2_CHK', 'VENT_NOX_CHK', 'VENT_SOOT_CHK']
            print(f"使用默认目标列: {target_cols}")
        
        prediction_dataset_path = os.path.join(output_dir, "pollution_prediction_dataset.joblib")
        prediction_dataset = self.build_prediction_dataset(
            emission_data, 
            external_data, 
            target_cols, 
            output_path=prediction_dataset_path
        )
        
        # 2. 构建NL2SQL数据集
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
        
        # 定义示例查询
        example_queries = [
            {
                "natural_language": "当前哪些机组实时污染物排放超标？",
                "sql_query": """
                SELECT f.FAC_NAME as 电厂名称, s.STEAMER_NAME as 机组名称, 
                       d1.RECTIME as 时间, d1.VENT_SO2_CHK as SO2排放浓度,
                       d2.VENT_NOX_CHK as NOx排放浓度,
                       d3.VENT_SOOT_CHK as 烟尘排放浓度,
                       CASE 
                           WHEN s.STEAMER_TYPE = '燃煤机' THEN 
                               CASE 
                                   WHEN d1.VENT_SO2_CHK > 35 OR d2.VENT_NOX_CHK > 50 OR d3.VENT_SOOT_CHK > 5 THEN '超标'
                                   ELSE '正常'
                               END
                           WHEN s.STEAMER_TYPE = '燃气机' THEN 
                               CASE 
                                   WHEN d1.VENT_SO2_CHK > 35 OR d2.VENT_NOX_CHK > 50 OR d3.VENT_SOOT_CHK > 5 THEN '超标'
                                   ELSE '正常'
                               END
                           ELSE '未知'
                       END as 排放状态
                FROM TB_FACTORY f
                JOIN TB_STEAMER s ON f.FAC_ID = s.FAC_ID
                JOIN td_dtl_2025 d1 ON s.STEAMER_ID = d1.STEAMER_ID
                JOIN td_dtx_2025 d2 ON s.STEAMER_ID = d2.STEAMER_ID AND d1.RECTIME = d2.RECTIME
                JOIN td_dcc_2025 d3 ON s.STEAMER_ID = d3.STEAMER_ID AND d1.RECTIME = d3.RECTIME
                WHERE d1.RECTIME >= DATEADD(HOUR, -1, GETDATE())
                AND (
                    (s.STEAMER_TYPE = '燃煤机' AND (d1.VENT_SO2_CHK > 35 OR d2.VENT_NOX_CHK > 50 OR d3.VENT_SOOT_CHK > 5))
                    OR
                    (s.STEAMER_TYPE = '燃气机' AND (d1.VENT_SO2_CHK > 35 OR d2.VENT_NOX_CHK > 50 OR d3.VENT_SOOT_CHK > 5))
                )
                ORDER BY d1.RECTIME DESC;
                """
            },
            {
                "natural_language": "过去24小时有哪些机组发生污染物小时浓度超标？",
                "sql_query": """
                SELECT f.FAC_NAME as 电厂名称, s.STEAMER_NAME as 机组名称, 
                       d1.RECTIME as 时间, d1.VENT_SO2_CHK as SO2排放浓度,
                       d2.VENT_NOX_CHK as NOx排放浓度,
                       d3.VENT_SOOT_CHK as 烟尘排放浓度,
                       CASE 
                           WHEN s.STEAMER_TYPE = '燃煤机' THEN 
                               CASE 
                                   WHEN d1.VENT_SO2_CHK > 35 OR d2.VENT_NOX_CHK > 50 OR d3.VENT_SOOT_CHK > 5 THEN '超标'
                                   ELSE '正常'
                               END
                           WHEN s.STEAMER_TYPE = '燃气机' THEN 
                               CASE 
                                   WHEN d1.VENT_SO2_CHK > 35 OR d2.VENT_NOX_CHK > 50 OR d3.VENT_SOOT_CHK > 5 THEN '超标'
                                   ELSE '正常'
                               END
                           ELSE '未知'
                       END as 排放状态,
                       d1.TIME_2TO30 + d1.TIME_30TO50 as SO2超标时长,
                       d2.TIME_2TO30 + d2.TIME_30TO50 as NOx超标时长,
                       d3.TIME_2TO30 + d3.TIME_30TO50 as 烟尘超标时长
                FROM TB_FACTORY f
                JOIN TB_STEAMER s ON f.FAC_ID = s.FAC_ID
                JOIN td_dtl_2025 d1 ON s.STEAMER_ID = d1.STEAMER_ID
                JOIN td_dtx_2025 d2 ON s.STEAMER_ID = d2.STEAMER_ID AND d1.RECTIME = d2.RECTIME
                JOIN td_dcc_2025 d3 ON s.STEAMER_ID = d3.STEAMER_ID AND d1.RECTIME = d3.RECTIME
                WHERE d1.RECTIME >= DATEADD(HOUR, -24, GETDATE())
                AND (
                    (s.STEAMER_TYPE = '燃煤机' AND (d1.VENT_SO2_CHK > 35 OR d2.VENT_NOX_CHK > 50 OR d3.VENT_SOOT_CHK > 5))
                    OR
                    (s.STEAMER_TYPE = '燃气机' AND (d1.VENT_SO2_CHK > 35 OR d2.VENT_NOX_CHK > 50 OR d3.VENT_SOOT_CHK > 5))
                )
                ORDER BY d1.RECTIME DESC;
                """
            },
            {
                "natural_language": "过去24小时单位发电量对应的污染物排放量排名最低的三台机组是哪三个？",
                "sql_query": """
                WITH EmissionPerKWH AS (
                    SELECT 
                        f.FAC_NAME as 电厂名称, 
                        s.STEAMER_NAME as 机组名称,
                        SUM(d1.VENT_SO2_T) as SO2总排放量,
                        SUM(d2.VENT_NOX_T) as NOx总排放量,
                        SUM(d3.VENT_SOOT_T) as 烟尘总排放量,
                        SUM(d1.FD_KWH) as 总发电量,
                        CASE 
                            WHEN SUM(d1.FD_KWH) > 0 THEN SUM(d1.VENT_SO2_T) / SUM(d1.FD_KWH)
                            ELSE 0 
                        END as 单位发电量SO2排放,
                        CASE 
                            WHEN SUM(d1.FD_KWH) > 0 THEN SUM(d2.VENT_NOX_T) / SUM(d1.FD_KWH)
                            ELSE 0 
                        END as 单位发电量NOx排放,
                        CASE 
                            WHEN SUM(d1.FD_KWH) > 0 THEN SUM(d3.VENT_SOOT_T) / SUM(d1.FD_KWH)
                            ELSE 0 
                        END as 单位发电量烟尘排放
                    FROM TB_FACTORY f
                    JOIN TB_STEAMER s ON f.FAC_ID = s.FAC_ID
                    JOIN td_dtl_2025 d1 ON s.STEAMER_ID = d1.STEAMER_ID
                    JOIN td_dtx_2025 d2 ON s.STEAMER_ID = d2.STEAMER_ID AND d1.RECTIME = d2.RECTIME
                    JOIN td_dcc_2025 d3 ON s.STEAMER_ID = d3.STEAMER_ID AND d1.RECTIME = d3.RECTIME
                    WHERE d1.RECTIME >= DATEADD(HOUR, -24, GETDATE())
                    GROUP BY f.FAC_NAME, s.STEAMER_NAME
                )
                
                SELECT TOP 3 电厂名称, 机组名称, 单位发电量SO2排放, 单位发电量NOx排放, 单位发电量烟尘排放
                FROM EmissionPerKWH
                WHERE 总发电量 > 0
                ORDER BY (单位发电量SO2排放 + 单位发电量NOx排放 + 单位发电量烟尘排放) ASC;
                """
            },
            {
                "natural_language": "分析江苏省燃煤机组2024年污染物排放趋势",
                "sql_query": """
                SELECT 
                    DATEPART(MONTH, d1.RECTIME) as 月份,
                    AVG(d1.VENT_SO2_CHK) as 平均SO2排放浓度,
                    AVG(d2.VENT_NOX_CHK) as 平均NOx排放浓度,
                    AVG(d3.VENT_SOOT_CHK) as 平均烟尘排放浓度
                FROM TB_FACTORY f
                JOIN TS_AREA a ON f.AREA_CODE = a.AREA_CODE
                JOIN TB_STEAMER s ON f.FAC_ID = s.FAC_ID
                JOIN td_dtl_2025 d1 ON s.STEAMER_ID = d1.STEAMER_ID
                JOIN td_dtx_2025 d2 ON s.STEAMER_ID = d2.STEAMER_ID AND d1.RECTIME = d2.RECTIME
                JOIN td_dcc_2025 d3 ON s.STEAMER_ID = d3.STEAMER_ID AND d1.RECTIME = d3.RECTIME
                WHERE a.PARENT_CODE = '01' -- 江苏省
                AND s.STEAMER_TYPE = '燃煤机'
                AND YEAR(d1.RECTIME) = 2024
                GROUP BY DATEPART(MONTH, d1.RECTIME)
                ORDER BY 月份;
                """
            },
            {
                "natural_language": "总结江苏省2024年12月发电情况",
                "sql_query": """
                WITH GenerationSummary AS (
                    SELECT 
                        COUNT(DISTINCT f.FAC_ID) as 电厂总数,
                        COUNT(DISTINCT CASE WHEN s.STEAMER_TYPE = '燃煤机' THEN s.STEAMER_ID END) as 煤机数量,
                        COUNT(DISTINCT CASE WHEN s.STEAMER_TYPE = '燃气机' THEN s.STEAMER_ID END) as 气机数量,
                        COUNT(DISTINCT CASE WHEN s.STEAMER_TYPE = '风电' THEN s.STEAMER_ID END) as 风电数量,
                        AVG(d1.VENT_SO2_CHK) as 平均SO2排放浓度,
                        AVG(d1.FGD_EFCY) as 平均脱硫效率,
                        AVG(d2.VENT_NOX_CHK) as 平均NOx排放浓度,
                        AVG(d2.SCR_EFCY) as 平均脱硝效率,
                        AVG(d3.VENT_SOOT_CHK) as 平均烟尘排放浓度,
                        AVG(s.RATING_FH) as 平均负荷
                    FROM TB_FACTORY f
                    JOIN TS_AREA a ON f.AREA_CODE = a.AREA_CODE
                    JOIN TB_STEAMER s ON f.FAC_ID = s.FAC_ID
                    LEFT JOIN td_dtl_2025 d1 ON s.STEAMER_ID = d1.STEAMER_ID AND YEAR(d1.RECTIME) = 2024 AND MONTH(d1.RECTIME) = 12
                    LEFT JOIN td_dtx_2025 d2 ON s.STEAMER_ID = d2.STEAMER_ID AND YEAR(d2.RECTIME) = 2024 AND MONTH(d2.RECTIME) = 12
                    LEFT JOIN td_dcc_2025 d3 ON s.STEAMER_ID = d3.STEAMER_ID AND YEAR(d3.RECTIME) = 2024 AND MONTH(d3.RECTIME) = 12
                    WHERE a.PARENT_CODE = '01' -- 江苏省
                )
                
                SELECT 
                    电厂总数,
                    煤机数量,
                    气机数量,
                    风电数量,
                    平均SO2排放浓度,
                    平均脱硫效率,
                    平均NOx排放浓度,
                    平均脱硝效率,
                    平均烟尘排放浓度,
                    平均负荷
                FROM GenerationSummary;
                """
            },
            {
                "natural_language": "盐城2024年11月燃煤、燃气、风电、光电负荷趋势",
                "sql_query": """
                WITH DailyLoad AS (
                    SELECT 
                        CONVERT(DATE, d1.RECTIME) as 日期,
                        s.STEAMER_TYPE as 机组类型,
                        AVG(s.RATING_FH) as 平均负荷
                    FROM TB_FACTORY f
                    JOIN TS_AREA a ON f.AREA_CODE = a.AREA_CODE
                    JOIN TB_STEAMER s ON f.FAC_ID = s.FAC_ID
                    JOIN td_dtl_2025 d1 ON s.STEAMER_ID = d1.STEAMER_ID
                    WHERE a.AREA_NAME = '盐城'
                    AND YEAR(d1.RECTIME) = 2024
                    AND MONTH(d1.RECTIME) = 11
                    AND s.STEAMER_TYPE IN ('燃煤机', '燃气机', '风电', '光电')
                    GROUP BY CONVERT(DATE, d1.RECTIME), s.STEAMER_TYPE
                )
                
                SELECT 
                    日期,
                    MAX(CASE WHEN 机组类型 = '燃煤机' THEN 平均负荷 ELSE 0 END) as 燃煤机负荷,
                    MAX(CASE WHEN 机组类型 = '燃气机' THEN 平均负荷 ELSE 0 END) as 燃气机负荷,
                    MAX(CASE WHEN 机组类型 = '风电' THEN 平均负荷 ELSE 0 END) as 风电负荷,
                    MAX(CASE WHEN 机组类型 = '光电' THEN 平均负荷 ELSE 0 END) as 光电负荷
                FROM DailyLoad
                GROUP BY 日期
                ORDER BY 日期;
                """
            }
        ]
        
        nl2sql_dataset_path = os.path.join(output_dir, "nl2sql_dataset.csv")
        nl2sql_dataset = self.build_nl2sql_dataset(schema_info, example_queries, nl2sql_dataset_path)
        
        # 3. 构建语音识别数据集
        # 定义文本示例
        text_examples = [
            {"text": "当前哪些机组实时污染物排放超标？", "domain": "pollution"},
            {"text": "过去24小时有哪些机组发生污染物小时浓度超标？", "domain": "pollution"},
            {"text": "过去24小时单位发电量对应的污染物排放量排名最低的三台机组是哪三个？", "domain": "pollution"},
            {"text": "分析江苏省燃煤机组2024年污染物排放趋势", "domain": "analysis"},
            {"text": "总结江苏省2024年12月发电情况", "domain": "summary"},
            {"text": "盐城2024年11月燃煤、燃气、风电、光电负荷趋势", "domain": "load"},
            {"text": "南京2023年各季度污染物排放情况对比", "domain": "comparison"},
            {"text": "江苏省2024年各地区发电量排名", "domain": "ranking"},
            {"text": "无锡市2024年1月至6月SO2排放趋势", "domain": "trend"},
            {"text": "常州电厂2024年脱硫效率分析", "domain": "efficiency"},
            {"text": "苏州地区风电场2024年发电量统计", "domain": "statistics"},
            {"text": "镇江2024年重污染天气期间电厂排放情况", "domain": "emergency"},
            {"text": "泰州2024年各机组单位发电量污染物排放对比", "domain": "comparison"},
            {"text": "徐州2024年煤电机组与气电机组排放对比", "domain": "comparison"},
            {"text": "连云港2024年风电装机容量变化", "domain": "capacity"},
            {"text": "淮安地区2024年光伏发电量月度变化", "domain": "trend"},
            {"text": "宿迁2024年各季度用电负荷分析", "domain": "load"},
            {"text": "江苏省2024年新能源发电占比分析", "domain": "proportion"},
            {"text": "南京地区2024年重大活动期间污染物排放变化", "domain": "event"},
            {"text": "苏州2024年环保政策实施前后排放对比", "domain": "policy"}
        ]
        
        speech_dataset_path = os.path.join(output_dir, "speech_recognition_dataset.csv")
        speech_dataset = self.build_speech_recognition_dataset(text_examples, speech_dataset_path)
        
        # 创建结果字典
        result = {
            'prediction_dataset': prediction_dataset,
            'nl2sql_dataset': nl2sql_dataset,
            'speech_dataset': speech_dataset
        }
        
        print("所有数据集构建完成")
        return result

# 测试代码
if __name__ == "__main__":
    # 创建数据集构建器
    builder = DatasetBuilder()
    
    # 设置输入和输出目录
    processed_dir = "./data/processed"
    external_dir = "./data/external"
    output_dir = "./data/datasets"
    
    # 构建所有数据集
    datasets = builder.build_all_datasets(processed_dir, external_dir, output_dir)

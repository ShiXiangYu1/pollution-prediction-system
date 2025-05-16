#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
外部数据获取模块 - 收集环保政策、天气数据等外部因素
"""

import pandas as pd
import numpy as np
import os
import requests
import json
import time
from datetime import datetime, timedelta
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from bs4 import BeautifulSoup
import re

class ExternalDataCollector:
    """
    外部数据收集类，用于获取环保政策、天气数据等外部因素
    """
    
    def __init__(self, data_dir='./data'):
        """
        初始化外部数据收集类
        
        参数:
            data_dir: 数据目录
        """
        self.data_dir = data_dir
        # 创建外部数据目录（如果不存在）
        os.makedirs(os.path.join(data_dir, 'external'), exist_ok=True)
        
        # 设置请求头
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
    
    def collect_weather_data(self, city, start_date, end_date, output_path=None):
        """
        收集天气数据
        
        参数:
            city: 城市名称
            start_date: 开始日期（YYYY-MM-DD）
            end_date: 结束日期（YYYY-MM-DD）
            output_path: 输出文件路径，默认为None
            
        返回:
            DataFrame: 天气数据
        """
        print(f"收集{city}从{start_date}到{end_date}的天气数据...")
        
        # 将日期字符串转换为datetime对象
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        # 创建日期范围
        date_range = pd.date_range(start=start_dt, end=end_dt, freq='D')
        
        # 创建结果DataFrame
        weather_data = pd.DataFrame(columns=['date', 'city', 'temperature', 'humidity', 'wind_speed', 'weather_condition'])
        
        # 模拟天气数据（实际应用中应使用真实API）
        for date in tqdm(date_range):
            date_str = date.strftime('%Y-%m-%d')
            
            # 模拟天气数据
            temperature = np.random.normal(25, 5)  # 平均气温25度，标准差5度
            humidity = np.random.uniform(40, 90)  # 湿度40%-90%
            wind_speed = np.random.exponential(2)  # 风速，指数分布
            
            # 天气状况
            weather_conditions = ['晴', '多云', '阴', '小雨', '中雨', '大雨', '雷阵雨', '雾']
            weather_condition = np.random.choice(weather_conditions, p=[0.3, 0.3, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05])
            
            # 添加到结果DataFrame
            weather_data = pd.concat([weather_data, pd.DataFrame({
                'date': [date_str],
                'city': [city],
                'temperature': [temperature],
                'humidity': [humidity],
                'wind_speed': [wind_speed],
                'weather_condition': [weather_condition]
            })], ignore_index=True)
            
            # 添加随机延迟，避免请求过快
            time.sleep(0.01)
        
        print(f"天气数据收集完成，共{len(weather_data)}条记录")
        
        # 保存数据
        if output_path is not None:
            # 确保输出目录存在
            output_dir = os.path.dirname(output_path)
            os.makedirs(output_dir, exist_ok=True)
            
            # 保存为CSV文件
            weather_data.to_csv(output_path, index=False)
            print(f"天气数据已保存到: {output_path}")
        
        return weather_data
    
    def collect_policy_data(self, keywords, start_date, end_date, output_path=None):
        """
        收集环保政策数据
        
        参数:
            keywords: 关键词列表
            start_date: 开始日期（YYYY-MM-DD）
            end_date: 结束日期（YYYY-MM-DD）
            output_path: 输出文件路径，默认为None
            
        返回:
            DataFrame: 政策数据
        """
        print(f"收集从{start_date}到{end_date}的环保政策数据...")
        
        # 将日期字符串转换为datetime对象
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        # 创建结果DataFrame
        policy_data = pd.DataFrame(columns=['date', 'title', 'source', 'content', 'url', 'keywords'])
        
        # 模拟政策数据（实际应用中应使用真实API或爬虫）
        # 示例政策标题和内容
        policy_titles = [
            "关于进一步加强电力行业污染物排放控制的通知",
            "江苏省大气污染防治条例实施细则",
            "关于推进燃煤电厂超低排放改造的指导意见",
            "电力行业碳达峰碳中和实施方案",
            "关于加强重污染天气应急管控的通知",
            "新能源发展十四五规划",
            "关于促进可再生能源发展的若干措施",
            "环保设施和排放口规范化整治实施方案",
            "关于加强工业炉窑大气污染综合治理的通知",
            "重点行业清洁生产技术推行方案"
        ]
        
        policy_sources = [
            "国家发改委",
            "江苏省生态环境厅",
            "生态环境部",
            "国家能源局",
            "江苏省人民政府",
            "工业和信息化部",
            "南京市生态环境局",
            "苏州市人民政府",
            "无锡市生态环境局",
            "江苏省发改委"
        ]
        
        policy_contents = [
            "为贯彻落实《中华人民共和国大气污染防治法》，进一步控制电力行业污染物排放，改善环境空气质量，现就有关事项通知如下：一、严格控制新建燃煤电厂。二、加快现役燃煤电厂超低排放改造。三、强化监督管理。",
            "为了防治大气污染，保护和改善环境，保障公众健康，推进生态文明建设，促进经济社会可持续发展，根据《中华人民共和国大气污染防治法》《江苏省大气污染防治条例》，制定本实施细则。",
            "为贯彻《国务院关于印发大气污染防治行动计划的通知》，大力推进燃煤电厂超低排放改造，制定本指导意见。到2020年，全省所有具备改造条件的燃煤电厂力争实现超低排放。",
            "为贯彻落实党中央、国务院关于碳达峰碳中和的重大战略决策，推动电力行业绿色低碳转型，制定本实施方案。到2025年，可再生能源发电量占比达到30%以上。",
            "为进一步做好重污染天气应对工作，减轻重污染天气影响，保障人民群众身体健康，现就加强重污染天气应急管控有关事项通知如下。",
            "为加快构建清洁低碳、安全高效的能源体系，推动能源高质量发展，制定本规划。到2025年，新能源发电装机容量达到12亿千瓦以上。",
            "为促进可再生能源持续健康发展，加快能源结构调整，提高能源安全保障能力，制定本措施。重点支持风电、太阳能发电、生物质能等可再生能源发展。",
            "为进一步规范排污单位环保设施和排放口设置，加强污染物排放监管，根据有关法律法规，制定本实施方案。",
            "为加强工业炉窑大气污染综合治理，提高环境空气质量，制定本通知。各地要全面排查工业炉窑，加大淘汰力度，加快治理步伐。",
            "为推动重点行业清洁生产，提高资源利用效率，减少污染物排放，制定本方案。重点推进电力、钢铁、水泥等行业清洁生产技术应用。"
        ]
        
        # 生成随机政策数据
        num_policies = 20  # 生成20条政策数据
        
        for _ in tqdm(range(num_policies)):
            # 随机选择政策标题和内容
            idx = np.random.randint(0, len(policy_titles))
            title = policy_titles[idx]
            source = policy_sources[np.random.randint(0, len(policy_sources))]
            content = policy_contents[idx]
            
            # 随机生成日期
            days_range = (end_dt - start_dt).days
            random_days = np.random.randint(0, days_range + 1)
            date = start_dt + timedelta(days=random_days)
            date_str = date.strftime('%Y-%m-%d')
            
            # 随机生成URL
            url = f"http://example.com/policy/{date.year}/{date.month:02d}/{date.day:02d}/{np.random.randint(1000, 9999)}.html"
            
            # 随机选择关键词
            selected_keywords = np.random.choice(keywords, size=min(3, len(keywords)), replace=False)
            keywords_str = ','.join(selected_keywords)
            
            # 添加到结果DataFrame
            policy_data = pd.concat([policy_data, pd.DataFrame({
                'date': [date_str],
                'title': [title],
                'source': [source],
                'content': [content],
                'url': [url],
                'keywords': [keywords_str]
            })], ignore_index=True)
        
        # 按日期排序
        policy_data = policy_data.sort_values('date')
        
        print(f"环保政策数据收集完成，共{len(policy_data)}条记录")
        
        # 保存数据
        if output_path is not None:
            # 确保输出目录存在
            output_dir = os.path.dirname(output_path)
            os.makedirs(output_dir, exist_ok=True)
            
            # 保存为CSV文件
            policy_data.to_csv(output_path, index=False)
            print(f"环保政策数据已保存到: {output_path}")
        
        return policy_data
    
    def collect_event_data(self, cities, start_date, end_date, output_path=None):
        """
        收集重大活动数据
        
        参数:
            cities: 城市列表
            start_date: 开始日期（YYYY-MM-DD）
            end_date: 结束日期（YYYY-MM-DD）
            output_path: 输出文件路径，默认为None
            
        返回:
            DataFrame: 活动数据
        """
        print(f"收集从{start_date}到{end_date}的重大活动数据...")
        
        # 将日期字符串转换为datetime对象
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        # 创建结果DataFrame
        event_data = pd.DataFrame(columns=['date', 'city', 'event_name', 'event_type', 'description', 'impact_level'])
        
        # 示例活动名称和类型
        event_names = [
            "江苏省经济发展论坛",
            "长三角一体化发展峰会",
            "国际环保技术展览会",
            "中国(南京)软件产业博览会",
            "江苏省运动会",
            "世界智能制造大会",
            "中国国际进口博览会",
            "长江经济带发展论坛",
            "国际生态环境保护大会",
            "中国(苏州)国际旅游节"
        ]
        
        event_types = ["会议", "展览", "体育赛事", "文化活动", "政治活动"]
        
        event_descriptions = [
            "汇聚省内外专家学者，共同探讨江苏经济高质量发展路径。",
            "推动长三角地区一体化发展，加强区域合作与交流。",
            "展示最新环保技术和产品，促进环保产业发展。",
            "展示软件和信息技术服务业最新成果，推动产业创新发展。",
            "全省规模最大的综合性运动会，展示体育健儿风采。",
            "聚焦智能制造前沿技术和应用，推动制造业转型升级。",
            "向世界展示中国开放新姿态，为全球贸易发展注入新动力。",
            "探讨长江经济带绿色发展新路径，推动生态优先战略。",
            "交流国际生态环境保护经验，共同应对全球环境挑战。",
            "展示苏州旅游资源和文化魅力，促进旅游业发展。"
        ]
        
        impact_levels = ["高", "中", "低"]
        
        # 生成随机活动数据
        num_events = 30  # 生成30条活动数据
        
        for _ in tqdm(range(num_events)):
            # 随机选择活动名称和描述
            idx = np.random.randint(0, len(event_names))
            event_name = event_names[idx]
            description = event_descriptions[idx]
            
            # 随机选择城市、活动类型和影响级别
            city = np.random.choice(cities)
            event_type = np.random.choice(event_types)
            impact_level = np.random.choice(impact_levels, p=[0.2, 0.5, 0.3])  # 高、中、低影响的概率
            
            # 随机生成日期
            days_range = (end_dt - start_dt).days
            random_days = np.random.randint(0, days_range + 1)
            date = start_dt + timedelta(days=random_days)
            date_str = date.strftime('%Y-%m-%d')
            
            # 添加到结果DataFrame
            event_data = pd.concat([event_data, pd.DataFrame({
                'date': [date_str],
                'city': [city],
                'event_name': [event_name],
                'event_type': [event_type],
                'description': [description],
                'impact_level': [impact_level]
            })], ignore_index=True)
        
        # 按日期排序
        event_data = event_data.sort_values('date')
        
        print(f"重大活动数据收集完成，共{len(event_data)}条记录")
        
        # 保存数据
        if output_path is not None:
            # 确保输出目录存在
            output_dir = os.path.dirname(output_path)
            os.makedirs(output_dir, exist_ok=True)
            
            # 保存为CSV文件
            event_data.to_csv(output_path, index=False)
            print(f"重大活动数据已保存到: {output_path}")
        
        return event_data
    
    def collect_renewable_energy_data(self, provinces, start_date, end_date, output_path=None):
        """
        收集新能源装机数据
        
        参数:
            provinces: 省份列表
            start_date: 开始日期（YYYY-MM-DD）
            end_date: 结束日期（YYYY-MM-DD）
            output_path: 输出文件路径，默认为None
            
        返回:
            DataFrame: 新能源装机数据
        """
        print(f"收集从{start_date}到{end_date}的新能源装机数据...")
        
        # 将日期字符串转换为datetime对象
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        # 创建月份范围
        month_range = pd.date_range(start=start_dt, end=end_dt, freq='MS')  # 月初
        
        # 创建结果DataFrame
        renewable_data = pd.DataFrame(columns=['date', 'province', 'wind_power', 'solar_power', 'biomass_power', 'total_renewable'])
        
        # 为每个省份生成数据
        for province in provinces:
            # 初始装机容量（兆瓦）
            initial_wind = np.random.uniform(5000, 10000)
            initial_solar = np.random.uniform(3000, 8000)
            initial_biomass = np.random.uniform(500, 2000)
            
            # 月增长率
            wind_growth_rate = np.random.uniform(0.01, 0.03)
            solar_growth_rate = np.random.uniform(0.02, 0.05)
            biomass_growth_rate = np.random.uniform(0.005, 0.015)
            
            # 为每个月生成数据
            for i, date in enumerate(month_range):
                # 计算当前月份的装机容量
                wind_power = initial_wind * (1 + wind_growth_rate) ** i
                solar_power = initial_solar * (1 + solar_growth_rate) ** i
                biomass_power = initial_biomass * (1 + biomass_growth_rate) ** i
                
                # 添加随机波动
                wind_power *= np.random.uniform(0.95, 1.05)
                solar_power *= np.random.uniform(0.95, 1.05)
                biomass_power *= np.random.uniform(0.95, 1.05)
                
                # 计算总装机容量
                total_renewable = wind_power + solar_power + biomass_power
                
                # 添加到结果DataFrame
                renewable_data = pd.concat([renewable_data, pd.DataFrame({
                    'date': [date.strftime('%Y-%m-%d')],
                    'province': [province],
                    'wind_power': [wind_power],
                    'solar_power': [solar_power],
                    'biomass_power': [biomass_power],
                    'total_renewable': [total_renewable]
                })], ignore_index=True)
        
        # 按日期和省份排序
        renewable_data = renewable_data.sort_values(['date', 'province'])
        
        print(f"新能源装机数据收集完成，共{len(renewable_data)}条记录")
        
        # 保存数据
        if output_path is not None:
            # 确保输出目录存在
            output_dir = os.path.dirname(output_path)
            os.makedirs(output_dir, exist_ok=True)
            
            # 保存为CSV文件
            renewable_data.to_csv(output_path, index=False)
            print(f"新能源装机数据已保存到: {output_path}")
        
        return renewable_data
    
    def collect_all_external_data(self, start_date, end_date, output_dir):
        """
        收集所有外部数据
        
        参数:
            start_date: 开始日期（YYYY-MM-DD）
            end_date: 结束日期（YYYY-MM-DD）
            output_dir: 输出目录
            
        返回:
            dict: 包含所有外部数据的字典
        """
        print(f"收集从{start_date}到{end_date}的所有外部数据...")
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 定义城市和省份列表
        cities = ["南京", "苏州", "无锡", "常州", "南通", "扬州", "镇江", "泰州", "盐城", "徐州", "连云港", "淮安", "宿迁"]
        provinces = ["江苏"]
        
        # 定义环保政策关键词
        policy_keywords = ["环保", "减排", "污染物", "大气污染", "超低排放", "脱硫", "脱硝", "除尘", 
                          "碳达峰", "碳中和", "新能源", "可再生能源", "清洁能源", "能源结构", "电力"]
        
        # 1. 收集天气数据
        weather_data = {}
        for city in cities:
            weather_file = os.path.join(output_dir, f"weather_{city}.csv")
            weather_data[city] = self.collect_weather_data(city, start_date, end_date, weather_file)
        
        # 2. 收集环保政策数据
        policy_file = os.path.join(output_dir, "environmental_policies.csv")
        policy_data = self.collect_policy_data(policy_keywords, start_date, end_date, policy_file)
        
        # 3. 收集重大活动数据
        event_file = os.path.join(output_dir, "major_events.csv")
        event_data = self.collect_event_data(cities, start_date, end_date, event_file)
        
        # 4. 收集新能源装机数据
        renewable_file = os.path.join(output_dir, "renewable_energy.csv")
        renewable_data = self.collect_renewable_energy_data(provinces, start_date, end_date, renewable_file)
        
        # 创建结果字典
        result = {
            'weather': weather_data,
            'policy': policy_data,
            'event': event_data,
            'renewable': renewable_data
        }
        
        print("所有外部数据收集完成")
        return result
    
    def merge_external_data_by_date(self, external_data, output_path=None):
        """
        按日期合并所有外部数据
        
        参数:
            external_data: 外部数据字典
            output_path: 输出文件路径，默认为None
            
        返回:
            DataFrame: 合并后的外部数据
        """
        print("按日期合并所有外部数据...")
        
        # 提取各类数据
        weather_data = external_data.get('weather', {})
        policy_data = external_data.get('policy', None)
        event_data = external_data.get('event', None)
        renewable_data = external_data.get('renewable', None)
        
        # 创建日期列表
        all_dates = set()
        
        # 添加天气数据的日期
        for city, df in weather_data.items():
            if df is not None:
                all_dates.update(df['date'].tolist())
        
        # 添加政策数据的日期
        if policy_data is not None:
            all_dates.update(policy_data['date'].tolist())
        
        # 添加活动数据的日期
        if event_data is not None:
            all_dates.update(event_data['date'].tolist())
        
        # 添加新能源数据的日期
        if renewable_data is not None:
            all_dates.update(renewable_data['date'].tolist())
        
        # 转换为列表并排序
        all_dates = sorted(list(all_dates))
        
        # 创建结果DataFrame
        merged_data = pd.DataFrame({'date': all_dates})
        
        # 合并天气数据
        for city, df in weather_data.items():
            if df is not None:
                # 选择需要的列
                weather_cols = ['temperature', 'humidity', 'wind_speed', 'weather_condition']
                city_weather = df[['date'] + weather_cols].copy()
                
                # 重命名列，添加城市前缀
                for col in weather_cols:
                    city_weather = city_weather.rename(columns={col: f"{city}_{col}"})
                
                # 合并数据
                merged_data = pd.merge(merged_data, city_weather, on='date', how='left')
        
        # 合并政策数据
        if policy_data is not None:
            # 计算每天的政策数量
            policy_counts = policy_data.groupby('date').size().reset_index(name='policy_count')
            merged_data = pd.merge(merged_data, policy_counts, on='date', how='left')
            
            # 填充缺失值
            merged_data['policy_count'] = merged_data['policy_count'].fillna(0)
            
            # 添加政策影响指标（简单示例：当天有政策为1，否则为0）
            merged_data['policy_impact'] = (merged_data['policy_count'] > 0).astype(int)
        
        # 合并活动数据
        if event_data is not None:
            # 计算每天的活动数量
            event_counts = event_data.groupby('date').size().reset_index(name='event_count')
            merged_data = pd.merge(merged_data, event_counts, on='date', how='left')
            
            # 填充缺失值
            merged_data['event_count'] = merged_data['event_count'].fillna(0)
            
            # 计算每天的高影响活动数量
            high_impact_events = event_data[event_data['impact_level'] == '高'].groupby('date').size().reset_index(name='high_impact_event_count')
            merged_data = pd.merge(merged_data, high_impact_events, on='date', how='left')
            
            # 填充缺失值
            merged_data['high_impact_event_count'] = merged_data['high_impact_event_count'].fillna(0)
        
        # 合并新能源数据
        if renewable_data is not None:
            # 计算每个省份的新能源总装机容量
            for province in renewable_data['province'].unique():
                province_data = renewable_data[renewable_data['province'] == province]
                province_data = province_data[['date', 'total_renewable']].rename(columns={'total_renewable': f"{province}_renewable_capacity"})
                merged_data = pd.merge(merged_data, province_data, on='date', how='left')
                
                # 使用前向填充处理缺失值（假设装机容量在月度数据之间保持不变）
                merged_data[f"{province}_renewable_capacity"] = merged_data[f"{province}_renewable_capacity"].fillna(method='ffill')
        
        # 将日期列转换为datetime类型
        merged_data['date'] = pd.to_datetime(merged_data['date'])
        
        # 添加时间特征
        merged_data['year'] = merged_data['date'].dt.year
        merged_data['month'] = merged_data['date'].dt.month
        merged_data['day'] = merged_data['date'].dt.day
        merged_data['dayofweek'] = merged_data['date'].dt.dayofweek
        merged_data['is_weekend'] = merged_data['dayofweek'] >= 5
        
        # 添加季节特征
        merged_data['season'] = merged_data['month'].apply(lambda x: 
                                                         1 if x in [3, 4, 5] else  # 春季
                                                         2 if x in [6, 7, 8] else  # 夏季
                                                         3 if x in [9, 10, 11] else  # 秋季
                                                         4)  # 冬季
        
        print(f"外部数据合并完成，共{len(merged_data)}条记录")
        
        # 保存数据
        if output_path is not None:
            # 确保输出目录存在
            output_dir = os.path.dirname(output_path)
            os.makedirs(output_dir, exist_ok=True)
            
            # 保存为CSV文件
            merged_data.to_csv(output_path, index=False)
            print(f"合并后的外部数据已保存到: {output_path}")
        
        return merged_data

# 测试代码
if __name__ == "__main__":
    # 创建外部数据收集器
    collector = ExternalDataCollector()
    
    # 设置日期范围
    start_date = "2024-01-01"
    end_date = "2024-12-31"
    
    # 设置输出目录
    output_dir = "./data/external"
    
    # 收集所有外部数据
    external_data = collector.collect_all_external_data(start_date, end_date, output_dir)
    
    # 按日期合并所有外部数据
    merged_data = collector.merge_external_data_by_date(external_data, os.path.join(output_dir, "merged_external_data.csv"))

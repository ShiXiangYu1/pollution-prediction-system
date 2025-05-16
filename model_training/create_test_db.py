#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
创建测试数据库和表结构
"""

import os
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import random

def create_test_database(db_path='./data/test_pollution_db.sqlite'):
    """
    创建测试数据库及相应表结构
    
    参数:
        db_path: 数据库文件路径
    """
    print(f"创建测试数据库: {db_path}")
    
    # 确保数据库目录存在
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    # 连接到数据库（如果不存在则创建）
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # 创建地区表(TS_AREA)
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS TS_AREA (
        AREA_CODE TEXT PRIMARY KEY,
        AREA_NAME TEXT NOT NULL,
        PARENT_CODE TEXT
    )
    ''')
    
    # 创建电厂表(TB_FACTORY)
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS TB_FACTORY (
        FAC_ID TEXT PRIMARY KEY,
        AREA_CODE TEXT NOT NULL,
        FAC_NAME TEXT NOT NULL,
        GROUP_ID TEXT,
        FAC_ADDR TEXT,
        X_MAP REAL,
        Y_MAP REAL,
        FAC_ALIAS TEXT,
        ACTIVE_FLAG INTEGER DEFAULT 1,
        FOREIGN KEY (AREA_CODE) REFERENCES TS_AREA (AREA_CODE)
    )
    ''')
    
    # 创建机组表(TB_STEAMER)
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS TB_STEAMER (
        STEAMER_ID TEXT PRIMARY KEY,
        STEAMER_NAME TEXT NOT NULL,
        FAC_ID TEXT NOT NULL,
        RATING_FH REAL,
        ACTIVE_FLAG INTEGER DEFAULT 1,
        STEAMER_TYPE TEXT,
        FOREIGN KEY (FAC_ID) REFERENCES TB_FACTORY (FAC_ID)
    )
    ''')
    
    # 创建电厂测点信息表(tb_rtu_channel)
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS tb_rtu_channel (
        FAC_ID TEXT NOT NULL,
        CHANNEL_NUM TEXT PRIMARY KEY,
        CHANNEL_NAME TEXT NOT NULL,
        KKS_WORK TEXT,
        ACTIVE_FLAG INTEGER DEFAULT 1,
        FOREIGN KEY (FAC_ID) REFERENCES TB_FACTORY (FAC_ID)
    )
    ''')
    
    # 创建测点采集数据表(td_hisdata)
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS td_hisdata (
        KKS_WORK TEXT NOT NULL,
        RECTIME TIMESTAMP NOT NULL,
        VALUE REAL,
        PRIMARY KEY (KKS_WORK, RECTIME)
    )
    ''')
    
    # 创建二氧化硫小时数据表(td_dtl_2025)
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS td_dtl_2025 (
        RECTIME TIMESTAMP NOT NULL,
        STEAMER_ID TEXT NOT NULL,
        VENT_SO2_CHK REAL,
        VENT_SO2_T REAL,
        FGD_EFCY REAL,
        FD_KWH REAL,
        CHK_TIME TIMESTAMP,
        STOP_TIME TIMESTAMP,
        STOP_CAUSE TEXT,
        OVER_MULTIPLE REAL,
        TIME_2TO30 INTEGER,
        TIME_30TO50 INTEGER,
        PRIMARY KEY (STEAMER_ID, RECTIME),
        FOREIGN KEY (STEAMER_ID) REFERENCES TB_STEAMER (STEAMER_ID)
    )
    ''')
    
    # 创建氮氧化物小时数据表(td_dtx_2025)
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS td_dtx_2025 (
        RECTIME TIMESTAMP NOT NULL,
        STEAMER_ID TEXT NOT NULL,
        VENT_NOX_CHK REAL,
        VENT_NOX_T REAL,
        SCR_EFCY REAL,
        FD_KWH REAL,
        CHK_TIME TIMESTAMP,
        STOP_TIME TIMESTAMP,
        STOP_CAUSE TEXT,
        OVER_MULTIPLE REAL,
        TIME_2TO30 INTEGER,
        TIME_30TO50 INTEGER,
        PRIMARY KEY (STEAMER_ID, RECTIME),
        FOREIGN KEY (STEAMER_ID) REFERENCES TB_STEAMER (STEAMER_ID)
    )
    ''')
    
    # 创建烟尘小时数据表(td_dcc_2025)
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS td_dcc_2025 (
        RECTIME TIMESTAMP NOT NULL,
        STEAMER_ID TEXT NOT NULL,
        VENT_SOOT_CHK REAL,
        VENT_SOOT_T REAL,
        FD_KWH REAL,
        CHK_TIME TIMESTAMP,
        STOP_TIME TIMESTAMP,
        STOP_CAUSE TEXT,
        OVER_MULTIPLE REAL,
        TIME_2TO30 INTEGER,
        TIME_30TO50 INTEGER,
        PRIMARY KEY (STEAMER_ID, RECTIME),
        FOREIGN KEY (STEAMER_ID) REFERENCES TB_STEAMER (STEAMER_ID)
    )
    ''')
    
    # 提交创建表操作
    conn.commit()
    
    print("数据库表结构创建完成")
    return conn

def generate_test_data(conn):
    """
    生成测试数据
    
    参数:
        conn: 数据库连接
    """
    print("生成测试数据...")
    
    cursor = conn.cursor()
    
    # 生成地区数据
    areas = [
        ('01', '江苏省', None),
        ('0101', '南京市', '01'),
        ('0102', '无锡市', '01'),
        ('0103', '徐州市', '01'),
        ('0104', '常州市', '01'),
        ('0105', '苏州市', '01'),
        ('0106', '南通市', '01'),
        ('0107', '连云港市', '01'),
        ('0108', '淮安市', '01'),
        ('0109', '盐城市', '01'),
        ('0110', '扬州市', '01'),
        ('0111', '镇江市', '01'),
        ('0112', '泰州市', '01'),
        ('0113', '宿迁市', '01')
    ]
    
    cursor.executemany("INSERT OR REPLACE INTO TS_AREA (AREA_CODE, AREA_NAME, PARENT_CODE) VALUES (?, ?, ?)", areas)
    
    # 生成电厂数据
    factories = [
        ('F001', '0101', '南京电厂', 'G001', '南京市浦口区', 118.78, 32.06, '南电', 1),
        ('F002', '0102', '无锡电厂', 'G001', '无锡市锡山区', 120.30, 31.59, '锡电', 1),
        ('F003', '0103', '徐州电厂', 'G002', '徐州市云龙区', 117.19, 34.27, '徐电', 1),
        ('F004', '0105', '苏州电厂', 'G002', '苏州市吴中区', 120.62, 31.30, '苏电', 1),
        ('F005', '0109', '盐城电厂', 'G003', '盐城市亭湖区', 120.16, 33.38, '盐电', 1)
    ]
    
    cursor.executemany("INSERT OR REPLACE INTO TB_FACTORY (FAC_ID, AREA_CODE, FAC_NAME, GROUP_ID, FAC_ADDR, X_MAP, Y_MAP, FAC_ALIAS, ACTIVE_FLAG) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", factories)
    
    # 生成机组数据
    steamers = [
        ('S001', '1号机组', 'F001', 600, 1, '燃煤机'),
        ('S002', '2号机组', 'F001', 600, 1, '燃煤机'),
        ('S003', '1号机组', 'F002', 350, 1, '燃气机'),
        ('S004', '2号机组', 'F002', 350, 1, '燃气机'),
        ('S005', '1号机组', 'F003', 1000, 1, '燃煤机'),
        ('S006', '1号机组', 'F004', 660, 1, '燃煤机'),
        ('S007', '2号机组', 'F004', 660, 1, '燃煤机'),
        ('S008', '1号机组', 'F005', 350, 1, '燃气机')
    ]
    
    cursor.executemany("INSERT OR REPLACE INTO TB_STEAMER (STEAMER_ID, STEAMER_NAME, FAC_ID, RATING_FH, ACTIVE_FLAG, STEAMER_TYPE) VALUES (?, ?, ?, ?, ?, ?)", steamers)
    
    # 生成测点信息数据
    channels = [
        ('F001', 'C001', 'SO2浓度', 'KKS001', 1),
        ('F001', 'C002', 'NOx浓度', 'KKS002', 1),
        ('F001', 'C003', '烟尘浓度', 'KKS003', 1),
        ('F002', 'C004', 'SO2浓度', 'KKS004', 1),
        ('F002', 'C005', 'NOx浓度', 'KKS005', 1),
        ('F002', 'C006', '烟尘浓度', 'KKS006', 1)
    ]
    
    cursor.executemany("INSERT OR REPLACE INTO tb_rtu_channel (FAC_ID, CHANNEL_NUM, CHANNEL_NAME, KKS_WORK, ACTIVE_FLAG) VALUES (?, ?, ?, ?, ?)", channels)
    
    # 生成污染物排放数据
    # 设置时间范围 - 最近7天的小时数据
    end_time = datetime.now().replace(minute=0, second=0, microsecond=0)
    start_time = end_time - timedelta(days=7)
    
    # 生成时间序列
    times = [start_time + timedelta(hours=i) for i in range(int((end_time - start_time).total_seconds() / 3600) + 1)]
    
    # 基本参数
    coal_so2_std = 35  # 燃煤机组SO2标准限值
    coal_nox_std = 50  # 燃煤机组NOx标准限值
    coal_soot_std = 5  # 燃煤机组烟尘标准限值
    
    gas_so2_std = 35  # 燃气机组SO2标准限值
    gas_nox_std = 50  # 燃气机组NOx标准限值
    gas_soot_std = 5  # 燃气机组烟尘标准限值
    
    # 为每个机组生成排放数据
    so2_data = []
    nox_data = []
    soot_data = []
    
    for steamer_id, _, _, rating, _, steamer_type in steamers:
        for t in times:
            # 生成一些随机波动，有时超标
            is_peak_hour = 8 <= t.hour <= 20  # 高峰时段
            is_weekend = t.weekday() >= 5  # 周末
            
            # 基础负荷和排放
            base_load = 0.7 + 0.2 * random.random()  # 基础负荷系数
            if is_peak_hour:
                base_load += 0.1  # 高峰时段负荷增加
            if is_weekend:
                base_load -= 0.1  # 周末负荷降低
                
            # 确保负荷在合理范围内
            base_load = max(0.4, min(0.95, base_load))
            
            # 计算发电量
            generation = rating * base_load
            
            # 随机生成一些超标情况
            is_so2_exceedance = random.random() < 0.05  # 5%概率SO2超标
            is_nox_exceedance = random.random() < 0.08  # 8%概率NOx超标
            is_soot_exceedance = random.random() < 0.03  # 3%概率烟尘超标
            
            # 根据机组类型选择不同的标准和基准排放
            if steamer_type == '燃煤机':
                so2_std = coal_so2_std
                nox_std = coal_nox_std
                soot_std = coal_soot_std
                
                base_so2 = 25 + 5 * random.random()  # 基础SO2浓度
                base_nox = 40 + 5 * random.random()  # 基础NOx浓度
                base_soot = 3 + 1 * random.random()  # 基础烟尘浓度
            else:  # 燃气机
                so2_std = gas_so2_std
                nox_std = gas_nox_std
                soot_std = gas_soot_std
                
                base_so2 = 15 + 5 * random.random()  # 基础SO2浓度
                base_nox = 30 + 5 * random.random()  # 基础NOx浓度
                base_soot = 2 + 0.5 * random.random()  # 基础烟尘浓度
            
            # 超标情况下增加排放值
            if is_so2_exceedance:
                so2_value = so2_std * (1.1 + 0.5 * random.random())  # 超标10%-60%
                so2_over = (so2_value / so2_std) - 1
                so2_time_2to30 = int(random.randint(10, 30)) if so2_over < 0.3 else 0
                so2_time_30to50 = int(random.randint(5, 15)) if so2_over >= 0.3 else 0
            else:
                so2_value = base_so2
                so2_over = 0
                so2_time_2to30 = 0
                so2_time_30to50 = 0
            
            if is_nox_exceedance:
                nox_value = nox_std * (1.1 + 0.6 * random.random())  # 超标10%-70%
                nox_over = (nox_value / nox_std) - 1
                nox_time_2to30 = int(random.randint(10, 30)) if nox_over < 0.3 else 0
                nox_time_30to50 = int(random.randint(5, 15)) if nox_over >= 0.3 else 0
            else:
                nox_value = base_nox
                nox_over = 0
                nox_time_2to30 = 0
                nox_time_30to50 = 0
            
            if is_soot_exceedance:
                soot_value = soot_std * (1.1 + 0.4 * random.random())  # 超标10%-50%
                soot_over = (soot_value / soot_std) - 1
                soot_time_2to30 = int(random.randint(10, 30)) if soot_over < 0.3 else 0
                soot_time_30to50 = int(random.randint(5, 15)) if soot_over >= 0.3 else 0
            else:
                soot_value = base_soot
                soot_over = 0
                soot_time_2to30 = 0
                soot_time_30to50 = 0
            
            # 计算排放量
            flue_gas_flow = generation * (12 if steamer_type == '燃煤机' else 8)  # 烟气流量，假设与发电量成正比
            so2_emission = so2_value * flue_gas_flow * 0.01  # SO2排放量
            nox_emission = nox_value * flue_gas_flow * 0.01  # NOx排放量
            soot_emission = soot_value * flue_gas_flow * 0.001  # 烟尘排放量
            
            # 脱硫脱硝效率
            fgd_efficiency = 90 + 5 * random.random()  # 脱硫效率
            scr_efficiency = 85 + 10 * random.random()  # 脱硝效率
            
            # 添加到数据列表
            so2_data.append((
                t.strftime('%Y-%m-%d %H:%M:%S'),
                steamer_id,
                round(so2_value, 2),
                round(so2_emission, 2),
                round(fgd_efficiency, 2),
                round(generation, 2),
                t.strftime('%Y-%m-%d %H:%M:%S'),
                None,
                None,
                round(so2_over, 2) if so2_over > 0 else None,
                so2_time_2to30,
                so2_time_30to50
            ))
            
            nox_data.append((
                t.strftime('%Y-%m-%d %H:%M:%S'),
                steamer_id,
                round(nox_value, 2),
                round(nox_emission, 2),
                round(scr_efficiency, 2),
                round(generation, 2),
                t.strftime('%Y-%m-%d %H:%M:%S'),
                None,
                None,
                round(nox_over, 2) if nox_over > 0 else None,
                nox_time_2to30,
                nox_time_30to50
            ))
            
            soot_data.append((
                t.strftime('%Y-%m-%d %H:%M:%S'),
                steamer_id,
                round(soot_value, 2),
                round(soot_emission, 2),
                round(generation, 2),
                t.strftime('%Y-%m-%d %H:%M:%S'),
                None,
                None,
                round(soot_over, 2) if soot_over > 0 else None,
                soot_time_2to30,
                soot_time_30to50
            ))
    
    # 批量插入数据
    cursor.executemany('''
    INSERT OR REPLACE INTO td_dtl_2025 (
        RECTIME, STEAMER_ID, VENT_SO2_CHK, VENT_SO2_T, FGD_EFCY, FD_KWH, 
        CHK_TIME, STOP_TIME, STOP_CAUSE, OVER_MULTIPLE, TIME_2TO30, TIME_30TO50
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', so2_data)
    
    cursor.executemany('''
    INSERT OR REPLACE INTO td_dtx_2025 (
        RECTIME, STEAMER_ID, VENT_NOX_CHK, VENT_NOX_T, SCR_EFCY, FD_KWH, 
        CHK_TIME, STOP_TIME, STOP_CAUSE, OVER_MULTIPLE, TIME_2TO30, TIME_30TO50
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', nox_data)
    
    cursor.executemany('''
    INSERT OR REPLACE INTO td_dcc_2025 (
        RECTIME, STEAMER_ID, VENT_SOOT_CHK, VENT_SOOT_T, FD_KWH, 
        CHK_TIME, STOP_TIME, STOP_CAUSE, OVER_MULTIPLE, TIME_2TO30, TIME_30TO50
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', soot_data)
    
    # 提交插入数据操作
    conn.commit()
    
    # 统计插入的数据量
    cursor.execute("SELECT COUNT(*) FROM td_dtl_2025")
    so2_count = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM td_dtx_2025")
    nox_count = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM td_dcc_2025")
    soot_count = cursor.fetchone()[0]
    
    print(f"测试数据生成完成，共插入SO2数据{so2_count}条，NOx数据{nox_count}条，烟尘数据{soot_count}条")

def main():
    """主函数"""
    db_path = './data/test_pollution_db.sqlite'
    
    # 创建数据库和表结构
    conn = create_test_database(db_path)
    
    # 生成测试数据
    generate_test_data(conn)
    
    # 关闭连接
    conn.close()
    
    print(f"测试数据库创建完成: {db_path}")

if __name__ == "__main__":
    main() 
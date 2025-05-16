#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据向量化模块 - 使用DeepSeek-8B模型处理外部文本数据
"""

import pandas as pd
import numpy as np
import os
import glob
import json
import torch
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModel
import joblib
from sklearn.metrics.pairwise import cosine_similarity

class TextVectorizer:
    """
    文本向量化类，用于处理外部文本数据并生成语义向量
    """
    
    def __init__(self, model_name="deepseek-ai/deepseek-llm-7b-base", data_dir='./data'):
        """
        初始化文本向量化类
        
        参数:
            model_name: 模型名称或路径
            data_dir: 数据目录
        """
        self.data_dir = data_dir
        self.model_name = model_name
        
        # 创建向量数据目录（如果不存在）
        os.makedirs(os.path.join(data_dir, 'vectors'), exist_ok=True)
        
        # 加载模型和分词器
        print(f"加载模型: {model_name}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            
            # 检查是否有GPU可用
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = self.model.to(self.device)
            
            print(f"模型加载成功，使用设备: {self.device}")
        except Exception as e:
            print(f"模型加载失败: {e}")
            self.tokenizer = None
            self.model = None
    
    def load_external_text_data(self, file_path):
        """
        加载外部文本数据
        
        参数:
            file_path: 文本数据文件路径
            
        返回:
            DataFrame: 处理后的文本数据
        """
        print(f"加载外部文本数据: {file_path}")
        
        try:
            # 根据文件扩展名选择加载方法
            ext = os.path.splitext(file_path)[1].lower()
            
            if ext == '.csv':
                df = pd.read_csv(file_path)
            elif ext == '.xlsx' or ext == '.xls':
                df = pd.read_excel(file_path)
            elif ext == '.json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                df = pd.DataFrame(data)
            elif ext == '.txt':
                # 假设文本文件每行是一个文档
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                df = pd.DataFrame({'text': lines})
            else:
                print(f"不支持的文件格式: {ext}")
                return None
            
            print(f"成功加载外部文本数据，形状: {df.shape}")
            return df
        except Exception as e:
            print(f"加载外部文本数据失败: {e}")
            return None
    
    def preprocess_text(self, text):
        """
        预处理文本
        
        参数:
            text: 输入文本
            
        返回:
            str: 预处理后的文本
        """
        if not isinstance(text, str):
            return ""
        
        # 基本清洗
        text = text.strip()
        
        # 移除多余的空白字符
        text = ' '.join(text.split())
        
        return text
    
    def generate_embeddings(self, texts, batch_size=8):
        """
        生成文本嵌入向量
        
        参数:
            texts: 文本列表
            batch_size: 批处理大小
            
        返回:
            numpy.ndarray: 嵌入向量数组
        """
        if self.model is None or self.tokenizer is None:
            print("模型未加载，无法生成嵌入向量")
            return None
        
        print(f"为 {len(texts)} 个文本生成嵌入向量...")
        
        # 预处理文本
        processed_texts = [self.preprocess_text(text) for text in texts]
        
        # 过滤空文本
        valid_indices = [i for i, text in enumerate(processed_texts) if text]
        valid_texts = [processed_texts[i] for i in valid_indices]
        
        if not valid_texts:
            print("没有有效的文本需要处理")
            return None
        
        # 分批处理
        embeddings = []
        
        for i in tqdm(range(0, len(valid_texts), batch_size)):
            batch_texts = valid_texts[i:i+batch_size]
            
            # 编码文本
            inputs = self.tokenizer(batch_texts, padding=True, truncation=True, 
                                   max_length=512, return_tensors="pt")
            
            # 将输入移动到设备
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 生成嵌入向量
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # 使用最后一层隐藏状态的平均值作为嵌入向量
            batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            embeddings.append(batch_embeddings)
        
        # 合并所有批次的嵌入向量
        all_embeddings = np.vstack(embeddings)
        
        # 创建完整的嵌入向量数组（包括空文本）
        result = np.zeros((len(texts), all_embeddings.shape[1]))
        for i, idx in enumerate(valid_indices):
            result[idx] = all_embeddings[i]
        
        print(f"嵌入向量生成完成，形状: {result.shape}")
        return result
    
    def extract_keywords(self, texts, top_k=5):
        """
        提取文本关键词
        
        参数:
            texts: 文本列表
            top_k: 每个文本提取的关键词数量
            
        返回:
            list: 关键词列表的列表
        """
        if self.model is None or self.tokenizer is None:
            print("模型未加载，无法提取关键词")
            return None
        
        print(f"为 {len(texts)} 个文本提取关键词...")
        
        # 预处理文本
        processed_texts = [self.preprocess_text(text) for text in texts]
        
        # 过滤空文本
        valid_indices = [i for i, text in enumerate(processed_texts) if text]
        valid_texts = [processed_texts[i] for i in valid_indices]
        
        if not valid_texts:
            print("没有有效的文本需要处理")
            return None
        
        # 使用模型提取关键词
        all_keywords = []
        
        for text in tqdm(valid_texts):
            # 构建提示
            prompt = f"请从以下文本中提取{top_k}个最重要的关键词，只返回关键词，用逗号分隔：\n\n{text}"
            
            # 编码提示
            inputs = self.tokenizer(prompt, padding=True, truncation=True, 
                                   max_length=512, return_tensors="pt")
            
            # 将输入移动到设备
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 生成关键词
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=50,
                    num_return_sequences=1,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True
                )
            
            # 解码输出
            keywords_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 提取关键词
            keywords = [kw.strip() for kw in keywords_text.split(',')]
            all_keywords.append(keywords[:top_k])  # 确保只取top_k个关键词
        
        # 创建完整的关键词列表（包括空文本）
        result = [[] for _ in range(len(texts))]
        for i, idx in enumerate(valid_indices):
            result[idx] = all_keywords[i]
        
        print(f"关键词提取完成")
        return result
    
    def classify_text(self, texts, categories):
        """
        对文本进行分类
        
        参数:
            texts: 文本列表
            categories: 类别列表
            
        返回:
            list: 分类结果列表
        """
        if self.model is None or self.tokenizer is None:
            print("模型未加载，无法进行文本分类")
            return None
        
        print(f"为 {len(texts)} 个文本进行分类...")
        
        # 预处理文本
        processed_texts = [self.preprocess_text(text) for text in texts]
        
        # 过滤空文本
        valid_indices = [i for i, text in enumerate(processed_texts) if text]
        valid_texts = [processed_texts[i] for i in valid_indices]
        
        if not valid_texts:
            print("没有有效的文本需要处理")
            return None
        
        # 生成类别嵌入向量
        category_embeddings = self.generate_embeddings(categories)
        
        # 生成文本嵌入向量
        text_embeddings = self.generate_embeddings(valid_texts)
        
        # 计算文本与类别之间的相似度
        similarities = cosine_similarity(text_embeddings, category_embeddings)
        
        # 获取最相似的类别
        category_indices = np.argmax(similarities, axis=1)
        classifications = [categories[idx] for idx in category_indices]
        
        # 创建完整的分类结果列表（包括空文本）
        result = ["" for _ in range(len(texts))]
        for i, idx in enumerate(valid_indices):
            result[idx] = classifications[i]
        
        print(f"文本分类完成")
        return result
    
    def sentiment_analysis(self, texts):
        """
        情感分析
        
        参数:
            texts: 文本列表
            
        返回:
            list: 情感分析结果列表（正面、负面、中性）
        """
        if self.model is None or self.tokenizer is None:
            print("模型未加载，无法进行情感分析")
            return None
        
        print(f"为 {len(texts)} 个文本进行情感分析...")
        
        # 预处理文本
        processed_texts = [self.preprocess_text(text) for text in texts]
        
        # 过滤空文本
        valid_indices = [i for i, text in enumerate(processed_texts) if text]
        valid_texts = [processed_texts[i] for i in valid_indices]
        
        if not valid_texts:
            print("没有有效的文本需要处理")
            return None
        
        # 使用模型进行情感分析
        sentiments = []
        
        for text in tqdm(valid_texts):
            # 构建提示
            prompt = f"请对以下文本进行情感分析，只返回'正面'、'负面'或'中性'之一：\n\n{text}"
            
            # 编码提示
            inputs = self.tokenizer(prompt, padding=True, truncation=True, 
                                   max_length=512, return_tensors="pt")
            
            # 将输入移动到设备
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 生成情感分析结果
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=10,
                    num_return_sequences=1,
                    temperature=0.3,
                    top_p=0.9,
                    do_sample=True
                )
            
            # 解码输出
            sentiment_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            
            # 标准化情感结果
            if "正面" in sentiment_text:
                sentiment = "正面"
            elif "负面" in sentiment_text:
                sentiment = "负面"
            else:
                sentiment = "中性"
            
            sentiments.append(sentiment)
        
        # 创建完整的情感分析结果列表（包括空文本）
        result = ["" for _ in range(len(texts))]
        for i, idx in enumerate(valid_indices):
            result[idx] = sentiments[i]
        
        print(f"情感分析完成")
        return result
    
    def process_external_data(self, df, text_col, date_col=None, output_path=None):
        """
        处理外部数据
        
        参数:
            df: 外部数据DataFrame
            text_col: 文本列名
            date_col: 日期列名，默认为None
            output_path: 输出文件路径，默认为None
            
        返回:
            DataFrame: 处理后的数据
        """
        if df is None or df.empty:
            print("无数据需要处理")
            return None
        
        if text_col not in df.columns:
            print(f"列 {text_col} 不存在")
            return df
        
        print(f"处理外部数据，文本列: {text_col}")
        
        # 复制数据，避免修改原始数据
        processed_df = df.copy()
        
        # 预处理文本
        processed_df['processed_text'] = processed_df[text_col].apply(self.preprocess_text)
        
        # 生成嵌入向量
        embeddings = self.generate_embeddings(processed_df[text_col].tolist())
        
        if embeddings is not None:
            # 将嵌入向量保存为单独的列
            for i in range(embeddings.shape[1]):
                processed_df[f'embedding_{i}'] = embeddings[:, i]
        
        # 提取关键词
        keywords = self.extract_keywords(processed_df[text_col].tolist())
        
        if keywords is not None:
            processed_df['keywords'] = keywords
        
        # 进行情感分析
        sentiments = self.sentiment_analysis(processed_df[text_col].tolist())
        
        if sentiments is not None:
            processed_df['sentiment'] = sentiments
        
        # 如果有日期列，确保是datetime类型
        if date_col is not None and date_col in processed_df.columns:
            if not pd.api.types.is_datetime64_any_dtype(processed_df[date_col]):
                processed_df[date_col] = pd.to_datetime(processed_df[date_col], errors='coerce')
        
        # 保存处理后的数据
        if output_path is not None:
            # 确保输出目录存在
            output_dir = os.path.dirname(output_path)
            os.makedirs(output_dir, exist_ok=True)
            
            # 保存为CSV文件
            processed_df.to_csv(output_path, index=False)
            print(f"处理后的数据已保存到: {output_path}")
        
        print(f"外部数据处理完成，形状: {processed_df.shape}")
        return processed_df
    
    def create_vector_database(self, texts, metadata=None, output_path=None):
        """
        创建向量数据库
        
        参数:
            texts: 文本列表
            metadata: 元数据列表，默认为None
            output_path: 输出文件路径，默认为None
            
        返回:
            dict: 向量数据库
        """
        print(f"为 {len(texts)} 个文本创建向量数据库...")
        
        # 生成嵌入向量
        embeddings = self.generate_embeddings(texts)
        
        if embeddings is None:
            print("生成嵌入向量失败，无法创建向量数据库")
            return None
        
        # 创建向量数据库
        vector_db = {
            'texts': texts,
            'embeddings': embeddings,
            'metadata': metadata if metadata is not None else [{}] * len(texts)
        }
        
        # 保存向量数据库
        if output_path is not None:
            # 确保输出目录存在
            output_dir = os.path.dirname(output_path)
            os.makedirs(output_dir, exist_ok=True)
            
            # 保存为joblib文件
            joblib.dump(vector_db, output_path)
            print(f"向量数据库已保存到: {output_path}")
        
        print(f"向量数据库创建完成")
        return vector_db
    
    def search_similar_texts(self, query, vector_db, top_k=5):
        """
        搜索相似文本
        
        参数:
            query: 查询文本
            vector_db: 向量数据库
            top_k: 返回的相似文本数量
            
        返回:
            list: 相似文本索引和相似度列表
        """
        if vector_db is None:
            print("向量数据库为空，无法搜索")
            return None
        
        print(f"搜索与查询相似的文本: {query}")
        
        # 生成查询嵌入向量
        query_embedding = self.generate_embeddings([query])
        
        if query_embedding is None:
            print("生成查询嵌入向量失败，无法搜索")
            return None
        
        # 计算相似度
        similarities = cosine_similarity(query_embedding, vector_db['embeddings'])[0]
        
        # 获取相似度最高的文本索引
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # 创建结果列表
        results = []
        for idx in top_indices:
            results.append({
                'index': idx,
                'text': vector_db['texts'][idx],
                'similarity': similarities[idx],
                'metadata': vector_db['metadata'][idx]
            })
        
        print(f"搜索完成，找到 {len(results)} 个相似文本")
        return results

# 测试代码
if __name__ == "__main__":
    # 创建文本向量化器
    vectorizer = TextVectorizer()
    
    # 设置输入和输出目录
    input_dir = "./data/external"
    output_dir = "./data/vectors"
    
    # 确保输入目录存在
    os.makedirs(input_dir, exist_ok=True)
    
    # 示例文本数据
    sample_texts = [
        "国家发改委发布关于进一步推进电力行业减排工作的通知，要求到2025年，全国煤电机组平均供电煤耗降至300克/千瓦时以下。",
        "江苏省生态环境厅发布2024年大气污染防治工作方案，加强对燃煤电厂的监管，推动超低排放改造。",
        "受强冷空气影响，江苏省多地气温骤降，电力负荷创新高，部分电厂满负荷运行。",
        "南京市举行重大活动期间，周边电厂执行临时减排措施，确保空气质量。",
        "新能源装机容量持续增长，2024年一季度江苏省风电、光伏发电量同比增长30%以上。"
    ]
    
    # 创建示例文本文件
    sample_file_path = os.path.join(input_dir, "sample_texts.txt")
    with open(sample_file_path, 'w', encoding='utf-8') as f:
        for text in sample_texts:
            f.write(text + '\n')
    
    # 加载示例文本数据
    df = vectorizer.load_external_text_data(sample_file_path)
    
    if df is not None:
        # 处理外部数据
        processed_df = vectorizer.process_external_data(
            df, 
            text_col='text', 
            output_path=os.path.join(output_dir, "processed_external_data.csv")
        )
        
        # 创建向量数据库
        vector_db = vectorizer.create_vector_database(
            processed_df['text'].tolist(),
            output_path=os.path.join(output_dir, "vector_database.joblib")
        )
        
        # 搜索相似文本
        if vector_db is not None:
            query = "江苏省环保政策对电厂的影响"
            results = vectorizer.search_similar_texts(query, vector_db, top_k=3)
            
            if results is not None:
                print("\n相似文本搜索结果:")
                for i, result in enumerate(results):
                    print(f"{i+1}. 相似度: {result['similarity']:.4f}")
                    print(f"   文本: {result['text']}")
                    print()

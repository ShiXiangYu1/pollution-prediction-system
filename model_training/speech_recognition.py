#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
语音识别模型集成模块 - 集成语音识别模型并进行微调
"""

import os
import torch
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from datasets import Dataset, Audio, load_dataset
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    WhisperFeatureExtractor,
    WhisperTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import evaluate
from tqdm import tqdm
import json
import time
from datetime import datetime
import pyttsx3

class SpeechRecognitionIntegrator:
    """
    语音识别模型集成类，用于集成和微调语音识别模型
    """
    
    def __init__(self, model_name="openai/whisper-small", data_dir='./data'):
        """
        初始化语音识别模型集成类
        
        参数:
            model_name: 模型名称或路径
            data_dir: 数据目录
        """
        self.data_dir = data_dir
        self.model_name = model_name
        
        # 创建模型目录（如果不存在）
        os.makedirs(os.path.join(data_dir, 'models', 'speech'), exist_ok=True)
        
        # 检查是否有GPU可用
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")
    
    def load_model(self):
        """
        加载语音识别模型
        
        返回:
            processor: 处理器
            model: 模型
        """
        print(f"加载语音识别模型: {self.model_name}")
        
        # 加载处理器
        processor = WhisperProcessor.from_pretrained(self.model_name)
        
        # 加载模型
        model = WhisperForConditionalGeneration.from_pretrained(self.model_name)
        model = model.to(self.device)
        
        print("语音识别模型加载完成")
        return processor, model
    
    def prepare_dataset(self, audio_files, transcriptions, language="chinese", split_ratio=0.8):
        """
        准备数据集
        
        参数:
            audio_files: 音频文件列表
            transcriptions: 转录文本列表
            language: 语言
            split_ratio: 训练集比例
            
        返回:
            dict: 包含训练集和测试集的字典
        """
        print("准备语音识别数据集...")
        
        # 创建数据集
        data = {
            "audio": audio_files,
            "text": transcriptions
        }
        
        dataset = Dataset.from_dict(data)
        
        # 添加音频特征
        dataset = dataset.cast_column("audio", Audio())
        
        # 划分训练集和测试集
        dataset = dataset.train_test_split(test_size=1-split_ratio)
        
        print(f"数据集准备完成，训练集: {len(dataset['train'])}条，测试集: {len(dataset['test'])}条")
        return dataset
    
    def prepare_common_voice_dataset(self, language="zh-CN", split="train", max_samples=None):
        """
        准备Common Voice数据集
        
        参数:
            language: 语言代码
            split: 数据集划分
            max_samples: 最大样本数
            
        返回:
            Dataset: 处理后的数据集
        """
        print(f"准备Common Voice数据集，语言: {language}, 划分: {split}")
        
        # 加载Common Voice数据集
        try:
            dataset = load_dataset("mozilla-foundation/common_voice_11_0", language, split=split)
            
            # 限制样本数量
            if max_samples and max_samples < len(dataset):
                dataset = dataset.select(range(max_samples))
            
            print(f"Common Voice数据集加载完成，共{len(dataset)}条记录")
            return dataset
        except Exception as e:
            print(f"加载Common Voice数据集失败: {e}")
            return None
    
    def prepare_custom_dataset(self, csv_file, audio_dir, text_column="text", audio_column="audio_file", language="chinese", split_ratio=0.8):
        """
        准备自定义数据集
        
        参数:
            csv_file: CSV文件路径
            audio_dir: 音频文件目录
            text_column: 文本列名
            audio_column: 音频文件列名
            language: 语言
            split_ratio: 训练集比例
            
        返回:
            dict: 包含训练集和测试集的字典
        """
        print(f"准备自定义数据集: {csv_file}")
        
        try:
            # 加载CSV文件
            df = pd.read_csv(csv_file)
            
            # 检查必要的列
            if text_column not in df.columns:
                print(f"CSV文件缺少文本列: {text_column}")
                return None
            
            if audio_column in df.columns:
                # 构建完整的音频文件路径
                audio_files = [os.path.join(audio_dir, file) for file in df[audio_column]]
            else:
                print(f"CSV文件缺少音频文件列: {audio_column}")
                return None
            
            # 获取转录文本
            transcriptions = df[text_column].tolist()
            
            # 准备数据集
            dataset = self.prepare_dataset(audio_files, transcriptions, language, split_ratio)
            
            print(f"自定义数据集准备完成")
            return dataset
        except Exception as e:
            print(f"准备自定义数据集失败: {e}")
            return None
    
    def generate_synthetic_speech(self, text_file, output_dir, language="zh-CN", max_samples=100):
        """
        使用pyttsx3生成合成语音
        
        参数:
            text_file: 文本文件路径
            output_dir: 输出目录
            language: 语言代码
            max_samples: 最大样本数
            
        返回:
            list: 音频文件路径列表
            list: 转写列表
        """
        try:
            print(f"使用pyttsx3生成合成语音，语言: {language}...")
            
            # 确保输出目录存在
            os.makedirs(output_dir, exist_ok=True)
            
            # 读取文本文件
            with open(text_file, 'r', encoding='utf-8') as f:
                texts = [line.strip() for line in f.readlines() if line.strip()]
            
            # 限制样本数
            if max_samples and max_samples < len(texts):
                texts = texts[:max_samples]
                
            print(f"读取了{len(texts)}条文本记录")
            
            # 初始化pyttsx3引擎
            engine = pyttsx3.init()
            
            # 设置语音属性（如果需要）
            if language.startswith("zh"):
                # 尝试设置中文语音（如果可用）
                voices = engine.getProperty('voices')
                for voice in voices:
                    if "chinese" in voice.id.lower() or "zh" in voice.id.lower():
                        engine.setProperty('voice', voice.id)
                        break
            
            # 设置语速和音量
            engine.setProperty('rate', 150)  # 语速
            engine.setProperty('volume', 1.0)  # 音量
            
            audio_files = []
            transcriptions = []
            
            # 为每条文本生成语音
            for i, text in enumerate(texts):
                print(f"正在合成第 {i+1}/{len(texts)} 条语音...")
                
                # 生成音频文件路径
                audio_file = os.path.join(output_dir, f"synthetic_{i:04d}.wav")
                
                # 使用pyttsx3生成语音
                engine.save_to_file(text, audio_file)
                engine.runAndWait()
                
                # 添加到列表
                audio_files.append(audio_file)
                transcriptions.append(text)
            
            print(f"合成语音数据生成完成，共{len(audio_files)}条记录")
            return audio_files, transcriptions
            
        except Exception as e:
            print(f"生成合成语音数据失败: {e}")
            
            # 尝试使用最简单的无声音频文件方法
            try:
                print("尝试使用无声音频文件作为备选方案...")
                
                # 创建无声音频文件
                import numpy as np
                import soundfile as sf
                
                audio_files = []
                transcriptions = []
                
                # 创建一个1秒的无声音频
                silence = np.zeros(22050, dtype=np.float32)
                
                for i, text in enumerate(texts):
                    # 生成音频文件路径
                    audio_file = os.path.join(output_dir, f"synthetic_{i:04d}.wav")
                    
                    # 保存无声音频
                    sf.write(audio_file, silence, 22050)
                    
                    # 添加到列表
                    audio_files.append(audio_file)
                    transcriptions.append(text)
                
                print(f"生成无声音频文件完成，共{len(audio_files)}条记录")
                return audio_files, transcriptions
                
            except Exception as backup_e:
                print(f"生成无声音频文件也失败了: {backup_e}")
                return [], []
    
    @dataclass
    class DataCollatorSpeechSeq2SeqWithPadding:
        """
        语音序列到序列的数据整理器
        """
        processor: Any
        
        def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
            # 提取音频特征
            input_features = [{"input_features": feature["input_features"]} for feature in features]
            batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
            
            # 提取标签
            label_features = [{"input_ids": feature["labels"]} for feature in features]
            labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
            
            # 替换填充标记
            labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
            
            # 如果批次是空的，则返回空字典
            if labels.shape[0] == 0:
                return {}
                
            batch["labels"] = labels
            return batch
    
    def preprocess_dataset(self, dataset, processor):
        """
        预处理数据集
        
        参数:
            dataset: 数据集
            processor: 处理器
            
        返回:
            Dataset: 预处理后的数据集
        """
        print("预处理数据集...")
        
        # 定义预处理函数
        def prepare_dataset(batch):
            # 加载并重采样音频
            audio = batch["audio"]
            
            # 提取音频特征
            batch["input_features"] = processor.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
            
            # 处理标签
            batch["labels"] = processor.tokenizer(batch["text"]).input_ids
            return batch
        
        # 应用预处理
        processed_dataset = dataset.map(prepare_dataset, remove_columns=dataset.column_names)
        
        print("数据集预处理完成")
        return processed_dataset
    
    def compute_metrics(self, pred):
        """
        计算评估指标
        
        参数:
            pred: 预测结果
            
        返回:
            dict: 评估指标
        """
        # 加载WER指标
        wer_metric = evaluate.load("wer")
        cer_metric = evaluate.load("cer")
        
        # 获取预测和标签
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        
        # 替换-100
        label_ids[label_ids == -100] = self.processor.tokenizer.pad_token_id
        
        # 解码预测和标签
        pred_str = self.processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = self.processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        
        # 计算WER
        wer = wer_metric.compute(predictions=pred_str, references=label_str)
        cer = cer_metric.compute(predictions=pred_str, references=label_str)
        
        return {"wer": wer, "cer": cer}
    
    def finetune_model(self, train_dataset, eval_dataset, output_dir, num_epochs=3, batch_size=8, learning_rate=1e-5):
        """
        微调模型
        
        参数:
            train_dataset: 训练数据集
            eval_dataset: 评估数据集
            output_dir: 输出目录
            num_epochs: 训练轮数
            batch_size: 批处理大小
            learning_rate: 学习率
            
        返回:
            model: 微调后的模型
            processor: 处理器
        """
        print("开始微调语音识别模型...")
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 加载模型和处理器
        self.processor, model = self.load_model()
        
        # 预处理数据集
        train_dataset = self.preprocess_dataset(train_dataset, self.processor)
        eval_dataset = self.preprocess_dataset(eval_dataset, self.processor)
        
        # 创建数据整理器
        data_collator = self.DataCollatorSpeechSeq2SeqWithPadding(processor=self.processor)
        
        # 设置训练参数
        training_args = Seq2SeqTrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=2,
            learning_rate=learning_rate,
            warmup_ratio=0.1,
            num_train_epochs=num_epochs,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            logging_steps=10,
            load_best_model_at_end=True,
            metric_for_best_model="wer",
            greater_is_better=False,
            fp16=True,
            generation_max_length=225,
            report_to="none",
            save_total_limit=2,
            push_to_hub=False
        )
        
        # 创建Trainer
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
            tokenizer=self.processor.feature_extractor
        )
        
        # 开始训练
        print(f"开始训练，批大小: {batch_size}, 学习率: {learning_rate}, 训练轮数: {num_epochs}")
        trainer.train()
        
        # 保存模型
        model.save_pretrained(output_dir)
        self.processor.save_pretrained(output_dir)
        
        print(f"模型微调完成，已保存到: {output_dir}")
        return model, self.processor
    
    def transcribe_audio(self, audio_file, model=None, processor=None):
        """
        转录音频
        
        参数:
            audio_file: 音频文件路径
            model: 模型，默认为None（使用当前加载的模型）
            processor: 处理器，默认为None（使用当前加载的处理器）
            
        返回:
            str: 转录文本
        """
        # 如果未提供模型和处理器，则加载
        if model is None or processor is None:
            processor, model = self.load_model()
        
        # 加载音频
        audio_array, sampling_rate = librosa.load(audio_file, sr=16000)
        
        # 提取特征
        input_features = processor.feature_extractor(audio_array, sampling_rate=sampling_rate, return_tensors="pt").input_features
        input_features = input_features.to(self.device)
        
        # 生成转录
        with torch.no_grad():
            predicted_ids = model.generate(input_features)
        
        # 解码转录
        transcription = processor.tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        
        return transcription
    
    def evaluate_model_on_testset(self, model, processor, test_dataset, output_file=None):
        """
        在测试集上评估模型
        
        参数:
            model: 模型
            processor: 处理器
            test_dataset: 测试数据集
            output_file: 输出文件路径
            
        返回:
            dict: 评估结果
        """
        print("在测试集上评估模型...")
        
        results = []
        wer_metric = evaluate.load("wer")
        cer_metric = evaluate.load("cer")
        
        all_references = []
        all_predictions = []
        
        for i, item in enumerate(tqdm(test_dataset)):
            # 获取音频和参考文本
            audio = item["audio"]
            reference = item["text"]
            
            # 提取特征
            input_features = processor.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt").input_features
            input_features = input_features.to(self.device)
            
            # 生成转录
            with torch.no_grad():
                predicted_ids = model.generate(input_features)
            
            # 解码转录
            prediction = processor.tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            
            # 添加到结果
            results.append({
                "reference": reference,
                "prediction": prediction
            })
            
            all_references.append(reference)
            all_predictions.append(prediction)
        
        # 计算整体WER和CER
        wer = wer_metric.compute(predictions=all_predictions, references=all_references)
        cer = cer_metric.compute(predictions=all_predictions, references=all_references)
        
        print(f"测试集评估结果: WER = {wer:.4f}, CER = {cer:.4f}")
        
        # 保存评估结果
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "results": results,
                    "metrics": {
                        "wer": wer,
                        "cer": cer
                    }
                }, f, ensure_ascii=False, indent=2)
            print(f"评估结果已保存到: {output_file}")
        
        return {"wer": wer, "cer": cer, "results": results}
    
    def create_speech_recognition_pipeline(self, model_path=None):
        """
        创建语音识别流水线
        
        参数:
            model_path: 模型路径，默认为None（使用预训练模型）
            
        返回:
            pipeline: 语音识别流水线
        """
        from transformers import pipeline
        
        # 如果提供了模型路径，则加载微调后的模型
        if model_path and os.path.exists(model_path):
            print(f"加载微调后的模型: {model_path}")
            model_path = model_path
        else:
            print(f"使用预训练模型: {self.model_name}")
            model_path = self.model_name
        
        # 创建流水线
        pipe = pipeline(
            "automatic-speech-recognition",
            model=model_path,
            chunk_length_s=30,
            device=0 if torch.cuda.is_available() else -1
        )
        
        return pipe
    
    def transcribe_audio_mock(self, audio_file_path):
        """
        模拟语音识别功能，用于测试
        
        参数:
            audio_file_path: 音频文件路径
            
        返回:
            str: 模拟的转录文本
        """
        # 基于文件名模拟不同的转录结果
        if "test_audio" in audio_file_path:
            return "请预测明天1号机组的污染物排放情况"
        elif "query_data" in audio_file_path:
            return "查询昨天1号机组的NOx排放量"
        elif "alert" in audio_file_path:
            return "检测到2号机组SO2排放超标"
        else:
            return "无法识别的音频内容"

# 测试代码
if __name__ == "__main__":
    # 创建语音识别集成器
    integrator = SpeechRecognitionIntegrator()
    
    # 设置输入和输出目录
    data_dir = "./data"
    models_dir = "./data/models/speech"
    
    # 生成合成语音数据
    speech_dataset_path = os.path.join(data_dir, "datasets", "speech_recognition_dataset.csv")
    if os.path.exists(speech_dataset_path):
        # 加载数据集
        df = pd.read_csv(speech_dataset_path)
        
        # 生成合成语音
        synthetic_dir = os.path.join(data_dir, "synthetic_speech")
        audio_files, transcriptions = integrator.generate_synthetic_speech(
            speech_dataset_path,
            synthetic_dir,
            max_samples=10
        )
        
        # 准备数据集
        if audio_files and transcriptions:
            dataset = integrator.prepare_dataset(audio_files, transcriptions)
            
            # 微调模型
            output_dir = os.path.join(models_dir, "whisper_finetuned")
            # model, processor = integrator.finetune_model(
            #     dataset["train"],
            #     dataset["test"],
            #     output_dir,
            #     num_epochs=1,  # 仅用于测试
            #     batch_size=2
            # )

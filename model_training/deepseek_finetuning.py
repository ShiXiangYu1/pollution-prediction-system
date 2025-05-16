#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepSeek-8B模型微调模块 - 使用LoRA/QLoRA方法微调DeepSeek-8B模型
"""

import os
import torch
import numpy as np
import pandas as pd
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel
)
from tqdm import tqdm
import wandb
import json
import time
from datetime import datetime

class DeepSeekFineTuner:
    """
    DeepSeek-8B模型微调类，使用LoRA/QLoRA方法微调模型
    """
    
    def __init__(self, model_name="deepseek-ai/deepseek-llm-7b-base", data_dir='./data'):
        """
        初始化DeepSeek-8B模型微调类
        
        参数:
            model_name: 模型名称或路径
            data_dir: 数据目录
        """
        self.data_dir = data_dir
        self.model_name = model_name
        
        # 创建模型目录（如果不存在）
        os.makedirs(os.path.join(data_dir, 'models'), exist_ok=True)
        
        # 设置日志级别
        logging.set_verbosity_info()
        
        # 检查是否有GPU可用
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")
        
        # 如果使用GPU，检查可用显存
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            print(f"可用GPU数量: {gpu_count}")
            
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                total_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)  # 转换为GB
                print(f"GPU {i}: {gpu_name}, 总显存: {total_memory:.2f} GB")
    
    def load_model_for_training(self, use_4bit=True, use_8bit=False):
        """
        加载用于训练的模型
        
        参数:
            use_4bit: 是否使用4位量化
            use_8bit: 是否使用8位量化
            
        返回:
            model: 加载的模型
            tokenizer: 分词器
        """
        print(f"加载模型: {self.model_name}")
        
        # 配置量化参数
        if use_4bit:
            print("使用4位量化")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        elif use_8bit:
            print("使用8位量化")
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True
            )
        else:
            print("不使用量化")
            quantization_config = None
        
        # 加载分词器
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # 设置特殊token
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        
        # 加载模型
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True
        )
        
        # 为k-bit训练准备模型
        if use_4bit or use_8bit:
            model = prepare_model_for_kbit_training(model)
        
        print("模型加载完成")
        return model, tokenizer
    
    def prepare_lora_config(self, r=16, lora_alpha=32, lora_dropout=0.05, bias="none"):
        """
        准备LoRA配置
        
        参数:
            r: LoRA秩
            lora_alpha: LoRA alpha参数
            lora_dropout: LoRA dropout率
            bias: 偏置参数
            
        返回:
            LoraConfig: LoRA配置
        """
        # 获取目标模块
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]
        
        # 创建LoRA配置
        lora_config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias=bias,
            task_type="CAUSAL_LM",
            target_modules=target_modules
        )
        
        print(f"LoRA配置: r={r}, alpha={lora_alpha}, dropout={lora_dropout}")
        return lora_config
    
    def prepare_training_data(self, data_path, instruction_template="指令: {instruction}\n\n回答: {response}"):
        """
        准备训练数据
        
        参数:
            data_path: 数据文件路径
            instruction_template: 指令模板
            
        返回:
            Dataset: 训练数据集
        """
        print(f"准备训练数据: {data_path}")
        
        # 加载数据
        try:
            # 根据文件扩展名选择加载方法
            ext = os.path.splitext(data_path)[1].lower()
            
            if ext == '.csv':
                df = pd.read_csv(data_path)
            elif ext == '.json':
                with open(data_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if isinstance(data, list):
                    df = pd.DataFrame(data)
                else:
                    df = pd.DataFrame([data])
            elif ext == '.jsonl':
                data = []
                with open(data_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        data.append(json.loads(line))
                df = pd.DataFrame(data)
            else:
                print(f"不支持的文件格式: {ext}")
                return None
            
            print(f"成功加载数据，形状: {df.shape}")
        except Exception as e:
            print(f"加载数据失败: {e}")
            return None
        
        # 检查必要的列
        required_cols = ['instruction', 'response']
        if not all(col in df.columns for col in required_cols):
            print(f"数据缺少必要的列: {required_cols}")
            return None
        
        # 格式化数据
        def format_instruction(row):
            return instruction_template.format(
                instruction=row['instruction'],
                response=row['response']
            )
        
        df['text'] = df.apply(format_instruction, axis=1)
        
        # 创建Dataset对象
        dataset = Dataset.from_pandas(df[['text']])
        
        print(f"训练数据准备完成，共{len(dataset)}条记录")
        return dataset
    
    def tokenize_dataset(self, dataset, tokenizer, max_length=512):
        """
        对数据集进行分词
        
        参数:
            dataset: 数据集
            tokenizer: 分词器
            max_length: 最大序列长度
            
        返回:
            Dataset: 分词后的数据集
        """
        print(f"对数据集进行分词，最大长度: {max_length}")
        
        def tokenize_function(examples):
            return tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            )
        
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["text"]
        )
        
        print("数据集分词完成")
        return tokenized_dataset
    
    def train_model(self, model, tokenizer, train_dataset, output_dir, 
                   batch_size=4, num_epochs=3, learning_rate=2e-4, 
                   warmup_ratio=0.03, max_steps=-1, logging_steps=10,
                   save_steps=100, use_wandb=False):
        """
        训练模型
        
        参数:
            model: 模型
            tokenizer: 分词器
            train_dataset: 训练数据集
            output_dir: 输出目录
            batch_size: 批处理大小
            num_epochs: 训练轮数
            learning_rate: 学习率
            warmup_ratio: 预热比例
            max_steps: 最大步数
            logging_steps: 日志记录步数
            save_steps: 保存步数
            use_wandb: 是否使用wandb
            
        返回:
            model: 训练后的模型
        """
        print("开始训练模型...")
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 初始化wandb
        if use_wandb:
            wandb.init(project="deepseek-finetune", name=f"lora-{datetime.now().strftime('%Y%m%d-%H%M%S')}")
        
        # 设置训练参数
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=4,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=True,
            logging_steps=logging_steps,
            save_steps=save_steps,
            warmup_ratio=warmup_ratio,
            max_steps=max_steps,
            report_to="wandb" if use_wandb else "none",
            save_total_limit=3,
            push_to_hub=False,
            remove_unused_columns=False
        )
        
        # 创建Trainer
        from transformers import Trainer
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=lambda data: {'input_ids': torch.stack([f['input_ids'] for f in data]),
                                       'attention_mask': torch.stack([f['attention_mask'] for f in data]),
                                       'labels': torch.stack([f['input_ids'] for f in data])}
        )
        
        # 开始训练
        print(f"开始训练，批大小: {batch_size}, 学习率: {learning_rate}, 训练轮数: {num_epochs}")
        trainer.train()
        
        # 保存模型
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        print(f"模型训练完成，已保存到: {output_dir}")
        return model
    
    def load_finetuned_model(self, model_path, use_4bit=True, use_8bit=False):
        """
        加载微调后的模型
        
        参数:
            model_path: 模型路径
            use_4bit: 是否使用4位量化
            use_8bit: 是否使用8位量化
            
        返回:
            model: 加载的模型
            tokenizer: 分词器
        """
        print(f"加载微调后的模型: {model_path}")
        
        # 配置量化参数
        if use_4bit:
            print("使用4位量化")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        elif use_8bit:
            print("使用8位量化")
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True
            )
        else:
            print("不使用量化")
            quantization_config = None
        
        # 加载基础模型
        base_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True
        )
        
        # 加载LoRA权重
        model = PeftModel.from_pretrained(base_model, model_path)
        
        # 加载分词器
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        print("微调后的模型加载完成")
        return model, tokenizer
    
    def generate_text(self, model, tokenizer, prompt, max_length=512, temperature=0.7, top_p=0.9, top_k=50):
        """
        生成文本
        
        参数:
            model: 模型
            tokenizer: 分词器
            prompt: 提示文本
            max_length: 最大生成长度
            temperature: 温度参数
            top_p: top-p采样参数
            top_k: top-k采样参数
            
        返回:
            str: 生成的文本
        """
        # 编码提示文本
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # 生成文本
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                num_return_sequences=1,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # 解码输出
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return generated_text
    
    def evaluate_model(self, model, tokenizer, eval_data, output_file=None):
        """
        评估模型
        
        参数:
            model: 模型
            tokenizer: 分词器
            eval_data: 评估数据
            output_file: 输出文件路径
            
        返回:
            dict: 评估结果
        """
        print("开始评估模型...")
        
        results = []
        
        for i, item in enumerate(tqdm(eval_data)):
            instruction = item.get('instruction', '')
            reference = item.get('response', '')
            
            # 构建提示
            prompt = f"指令: {instruction}\n\n回答: "
            
            # 生成回答
            generated = self.generate_text(model, tokenizer, prompt)
            
            # 提取生成的回答部分
            if "回答: " in generated:
                generated = generated.split("回答: ", 1)[1].strip()
            
            # 保存结果
            results.append({
                'instruction': instruction,
                'reference': reference,
                'generated': generated
            })
        
        # 保存评估结果
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"评估结果已保存到: {output_file}")
        
        print("模型评估完成")
        return results
    
    def create_nl2sql_training_data(self, schema_info, example_queries, output_path=None):
        """
        创建NL2SQL训练数据
        
        参数:
            schema_info: 数据库模式信息
            example_queries: 示例查询列表
            output_path: 输出文件路径
            
        返回:
            list: 训练数据列表
        """
        print("创建NL2SQL训练数据...")
        
        training_data = []
        
        for example in example_queries:
            nl = example.get('natural_language', '')
            sql = example.get('sql_query', '')
            
            if nl and sql:
                # 构建指令
                instruction = f"根据以下数据库模式，将自然语言查询转换为SQL查询。\n\n数据库模式:\n{schema_info}\n\n自然语言查询: {nl}"
                
                # 构建回答
                response = f"SQL查询:\n{sql}"
                
                # 添加到训练数据
                training_data.append({
                    'instruction': instruction,
                    'response': response
                })
        
        print(f"NL2SQL训练数据创建完成，共{len(training_data)}条记录")
        
        # 保存训练数据
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(training_data, f, ensure_ascii=False, indent=2)
            print(f"NL2SQL训练数据已保存到: {output_path}")
        
        return training_data
    
    def create_pollution_prediction_training_data(self, external_data_examples, output_path=None):
        """
        创建污染物排放预测训练数据
        
        参数:
            external_data_examples: 外部数据示例列表
            output_path: 输出文件路径
            
        返回:
            list: 训练数据列表
        """
        print("创建污染物排放预测训练数据...")
        
        training_data = []
        
        for example in external_data_examples:
            context = example.get('context', '')
            prediction = example.get('prediction', '')
            
            if context and prediction:
                # 构建指令
                instruction = f"根据以下环境和运行数据，预测未来24小时的污染物排放水平。\n\n数据:\n{context}"
                
                # 构建回答
                response = f"预测结果:\n{prediction}"
                
                # 添加到训练数据
                training_data.append({
                    'instruction': instruction,
                    'response': response
                })
        
        print(f"污染物排放预测训练数据创建完成，共{len(training_data)}条记录")
        
        # 保存训练数据
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(training_data, f, ensure_ascii=False, indent=2)
            print(f"污染物排放预测训练数据已保存到: {output_path}")
        
        return training_data
    
    def finetune_for_nl2sql(self, nl2sql_data_path, output_dir):
        """
        针对NL2SQL任务微调模型
        
        参数:
            nl2sql_data_path: NL2SQL数据路径
            output_dir: 输出目录
            
        返回:
            model: 微调后的模型
            tokenizer: 分词器
        """
        print(f"针对NL2SQL任务微调模型，数据: {nl2sql_data_path}")
        
        # 加载模型
        model, tokenizer = self.load_model_for_training(use_4bit=True)
        
        # 准备LoRA配置
        lora_config = self.prepare_lora_config(r=16, lora_alpha=32)
        
        # 应用LoRA配置
        model = get_peft_model(model, lora_config)
        
        # 准备训练数据
        train_dataset = self.prepare_training_data(nl2sql_data_path)
        
        # 对数据集进行分词
        tokenized_dataset = self.tokenize_dataset(train_dataset, tokenizer, max_length=512)
        
        # 训练模型
        model = self.train_model(
            model, 
            tokenizer, 
            tokenized_dataset, 
            output_dir,
            batch_size=4,
            num_epochs=3,
            learning_rate=2e-4
        )
        
        print(f"NL2SQL模型微调完成，已保存到: {output_dir}")
        return model, tokenizer
    
    def finetune_for_pollution_prediction(self, prediction_data_path, output_dir):
        """
        针对污染物排放预测任务微调模型
        
        参数:
            prediction_data_path: 预测数据路径
            output_dir: 输出目录
            
        返回:
            model: 微调后的模型
            tokenizer: 分词器
        """
        print(f"针对污染物排放预测任务微调模型，数据: {prediction_data_path}")
        
        # 加载模型
        model, tokenizer = self.load_model_for_training(use_4bit=True)
        
        # 准备LoRA配置
        lora_config = self.prepare_lora_config(r=16, lora_alpha=32)
        
        # 应用LoRA配置
        model = get_peft_model(model, lora_config)
        
        # 准备训练数据
        train_dataset = self.prepare_training_data(prediction_data_path)
        
        # 对数据集进行分词
        tokenized_dataset = self.tokenize_dataset(train_dataset, tokenizer, max_length=512)
        
        # 训练模型
        model = self.train_model(
            model, 
            tokenizer, 
            tokenized_dataset, 
            output_dir,
            batch_size=4,
            num_epochs=3,
            learning_rate=2e-4
        )
        
        print(f"污染物排放预测模型微调完成，已保存到: {output_dir}")
        return model, tokenizer

# 测试代码
if __name__ == "__main__":
    # 创建DeepSeek微调器
    finetuner = DeepSeekFineTuner()
    
    # 设置输入和输出目录
    data_dir = "./data"
    models_dir = "./data/models"
    
    # 创建NL2SQL训练数据
    schema_info = """
    地区表(TS_AREA): 地区编号(AREA_CODE), 地区名称(AREA_NAME), 父地区编号(PARENT_CODE)
    电厂表(TB_FACTORY): 电厂编号(FAC_ID), 地区编号(AREA_CODE), 电厂名称(FAC_NAME), 归属集团(GROUP_ID), 地址(FAC_ADDR), 电厂坐标(X_MAP, Y_MAP), 电厂简称(FAC_ALIAS), 电厂是否启用(ACTIVE_FLAG)
    机组表(TB_STEAMER): 机组编号(STEAMER_ID), 机组名称(STEAMER_NAME), 机组所属电厂编号(FAC_ID), 机组负荷(RATING_FH), 机组是否启用(ACTIVE_FLAG), 机组类型(STEAMER_TYPE)
    """
    
    example_queries = [
        {
            "natural_language": "当前哪些机组实时污染物排放超标？",
            "sql_query": "SELECT f.FAC_NAME, s.STEAMER_NAME FROM TB_FACTORY f JOIN TB_STEAMER s ON f.FAC_ID = s.FAC_ID WHERE s.ACTIVE_FLAG = '是'"
        },
        {
            "natural_language": "南京地区的电厂有哪些？",
            "sql_query": "SELECT f.FAC_NAME FROM TB_FACTORY f JOIN TS_AREA a ON f.AREA_CODE = a.AREA_CODE WHERE a.AREA_NAME = '南京'"
        }
    ]
    
    nl2sql_data_path = os.path.join(data_dir, "nl2sql_training_data.json")
    finetuner.create_nl2sql_training_data(schema_info, example_queries, nl2sql_data_path)
    
    # 微调NL2SQL模型
    nl2sql_model_dir = os.path.join(models_dir, "nl2sql_model")
    # finetuner.finetune_for_nl2sql(nl2sql_data_path, nl2sql_model_dir)

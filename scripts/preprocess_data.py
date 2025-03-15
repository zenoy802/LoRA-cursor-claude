#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
预处理下载的数据集，准备训练和评估
"""

import os
import argparse
import random
from datasets import load_from_disk, Dataset, DatasetDict
from transformers import LlamaTokenizer

def format_instruction(row):
    """将BELLE数据集格式化为指令格式"""
    return {
        "text": f"### 指令：\n{row['instruction']}\n\n### 回答：\n{row['output']}"
    }

def prepare_belle_dataset(dataset_path, tokenizer, max_length, val_ratio=0.05):
    """准备BELLE数据集用于训练"""
    print("正在准备BELLE数据集...")
    
    # 加载数据集
    belle_dataset = load_from_disk(dataset_path)
    
    # 格式化为指令格式
    belle_dataset = belle_dataset.map(format_instruction, remove_columns=belle_dataset.column_names)
    
    # 分割训练集和验证集
    train_val = belle_dataset.train_test_split(test_size=val_ratio, seed=42)
    
    # 对数据集进行编码
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt"
        )
    
    train_val = train_val.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing the dataset",
    )
    
    # 添加labels（用于自回归训练）
    train_val = train_val.map(
        lambda examples: {"labels": examples["input_ids"].copy()},
        batched=True,
        desc="Creating labels",
    )
    
    print(f"训练集大小: {len(train_val['train'])}")
    print(f"验证集大小: {len(train_val['test'])}")
    
    return train_val

def prepare_ceval_dataset(dataset_path, tokenizer):
    """准备C-Eval数据集用于评估"""
    print("正在准备C-Eval数据集...")
    
    # 加载数据集
    ceval_dataset = load_from_disk(dataset_path)
    
    # 只使用验证集和测试集
    return DatasetDict({
        "validation": ceval_dataset["validation"],
        "test": ceval_dataset["test"]
    })

def prepare_mmlu_dataset(dataset_path, tokenizer):
    """准备MMLU数据集用于评估"""
    print("正在准备MMLU数据集...")
    
    # 加载数据集
    mmlu_dataset = load_from_disk(dataset_path)
    
    # 只使用验证集和测试集
    return DatasetDict({
        "validation": mmlu_dataset["validation"],
        "test": mmlu_dataset["test"]
    })

def main(args):
    # 加载tokenizer
    tokenizer = LlamaTokenizer.from_pretrained(args.model_name_or_path)
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 准备训练数据集
    train_dataset = prepare_belle_dataset(
        os.path.join(args.data_dir, "belle_dataset"),
        tokenizer,
        args.max_length,
        args.val_ratio
    )
    train_dataset.save_to_disk(os.path.join(args.output_dir, "train_dataset"))
    
    # 准备评估数据集
    ceval_dataset = prepare_ceval_dataset(
        os.path.join(args.data_dir, "ceval_dataset"),
        tokenizer
    )
    ceval_dataset.save_to_disk(os.path.join(args.output_dir, "ceval_dataset"))
    
    mmlu_dataset = prepare_mmlu_dataset(
        os.path.join(args.data_dir, "mmlu_dataset"),
        tokenizer
    )
    mmlu_dataset.save_to_disk(os.path.join(args.output_dir, "mmlu_dataset"))
    
    # 可以继续准备其他数据集...
    
    print("所有数据集预处理完成！")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="预处理下载的数据集")
    parser.add_argument("--data_dir", type=str, default="data", help="原始数据集目录")
    parser.add_argument("--output_dir", type=str, default="processed_data", help="处理后数据集保存目录")
    parser.add_argument("--model_name_or_path", type=str, default="meta-llama/Llama-2-7b-hf", 
                        help="模型名称或路径，用于加载tokenizer")
    parser.add_argument("--max_length", type=int, default=2048, help="最大序列长度")
    parser.add_argument("--val_ratio", type=float, default=0.05, help="验证集比例")
    args = parser.parse_args()
    
    main(args) 
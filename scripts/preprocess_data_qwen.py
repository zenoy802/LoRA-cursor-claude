#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
预处理下载的数据集，准备Qwen模型的训练和评估
"""

import os
import argparse
from datasets import load_from_disk, Dataset, DatasetDict
from transformers import AutoTokenizer

def format_instruction(row):
    """将BELLE数据集格式化为Qwen支持的指令格式"""
    # Qwen模型推荐的指令格式
    return {
        "text": f"<|im_start|>user\n{row['instruction']}<|im_end|>\n<|im_start|>assistant\n{row['output']}<|im_end|>"
    }

def prepare_belle_dataset(dataset_path, tokenizer, max_length, val_ratio=0.05):
    """准备BELLE数据集用于训练"""
    print("正在准备BELLE数据集...")
    
    # 加载数据集
    belle_dataset = load_from_disk(dataset_path)
    
    # 格式化为Qwen指令格式
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

def prepare_evaluation_dataset(dataset_path, dataset_name):
    """准备评估数据集"""
    print(f"正在准备{dataset_name}数据集...")
    
    # 判断是否为中文多任务数据集（CEVAL或CMMLU）
    if dataset_name in ["C-Eval", "CMMLU"]:
        # 中文数据集是按任务分开存储的，需要遍历所有任务目录
        import glob
        from datasets import concatenate_datasets
        
        # 获取所有任务目录
        task_dirs = glob.glob(os.path.join(dataset_path, "*"))
        
        if not task_dirs:
            raise ValueError(f"在 {dataset_path} 中没有找到任何{dataset_name}任务数据集")
        
        # 初始化存放所有任务的验证集和测试集
        all_validation = []
        all_test = []
        
        # 遍历并加载每个任务
        for task_dir in task_dirs:
            if os.path.isdir(task_dir):
                task_name = os.path.basename(task_dir)
                print(f"  加载任务: {task_name}")
                try:
                    # 加载单个任务数据集
                    task_dataset = load_from_disk(task_dir)
                    
                    # 添加任务名称字段，用于后续分析
                    if "validation" in task_dataset:
                        task_dataset["validation"] = task_dataset["validation"].map(
                            lambda x: {"task": task_name}, remove_columns=[]
                        )
                        all_validation.append(task_dataset["validation"])
                    
                    if "test" in task_dataset:
                        task_dataset["test"] = task_dataset["test"].map(
                            lambda x: {"task": task_name}, remove_columns=[]
                        )
                        all_test.append(task_dataset["test"])
                except Exception as e:
                    print(f"  加载任务 {task_name} 失败: {str(e)}")
        
        # 合并所有任务的数据集
        combined_validation = concatenate_datasets(all_validation) if all_validation else None
        combined_test = concatenate_datasets(all_test) if all_test else None
        
        if combined_validation is None and combined_test is None:
            raise ValueError(f"无法加载任何{dataset_name}数据集")
        
        # 返回合并后的数据集
        return DatasetDict({
            "validation": combined_validation if combined_validation else None,
            "test": combined_test if combined_test else None
        })
    else:
        # 英文数据集（如MMLU）直接加载
        dataset = load_from_disk(dataset_path)
        
        # 只使用验证集和测试集
        return DatasetDict({
            "validation": dataset["validation"],
            "test": dataset["test"]
        })

def format_multiple_choice_prompt(item):
    """将多选题格式化为Qwen支持的格式"""
    # 为Qwen模型定制的多选题提示格式
    prompt = f"<|im_start|>user\n问题: {item['question']}\n\n选项:\nA. {item['choices'][0]}\nB. {item['choices'][1]}\nC. {item['choices'][2]}\nD. {item['choices'][3]}\n\n请直接回答选项字母。<|im_end|>\n<|im_start|>assistant\n"
    return prompt

def main(args):
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    
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
    ceval_dataset = prepare_evaluation_dataset(
        os.path.join(args.data_dir, "ceval_dataset"),
        "C-Eval"
    )
    ceval_dataset.save_to_disk(os.path.join(args.output_dir, "ceval_dataset"))
    
    mmlu_dataset = prepare_evaluation_dataset(
        os.path.join(args.data_dir, "mmlu_dataset"),
        "MMLU"
    )
    mmlu_dataset.save_to_disk(os.path.join(args.output_dir, "mmlu_dataset"))
    
    # 可以继续准备其他数据集...
    
    print("所有数据集预处理完成！")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="预处理下载的数据集用于Qwen模型")
    parser.add_argument("--data_dir", type=str, default="data", help="原始数据集目录")
    parser.add_argument("--output_dir", type=str, default="processed_data_qwen", help="处理后数据集保存目录")
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen-7B", 
                        help="模型名称或路径，用于加载tokenizer")
    parser.add_argument("--max_length", type=int, default=2048, help="最大序列长度")
    parser.add_argument("--val_ratio", type=float, default=0.05, help="验证集比例")
    args = parser.parse_args()
    
    main(args) 
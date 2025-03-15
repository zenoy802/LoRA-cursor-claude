#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
评估基础Qwen模型在各个数据集上的性能，作为实验基线
"""

import os
import argparse
import json
import torch
import numpy as np
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
from tqdm import tqdm

def evaluate_multiple_choice(model, tokenizer, dataset, max_length=2048, batch_size=1, device="cuda"):
    """评估模型在多选题数据集上的性能"""
    model.eval()
    correct = 0
    total = 0
    
    for i in tqdm(range(0, len(dataset), batch_size), desc="Evaluating"):
        batch = dataset[i:i+batch_size]
        
        for item in batch:
            # 构造提示（使用Qwen的聊天格式）
            prompt = f"<|im_start|>user\n问题: {item['question']}\n\n选项:\nA. {item['choices'][0]}\nB. {item['choices'][1]}\nC. {item['choices'][2]}\nD. {item['choices'][3]}\n\n请直接回答选项字母。<|im_end|>\n<|im_start|>assistant\n"
            
            inputs = tokenizer(prompt, return_tensors="pt", max_length=max_length, truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=10,
                    temperature=0.1,
                    do_sample=False,
                )
            
            # 解码生成的文本
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 处理Qwen特有的token格式，获取assistant回答部分
            assistant_text = ""
            if "<|im_start|>assistant" in generated_text:
                assistant_text = generated_text.split("<|im_start|>assistant")[-1]
                if "<|im_end|>" in assistant_text:
                    assistant_text = assistant_text.split("<|im_end|>")[0]
            else:
                # 如果没有特定标记，尝试获取生成的部分
                assistant_text = generated_text[len(prompt):].strip()
            
            # 提取答案
            options = ["A", "B", "C", "D"]
            predicted_answer = None
            
            for opt in options:
                if opt in assistant_text[:10]:  # 只查看前10个字符
                    predicted_answer = opt
                    break
            
            # 如果未能提取答案，尝试使用第一个字母
            if predicted_answer is None and len(assistant_text) > 0:
                first_char = assistant_text[0].upper()
                if first_char in options:
                    predicted_answer = first_char
            
            # 默认选A
            if predicted_answer is None:
                predicted_answer = "A"
            
            # 计算正确率
            correct_answer = options[item['answer']]
            if predicted_answer == correct_answer:
                correct += 1
            
            total += 1
    
    accuracy = correct / total if total > 0 else 0
    return {"accuracy": accuracy, "correct": correct, "total": total}

def evaluate_datasets(model, tokenizer, eval_datasets, results_dir, device="cuda"):
    """评估模型在多个数据集上的性能"""
    results = {}
    
    # 评估基础模型
    print("正在评估基础Qwen模型...")
    base_results = {}
    
    for dataset_name, dataset in eval_datasets.items():
        print(f"评估基础模型在 {dataset_name} 上的性能...")
        dataset_result = evaluate_multiple_choice(
            model, tokenizer, dataset["test"], device=device
        )
        base_results[dataset_name] = dataset_result
        print(f"基础模型在 {dataset_name} 上的准确率: {dataset_result['accuracy']:.4f}")
    
    results["base_model"] = base_results
    
    # 保存结果
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "qwen_baseline_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # 绘制柱状图
    plot_bar_chart(base_results, os.path.join(results_dir, "qwen_baseline_chart.png"))
    
    return results

def plot_bar_chart(results, output_path):
    """绘制基线性能柱状图"""
    # 提取数据
    categories = list(results.keys())
    accuracies = [results[cat]["accuracy"] for cat in categories]
    
    # 绘制柱状图
    plt.figure(figsize=(10, 6))
    bars = plt.bar(categories, accuracies, color='skyblue')
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom', rotation=0)
    
    # 设置标题和标签
    plt.title('Qwen基础模型在各数据集上的准确率', fontsize=15)
    plt.xlabel('数据集', fontsize=12)
    plt.ylabel('准确率', fontsize=12)
    plt.ylim(0, 1)
    
    # 保存图表
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def main(args):
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    
    # 加载基础模型
    print(f"加载基础模型: {args.model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # 加载评估数据集
    eval_datasets = {}
    
    # 加载C-Eval数据集（中文能力）
    print("加载C-Eval数据集...")
    ceval_dataset = load_from_disk(args.ceval_path)
    eval_datasets["C-Eval"] = ceval_dataset
    
    # 加载MMLU数据集（英文能力）
    print("加载MMLU数据集...")
    mmlu_dataset = load_from_disk(args.mmlu_path)
    eval_datasets["MMLU"] = mmlu_dataset
    
    # 尝试加载其他数据集（如果可用）
    if args.gsm8k_path and os.path.exists(args.gsm8k_path):
        print("加载GSM8K数据集...")
        gsm8k_dataset = load_from_disk(args.gsm8k_path)
        eval_datasets["GSM8K"] = gsm8k_dataset
    
    if args.hellaswag_path and os.path.exists(args.hellaswag_path):
        print("加载HellaSwag数据集...")
        hellaswag_dataset = load_from_disk(args.hellaswag_path)
        eval_datasets["HellaSwag"] = hellaswag_dataset
    
    # 进行评估
    results = evaluate_datasets(
        model,
        tokenizer,
        eval_datasets,
        args.results_dir,
        device=device
    )
    
    print("Qwen基线评估完成！")
    print(f"结果已保存到: {args.results_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="评估基础Qwen模型在各个数据集上的性能")
    parser.add_argument("--model_path", type=str, default="models/base_model", 
                        help="基础模型路径")
    parser.add_argument("--ceval_path", type=str, default="processed_data_qwen/ceval_dataset", 
                        help="C-Eval数据集路径")
    parser.add_argument("--mmlu_path", type=str, default="processed_data_qwen/mmlu_dataset", 
                        help="MMLU数据集路径")
    parser.add_argument("--gsm8k_path", type=str, default="processed_data_qwen/gsm8k_dataset", 
                        help="GSM8K数据集路径（可选）")
    parser.add_argument("--hellaswag_path", type=str, default="processed_data_qwen/hellaswag_dataset", 
                        help="HellaSwag数据集路径（可选）")
    parser.add_argument("--results_dir", type=str, default="evaluation_results/qwen_baseline", 
                        help="结果保存目录")
    args = parser.parse_args()
    
    main(args) 
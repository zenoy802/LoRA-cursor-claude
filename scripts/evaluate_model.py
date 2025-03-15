#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
评估模型在中文和英文数据集上的能力，分析灾难性遗忘情况
"""

import os
import argparse
import json
import torch
import numpy as np
from datasets import load_from_disk
from transformers import LlamaForCausalLM, LlamaTokenizer
from peft import PeftModel
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
            # 构造提示
            prompt = f"问题: {item['question']}\n\n选项:\nA. {item['choices'][0]}\nB. {item['choices'][1]}\nC. {item['choices'][2]}\nD. {item['choices'][3]}\n\n请直接回答选项字母。"
            
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
            answer_text = generated_text[len(prompt):].strip()
            
            # 提取答案
            options = ["A", "B", "C", "D"]
            predicted_answer = None
            
            for opt in options:
                if opt in answer_text[:10]:  # 只查看前10个字符
                    predicted_answer = opt
                    break
            
            # 如果未能提取答案，尝试使用第一个字母
            if predicted_answer is None and len(answer_text) > 0:
                first_char = answer_text[0].upper()
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

def evaluate_datasets(base_model, lora_model, tokenizer, eval_datasets, results_dir, device="cuda"):
    """评估模型在多个数据集上的性能"""
    results = {}
    
    # 评估基础模型
    print("正在评估基础模型...")
    base_results = {}
    
    for dataset_name, dataset in eval_datasets.items():
        print(f"评估基础模型在 {dataset_name} 上的性能...")
        dataset_result = evaluate_multiple_choice(
            base_model, tokenizer, dataset["test"], device=device
        )
        base_results[dataset_name] = dataset_result
        print(f"基础模型在 {dataset_name} 上的准确率: {dataset_result['accuracy']:.4f}")
    
    results["base_model"] = base_results
    
    # 评估LoRA微调后的模型
    print("正在评估LoRA微调后的模型...")
    lora_results = {}
    
    for dataset_name, dataset in eval_datasets.items():
        print(f"评估LoRA模型在 {dataset_name} 上的性能...")
        dataset_result = evaluate_multiple_choice(
            lora_model, tokenizer, dataset["test"], device=device
        )
        lora_results[dataset_name] = dataset_result
        print(f"LoRA模型在 {dataset_name} 上的准确率: {dataset_result['accuracy']:.4f}")
    
    results["lora_model"] = lora_results
    
    # 分析结果
    print("\n性能变化分析:")
    changes = {}
    
    for dataset_name in eval_datasets.keys():
        base_acc = base_results[dataset_name]["accuracy"]
        lora_acc = lora_results[dataset_name]["accuracy"]
        change = (lora_acc - base_acc) / base_acc * 100 if base_acc > 0 else float('inf')
        
        changes[dataset_name] = {
            "base_accuracy": base_acc,
            "lora_accuracy": lora_acc,
            "absolute_change": lora_acc - base_acc,
            "percentage_change": change
        }
        
        print(f"{dataset_name}: 基础准确率 {base_acc:.4f}, LoRA准确率 {lora_acc:.4f}, "
              f"变化 {lora_acc - base_acc:.4f} ({change:.2f}%)")
    
    results["changes"] = changes
    
    # 保存结果
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "evaluation_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # 绘制雷达图
    plot_radar_chart(changes, os.path.join(results_dir, "radar_chart.png"))
    
    return results

def plot_radar_chart(changes, output_path):
    """绘制能力变化雷达图"""
    # 提取数据
    categories = list(changes.keys())
    base_values = [changes[cat]["base_accuracy"] for cat in categories]
    lora_values = [changes[cat]["lora_accuracy"] for cat in categories]
    
    # 计算角度
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # 闭合图形
    
    # 添加数据
    base_values += base_values[:1]
    lora_values += lora_values[:1]
    
    # 绘制雷达图
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    # 基础模型
    ax.plot(angles, base_values, 'o-', linewidth=2, label='基础模型')
    ax.fill(angles, base_values, alpha=0.25)
    
    # LoRA模型
    ax.plot(angles, lora_values, 'o-', linewidth=2, label='LoRA微调模型')
    ax.fill(angles, lora_values, alpha=0.25)
    
    # 添加类别标签
    ax.set_thetagrids(np.array(angles[:-1]) * 180 / np.pi, categories)
    
    # 设置y轴范围
    ax.set_ylim(0, 1)
    
    # 添加图例和标题
    ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title('模型在不同数据集上的性能对比', size=15, y=1.1)
    
    # 保存图表
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def main(args):
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载tokenizer
    tokenizer = LlamaTokenizer.from_pretrained(args.base_model_path)
    tokenizer.pad_token = tokenizer.eos_token
    
    # 加载基础模型
    print(f"加载基础模型: {args.base_model_path}")
    base_model = LlamaForCausalLM.from_pretrained(
        args.base_model_path,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    
    # 加载LoRA模型
    print(f"加载LoRA模型: {args.lora_model_path}")
    lora_model = LlamaForCausalLM.from_pretrained(
        args.base_model_path,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    lora_model = PeftModel.from_pretrained(
        lora_model,
        args.lora_model_path,
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
    
    # 进行评估
    results = evaluate_datasets(
        base_model,
        lora_model,
        tokenizer,
        eval_datasets,
        args.results_dir,
        device=device
    )
    
    # 分析灾难性遗忘
    print("\n灾难性遗忘分析:")
    
    # 中文能力提升
    chinese_datasets = ["C-Eval"]
    chinese_improvement = sum(results["changes"][ds]["absolute_change"] for ds in chinese_datasets) / len(chinese_datasets)
    print(f"中文能力平均提升: {chinese_improvement:.4f}")
    
    # 英文能力变化
    english_datasets = ["MMLU"]
    english_change = sum(results["changes"][ds]["absolute_change"] for ds in english_datasets) / len(english_datasets)
    print(f"英文能力平均变化: {english_change:.4f}")
    
    # 判断是否发生灾难性遗忘
    catastrophic_forgetting = english_change < -0.05  # 阈值可根据实际情况调整
    
    if catastrophic_forgetting:
        print("警告: 检测到灾难性遗忘现象!")
    else:
        print("未检测到明显的灾难性遗忘现象.")
    
    print("评估完成！")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="评估模型在中文和英文数据集上的能力")
    parser.add_argument("--base_model_path", type=str, default="meta-llama/Llama-2-7b-hf", 
                        help="基础模型路径")
    parser.add_argument("--lora_model_path", type=str, default="results/lora-chinese-llama", 
                        help="LoRA模型路径")
    parser.add_argument("--ceval_path", type=str, default="processed_data/ceval_dataset", 
                        help="C-Eval数据集路径")
    parser.add_argument("--mmlu_path", type=str, default="processed_data/mmlu_dataset", 
                        help="MMLU数据集路径")
    parser.add_argument("--results_dir", type=str, default="evaluation_results", 
                        help="结果保存目录")
    args = parser.parse_args()
    
    main(args) 
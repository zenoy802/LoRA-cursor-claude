#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
评估LoRA训练过程中保存的多个检查点，跟踪模型性能随训练进度的变化
"""

import os
import re
import argparse
import json
import torch
import numpy as np
from datasets import load_from_disk
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import matplotlib.pyplot as plt
from tqdm import tqdm
import glob
from datetime import datetime

def evaluate_multiple_choice_chinese(model, tokenizer, dataset, max_length=2048, batch_size=1, device="cuda", model_type="llama"):
    """评估模型在多选题数据集上的性能"""
    model.eval()
    correct = 0
    total = 0
    
    for i in tqdm(range(0, len(dataset), batch_size), desc="Evaluating", leave=False):
        batch = dataset[i:i+batch_size]
        
        for i in range(batch_size):
            question = batch['question'][i]
            answer = batch['answer'][i]
            # 构造提示（根据模型类型选择格式）
            if model_type.lower() == "qwen":
                prompt = f"<|im_start|>user\n问题: {question}\n\n请直接回答选项字母。<|im_end|>\n<|im_start|>assistant\n"
            else:  # llama
                prompt = f"问题: {question}\n\n请直接回答选项字母。"
            
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
            
            # 根据模型类型提取答案
            if model_type.lower() == "qwen":
                # 处理Qwen特有的token格式，获取assistant回答部分
                assistant_text = ""
                if "<|im_start|>assistant" in generated_text:
                    assistant_text = generated_text.split("<|im_start|>assistant")[-1]
                    if "<|im_end|>" in assistant_text:
                        assistant_text = assistant_text.split("<|im_end|>")[0]
                else:
                    # 如果没有特定标记，尝试获取生成的部分
                    assistant_text = generated_text[len(prompt):].strip()
                answer_text = assistant_text
            else:  # llama
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
            
            # # 默认选A
            # if predicted_answer is None:
            #     predicted_answer = "A"
            
            # 计算正确率
            correct_answer = answer
            if predicted_answer is not None and predicted_answer == correct_answer:
                correct += 1
            
            total += 1
    
    accuracy = correct / total if total > 0 else 0
    return {"accuracy": accuracy, "correct": correct, "total": total}

def evaluate_multiple_choice_english(model, tokenizer, dataset, max_length=2048, batch_size=1, device="cuda", model_type="llama"):
    """评估模型在多选题数据集上的性能"""
    model.eval()
    correct = 0
    total = 0
    
    for i in tqdm(range(0, len(dataset), batch_size), desc="Evaluating", leave=False):
        batch = dataset[i:i+batch_size]
        
        for i in range(batch_size):
            question = batch['question'][i]
            choices = batch['choices'][i]
            answer = batch['answer'][i]
            answer_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
            if answer in answer_map:
                answer = answer_map[answer]
            # 构造提示（根据模型类型选择格式）
            if model_type.lower() == "qwen":
                prompt = f"<|im_start|>user\nQuestion: {question}\n\nChoices:\nA. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}\n\nPlease answer the option letter directly.<|im_end|>\n<|im_start|>assistant\n"
            else:  # llama
                prompt = f"Question: {question}\n\nChoices:\nA. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}\n\nPlease answer the option letter directly."
            
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
            
            # 根据模型类型提取答案
            if model_type.lower() == "qwen":
                # 处理Qwen特有的token格式，获取assistant回答部分
                assistant_text = ""
                if "<|im_start|>assistant" in generated_text:
                    assistant_text = generated_text.split("<|im_start|>assistant")[-1]
                    if "<|im_end|>" in assistant_text:
                        assistant_text = assistant_text.split("<|im_end|>")[0]
                else:
                    # 如果没有特定标记，尝试获取生成的部分
                    assistant_text = generated_text[len(prompt):].strip()
                answer_text = assistant_text
            else:  # llama
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
            
            # # 默认选A
            # if predicted_answer is None:
            #     predicted_answer = "A"
            
            # 计算正确率
            correct_answer = answer
            if predicted_answer is not None and predicted_answer == correct_answer:
                correct += 1
            
            total += 1
    
    accuracy = correct / total if total > 0 else 0
    return {"accuracy": accuracy, "correct": correct, "total": total}

def evaluate_checkpoint(base_model, checkpoint_path, tokenizer, eval_datasets, model_type="llama", device="cuda"):
    """评估单个检查点的性能"""
    print(f"评估检查点: {checkpoint_path}")
    
    # 加载LoRA模型
    if model_type.lower() == "qwen":
        lora_model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
    else:  # llama
        lora_model = LlamaForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16,
            device_map="auto",
        )
    
    lora_model = PeftModel.from_pretrained(
        lora_model,
        checkpoint_path,
    )
    
    # 评估结果
    results = {}
    
    for dataset_name, dataset in eval_datasets.items():
        print(f"  评估数据集: {dataset_name}")
        
        # 检查数据集是否有task字段，如果有，说明是CEVAL或CMMLU等中文数据集，需要分任务评估
        if "task" in dataset["test"].column_names:
            # 按任务分组评估
            tasks = dataset["test"].unique("task")
            task_results = {}
            total_correct = 0
            total_samples = 0
            
            for task in tasks:
                task_dataset = dataset["validation"].filter(lambda x: x["task"] == task)
                print(f"    任务: {task} (样本数: {len(task_dataset)})")
                
                task_result = evaluate_multiple_choice_chinese(
                    lora_model, tokenizer, task_dataset, device=device, model_type=model_type
                )
                
                task_results[task] = task_result
                total_correct += task_result["correct"]
                total_samples += task_result["total"]
                
                print(f"    任务 {task} 准确率: {task_result['accuracy']:.4f}")
            
            # 计算整体准确率
            overall_accuracy = total_correct / total_samples if total_samples > 0 else 0
            
            # 保存各任务结果和整体结果
            results[dataset_name] = {
                "task_results": task_results,
                "overall": {
                    "accuracy": overall_accuracy,
                    "correct": total_correct,
                    "total": total_samples
                }
            }
            
            print(f"  {dataset_name} 整体准确率: {overall_accuracy:.4f}")
        else:
            # 常规评估（如MMLU等英文数据集）
            dataset_result = evaluate_multiple_choice_english(
                lora_model, tokenizer, dataset["test"], device=device, model_type=model_type
            )
            results[dataset_name] = {
                "overall": dataset_result
            }
            print(f"  {dataset_name} 准确率: {dataset_result['accuracy']:.4f}")
    
    # 释放GPU内存
    del lora_model
    torch.cuda.empty_cache()
    
    return results

def extract_step_from_checkpoint(checkpoint_path):
    """从检查点路径中提取步数"""
    match = re.search(r'checkpoint-(\d+)', checkpoint_path)
    if match:
        return int(match.group(1))
    return 0

def plot_performance_curve(checkpoint_results, output_path):
    """绘制性能曲线图"""
    steps = sorted(list(checkpoint_results.keys()))
    
    # 为每个数据集创建一个图表
    datasets = list(checkpoint_results[steps[0]].keys())
    
    plt.figure(figsize=(12, 8))
    
    for dataset_name in datasets:
        accuracies = [checkpoint_results[step][dataset_name]["overall"]["accuracy"] for step in steps]
        plt.plot(steps, accuracies, 'o-', linewidth=2, label=dataset_name)
    
    plt.xlabel('训练步数', fontsize=12)
    plt.ylabel('准确率', fontsize=12)
    plt.title('模型性能随训练步数的变化', fontsize=15)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 保存图表
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def main(args):
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 确定模型类型
    model_type = "qwen" if "qwen" in args.base_model_path.lower() else "llama"
    print(f"模型类型: {model_type}")
    
    # 加载tokenizer
    if model_type == "qwen":
        tokenizer = AutoTokenizer.from_pretrained(f"{args.base_model_path}_tokenizer", trust_remote_code=True)
    else:  # llama
        tokenizer = LlamaTokenizer.from_pretrained(f"{args.base_model_path}_tokenizer")
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载基础模型
    print(f"加载基础模型: {args.base_model_path}")
    if model_type == "qwen":
        base_model = f"{args.base_model_path}_base_model"  # 只保存路径，实际模型在评估每个检查点时加载
    else:  # llama
        base_model = f"{args.base_model_path}_base_model"
    
    # 加载评估数据集
    eval_datasets = {}
    
    # 加载C-Eval数据集（中文能力）
    print("加载C-Eval数据集...")
    ceval_dataset = load_from_disk(args.ceval_path)
    eval_datasets["C-Eval"] = ceval_dataset
    
    # 加载CMMLU数据集（如果提供）
    if args.cmmlu_path and os.path.exists(args.cmmlu_path):
        print("加载CMMLU数据集...")
        cmmlu_dataset = load_from_disk(args.cmmlu_path)
        eval_datasets["CMMLU"] = cmmlu_dataset
    
    # 加载MMLU数据集（英文能力）
    if args.mmlu_path and os.path.exists(args.mmlu_path):
        print("加载MMLU数据集...")
        mmlu_dataset = load_from_disk(args.mmlu_path)
        eval_datasets["MMLU"] = mmlu_dataset
    
    # 查找所有检查点
    checkpoint_dirs = sorted(glob.glob(os.path.join(args.checkpoints_dir, "checkpoint-*")))
    
    if not checkpoint_dirs:
        print(f"在 {args.checkpoints_dir} 中未找到任何检查点")
        return
    
    print(f"找到 {len(checkpoint_dirs)} 个检查点")
    
    # 如果指定了步数间隔，筛选检查点
    if args.step_interval > 0:
        steps = [extract_step_from_checkpoint(cp) for cp in checkpoint_dirs]
        selected_indices = list(range(0, len(steps), args.step_interval))
        selected_indices.append(len(steps) - 1)  # 确保包含最后一个检查点
        selected_indices = sorted(list(set(selected_indices)))  # 去重并排序
        checkpoint_dirs = [checkpoint_dirs[i] for i in selected_indices]
        print(f"根据步数间隔筛选后，将评估 {len(checkpoint_dirs)} 个检查点")
    
    # 创建结果目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(args.results_dir, f"checkpoint_eval_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    
    # 评估每个检查点
    checkpoint_results = {}
    
    for checkpoint_dir in checkpoint_dirs:
        step = extract_step_from_checkpoint(checkpoint_dir)
        results = evaluate_checkpoint(
            base_model,
            checkpoint_dir,
            tokenizer,
            eval_datasets,
            model_type=model_type,
            device=device
        )
        checkpoint_results[step] = results
        
        # 保存当前检查点的结果
        with open(os.path.join(results_dir, f"step_{step}_results.json"), "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    
    # 保存所有检查点的汇总结果
    with open(os.path.join(results_dir, "all_checkpoints_results.json"), "w", encoding="utf-8") as f:
        json.dump(checkpoint_results, f, indent=2, ensure_ascii=False)
    
    # 绘制性能曲线
    plot_performance_curve(checkpoint_results, os.path.join(results_dir, "performance_curve.png"))
    
    # 输出最佳检查点
    best_checkpoints = {}
    for dataset_name in eval_datasets.keys():
        best_step = max(checkpoint_results.keys(), 
                        key=lambda s: checkpoint_results[s][dataset_name]["overall"]["accuracy"])
        best_accuracy = checkpoint_results[best_step][dataset_name]["overall"]["accuracy"]
        best_checkpoints[dataset_name] = {
            "step": best_step,
            "accuracy": best_accuracy,
            "checkpoint_path": os.path.join(args.checkpoints_dir, f"checkpoint-{best_step}")
        }
        print(f"{dataset_name} 最佳检查点: 步数 {best_step}, 准确率 {best_accuracy:.4f}")
    
    # 保存最佳检查点信息
    with open(os.path.join(results_dir, "best_checkpoints.json"), "w", encoding="utf-8") as f:
        json.dump(best_checkpoints, f, indent=2, ensure_ascii=False)
    
    print(f"评估完成！结果已保存到: {results_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="评估LoRA训练过程中保存的多个检查点")
    parser.add_argument("--base_model_path", type=str, required=True,
                        help="基础模型路径")
    parser.add_argument("--checkpoints_dir", type=str, required=True,
                        help="包含检查点的目录路径")
    parser.add_argument("--ceval_path", type=str, required=True,
                        help="C-Eval数据集路径")
    parser.add_argument("--cmmlu_path", type=str, default="",
                        help="CMMLU数据集路径（可选）")
    parser.add_argument("--mmlu_path", type=str, default="",
                        help="MMLU数据集路径（可选）")
    parser.add_argument("--results_dir", type=str, default="evaluation_results/checkpoints",
                        help="结果保存目录")
    parser.add_argument("--step_interval", type=int, default=0,
                        help="评估检查点的步数间隔，0表示评估所有检查点")
    args = parser.parse_args()
    
    main(args) 
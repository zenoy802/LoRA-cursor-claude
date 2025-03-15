#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
下载实验所需的所有数据集
"""

import os
import argparse
from datasets import load_dataset

def main(args):
    # 创建数据保存目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("开始下载数据集...")
    
    # 下载中文指令微调数据集 - 使用正确的BELLE数据集ID
    print("下载BELLE中文指令微调数据集...")
    try:
        # 尝试下载BELLE-2M数据集
        belle_dataset = load_dataset("BelleGroup/train_2M_CN", split="train")
        print("成功加载BELLE-2M数据集")
    except Exception as e:
        print(f"无法加载BELLE-2M，尝试加载BELLE-1M: {str(e)}")
        try:
            # 如果失败，尝试下载BELLE-1M数据集
            belle_dataset = load_dataset("BelleGroup/train_1M_CN", split="train")
            print("成功加载BELLE-1M数据集")
        except Exception as e2:
            print(f"无法加载BELLE-1M，尝试加载更小的BELLE数据集: {str(e2)}")
            # 如果再次失败，尝试下载更小的数据集
            belle_dataset = load_dataset("BelleGroup/generated_chat_0.4M", split="train")
            print("成功加载BELLE-0.4M数据集")
    
    # 如果只需要子集，可以采样
    if args.belle_sample_size > 0:
        belle_dataset = belle_dataset.shuffle(seed=42).select(range(args.belle_sample_size))
    belle_dataset.save_to_disk(os.path.join(args.output_dir, "belle_dataset"))
    print(f"BELLE数据集已保存，共{len(belle_dataset)}条样本")
    
    # 下载中文评估数据集
    print("下载中文评估数据集...")
    
    # C-Eval数据集
    print("下载C-Eval数据集...")
    try:
        ceval_dataset = load_dataset("ceval/ceval", "main")
    except Exception as e:
        print(f"无法直接加载C-Eval，尝试备用链接: {str(e)}")
        ceval_dataset = load_dataset("SJTU-LIT/ceval")
    ceval_dataset.save_to_disk(os.path.join(args.output_dir, "ceval_dataset"))
    print(f"C-Eval数据集已保存")
    
    # CMMLU数据集
    print("下载CMMLU数据集...")
    try:
        cmmlu_dataset = load_dataset("haonan-li/cmmlu", "main")
    except Exception as e:
        print(f"无法加载CMMLU，尝试备用链接: {str(e)}")
        try:
            cmmlu_dataset = load_dataset("SJTU-LIT/cmmlu")
        except Exception as e2:
            print(f"备用CMMLU也无法加载，跳过此数据集: {str(e2)}")
            cmmlu_dataset = None
    
    if cmmlu_dataset is not None:
        cmmlu_dataset.save_to_disk(os.path.join(args.output_dir, "cmmlu_dataset"))
        print(f"CMMLU数据集已保存")
    
    # 下载英文评估数据集（用于监测遗忘）
    print("下载英文评估数据集...")
    
    # MMLU数据集
    print("下载MMLU数据集...")
    try:
        mmlu_dataset = load_dataset("cais/mmlu", "all")
    except Exception as e:
        print(f"无法以默认配置加载MMLU，尝试备用方式: {str(e)}")
        try:
            mmlu_dataset = load_dataset("lukaemon/mmlu", "all")
        except Exception as e2:
            print(f"备用MMLU也无法加载，尝试加载MMLU的一个子集: {str(e2)}")
            # 尝试加载MMLU的子集
            mmlu_dataset = load_dataset("cais/mmlu", "abstract_algebra")
    
    mmlu_dataset.save_to_disk(os.path.join(args.output_dir, "mmlu_dataset"))
    print(f"MMLU数据集已保存")
    
    # GSM8K数据集
    print("下载GSM8K数据集...")
    try:
        gsm8k_dataset = load_dataset("gsm8k", "main")
        gsm8k_dataset.save_to_disk(os.path.join(args.output_dir, "gsm8k_dataset"))
        print(f"GSM8K数据集已保存")
    except Exception as e:
        print(f"无法加载GSM8K数据集，跳过: {str(e)}")
    
    # HellaSwag数据集
    print("下载HellaSwag数据集...")
    try:
        hellaswag_dataset = load_dataset("Rowan/hellaswag")
        hellaswag_dataset.save_to_disk(os.path.join(args.output_dir, "hellaswag_dataset"))
        print(f"HellaSwag数据集已保存")
    except Exception as e:
        print(f"无法加载HellaSwag数据集，跳过: {str(e)}")
    
    print("所有数据集下载完成！")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="下载实验所需的所有数据集")
    parser.add_argument("--output_dir", type=str, default="data", help="数据保存目录")
    parser.add_argument("--belle_sample_size", type=int, default=50000, 
                        help="从BELLE数据集中采样的样本数，设为-1表示使用全部数据")
    args = parser.parse_args()
    
    main(args) 
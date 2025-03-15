#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
下载基础大语言模型
"""

import os
import argparse
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer

def main(args):
    # 创建模型保存目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"开始下载模型: {args.model_name_or_path}")
    
    # 下载并保存tokenizer
    print("下载tokenizer...")
    tokenizer = LlamaTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.save_pretrained(os.path.join(args.output_dir, f"{args.model_name_or_path}_tokenizer"))
    
    # 下载并保存模型
    print("下载模型...")
    model = LlamaForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.float16 if args.use_fp16 else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    
    model_save_path = os.path.join(args.output_dir, f"{args.model_name_or_path}_base_model")
    model.save_pretrained(model_save_path)
    print(f"模型已保存到: {model_save_path}")
    
    print("模型下载完成！")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="下载基础大语言模型")
    parser.add_argument("--model_name_or_path", type=str, default="meta-llama/Llama-2-7b-hf", 
                        help="模型名称或路径")
    parser.add_argument("--output_dir", type=str, default="models", 
                        help="模型保存目录")
    parser.add_argument("--use_fp16", action="store_true", default=True,
                        help="是否使用FP16精度保存模型")
    args = parser.parse_args()
    
    main(args) 
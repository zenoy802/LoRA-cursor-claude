#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用LoRA技术对Llama模型进行中文能力微调
"""

import os
import argparse
import torch
from datasets import load_from_disk
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType

def main(args):
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载tokenizer
    print(f"加载tokenizer: {args.model_path}")
    tokenizer = LlamaTokenizer.from_pretrained(f"{args.model_path}_tokenizer",)
    tokenizer.pad_token = tokenizer.eos_token
    
    # 加载模型
    print(f"加载基础模型: {args.model_path}")
    model = LlamaForCausalLM.from_pretrained(
        f"{args.model_path}_base_model",
        torch_dtype=torch.float16,
        device_map="auto",
    )
    
    # 配置LoRA
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    # 应用LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # 加载数据集
    print("加载训练数据集...")
    dataset = load_from_disk(args.dataset_path)
    
    # 数据收集器
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 训练参数
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_steps=args.logging_steps,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        fp16=True,
        report_to="wandb",
        run_name="LoRA-Llama-2-7b-hf"
    )
    
    # 初始化Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=data_collator,
    )
    
    # 开始训练
    print("开始LoRA微调...")
    trainer.train()
    
    # 保存最终模型
    print(f"保存模型到: {args.output_dir}")
    trainer.save_model(args.output_dir)
    
    print("训练完成！")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用LoRA技术对Llama模型进行中文能力微调")
    parser.add_argument("--model_path", type=str, default="models/base_model", 
                        help="已下载的基础模型路径")
    parser.add_argument("--dataset_path", type=str, default="processed_data/train_dataset", 
                        help="预处理后的数据集路径")
    parser.add_argument("--output_dir", type=str, default="results/lora-chinese-llama", 
                        help="模型保存目录")
    parser.add_argument("--lora_rank", type=int, default=16, 
                        help="LoRA秩")
    parser.add_argument("--lora_alpha", type=int, default=32, 
                        help="LoRA alpha参数")
    parser.add_argument("--lora_dropout", type=float, default=0.05, 
                        help="LoRA dropout")
    parser.add_argument("--num_epochs", type=int, default=3, 
                        help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=4, 
                        help="批大小")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, 
                        help="梯度累积步数")
    parser.add_argument("--learning_rate", type=float, default=2e-5, 
                        help="学习率")
    parser.add_argument("--weight_decay", type=float, default=0.01, 
                        help="权重衰减")
    parser.add_argument("--warmup_ratio", type=float, default=0.03, 
                        help="预热比例")
    parser.add_argument("--logging_steps", type=int, default=50, 
                        help="日志记录步数")
    parser.add_argument("--eval_steps", type=int, default=500, 
                        help="评估步数")
    parser.add_argument("--save_steps", type=int, default=500, 
                        help="保存步数")
    args = parser.parse_args()
    
    main(args) 
# LoRA中文能力微调实验

本项目旨在使用LoRA技术对Llama模型进行中文能力微调，同时监测和避免灾难性遗忘。

## 项目结构

```
LoRA-cursor-claude/
│
├── scripts/
│ ├── download_data.py      # 下载实验所需的所有数据集
│ ├── download_model.py     # 下载基础大语言模型
│ ├── preprocess_data.py    # 预处理数据集为训练和评估格式
│ ├── evaluate_baseline.py  # 评估基础模型性能（微调前）
│ ├── train_lora.py         # 使用LoRA技术微调模型
│ └── evaluate_model.py     # 评估微调后模型性能并分析灾难性遗忘
│
├── data/                   # 存储原始数据集
│
├── models/                 # 存储基础模型
│
├── processed_data/         # 存储预处理后的数据集
│
├── results/                # 存储训练结果和模型
│
└── evaluation_results/     # 存储评估结果和可视化
```

## 安装依赖
```bash
pip install -r requirements.txt
```

## 实验流程

### 1. 下载数据集

```bash
python scripts/download_data.py --output_dir data --belle_sample_size 50000
```

### 2. 预处理数据集
```bash
python scripts/preprocess_data.py --data_dir data --output_dir processed_data --model_name_or_path meta-llama/Llama-2-7b-hf
```

### 3. 下载基础模型
```bash
python scripts/download_model.py --model_name_or_path meta-llama/Llama-2-7b-hf --output_dir models
```

### 4. 评估基础模型（微调前）
```bash
python scripts/evaluate_baseline.py --model_path models/meta-llama/Llama-2-7b-hf --results_dir evaluation_results/baseline
```

### 5. LoRA微调
```bash
python scripts/train_lora.py --model_path models/meta-llama/Llama-2-7b-hf --dataset_path processed_data/train_dataset --output_dir results/lora-chinese-llama --lora_rank 16
```

### 6. 评估微调后模型
```bash
python scripts/evaluate_model.py --base_model_path models/meta-llama/Llama-2-7b-hf --lora_model_path results/lora-chinese-llama --results_dir evaluation_results/lora
```

## 实验设计

本实验旨在验证以下假设：

1. LoRA微调可以有效提升Llama模型的中文能力
2. 适当参数配置的LoRA微调可以避免灾难性遗忘现象

实验使用的主要数据集：
- **训练集**：BELLE中文指令数据集
- **中文评估**：C-Eval、CMMLU
- **英文评估**：MMLU、GSM8K等

LoRA配置参数可调整范围：
- rank: 4, 8, 16, 32, 64, 128
- alpha: 16, 32, 64
- dropout: 0.05, 0.1

## 结果分析

评估结果会自动生成以下内容：
1. 基础模型性能基线（JSON格式和图表）
2. 微调后模型性能（JSON格式）
3. 中文和英文能力雷达图对比
4. 灾难性遗忘现象分析

-------------------------------------------------

# Qwen LoRA中文微调

除了Llama模型外，本项目还支持使用LoRA技术对Qwen系列模型进行中文能力增强微调，并评估微调效果。

## Qwen微调流程

### 1. 下载Qwen基础模型

```bash
python scripts/download_qwen_model.py --model_name_or_path "Qwen/Qwen2.5-7B-Instruct" --output_dir "models"
```

### 2. 预处理数据(Qwen格式)

```bash
python scripts/preprocess_data_qwen.py --data_dir "data" --output_dir "processed_data_qwen" --model_name_or_path "models/base_model"
```

### 3. 评估Qwen基线性能

```bash
python scripts/evaluate_qwen_baseline.py --model_path "models/Qwen/Qwen2.5-7B-Instruct" --ceval_path "processed_data_qwen/ceval_dataset" --mmlu_path "processed_data_qwen/mmlu_dataset" --results_dir "evaluation_results/qwen_baseline"
```

### 4. 使用LoRA进行Qwen微调

```bash
python scripts/train_qwen_lora.py --model_path "models/base_model" --dataset_path "processed_data_qwen/train_dataset" --output_dir "results/lora-chinese-qwen" --num_epochs 3 --batch_size 4 --gradient_accumulation_steps 8 --learning_rate 1e-4
```

### 5. 评估微调后的Qwen模型

```bash
python scripts/evaluate_qwen_model.py --base_model_path "models/base_model" --lora_model_path "results/lora-chinese-qwen" --ceval_path "processed_data_qwen/ceval_dataset" --mmlu_path "processed_data_qwen/mmlu_dataset" --results_dir "evaluation_results/qwen"
```

## Qwen与Llama的主要差异

### 模型处理方式
- Qwen模型需要设置`trust_remote_code=True`
- 使用`AutoModelForCausalLM`和`AutoTokenizer`而非Llama专用类

### 指令格式
Qwen模型使用特殊的聊天格式：
```
<|im_start|>user
指令内容
<|im_end|>
<|im_start|>assistant
回答内容
<|im_end|>
```

### LoRA参数调整
针对Qwen模型的LoRA微调，主要调整以下参数：
- `lora_rank`: LoRA矩阵的秩，默认为8，可根据需要调整至4-32
- `lora_alpha`: LoRA缩放因子，默认为32
- `learning_rate`: 学习率，默认为1e-4，比Llama默认值(2e-5)高
- `target_modules`: 应用LoRA的模块设置为 `["c_attn", "c_proj", "w1", "w2"]`，适配Qwen架构

## 注意事项

1. Qwen处理结果时需要特别解析`<|im_start|>assistant`和`<|im_end|>`等标记
2. 建议为Qwen和Llama使用不同的预处理数据目录，避免格式混淆
3. 评估结果会保存在各自的目录下，便于比较两种模型的性能差异

## 许可证

MIT

## 参考资料

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [BELLE项目](https://github.com/LianjiaTech/BELLE)
- [Hugging Face PEFT库](https://github.com/huggingface/peft)
- [Qwen官方仓库](https://github.com/QwenLM/Qwen)
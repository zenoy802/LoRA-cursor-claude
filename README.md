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
python scripts/evaluate_baseline.py --model_path models/base_model --results_dir evaluation_results/baseline
```

### 5. LoRA微调
```bash
python scripts/train_lora.py --model_path models/base_model --dataset_path processed_data/train_dataset --output_dir results/lora-chinese-llama --lora_rank 16
```

### 6. 评估微调后模型
```bash
python scripts/evaluate_model.py --base_model_path models/base_model --lora_model_path results/lora-chinese-llama --results_dir evaluation_results/lora
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

## 许可证

MIT

## 参考资料

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [BELLE项目](https://github.com/LianjiaTech/BELLE)
- [Hugging Face PEFT库](https://github.com/huggingface/peft)
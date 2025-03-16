#!/bin/bash
# 运行LoRA检查点评估脚本的示例

# 设置通用参数
RESULTS_DIR="evaluation_results/checkpoints"
STEP_INTERVAL=0  # 每隔2个检查点评估一次，设为0则评估所有检查点

# 创建结果目录
mkdir -p $RESULTS_DIR

# ===== Llama模型检查点评估 =====
echo "开始评估Llama模型检查点..."

# 设置Llama特定参数
LLAMA_BASE_MODEL="models/meta-llama/Llama-2-7b-hf"  # 基础模型路径
LLAMA_CHECKPOINTS_DIR="results/lora-chinese-llama"  # 检查点目录
LLAMA_CEVAL_PATH="processed_data/ceval_dataset"  # C-Eval数据集路径
# LLAMA_CMMLU_PATH="processed_data/cmmlu_dataset"  # CMMLU数据集路径
LLAMA_MMLU_PATH="processed_data/mmlu_dataset"  # MMLU数据集路径

# 运行Llama检查点评估
python scripts/evaluate_checkpoints.py \
  --base_model_path $LLAMA_BASE_MODEL \
  --checkpoints_dir $LLAMA_CHECKPOINTS_DIR \
  --ceval_path $LLAMA_CEVAL_PATH \
  # --cmmlu_path $LLAMA_CMMLU_PATH \
  --mmlu_path $LLAMA_MMLU_PATH \
  --results_dir $RESULTS_DIR \
  --step_interval $STEP_INTERVAL

# ===== Qwen模型检查点评估 =====
# echo "开始评估Qwen模型检查点..."

# # 设置Qwen特定参数
# QWEN_BASE_MODEL="models/Qwen/Qwen2.5-7B-Instruct"  # 基础模型路径
# QWEN_CHECKPOINTS_DIR="results/lora-chinese-qwen"  # 检查点目录
# QWEN_CEVAL_PATH="processed_data_qwen/ceval_dataset"  # C-Eval数据集路径
# QWEN_CMMLU_PATH="processed_data_qwen/cmmlu_dataset"  # CMMLU数据集路径
# QWEN_MMLU_PATH="processed_data_qwen/mmlu_dataset"  # MMLU数据集路径

# # 运行Qwen检查点评估
# python scripts/evaluate_checkpoints.py \
#   --base_model_path $QWEN_BASE_MODEL \
#   --checkpoints_dir $QWEN_CHECKPOINTS_DIR \
#   --ceval_path $QWEN_CEVAL_PATH \
#   --cmmlu_path $QWEN_CMMLU_PATH \
#   --mmlu_path $QWEN_MMLU_PATH \
#   --results_dir $RESULTS_DIR \
#   --step_interval $STEP_INTERVAL

# echo "检查点评估完成！" 
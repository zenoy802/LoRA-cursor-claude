# LoRA检查点评估工具

这个工具用于评估LoRA训练过程中保存的多个检查点，帮助您跟踪模型性能随训练进度的变化，并找出最佳检查点。

## 功能特点

- 自动发现并评估训练过程中保存的所有检查点
- 支持按步数间隔筛选检查点，减少评估时间
- 支持Llama和Qwen模型架构
- 支持多个评估数据集（C-Eval、CMMLU、MMLU等）
- 按任务分析中文评估数据集的性能
- 生成性能随训练步数变化的曲线图
- 自动识别每个数据集上的最佳检查点

## 使用方法

### 直接运行评估脚本

```bash
python scripts/evaluate_checkpoints.py \
  --base_model_path <基础模型路径> \
  --checkpoints_dir <检查点目录> \
  --ceval_path <C-Eval数据集路径> \
  --cmmlu_path <CMMLU数据集路径> \
  --mmlu_path <MMLU数据集路径> \
  --results_dir <结果保存目录> \
  --step_interval <步数间隔>
```

### 使用示例脚本

我们提供了一个示例脚本，可以直接运行：

```bash
bash scripts/run_checkpoint_evaluation.sh
```

您可以根据需要修改脚本中的参数。

## 参数说明

- `--base_model_path`：基础模型路径，必需
- `--checkpoints_dir`：包含检查点的目录路径，必需
- `--ceval_path`：C-Eval数据集路径，必需
- `--cmmlu_path`：CMMLU数据集路径，可选
- `--mmlu_path`：MMLU数据集路径，可选
- `--results_dir`：结果保存目录，默认为"evaluation_results/checkpoints"
- `--step_interval`：评估检查点的步数间隔，0表示评估所有检查点，默认为0

## 输出结果

评估完成后，将在结果目录中生成以下文件：

- `step_<步数>_results.json`：每个检查点的详细评估结果
- `all_checkpoints_results.json`：所有检查点的汇总结果
- `best_checkpoints.json`：每个数据集上的最佳检查点信息
- `performance_curve.png`：性能随训练步数变化的曲线图

## 示例输出

```
找到 10 个检查点
根据步数间隔筛选后，将评估 6 个检查点
评估检查点: results/lora-chinese-llama/checkpoint-500
  评估数据集: C-Eval
    任务: computer_network (样本数: 34)
    任务 computer_network 准确率: 0.5294
    ...
  C-Eval 整体准确率: 0.4872
  评估数据集: MMLU
  MMLU 准确率: 0.6123
...
C-Eval 最佳检查点: 步数 2500, 准确率 0.5632
MMLU 最佳检查点: 步数 1500, 准确率 0.6245
评估完成！结果已保存到: evaluation_results/checkpoints/checkpoint_eval_20230815_123456
```

## 注意事项

1. 评估过程可能需要较长时间，特别是当检查点数量多或评估数据集大时
2. 建议使用`--step_interval`参数减少评估的检查点数量
3. 评估需要足够的GPU内存，如果内存不足，可能需要调整批处理大小或使用更小的模型 
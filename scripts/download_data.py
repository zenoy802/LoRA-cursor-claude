#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
下载实验所需的所有数据集
"""

import os
import argparse
from datasets import load_dataset

os.environ['HF_HOME'] = '/root/autodl-tmp/.cache/'

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
    # TODO: 清洗部分数据，如翻译任务
    if args.belle_sample_size > 0:
        belle_dataset = belle_dataset.shuffle(seed=42).select(range(args.belle_sample_size))
    belle_dataset.save_to_disk(os.path.join(args.output_dir, "belle_dataset"))
    print(f"BELLE数据集已保存，共{len(belle_dataset)}条样本")
    
    # 下载中文评估数据集
    print("下载中文评估数据集...")
    
    # C-Eval数据集
    print("下载C-Eval数据集...")
    task_list = ['computer_network', 'operating_system', 'computer_architecture', 'college_programming', 'college_physics', 'college_chemistry', 'advanced_mathematics', 'probability_and_statistics', 'discrete_mathematics', 'electrical_engineer', 'metrology_engineer', 'high_school_mathematics', 'high_school_physics', 'high_school_chemistry', 'high_school_biology', 'middle_school_mathematics', 'middle_school_biology', 'middle_school_physics', 'middle_school_chemistry', 'veterinary_medicine', 'college_economics', 'business_administration', 'marxism', 'mao_zedong_thought', 'education_science', 'teacher_qualification', 'high_school_politics', 'high_school_geography', 'middle_school_politics', 'middle_school_geography', 'modern_chinese_history', 'ideological_and_moral_cultivation', 'logic', 'law', 'chinese_language_and_literature', 'art_studies', 'professional_tour_guide', 'legal_professional', 'high_school_chinese', 'high_school_history', 'middle_school_history', 'civil_servant', 'sports_science', 'plant_protection', 'basic_medicine', 'clinical_medicine', 'urban_and_rural_planner', 'accountant', 'fire_engineer', 'environmental_impact_assessment_engineer', 'tax_accountant', 'physician']
    for task in task_list:
        try:
            ceval_dataset = load_dataset("ceval/ceval-exam", name=task, trust_remote_code=True)
        except Exception as e:
            print(f"无法直接加载C-Eval: {str(e)}")
            ceval_dataset = None
        if ceval_dataset is not None:
            ceval_dataset.save_to_disk(os.path.join(args.output_dir, f"ceval_dataset/{task}"))
            print(f"C-Eval数据集 {task} 已保存")
    print(f"C-Eval数据集已保存")
    
    # CMMLU数据集
    print("下载CMMLU数据集...")
    task_list = ['agronomy', 'anatomy', 'ancient_chinese', 'arts', 'astronomy', 'business_ethics', 'chinese_civil_service_exam', 'chinese_driving_rule', 'chinese_food_culture', 'chinese_foreign_policy', 'chinese_history', 'chinese_literature', 
'chinese_teacher_qualification', 'clinical_knowledge', 'college_actuarial_science', 'college_education', 'college_engineering_hydrology', 'college_law', 'college_mathematics', 'college_medical_statistics', 'college_medicine', 'computer_science',
'computer_security', 'conceptual_physics', 'construction_project_management', 'economics', 'education', 'electrical_engineering', 'elementary_chinese', 'elementary_commonsense', 'elementary_information_and_technology', 'elementary_mathematics', 
'ethnology', 'food_science', 'genetics', 'global_facts', 'high_school_biology', 'high_school_chemistry', 'high_school_geography', 'high_school_mathematics', 'high_school_physics', 'high_school_politics', 'human_sexuality',
'international_law', 'journalism', 'jurisprudence', 'legal_and_moral_basis', 'logical', 'machine_learning', 'management', 'marketing', 'marxist_theory', 'modern_chinese', 'nutrition', 'philosophy', 'professional_accounting', 'professional_law', 
'professional_medicine', 'professional_psychology', 'public_relations', 'security_study', 'sociology', 'sports_science', 'traditional_chinese_medicine', 'virology', 'world_history', 'world_religions']
    for task in task_list:
        try:
            cmmlu_dataset = load_dataset(r"haonan-li/cmmlu", task, trust_remote_code=True)
        except Exception as e:
            print(f"无法加载CMMLU: {str(e)}")
            cmmlu_dataset = None
        
        if cmmlu_dataset is not None:
            cmmlu_dataset.save_to_disk(os.path.join(args.output_dir, f"cmmlu_dataset/{task}"))
            print(f"CMMLU数据集 {task} 已保存")
    print(f"CMMLU数据集已保存")
    
    # 下载英文评估数据集（用于监测遗忘）
    print("下载英文评估数据集...")
    
    # MMLU数据集
    print("下载MMLU数据集...")
    try:
        mmlu_dataset = load_dataset("cais/mmlu", "all", trust_remote_code=True)
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
        gsm8k_dataset = load_dataset("gsm8k", "main", trust_remote_code=True)
        gsm8k_dataset.save_to_disk(os.path.join(args.output_dir, "gsm8k_dataset"))
        print(f"GSM8K数据集已保存")
    except Exception as e:
        print(f"无法加载GSM8K数据集，跳过: {str(e)}")
    
    # HellaSwag数据集
    print("下载HellaSwag数据集...")
    try:
        hellaswag_dataset = load_dataset("Rowan/hellaswag", trust_remote_code=True)
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
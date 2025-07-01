#!/bin/bash

# 确保在出错时脚本会退出
set -e

# --- 1. 路径配置 ---
# 你的基础模型路径
export BASE_MODEL_PATH="./merged-model"
# 你SFT阶段训练好的LoRA适配器路径 (用于生成数据和作为GRPO的起点)
export SFT_ADAPTER_PATH="../checkpoint-1405"
# 你的原始SFT数据路径 (多个文件用逗号隔开)
export DATA_PATH="../processed_data/train_data.jsonl"
# 最终GRPO LoRA适配器的输出目录
export OUTPUT_DIR="./results/qwen2-math-grpo-final"
# DeepSpeed配置文件的路径
export DS_CONFIG_PATH="./ds_config_stage2.json"



# --- 4. 启动命令 ---
# CUDA_VISIBLE_DEVICES=0 trl vllm-serve  --model ./merged_model --dtype auto先用此命令加载vllm的推理服务
# 使用 deepspeed 命令直接启动脚本
# --num_gpus 2: 指定使用2个GPU，这对应于你 accelerate_config.yaml 中的 num_processes: 2
# --master_port: 指定主进程端口，避免冲突，等同于 accelerate 的 --main_process_port
deepspeed  --include 'localhost:1,2' --master_port 29501 grpo_deepspeed.py \
    --deepspeed ${DS_CONFIG_PATH} \
    --model_name_or_path ${BASE_MODEL_PATH} \
    --sft_adapter_path ${SFT_ADAPTER_PATH} \
    --data_path ${DATA_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --model_max_length 4096 \
    --vllm_gpu_memory_utilization 0.8
    # ▲▲▲ 【新增和修改】在这里加入了3个新参数 ▲▲▲

echo "GRPO training finished. Final adapter saved to ${OUTPUT_DIR}"
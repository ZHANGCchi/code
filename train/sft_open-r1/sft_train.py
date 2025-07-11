# 最简短的SFT训练脚本
import pandas as pd
import os
from datasets import Dataset, load_dataset
from trl import SFTTrainer, SFTConfig
from swanlab.integration.transformers import SwanLabCallback
from transformers import TextStreamer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

DS_CONFIG = "ds_z2_offload_config.json"
from peft import LoraConfig, get_peft_model, TaskType 
import sys
from typing import Optional, Union

model_name = "Qwen/Qwen2.5-7B"
tokenizer = AutoTokenizer.from_pretrained(model_name)

device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)} 

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map = device_map,
    # attn_implementation="flash_attention_2"
)

model.enable_input_require_grads()

config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    modules_to_save=["lm_head", "embed_token"],
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    lora_dropout=0.05,
)
model = get_peft_model(model, config)

def format_user_assistant_template(example):
    """
    处理包含 "instruction" 和 "response" 字段的数据集。
    它会手动构建一个包含 system prompt 的完整对话结构，然后应用模板。
    """
    system_prompt = "Please reason step by step, and put your final answer within \\boxed{}."
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": example['instruction']},
        {"role": "assistant", "content": example['response']}
    ]
    return {"messages": messages}

print("正在使用新格式 ('instruction' / 'response' 字段) 处理数据...")
# 1. 加载新格式数据集 (请确保路径正确)
dataset_path = "/root/autodl-tmp/openr1/open-r1/output1.jsonl" # <--- 新格式数据路径
full_dataset = load_dataset("json", data_files=dataset_path, split="train")

# 2. 应用新格式的处理函数
processed_ds = full_dataset.map(
    format_user_assistant_template, 
    remove_columns=full_dataset.column_names
)

# --- 后续步骤是共通的 ---
# 3. 使用 datasets 内置的方法来拆分训练集和验证集
# ds_splits = processed_ds.train_test_split(test_size=0.01, seed=42)
# train_ds = ds_splits["train"]
# val_ds = ds_splits["test"]
train_ds = processed_ds

swanlab_callback = SwanLabCallback(
    project_name="qwen25",
    experiment_name="sft",
    config={
        "model_name": model_name,
        "dataset_name": "qwen-max-dataset",
        "task_type": "sft",
        "framework": "trl",
        "model_id": "qwen_F",
        "lora_rank": 32,
        "lora_alpha": 64,
        "lora_dropout": 0.05,
    },
)

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer, 

    train_dataset = train_ds,
    # eval_dataset = val_ds,
    callbacks=[swanlab_callback],
    args=SFTConfig(
        output_dir="./output1-mix-Math-sft",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        warmup_ratio = 0.03, 
        num_train_epochs=5,
        learning_rate=4.0e-05,
        optim="adamw_torch",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed = 42,
        logging_steps=1,
        save_steps=200,
        save_total_limit=3,
        fp16=True,
        max_grad_norm = 0.2,
        max_length=4096,
        report_to="none",
        deepspeed=DS_CONFIG,
        gradient_checkpointing=True, 
        gradient_checkpointing_kwargs={'use_reentrant':False},
        eos_token="<|im_end|>"
        # eval_strategy="steps",
        # eval_steps=200,      
    ),
)

trainer.train()

# 尽量简短的一个sft脚本，使用Qwen2.5-7B模型，支持两种数据格式
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
    torch_dtype=torch.bfloat16,
    device_map = device_map,
)

model.enable_input_require_grads()

config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    lora_dropout=0.05,
)
model = get_peft_model(model, config)

#################################################################################
#### 这是整合了两种数据处理方式的、更灵活的代码块 #############################
#################################################################################

# ==============================================================================
# ==== 配置开关: 请在这里选择要使用的数据格式 ====================================
# ==============================================================================
# 如果您的数据集含有 "messages" 字段 (旧格式), 请设置为 True
# 如果您的数据集含有 "user_content" 和 "qwen_response" 字段 (新格式), 请设置为 False
USE_MESSAGES_FORMAT = False 
# ==============================================================================

# --- 定义第一种数据处理函数 (旧格式) ---
def format_chat_template(example):
    """
    处理包含 "messages" 字段的数据集。
    它会在对话列表前加上一个固定的 system prompt，然后应用模板。
    """
    messages = example['messages']
    messages_with_system = [
        {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."}
    ] + messages
    return {
        "text": tokenizer.apply_chat_template(
            messages_with_system,  
            tokenize=False,
        )
    }

# --- 定义第二种数据处理函数 (新格式) ---
def format_user_assistant_template(example):
    """
    处理包含 "user_content" 和 "qwen_response" 字段的数据集。
    它会手动构建一个包含 system prompt 的完整对话结构，然后应用模板。
    """
    system_prompt = "Please reason step by step, and put your final answer within \\boxed{}."
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": example['user_content']},
        {"role": "assistant", "content": example['qwen_response']}
    ]
    return {
        "text": tokenizer.apply_chat_template(
            messages,
            tokenize=False,
        )
    }

# --- 根据开关变量，执行相应的数据加载和处理流程 ---
if USE_MESSAGES_FORMAT:
    print("正在使用旧格式 ('messages' 字段) 处理数据...")
    # 1. 加载旧格式数据集 (请确保路径正确)
    dataset_path = "/root/autodl-tmp/openr1/open-r1/Mix-T_M_filtered.jsonl" # <--- 旧格式数据路径
    full_dataset = load_dataset("json", data_files=dataset_path, split="train")
    
    # 2. 应用旧格式的处理函数
    processed_ds = full_dataset.map(
        format_chat_template,  
        remove_columns=full_dataset.column_names
    )
else:
    print("正在使用新格式 ('user_content' / 'qwen_response' 字段) 处理数据...")
    # 1. 加载新格式数据集 (请确保路径正确)
    dataset_path = "/root/autodl-tmp/openr1/open-r1/data_temp/qwen-max-dataset/*.jsonl" # <--- 新格式数据路径
    full_dataset = load_dataset("json", data_files=dataset_path, split="train")
    
    # 2. 应用新格式的处理函数
    processed_ds = full_dataset.map(
        format_user_assistant_template, 
        remove_columns=full_dataset.column_names
    )

# --- 后续步骤是共通的 ---
# 3. 使用 datasets 内置的方法来拆分训练集和验证集
ds_splits = processed_ds.train_test_split(test_size=0.01, seed=42)
train_ds = ds_splits["train"]
val_ds = ds_splits["test"]

print("\n数据处理完成，最终数据集结构如下:")
print(train_ds)
print(len(train_ds))
if len(train_ds) > 0:
    print(f"\n一条训练数据示例: \n{train_ds[0]['text']}")

#################################################
#### 新代码到这里结束 ###########################
#################################################


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
    eval_dataset = val_ds,
    callbacks=[swanlab_callback],
    args=SFTConfig(
        output_dir="./output_qwen-max-mix-sft",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        warmup_steps = 20, 
        num_train_epochs=5,
        learning_rate=2e-4,
        optim="adamw_torch",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed = 42,
        logging_steps=5,
        save_steps=200,
        save_total_limit=3,
        fp16=True,
        max_grad_norm= 1.0,
        max_length=8192,
        report_to="none",
        deepspeed=DS_CONFIG,
        gradient_checkpointing=True, 
        # eval_strategy="steps",   # 告诉 Trainer 按步数进行验证
        # eval_steps=200,                # 每200步验证一次，和 save_steps 保持一致
    ),
)


gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 , 1)
max_memory = round(gpu_stats.total_memory / 1024 , 1)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

trainer_stats = trainer.train()

# 显示最终内存和时间统计
used_memory = round(torch.cuda.max_memory_reserved() / 1024  , 1)
used_memory_for_lora = round(used_memory - start_gpu_memory, 1)
used_percentage = round(used_memory / max_memory * 100, 1)
lora_percentage = round(used_memory_for_lora / max_memory * 100, 1)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(
    f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training."
)
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")


class CaptureStreamer(TextStreamer):
    def __init__(self, tokenizer, skip_prompt: bool = False, **kwargs):
        super().__init__(tokenizer, skip_prompt=skip_prompt, **kwargs)
        self.generated_text = ""  # 用于存储完整输出

    def on_finalized_text(self, text: str, stream_end: bool = False):
        """重写方法捕获最终文本"""
        self.generated_text += text  # 累积输出
        super().on_finalized_text(text, stream_end=stream_end)  # 保持原样输出到终端

    def get_output(self) -> str:
        """获取完整生成内容"""
        return self.generated_text.strip()

def ask(question, is_thinking=True, save_to_file=None):
    messages = [{"role": "user", "content": question}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # 使用自定义的 CaptureStreamer
    streamer = CaptureStreamer(tokenizer, skip_prompt=True)

    # 生成响应
    model.eval()  # 确保模型在推理模式
    with torch.no_grad():
        _ = model.generate(
            **tokenizer(text, return_tensors="pt").to("cuda"),
            max_new_tokens=1024,
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            streamer=streamer,  # 关键：使用自定义的 streamer
        )

    # 获取完整输出
    full_output = streamer.get_output()

    # 保存到文件
    if save_to_file:
        try:
            with open(save_to_file, "w", encoding="utf-8") as f:
                f.write(full_output)
            print(f"✅ 成功写入文件: {save_to_file}")
        except Exception as e:
            print(f"❌ 写入文件失败: {e}")

    return full_output

# 测试调用
ask(" What are all values of $p$ such that for every $q>0$, we have   $$\\frac{3(pq^2+p^2q+3q^2+3pq)}{p+q}>2p^2q?$$ Express your answer in interval notation in decimal form.",
    save_to_file='./output.txt')
print("#############################################################################################")
print("#############################################################################################")
print("#############################################################################################")
print("#############################################################################################")
print("#############################################################################################")


# ask("根据描述，一个1岁的孩子在夏季头皮出现多处小结节，长期不愈合，且现在疮大如梅，溃破流脓，口不收敛，头皮下有空洞，患处皮肤增厚。这种病症在中医中诊断为什么病？",is_thinking=False)

model.save_pretrained("lora_model")  # Local saving
tokenizer.save_pretrained("lora_model")
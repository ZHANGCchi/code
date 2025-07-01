# grpo_deepspeed.py

# ----------------------------------------------------------------
# 导入所有必要的库
# ----------------------------------------------------------------
import os
import io
import json
import logging
import random
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence
import sys

# 打印环境信息
print(f"✅ 正在运行的 Python 文件是: {__file__ if '__file__' in locals() else 'Jupyter/Interactive session'}")
print(f"✅ Python 模块搜索路径 sys.path = {sys.path}")

import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from functools import partial

import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    set_seed,
    TrainerCallback, # ▲▲▲ 【新增】导入Callback ▲▲▲
)
from peft import LoraConfig, PeftModel
from trl import GRPOTrainer, GRPOConfig

import swanlab # ▲▲▲ 【新增】导入swanlab ▲▲▲

# ----------------------------------------------------------------
# 1. 参数定义 (Dataclasses) - 已更新
# ----------------------------------------------------------------
@dataclass
class ModelArguments:
    """模型相关参数"""
    model_name_or_path: str = field(
        metadata={"help": "基础大模型的路径"}
    )
    sft_adapter_path: Optional[str] = field(
        default=None,
        metadata={"help": "SFT阶段训练好的LoRA适配器路径，作为GRPO训练的起点。"}
    )

@dataclass
class DataArguments:
    """数据相关参数"""
    data_path: str = field(
        metadata={"help": "原始SFT数据文件路径，多个文件用逗号隔开。"}
    )
    validation_split_percentage: int = field(
        default=5,
        metadata={"help": "从数据集中划分出多少百分比作为验证集。"}
    )


@dataclass
class TrainingArguments(GRPOConfig):
    """GRPO训练参数 (继承自trl.GRPOConfig) - 已更新"""
    output_dir: str = field(default="./results_qwen_grpo")
    num_train_epochs: float = field(default=3.0)
    # GRPOTrainer会为每个prompt生成num_completions个答案
    num_generations: int = field(
        default=6,
        metadata={"help": "为每个问题生成多少个候选答案用于排序。"}
    )
    beta: float = field(default=0.0, metadata={"help": "GRPO的正则化系数。"})
    per_device_train_batch_size: int = field(default=3)
    per_device_eval_batch_size: int = field(default=1)
    gradient_accumulation_steps: int = field(default=3)
    gradient_checkpointing: bool = field(default=True)
    learning_rate: float = field(default=5e-6)
    lr_scheduler_type: str = field(default="cosine")
    warmup_ratio: float = field(default=0.05)
    logging_strategy: str = field(default="steps")
    logging_steps: int = field(default=5)
    # evaluation_strategy: str = field(default="steps", metadata={"help": "必须设为 'steps' 才能在训练中进行验证。"})
    # eval_steps: int = field(default=50)
    save_strategy: str = field(default="steps")
    save_steps: int = field(default=50)
    save_total_limit: int = field(default=10)
    bf16: bool = field(default=True, metadata={"help": "在支持的硬件上（如A100）建议开启。"})
    fp16: bool = field(default=False)
    deepspeed: Optional[str] = field(default=None, metadata={"help": "DeepSpeed配置文件的路径。"})
    optim: str = field(default="adamw_torch", metadata={"help": "推荐使用分页优化器以节省显存。"})
    model_max_length: int = field(default=4096, metadata={"help": "模型的最大输入长度，默认为4096。"})
    temperature: float = field(default=0.7)
    top_p: float = field(default=0.9)
    max_completion_length: int = field(default=3584, metadata={"help": "模型采样的最大生成长度，默认为3584。"})
    seed: int = field(default=42)
    remove_unused_columns: bool = field(default=False, metadata={"help": "必须设为False，否则trl会丢弃'answer'等重要列。"})
    report_to: str = field(default="none", metadata={"help": "关闭默认的wandb集成"}) # ▲▲▲ 【修改】关闭wandb集成 ▲▲▲
    use_vllm: bool = field(default=True)
    vllm_mode: str = field(default="server")
    vllm_server_base_url: str = field(default="http://0.0.0.0:8000")

# ▲▲▲ 【新增】SwanLab回调类 ▲▲▲
class SwanLabCallback(TrainerCallback):
    """
    一个Trainer回调，用于在每次日志记录时将指标同步到swanlab。
    """
    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_world_process_zero:  # 只在主进程中记录
            if logs is not None:
                # logs 字典包含了 "loss", "learning_rate" 等信息
                swanlab.log(logs)

# ----------------------------------------------------------------
# 2. 工具函数和数据处理类 (无变化)
# ----------------------------------------------------------------
PROMPT_DICT = {
    "prompt_input": """<|im_start|>system
You are a helpful math assistant. Solve the problem step by step.
At the end, output the final answer in the following format:
**Answer:** \\boxed{{your_final_numeric_answer}}
Do NOT include any text after the boxed answer.<|im_end|>
<|im_start|>user
{input}<|im_end|>
<|im_start|>assistant
""",
}

def jload(f, mode="r", jsonl=True):
    """加载JSON或JSONL文件"""
    if not isinstance(f, io.IOBase):
        with open(f, mode=mode, encoding="utf-8") as f_obj:
            if jsonl:
                return [json.loads(line) for line in f_obj if line.strip()]
            else:
                return json.load(f_obj)
    else:
        if jsonl:
            return [json.loads(line) for line in f if line.strip()]
        else:
            return json.load(f)

# ===================================================================
#                      答案评估与提取模块 (无变化)
# ===================================================================
def normalize_number(s: any) -> Optional[float or str]:
    """一个更健壮的数字标准化函数。"""
    if isinstance(s, (int, float)):
        return s
    if not isinstance(s, str):
        return s
    s = s.strip().replace(",", "")
    if s.endswith("%"):
        try:
            return float(s[:-1]) / 100.0
        except ValueError:
            return s
    try:
        return float(s)
    except ValueError:
        return s

def extract_final_boxed_content(text: str) -> Optional[str]:
    """健壮地提取最后一个 \\boxed{...} 中的内容。"""
    last_boxed_start = text.rfind('\\boxed{')
    if last_boxed_start == -1:
        return None
    start_idx = last_boxed_start + len('\\boxed{')
    bracket_count = 1
    end_idx = start_idx
    while end_idx < len(text) and bracket_count > 0:
        if text[end_idx] == '{':
            bracket_count += 1
        elif text[end_idx] == '}':
            bracket_count -= 1
        end_idx += 1
    if bracket_count == 0:
        return text[start_idx:end_idx-1]
    return None

def are_answers_equal(model_ans_str: str, ground_truth_ans_str: str) -> bool:
    """使用健壮的逻辑比较两个答案。"""
    if model_ans_str is None or ground_truth_ans_str is None:
        return False
    try:
        # 统一 \dfrac 为 \frac
        s_val1 = re.sub(r'\\dfrac{(.*?)}{(.*?)}', r'\\frac{\1}{\2}', model_ans_str)
        s_val2 = re.sub(r'\\dfrac{(.*?)}{(.*?)}', r'\\frac{\1}{\2}', ground_truth_ans_str)

        # 尝试解析分数
        frac_match1 = re.match(r'\\frac{(.*)}{(.*)}', s_val1)
        frac_match2 = re.match(r'\\frac{(.*)}{(.*)}', s_val2)

        if frac_match1:
            num = normalize_number(frac_match1.group(1))
            den = normalize_number(frac_match1.group(2))
            if isinstance(num, (int, float)) and isinstance(den, (int, float)) and den != 0:
                s_val1 = num / den
        if frac_match2:
            num = normalize_number(frac_match2.group(1))
            den = normalize_number(frac_match2.group(2))
            if isinstance(num, (int, float)) and isinstance(den, (int, float)) and den != 0:
                s_val2 = num / den

        # 标准化并比较
        num1 = normalize_number(s_val1)
        num2 = normalize_number(s_val2)

        if isinstance(num1, (int, float)) and isinstance(num2, (int, float)):
            return abs(num1 - num2) < 1e-6 # 浮点数比较
        else:
            return str(num1).strip() == str(num2).strip() # 字符串比较
    except Exception as e:
        logging.error(f"比较答案时出错: val1='{model_ans_str}', val2='{ground_truth_ans_str}'. 错误: {e}")
        return False


# ===================================================================
#                 【重构后】GRPOTrainer 使用的奖励函数
# ===================================================================
def compute_final_reward(prompts: List[str], completions: List[str], answers: List[str], **kwargs) -> List[float]:
    """
    重构后的奖励函数。
    它接收由Trainer生成的completions和从数据集中传入的ground truth answers。
    """
    rewards = []

    # 提取标准答案中的 boxed 内容
    # 由于对于同一个prompt的所有completions，其标准答案是相同的，所以可以提前提取
    ground_truth_ans_str = extract_final_boxed_content(answers[0])

    if ground_truth_ans_str is None:
        logging.warning(f"无法从标准答案中提取box内容，将为所有候选答案打0分。标准答案: {answers[0]}")
        return [0.0] * len(completions)

    for comp in completions:
        # 步骤 1: 检查当前生成答案的格式 (此检查可选，但有助于奖励塑形)
        if comp.count('---') != 1:
            rewards.append(0.0)
            continue

        # 步骤 2: 从当前生成的答案中提取数值
        model_ans_str = extract_final_boxed_content(comp)
        if model_ans_str is None:
            rewards.append(0.0) # 格式错误，无法提取
            continue

        # 步骤 3: 比较并返回分数
        if are_answers_equal(model_ans_str, ground_truth_ans_str):
            rewards.append(2.0) # 答案正确
        else:
            rewards.append(0.2) # 答案错误但格式正确

    return rewards

# ===================================================================
#                      【重构后】Dataset 类
# ===================================================================
class GRPODataset(Dataset):
    """
    一个简单的Dataset类，用于包装预处理过的数据。
    __getitem__ 返回一个字典，包含 'prompt' 和 'answer'，
    以便Trainer可以自动将 'answer' 传递给奖励函数。
    """
    def __init__(self, processed_data: List[Dict[str, str]], tokenizer: transformers.PreTrainedTokenizer):
        self.data = processed_data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index) -> Dict[str, str]:
        # 返回的数据将由DataCollator处理，并传递给模型和奖励函数
        return self.data[index]

# ----------------------------------------------------------------
# 3. 训练主函数 (已重构)
# ----------------------------------------------------------------
def train():
    """GRPO训练主函数"""
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    set_seed(training_args.seed)

    # --- 步骤 1: 加载基础模型和SFT LoRA适配器 ---
    logging.info("步骤 1: 在GPU上加载基础模型。")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if training_args.bf16 else torch.float16,
        device_map={"": torch.cuda.current_device()} # 确保模型加载到GPU
    )

    model_to_train = base_model
    if model_args.sft_adapter_path:
    #     logging.info(f"步骤 1.1: 加载SFT LoRA适配器 {model_args.sft_adapter_path}。")
    #     model_to_train = PeftModel.from_pretrained(base_model, model_args.sft_adapter_path)
    #     logging.info("已成功将SFT LoRA适配器合并到基础模型上。")
    # else:
        logging.warning("警告：未提供SFT适配器路径 (--sft_adapter_path)，将直接在基础模型上进行GRPO训练。")

    # --- 步骤 2: 加载分词器 ---
    logging.info("步骤 2: 加载分词器。")
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        model_max_length=training_args.model_max_length,
        padding_side="left", # 对生成任务，padding在左侧是标准做法
        use_fast=False,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logging.info("分词器的 pad_token 未设置，已将其设置为 eos_token。")

    # --- 步骤 3: 加载并处理原始数据 ---
    logging.info(f"步骤 3: 从 {data_args.data_path} 加载并处理数据。")
    raw_data = []
    for path in data_args.data_path.split(','):
        raw_data.extend(jload(path.strip()))

    processed_data = []
    for item in tqdm(raw_data, desc="正在处理数据集"):
        question = item.get("question")
        # 从嵌套结构中获取推理和最终答案
        message = item.get("message", {})
        reasoning = message.get("reasoning_content", "")
        content = message.get("content", "")

        if not question or not content:
            continue

        prompt = PROMPT_DICT["prompt_input"].format(input=question)
        # 将推理过程和最终答案组合成完整的 "answer"
        full_answer = f"{reasoning}\n\n---\n{content}"

        processed_data.append({"prompt": prompt, "answers": full_answer})

    if not processed_data:
        raise ValueError(f"错误: 从 {data_args.data_path} 加载并处理后，数据集为空！请检查数据格式和路径。")
    logging.info(f"数据处理完成，共得到 {len(processed_data)} 条有效数据。")


    # --- 步骤 4: 分割数据集并创建Dataset实例 ---
    logging.info("步骤 4: 分割训练集和验证集。")
    eval_size = int(len(processed_data) * (data_args.validation_split_percentage / 100.0))
    train_size = len(processed_data) - eval_size

    # 使用torch.utils.data.random_split进行可复现的分割
    train_subset, eval_subset = torch.utils.data.random_split(
        processed_data, [train_size, eval_size],
        generator=torch.Generator().manual_seed(training_args.seed)
    )

    train_dataset = GRPODataset(list(train_subset), tokenizer)
    eval_dataset = GRPODataset(list(eval_subset), tokenizer)
    logging.info(f"数据集分割完成: {len(train_dataset)}条训练数据, {len(eval_dataset)}条验证数据。")


    # --- 步骤 5: 初始化 GRPOTrainer ---
    logging.info("步骤 5: 初始化GRPOTrainer。")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"], # 针对更多层以提高性能
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # ▲▲▲ 【新增】初始化 SwanLab ▲▲▲
    # 将会自动从环境变量 SWANLAB_API_KEY, SWANLAB_PROJECT 等读取配置
    swanlab.init(
        config=training_args.to_dict(), # 记录所有训练超参数
        # project="Your-Project-Name", # 可选：指定项目名称
        # experiment_name="qwen2-grpo-run-1" # 可选：指定实验名称
    )


    trainer = GRPOTrainer(
        model=model_to_train,
        reward_funcs=compute_final_reward, # 使用我们重构后的奖励函数
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=lora_config,
        callbacks=[SwanLabCallback()], # ▲▲▲ 【修改】添加回调 ▲▲▲
    )

    # ===================================================================
    #                      【请将下面的代码添加到此处】
    # ===================================================================
    # 导入math库用于向上取整
    import math
    import os

    # 从DeepSpeed/Accelerate的环境变量中获取GPU数量
    # 当您使用 deepspeed --num_gpus 2 ... 启动时, 这个环境变量会被设置为 2
    num_gpus = int(os.environ.get("WORLD_SIZE", 1))

    train_dataset_size = len(train_dataset)
    num_epochs = training_args.num_train_epochs
    
    # 【关键】直接从 Hugging Face Trainer 的最终配置中获取梯度累积步数
    # 这样可以反映出所有配置（脚本、DeepSpeed JSON）合并后的最终结果
    grad_acc_steps = trainer.args.gradient_accumulation_steps
    
    # 计算每个GPU处理的数据量
    samples_per_gpu = math.ceil(train_dataset_size / num_gpus)
    
    # 计算每个epoch的更新次数
    updates_per_epoch = math.ceil(samples_per_gpu / grad_acc_steps)
    
    # 计算总步数
    total_steps = updates_per_epoch * num_epochs

    # 使用 logging 打印，确保在分布式训练中能清晰看到
    logging.info("--------------------------------------------------")
    logging.info("✅ 总训练步数计算详情:")
    logging.info(f"    - 训练数据集大小: {train_dataset_size}")
    logging.info(f"    - GPU 数量 (WORLD_SIZE): {num_gpus}")
    logging.info(f"    - 训练周期数 (num_train_epochs): {num_epochs}")
    logging.info(f"    - 【最终生效】梯度累积步数: {grad_acc_steps}")
    logging.info("---")
    logging.info("    计算过程:")
    logging.info(f"    1. 每个GPU处理的样本数 = ceil({train_dataset_size} / {num_gpus}) = {samples_per_gpu}")
    logging.info(f"    2. 每个Epoch的更新次数 = ceil({samples_per_gpu} / {grad_acc_steps}) = {updates_per_epoch}")
    logging.info(f"    3. 总步数 = {updates_per_epoch} (每轮步数) * {num_epochs} (周期) = {total_steps}")
    logging.info("--------------------------------------------------")
    # ===================================================================
    #                      【添加到此结束】
    # ===================================================================
    
    # --- 步骤 6: 开始训练 ---
    logging.info("步骤 6: 开始GRPO训练...")
    trainer.train()

    # --- 步骤 7: 保存最终模型 ---
    logging.info("步骤 7: 训练完成，保存最终适配器。")
    final_output_dir = os.path.join(training_args.output_dir, "final_checkpoint")
    trainer.save_model(output_dir=final_output_dir)
    logging.warning(f"最终GRPO LoRA适配器已保存至: {final_output_dir}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
    )
    train()
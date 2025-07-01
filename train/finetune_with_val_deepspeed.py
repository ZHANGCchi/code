# 导入必要的库
import copy
import os
import io
import json
import logging
import random  # 新增：用于混洗数据集
from dataclasses import dataclass, field, asdict
from typing import Dict, Optional, Sequence

import torch
from torch.utils.data import Dataset
try:
    import openmind as tf_module
except:
    import transformers as tf_module
import transformers

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from swanlab.integration.transformers import SwanLabCallback

# 定义忽略标记，用于屏蔽损失计算中的某些标记(如输入部分)
IGNORE_INDEX = -100

# 定义模型的提示模板，用于构建输入
PROMPT_DICT = {
    "prompt_no_input": """<|im_start|>system\n{instruction}<|im_end|>\n<|im_start|>user\n<|im_end|>\n<|im_start|>assistant\n""",
    "prompt_input": """<|im_start|>system\n{instruction}<|im_end|>\n<|im_start|>user\n{input}<|im_end|>\n<|im_start|>assistant\n""",
}


@dataclass
class ModelArguments:
    """模型相关参数"""
    model_name_or_path: Optional[str] = field(
        default="./qwen/Qwen2___5-7B-Instruct"
    )


@dataclass
class DataArguments:
    """数据相关参数"""
    data_path: str = field(
        default="./hard_train_gsm8k_filtered.jsonl, ./hard_train_math_filtered.jsonl, ./hard_train_prm_filtered.jsonl",
        metadata={"help": "训练数据文件的路径。可以提供多个路径，用逗号分隔。"},
    )


# VVVVVVVVVVVVVVVVVVVVVV  修改部分 1: 更新训练参数以包含验证设置 VVVVVVVVVVVVVVVVVVVVVV
@dataclass
class TrainingArguments(transformers.TrainingArguments):
    """
    训练相关参数，包含推荐的默认值和验证配置。
    """
    # --- 核心训练参数 ---
    output_dir: str = field(
        default="./results_qwen_finetuned",
        metadata={"help": "模型检查点和最终模型的输出目录。"}
    )
    num_train_epochs: int = field(
        default=7,
        metadata={"help": "训练的总轮数 (epochs)。"}
    )

    # --- 显存与批次大小相关参数 ---
    per_device_train_batch_size: int = field(
        default=4,  # 保持我们希望的训练批量大小
        metadata={"help": "每个GPU设备上的训练批次大小。"}
    )
    # 新增或修改下面这个参数，为验证设置一个极小的批量
    per_device_eval_batch_size: int = field(
        default=1,  # 明确设置验证的批量大小为1，解决OOM问题
        metadata={"help": "每个GPU设备上的验证批次大小。"}
    )
    
    gradient_accumulation_steps: int = field(
        default=1,  # 保持为1，避免之前的 no_sync 错误
        metadata={"help": "梯度累积的步数。"}
    )
    # DEEPSPEED修改 1: (可选但推荐) 开启梯度检查点进一步节省显存
    gradient_checkpointing: bool = field(
        default=True,
        metadata={"help": "是否启用梯度检查点来节省显存。"}
    )

    # --- 学习率与调度器相关参数 ---
    learning_rate: float = field(default=2e-4, metadata={"help": "初始学习率。"})
    weight_decay: float = field(default=0.01, metadata={"help": "权重衰减值。"})
    lr_scheduler_type: str = field(default="cosine", metadata={"help": "学习率调度器类型。"})
    warmup_ratio: float = field(default=0.05, metadata={"help": "学习率预热的比例。"})

    # --- 日志与保存、评估相关参数 ---
    logging_strategy: str = field(default="steps", metadata={"help": "日志记录策略。"})
    logging_steps: int = field(default=20, metadata={"help": "每隔 N 步记录一次训练日志。"})
    evaluation_strategy: str = field(default="steps", metadata={"help": "验证策略 ('steps' 或 'epoch')。"})
    eval_steps: int = field(default=100, metadata={"help": "每隔 N 步进行一次验证。"})
    save_strategy: str = field(default="steps", metadata={"help": "模型保存策略。"})
    save_steps: int = field(default=100, metadata={"help": "每隔 N 步保存一次检查点。"})
    save_total_limit: int = field(default=3, metadata={"help": "最多保存的检查点数量。"})

    # --- 性能优化参数 ---
    bf16: bool = field(default=True, metadata={"help": "是否使用 bfloat16。"})
    fp16: bool = field(default=False, metadata={"help": "是否使用 float16。"})

    # --- DeepSpeed 相关参数 ---
    # DEEPSPEED修改 2: 添加deepspeed配置项，指向我们的json文件
    deepspeed: Optional[str] = field(
        default="./ds_config.json", # 默认指向当前目录的ds_config.json
        metadata={"help": "DeepSpeed配置文件的路径。"}
    )

    # --- 其他参数 ---
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(default=4096, metadata={"help": "最大序列长度。"})
    seed: int = field(default=42, metadata={"help": "用于复现的随机种子。"})

# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


def _tokenize_fn(strings: Sequence[str], tokenizer) -> Dict:
    """将字符串列表转换为token"""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


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

# vvvvvvvvvvvvvv 在这里添加下面的 jdump 函数 vvvvvvvvvvvvvv
def jdump(obj, f, mode="w", jsonl=True):
    """将对象保存为JSON或JSONL文件"""
    if not isinstance(f, io.IOBase):
        with open(f, mode=mode, encoding="utf-8") as f_obj:
            if jsonl:
                for item in obj:
                    f_obj.write(json.dumps(item, ensure_ascii=False) + "\n")
            else:
                json.dump(obj, f_obj, ensure_ascii=False, indent=4)
    else:
        if jsonl:
            for item in obj:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        else:
            json.dump(obj, f, ensure_ascii=False, indent=4)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer,
) -> Dict:
    """预处理数据，将源文本和目标文本转换为token，并设置标签"""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [
        _tokenize_fn(strings, tokenizer) for strings in (examples, sources)
    ]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


# VVVVVVVVVVVVVVVVVVVVVV  修改部分 2: 使数据集类接收列表而非文件路径 VVVVVVVVVVVVVVVVVVVVVV
class SupervisedDataset(Dataset):
    """监督微调数据集"""

    def __init__(self, list_data_dict: list, tokenizer):
        """
        修改后的初始化方法
        Args:
            list_data_dict: 包含数据样本的列表，而不是文件路径
            tokenizer: 分词器
        """
        super(SupervisedDataset, self).__init__()
        logging.warning("Formatting inputs...")

        # 1. 在这里定义您用于生成数据的、固定的系统提示
        system_prompt = (
            "You are a helpful math assistant. Solve the problem step by step.\n"
            "At the end, output the final answer in the following format:\n"
            "**Answer:** \\boxed{your_final_numeric_answer}\n"
            "Do NOT include any text after the boxed answer."
        )
        
        # 2. 选择带有 input 的提示模板
        prompt_input = PROMPT_DICT["prompt_input"]

        internal_data_list = []
        for example in list_data_dict:
            message_obj = example.get("message", {})
            reasoning = message_obj.get("reasoning_content", "")
            final_answer = message_obj.get("content", "")
            combined_output = f"{reasoning}\n\n---\n{final_answer}"
            
            # 3. 将系统提示和问题分别填入 instruction 和 input
            internal_data_list.append({
                "instruction": system_prompt,              # 使用固定的系统指令
                "input": example.get("question", ""),      # 将原始问题作为用户输入
                "output": combined_output
            })

        # 4. 使用 prompt_input 来格式化源文本，这样能同时包含 instruction 和 input
        sources = [
            prompt_input.format_map(example)
            for example in internal_data_list
        ]
        targets = [
            f"{example['output']}{tokenizer.eos_token}"
            for example in internal_data_list
        ]

        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer)

        try:
            self.input_ids = data_dict["input_ids"]
        except KeyError as e:
            raise KeyError("input_ids is invalid") from e
        try:
            self.labels = data_dict["labels"]
        except KeyError as e:
            raise KeyError("labels is invalid") from e

    def __len__(self):
        """返回数据集长度"""
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        """获取数据集中的单个样本"""
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


@dataclass
class DataCollatorForSupervisedDataset(object):
    """监督微调数据集的数据整理器，用于批处理"""

    tokenizer: object

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        """将多个样本整理为批次"""
        input_ids, labels = tuple(
            [instance[key] for instance in instances] for key in ("input_ids", "labels")
        )
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


# VVVVVVVVVVVVVVVVVVVVVV  修改部分: 加载多个数据集 VVVVVVVVVVVVVVVVVVVVVV
def make_supervised_data_module(tokenizer, data_args) -> Dict:
    """
    一个更智能的数据模块创建函数。
    它会检查是否存在已处理好的数据集，如果没有，则创建并保存它们。
    """
    
    # 1. 定义处理后数据的固定存放路径
    # 将处理好的数据统一存放在一个子目录中，保持整洁
    processed_dir = "./processed_data"
    train_path = os.path.join(processed_dir, "train_data.jsonl")
    eval_path = os.path.join(processed_dir, "validation_data.jsonl")
    test_path = os.path.join(processed_dir, "test_data.jsonl")

    # 2. 检查已处理的数据集是否存在
    if os.path.exists(train_path) and os.path.exists(eval_path) and os.path.exists(test_path):
        # 如果存在，直接加载
        logging.warning(f"发现已存在的已处理数据集，将从 '{processed_dir}' 目录直接加载...")
        train_list = jload(train_path)
        eval_list = jload(eval_path)
        # 测试集在此函数中不再需要加载，因为它仅用于训练结束后的独立评估
        logging.warning(f"成功加载 {len(train_list)} 条训练样本和 {len(eval_list)} 条验证样本。")

    else:
        # 如果不存在，则执行一次性处理流程
        logging.warning("未发现已处理的数据集，将执行一次性混洗、分割和保存...")
        
        # 确保输出目录存在
        os.makedirs(processed_dir, exist_ok=True)

        # a. 从原始路径加载所有数据集
        logging.warning("从原始路径加载数据...")
        data_paths = data_args.data_path.split(',')
        full_data_list = []
        for path in data_paths:
            path = path.strip()
            if not path: continue
            logging.warning(f"Loading data from {path}...")
            full_data_list.extend(jload(path))

        logging.warning(f"Totally loaded {len(full_data_list)} examples from all sources.")

        # b. 随机混洗 (请确保在train()函数开头设置了随机种子以保证复现性)
        random.shuffle(full_data_list)

        # c. 分割数据集
        train_ratio, eval_ratio = 0.8, 0.1 # 80%训练, 10%验证, 剩下10%为测试
        train_end_idx = int(len(full_data_list) * train_ratio)
        eval_end_idx = int(len(full_data_list) * (train_ratio + eval_ratio))
        
        train_list = full_data_list[:train_end_idx]
        eval_list = full_data_list[train_end_idx:eval_end_idx]
        test_list = full_data_list[eval_end_idx:]

        # d. 将分割后的三个文件保存到磁盘
        logging.warning(f"正在保存已分割的数据集至 '{processed_dir}' 以备将来使用...")
        jdump(train_list, train_path)
        jdump(eval_list, eval_path)
        jdump(test_list, test_path)
        logging.warning("数据集保存完毕。")

    # 3. 后续步骤：为加载好的训练集和验证集创建 Dataset 对象
    logging.warning(f"Splitting data: {len(train_list)} for training, {len(eval_list)} for evaluation.")

    train_dataset = SupervisedDataset(
        list_data_dict=train_list, tokenizer=tokenizer
    )
    eval_dataset = SupervisedDataset(
        list_data_dict=eval_list, tokenizer=tokenizer
    )

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    return dict(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator
    )
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


def train():
    """训练主函数（已更新为支持续训和可复现）"""
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # --- 修改一：在所有操作前设置随机种子 ---
    # 这是确保所有随机操作（包括数据划分）可复现的关键
    transformers.set_seed(training_args.seed)

    # --- 后续的模型和分词器加载逻辑保持不变 ---
    model = tf_module.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        trust_remote_code=True,
    )

    model.enable_input_require_grads()

    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    model.print_trainable_parameters()

    tokenizer = tf_module.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

    # --- SwanLab回调逻辑保持不变 ---
    if not os.environ.get("DEEPSPEED_ZERO_STAGE", "0") == "3" or torch.distributed.get_rank() == 0:
        swanlab_call = SwanLabCallback(
            project="Ascend_finetune_v2_deepspeed",
            experiment_name=os.path.basename(os.path.normpath(training_args.output_dir)),
            config=asdict(data_args) | asdict(model_args) | asdict(training_args) | asdict(lora_config),
            public=True,
        )
        callbacks = [swanlab_call]
    else:
        callbacks = []

    # --- 创建Trainer的逻辑保持不变 ---
    trainer = tf_module.Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        callbacks=callbacks,
        **data_module,
    )
    
    # --- 修改二：使用更稳健的强制续训逻辑 ---
    # 自动寻找最新的检查点
    last_checkpoint = transformers.trainer_utils.get_last_checkpoint(training_args.output_dir)
    
    if last_checkpoint:
        logging.warning(f"检测到检查点，将从 {last_checkpoint} 恢复训练...")
    else:
        # 这是一个安全检查，如果目录为空，就从头开始
        logging.warning("未检测到检查点，将从头开始训练。")
    
    # 开始训练，并明确指定从哪个检查点恢复
    # 如果last_checkpoint为None，trainer.train的行为等同于从头开始，逻辑正确
    trainer.train(resume_from_checkpoint=last_checkpoint)

    # --- 保存最终模型的逻辑保持不变 ---
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
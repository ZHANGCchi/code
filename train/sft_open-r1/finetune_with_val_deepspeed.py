# 使用部分/Mixture-of-Thoughts_math_dataset数据进行Lora微调，代码整体还是和train下的finetune_with_val_deepspeed.py类似
# 此微调只学prompt之后的回答
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

# 新增：支持arrow格式
import datasets
from datasets import Dataset as HFDataset, load_from_disk

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from swanlab.integration.transformers import SwanLabCallback

# 定义忽略标记，用于屏蔽损失计算中的某些标记(如输入部分)
IGNORE_INDEX = -100


@dataclass
class ModelArguments:
    """模型相关参数"""
    model_name_or_path: Optional[str] = field(
        default="../../qwen/Qwen2___5-7B-Instruct"
    )


@dataclass
class DataArguments:
    """数据相关参数"""
    data_path: str = field(
        default="/root/Mixture-of-Thoughts_math_dataset",  # 支持arrow格式
        metadata={"help": "训练数据文件的路径。支持.jsonl、.json或.arrow格式。可以提供多个路径，用逗号分隔。"},
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
        default=5,
        metadata={"help": "训练的总轮数 (epochs)。"}
    )

    # --- 显存与批次大小相关参数 ---
    per_device_train_batch_size: int = field(
        default=1,  # 保持我们希望的训练批量大小
        metadata={"help": "每个GPU设备上的训练批次大小。"}
    )
    # 新增或修改下面这个参数，为验证设置一个极小的批量
    per_device_eval_batch_size: int = field(
        default=1,  # 明确设置验证的批量大小为1，解决OOM问题
        metadata={"help": "每个GPU设备上的验证批次大小。"}
    )
    
    gradient_accumulation_steps: int = field(
        default=4,  # 保持为1，避免之前的 no_sync 错误
        metadata={"help": "梯度累积的步数。"}
    )
    # DEEPSPEED修改 1: (可选但推荐) 开启梯度检查点进一步节省显存
    gradient_checkpointing: bool = field(
        default=True,
        metadata={"help": "是否启用梯度检查点来节省显存。"}
    )

    # --- 学习率与调度器相关参数 ---
    learning_rate: float = field(default=2e-5, metadata={"help": "初始学习率。"})
    weight_decay: float = field(default=0.01, metadata={"help": "权重衰减值。"})
    lr_scheduler_type: str = field(default="cosine", metadata={"help": "学习率调度器类型。"})
    warmup_ratio: float = field(default=0.05, metadata={"help": "学习率预热的比例。"})

    # --- 日志与保存、评估相关参数 ---
    logging_strategy: str = field(default="steps", metadata={"help": "日志记录策略。"})
    logging_steps: int = field(default=10, metadata={"help": "每隔 N 步记录一次训练日志。"})
    do_eval: bool = field(default=True)
    evaluation_strategy: str = field(default="steps", metadata={"help": "验证策略 ('steps' 或 'epoch')。"})
    eval_steps: int = field(default=100, metadata={"help": "每隔 N 步进行一次验证。"})
    save_strategy: str = field(default="steps", metadata={"help": "模型保存策略。"})
    save_steps: int = field(default=100, metadata={"help": "每隔 N 步保存一次检查点。"})
    save_total_limit: int = field(default=5, metadata={"help": "最多保存的检查点数量。"})

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
        # 明确禁用Trainer的自动报告功能，这样我们手动创建的SwanLabCallback才能生效
    report_to: str = field(
        default="none",
        metadata={"help": "设置为'none'以禁用自动报告，并使用手动传入的callbacks。"}
    )
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(default=4096, metadata={"help": "最大序列长度。"})
    seed: int = field(default=42, metadata={"help": "用于复现的随机种子。"})

# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


def jload(f, mode="r", jsonl=True):
    """加载JSON、JSONL或Arrow格式文件"""
    # 检查是否是arrow格式文件或目录
    if f.endswith('.arrow') or os.path.isdir(f):
        try:
            # 尝试作为HuggingFace Dataset加载
            dataset = load_from_disk(f)
            data_list = dataset.to_list()
            logging.warning(f"Successfully loaded {len(data_list)} examples from arrow dataset {f}")
            return data_list
        except Exception as e:
            logging.warning(f"Failed to load as HuggingFace dataset: {e}")
            try:
                # 如果失败，尝试用pyarrow直接读取
                import pyarrow as pa
                if f.endswith('.arrow'):
                    # 单个arrow文件
                    table = pa.ipc.open_file(f).read_all()
                else:
                    # arrow目录，尝试读取parquet文件
                    import pyarrow.parquet as pq
                    table = pq.read_table(f)
                data_list = table.to_pylist()
                logging.warning(f"Successfully loaded {len(data_list)} examples using pyarrow from {f}")
                return data_list
            except Exception as e2:
                logging.error(f"Failed to load arrow file {f}: {e2}")
                raise e2
    
    # 原有的JSON/JSONL处理逻辑
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

# VVVVVVVVVVVVVVVVVVVVVV  修改部分 2: 使数据集类接收列表而非文件路径 VVVVVVVVVVVVVVVVVVVVVV
class SupervisedDataset(Dataset):
    """监督微调数据集"""

    def __init__(self, list_data_dict: list, tokenizer):
        """
        修改后的初始化方法，适配新的数据格式
        Args:
            list_data_dict: 包含数据样本的列表
            tokenizer: 分词器
        """
        super(SupervisedDataset, self).__init__()
        logging.warning("Formatting inputs...")

        # 定义系统提示
        system_prompt = (
            "You are a helpful math assistant. Solve the problem step by step.\n"
            "At the end, output the final answer in the following format:\n"
            "**Answer:** \\boxed{your_final_numeric_answer}\n"
            "Do NOT include any text after the boxed answer."
        )
        # 在处理循环开始前添加调试
        logging.warning("=== 查看前几个样本的具体内容 ===")
        for i, example in enumerate(list_data_dict[:3]):
            messages = example.get("messages", [])
            if messages and len(messages) == 2:
                logging.warning(f"Sample {i} user: {messages[0]['content'][:100]}...")
                logging.warning(f"Sample {i} assistant: {messages[1]['content'][:100]}...")
        logging.warning("=== 调试结束 ===")

        input_ids_list = []
        labels_list = []
        skipped_count = 0
        
        for example in list_data_dict:
            # 修改：使用正确的字段名 "messages"
            messages = example.get("messages", [])
            
            # 验证数据格式
            if not isinstance(messages, list) or len(messages) != 2:
                logging.warning(f"跳过格式错误的样本: {messages}")
                skipped_count += 1
                continue
                
            if messages[0]["role"] != "user" or messages[1]["role"] != "assistant":
                logging.warning(f"跳过角色错误的样本: {messages}")
                skipped_count += 1
                continue
            
            user_content = messages[0]["content"]
            assistant_content = messages[1]["content"]
            
            # 跳过空内容的样本
            if not user_content.strip() or not assistant_content.strip():
                logging.warning(f"跳过空内容的样本")
                skipped_count += 1
                continue
            
            # 构建完整的对话
            full_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content}
            ]
            
            # 构建source部分（不包含assistant回答）
            source_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ]
            
            try:
                # 使用chat_template处理
                full_text = tokenizer.apply_chat_template(
                    full_messages, 
                    tokenize=False, 
                    add_generation_prompt=False
                )
                
                source_text = tokenizer.apply_chat_template(
                    source_messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                
                # 分别tokenize
                full_tokenized = tokenizer(
                    full_text,
                    truncation=True,
                    max_length=tokenizer.model_max_length,
                    padding=False,
                    return_tensors="pt"
                )
                
                source_tokenized = tokenizer(
                    source_text,
                    truncation=True,
                    max_length=tokenizer.model_max_length,
                    padding=False,
                    return_tensors="pt"
                )
                
                input_ids = full_tokenized["input_ids"][0]
                source_len = source_tokenized["input_ids"].shape[1]
                
                # 创建labels，前source_len个位置设为IGNORE_INDEX
                labels = input_ids.clone()
                labels[:source_len] = IGNORE_INDEX
                
                input_ids_list.append(input_ids)
                labels_list.append(labels)
                
            except Exception as e:
                logging.warning(f"处理样本时出错: {e}")
                skipped_count += 1
                continue

        logging.warning(f"处理完成：有效样本数 {len(input_ids_list)}，跳过样本数 {skipped_count}")
        
        if len(input_ids_list) == 0:
            raise ValueError("没有有效的训练样本！请检查数据格式。")
        
        self.input_ids = input_ids_list
        self.labels = labels_list

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


# VVVVVVVVVVVVVVVVVVVVVV  修改部分: 支持arrow格式数据加载 VVVVVVVVVVVVVVVVVVVVVV
def make_supervised_data_module(tokenizer, data_args) -> Dict:
    """
    一个更智能的数据模块创建函数，支持arrow格式。
    它会检查是否存在已处理好的数据集，如果没有，则创建并保存它们。
    """
    
    # 1. 定义处理后数据的固定存放路径
    processed_dir = "/root/processed-data" ## 新的路径
    train_path = os.path.join(processed_dir, "train_data.jsonl")
    eval_path = os.path.join(processed_dir, "validation_data.jsonl")
    test_path = os.path.join(processed_dir, "test_data.jsonl")

    # 2. 检查已处理的数据集是否存在
    if os.path.exists(train_path) and os.path.exists(eval_path) and os.path.exists(test_path):
        # 如果存在，直接加载
        logging.warning(f"发现已存在的已处理数据集，将从 '{processed_dir}' 目录直接加载...")
        train_list = jload(train_path)
        eval_list = jload(eval_path)
        logging.warning(f"成功加载 {len(train_list)} 条训练样本和 {len(eval_list)} 条验证样本。")

    else:
        # 如果不存在，则执行一次性处理流程
        logging.warning("未发现已处理的数据集，将执行一次性混洗、分割和保存...")
        
        # 确保输出目录存在
        os.makedirs(processed_dir, exist_ok=True)

        # a. 从原始路径加载所有数据集（支持arrow格式）
        logging.warning("从原始路径加载数据...")
        data_paths = data_args.data_path.split(',')
        full_data_list = []
        
        for path in data_paths:
            path = path.strip()
            if not path: 
                continue
                
            logging.warning(f"Loading data from {path}...")
            
            try:
                data_list = jload(path)  # jload函数已经支持arrow格式
                full_data_list.extend(data_list)
                logging.warning(f"Successfully loaded {len(data_list)} examples from {path}")
            except Exception as e:
                logging.error(f"Failed to load data from {path}: {e}")
                continue

        logging.warning(f"Totally loaded {len(full_data_list)} examples from all sources.")

        # === 新增：限制数据集大小用于测试 ===
        if len(full_data_list) > 20000:  # 如果超过1万条
            logging.warning(f"数据集较大({len(full_data_list)}条)，使用前20000条数据进行测试...")
            full_data_list = full_data_list[:20000]
        
        # 打印几个样本来验证数据格式
        if full_data_list:
            logging.warning("数据样本示例:")
            for i, sample in enumerate(full_data_list[:3]):  # 打印前3个样本
                logging.warning(f"Sample {i}: {sample}")
        else:
            logging.error("没有加载到任何数据！请检查数据路径和格式。")
            raise ValueError("没有加载到任何数据")

        # === 筛选数据 ===
        logging.warning("开始筛选数据...")
        filtered_data_list = []
        skipped_long = 0
        skipped_empty = 0
        model_max_length = 4096  # 从TrainingArguments中的默认值获取
        
        for idx, example in enumerate(full_data_list):
            if idx % 1000 == 0:
                logging.warning(f"筛选进度: {idx}/{len(full_data_list)}")
            
            # 检查是否有num_tokens字段
            if "num_tokens" in example:
                if example["num_tokens"] > model_max_length:
                    skipped_long += 1
                    continue
            
            # 检查messages格式
            messages = example.get("messages", [])
            if not isinstance(messages, list) or len(messages) != 2:
                skipped_empty += 1
                continue
                
            if (messages[0].get("role") != "user" or 
                messages[1].get("role") != "assistant"):
                skipped_empty += 1
                continue
            
            user_content = messages[0].get("content", "")
            assistant_content = messages[1].get("content", "")
            
            # 跳过空内容的样本
            if not user_content.strip() or not assistant_content.strip():
                skipped_empty += 1
                continue
            
            # 如果没有num_tokens字段，手动计算token长度进行筛选
            if "num_tokens" not in example:
                # 简单估算：用内容长度估算token数量（大概4个字符=1个token）
                estimated_tokens = (len(user_content) + len(assistant_content)) // 4
                if estimated_tokens > model_max_length:
                    skipped_long += 1
                    continue
            
            filtered_data_list.append(example)
        
        logging.warning(f"筛选完成: 保留 {len(filtered_data_list)} 个样本")
        logging.warning(f"跳过过长样本: {skipped_long} 个")
        logging.warning(f"跳过格式错误/空内容样本: {skipped_empty} 个")
        
        if len(filtered_data_list) == 0:
            logging.error("筛选后没有剩余数据！请检查筛选条件。")
            raise ValueError("筛选后没有剩余数据")

        # b. 随机混洗
        random.shuffle(filtered_data_list)

        # c. 分割数据集
        train_ratio, eval_ratio = 0.8, 0.1 # 80%训练, 10%验证, 剩下10%为测试
        train_end_idx = int(len(filtered_data_list) * train_ratio)
        eval_end_idx = int(len(filtered_data_list) * (train_ratio + eval_ratio))
        
        train_list = filtered_data_list[:train_end_idx]
        eval_list = filtered_data_list[train_end_idx:eval_end_idx]
        test_list = filtered_data_list[eval_end_idx:]

        # d. 将分割后的三个文件保存到磁盘
        logging.warning(f"正在保存已分割的数据集至 '{processed_dir}' 以备将来使用...")
        logging.warning(f"训练集: {len(train_list)} 样本")
        logging.warning(f"验证集: {len(eval_list)} 样本") 
        logging.warning(f"测试集: {len(test_list)} 样本")
        
        jdump(train_list, train_path)
        jdump(eval_list, eval_path)
        jdump(test_list, test_path)
        logging.warning("数据集保存完毕。")

    # 验证数据集不为空
    if len(train_list) == 0:
        logging.error("训练集为空！请检查数据格式。")
        raise ValueError("训练集为空")
    
    if len(eval_list) == 0:
        logging.warning("验证集为空！请检查数据格式。")
        raise ValueError("验证集为空")

    # 3. 后续步骤：为加载好的训练集和验证集创建 Dataset 对象
    logging.warning(f"最终数据分割: {len(train_list)} 训练样本, {len(eval_list)} 验证样本.")

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
    
    # === 禁用Wandb ===
    os.environ["WANDB_MODE"] = "disabled"
    os.environ["WANDB_DISABLED"] = "true"
    
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # --- 修改一：在所有操作前设置随机种子 ---
    transformers.set_seed(training_args.seed)

    # 确保输出目录存在
    os.makedirs(training_args.output_dir, exist_ok=True)

    # ... 模型和tokenizer加载代码保持不变 ...
    model = tf_module.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        trust_remote_code=True,
    )

    model.enable_input_require_grads()

    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
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

    # === 验证数据集检查 ===
    logging.warning(f"训练数据集大小: {len(data_module['train_dataset'])}")
    logging.warning(f"验证数据集大小: {len(data_module['eval_dataset'])}")

    # --- 修改SwanLab配置为英文项目名称 ---
    try:
        # 检查是否在分布式环境中
        if torch.distributed.is_initialized():
            # 只在rank 0进程初始化SwanLab
            if torch.distributed.get_rank() == 0:
                swanlab_call = SwanLabCallback(
                    project="Math-Reasoning-Finetuning",  # 改为英文项目名
                    experiment_name=f"Qwen7B-Math-{os.path.basename(training_args.output_dir)}",  # 英文实验名
                    config={
                        **asdict(data_args),
                        **asdict(model_args), 
                        **asdict(training_args),
                        **asdict(lora_config),
                        "base_model": "Qwen2.5-7B-Instruct",
                        "task_type": "math_reasoning_finetuning",
                        "dataset": "OpenR1_Math_Dataset",
                        "method": "LoRA",
                        "framework": "DeepSpeed",
                        # 中文描述可以放在这里
                        "description_zh": "使用LoRA方法对通义千问2.5-7B模型进行数学推理能力微调",
                        "task_zh": "数学推理微调",
                    },
                    description="Fine-tuning Qwen2.5-7B for math reasoning using LoRA method",
                    tags=["math-reasoning", "qwen", "lora", "deepspeed"],  # 英文标签
                )
                callbacks = [swanlab_call]
                logging.warning("SwanLab已为主进程初始化")
            else:
                callbacks = []
        else:
            # 单GPU训练
            swanlab_call = SwanLabCallback(
                project="Math-Reasoning-Finetuning",  # 改为英文项目名
                experiment_name=f"Qwen7B-Math-{os.path.basename(training_args.output_dir)}",  # 英文实验名
                config={
                    **asdict(data_args),
                    **asdict(model_args), 
                    **asdict(training_args),
                    **asdict(lora_config),
                    "base_model": "Qwen2.5-7B-Instruct",
                    "task_type": "math_reasoning_finetuning", 
                    "dataset": "OpenR1_Math_Dataset",
                    "method": "LoRA",
                    "framework": "DeepSpeed",
                    # 中文描述可以放在这里
                    "description_zh": "使用LoRA方法对通义千问2.5-7B模型进行数学推理能力微调",
                    "task_zh": "数学推理微调",
                },
                description="Fine-tuning Qwen2.5-7B for math reasoning using LoRA method",
                tags=["math-reasoning", "qwen", "lora", "deepspeed"],  # 英文标签
            )
            callbacks = [swanlab_call]
            logging.warning("SwanLab已为单GPU训练初始化")
            
    except Exception as e:
        logging.warning(f"SwanLab初始化失败: {e}")
        logging.warning("训练将继续进行，但不会记录到SwanLab")
        callbacks = []

    # --- 创建Trainer ---
    trainer = tf_module.Trainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        callbacks=callbacks,
        **data_module,
    )
    
    # === 验证trainer配置 ===
    logging.warning(f"Trainer验证配置:")
    logging.warning(f"  - evaluation_strategy: {trainer.args.evaluation_strategy}")
    logging.warning(f"  - eval_steps: {trainer.args.eval_steps}")
    logging.warning(f"  - do_eval: {trainer.args.do_eval}")
    
    # --- 检查点逻辑 ---
    last_checkpoint = None
    if os.path.exists(training_args.output_dir) and os.listdir(training_args.output_dir):
        last_checkpoint = transformers.trainer_utils.get_last_checkpoint(training_args.output_dir)
    
    if last_checkpoint:
        logging.warning(f"检测到检查点，将从 {last_checkpoint} 恢复训练...")
    else:
        logging.warning("未检测到检查点，将从头开始训练。")
    
    # 开始训练
    trainer.train(resume_from_checkpoint=last_checkpoint)

    # --- 保存最终模型 ---
    trainer.save_model(output_dir=training_args.output_dir)
    
    # 训练结束后记录最终状态
    if callbacks and len(callbacks) > 0:
        try:
            import swanlab
            # 这里可以用英文记录，中文放在描述里
            swanlab.log({
                "training_status": "completed",
                "final_epochs": training_args.num_train_epochs,
                "model_saved": True,
                "save_path": training_args.output_dir,
                "status_zh": "训练已完成",  # 中文状态
            })
            swanlab.finish()
            logging.warning("训练完成，SwanLab记录已结束")
        except Exception as e:
            logging.warning(f"SwanLab结束记录时出错: {e}")


if __name__ == "__main__":
    train()
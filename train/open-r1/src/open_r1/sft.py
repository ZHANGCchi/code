import logging
import os
import sys
import datasets
import transformers
import swanlab
from transformers import set_seed, TrainerCallback
from transformers.trainer_utils import get_last_checkpoint

from open_r1.configs import ScriptArguments, SFTConfig
from open_r1.utils import get_dataset, get_model, get_tokenizer
from open_r1.utils.callbacks import get_callbacks
from trl import ModelConfig, SFTTrainer, TrlParser, get_peft_config, setup_chat_format

logger = logging.getLogger(__name__)

# --- SwanLabCallback Class Definition (不变) ---
class SwanLabCallback(TrainerCallback):
    def __init__(self, **kwargs):
        super().__init__()
        self.is_initialized = False
        self.init_kwargs = kwargs

    def on_init_end(self, args, state, control, **kwargs):
        if state.is_world_process_zero and not self.is_initialized:
            swanlab.init(config=args.to_sanitized_dict(), **self.init_kwargs)
            self.is_initialized = True

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_world_process_zero and self.is_initialized:
            metrics = {k: v for k, v in logs.items() if isinstance(v, (int, float))}
            swanlab.log(metrics, step=state.global_step)

    def on_train_end(self, args, state, control, **kwargs):
        if state.is_world_process_zero and self.is_initialized:
            swanlab.finish()

# --- 数据集格式化函数 (硬编码版本) ---
def format_dataset_with_template(example, tokenizer):
    # 1. 创建消息列表
    messages = [
        {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
        {"role": "user", "content": example["user_content"]},
        {"role": "assistant", "content": example["qwen_response"]},
    ]
    # 2. 调用分词器的模板应用功能，生成最终文本，并存入 'text' 字段
    # tokenize=False 表示只生成字符串，而不是 token IDs
    example["text"] = tokenizer.apply_chat_template(messages, tokenize=False)
    return example

def main(script_args, training_args, model_args):
    set_seed(training_args.seed)

    # --- 日志设置 (不变) ---
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.info(f"Model parameters {model_args}")
    logger.info(f"Script parameters {script_args}")
    logger.info(f"Training parameters {training_args}")

    # --- 检查点 (不变) ---
    last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    # --- 加载数据集、分词器和模型 ---
    # dataset = get_dataset(script_args)
    from datasets import load_dataset # 确保导入
    
    logger.info(f"Loading dataset from local file: {training_args.train_file}")
    # SFTConfig (即 training_args) 中包含了 train_file 的路径
    # 我们直接用它来加载 JSON 数据集
    dataset = load_dataset("json", data_files={"train": training_args.train_file})

    tokenizer = get_tokenizer(model_args, training_args)
    model = get_model(model_args, training_args)

    # --- 应用聊天模板格式化 ---
    logger.info("Applying chat template to the dataset (hardcoded for 'user_content' and 'qwen_response')...")
    from functools import partial # 确保在文件顶部或此处导入
    # 使用 partial 将 tokenizer 固定为我们新函数的第一个参数
    formatting_func = partial(format_dataset_with_template, tokenizer=tokenizer)

    dataset = dataset.map(
        formatting_func,
        num_proc=training_args.dataset_num_proc
    )

    if tokenizer.chat_template is None:
        logger.info("No chat template provided, defaulting to ChatML.")
        model, tokenizer = setup_chat_format(model, tokenizer, format="chatml")

    # --- 初始化 SFT Trainer ---
    swanlab_callback = SwanLabCallback(
        project="OpenR1-Finetune-SwanLab",
        experiment_name=f"sft-{training_args.run_name}" if training_args.run_name else "sft-experiment"
    )
    all_callbacks = get_callbacks(training_args, model_args)
    all_callbacks.append(swanlab_callback)

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=(dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None),
        peft_config=get_peft_config(model_args),
        callbacks=all_callbacks
        # dataset_text_field="messages",
    )

    # --- 训练循环 (不变) ---
    logger.info("*** Train ***")
    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(dataset[script_args.dataset_train_split])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    # --- 保存模型 (不变) ---
    logger.info("*** Save model ***")
    trainer.model.generation_config.eos_token_id = tokenizer.eos_token_id
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    kwargs = {"dataset_name": script_args.dataset_name, "tags": ["open-r1"]}
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    # --- 评估 (不变) ---
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(dataset[script_args.dataset_test_split])
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # --- 推送至Hub (不变) ---
    if training_args.push_to_hub:
        logger.info("Pushing to hub...")
        trainer.push_to_hub(**kwargs)

if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
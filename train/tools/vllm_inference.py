#   使用vllm进行批量推理
#   支持基础模型和微调后的LoRA模型，通过--use_lora参数切换
#   输出结果包含每个问题的预测答案和token使用情况
import json
import torch
import argparse
import os
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
# --- 提示词和系统提示与原脚本保持一致 ---
PROMPT_DICT = {
    "prompt_input": """<|im_start|>system
{instruction}<|im_end|>
<|im_start|>user
{input}<|im_end|>
<|im_start|>assistant
""",
}
SYSTEM_PROMPT = (
 "Please reason step by step, and put your final answer within \\boxed{}."
)

def vllm_batch_inference(args, use_lora_adapter=False):
    """
    使用 vLLM 对模型进行批量推理。

    Args:
        args: 命令行参数。
        use_lora_adapter (bool): 是否使用 LoRA 适配器。
    """
    # --- 1. 设置模型标识符和模式信息 ---
    if use_lora_adapter:
        model_identifier = args.model_path # vLLM 从基础模型加载
        print(f"模式: 使用 vLLM + LoRA 适配器 ('{args.lora_path}') 进行推理")
        if not args.lora_path:
            raise ValueError("使用 --use_lora 时必须提供 --lora_path。")
    else:
        model_identifier = args.model_path
        print(f"模式: 使用 vLLM 基础模型 ('{args.model_path}') 进行推理")

    # --- 2. 使用 vLLM 加载模型 ---
    print("正在使用 vLLM 加载模型...")
    llm = LLM(
        model=model_identifier,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_lora=use_lora_adapter,
        max_loras=1,
        max_lora_rank=args.max_lora_rank, # <--- 在此添加参数
        trust_remote_code=True,
        # max_model_len=args.max_model_len
    )
    
    # 加载 Tokenizer 用于准备提示和获取特殊 token ID
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- 3. 准备推理参数 ---
    sampling_params = SamplingParams(
        temperature=0,  # 使用贪心解码，与原脚本的 do_sample=False 一致
        max_tokens=args.max_new_tokens,
        stop_token_ids=[tokenizer.eos_token_id]
    )
    
    # 准备 LoRA 请求 (如果需要)
    lora_request = None
    if use_lora_adapter:
        # lora_name: 适配器的唯一名称
        # lora_int_id: 适配器的唯一整数ID
        # lora_local_path: 适配器权重文件路径
        lora_request = LoRARequest(lora_name="adapter", lora_int_id=1, lora_local_path=args.lora_path)

    # --- 4. 加载并准备数据 ---
    test_data = []
    with open(args.test_file, 'r', encoding='utf-8') as f:
        for line in f:
            test_data.append(json.loads(line))
    
    # 一次性准备所有提示
    prompts = [
        PROMPT_DICT["prompt_input"].format(instruction=SYSTEM_PROMPT, input=item['problem'])
        for item in test_data
    ]
    
    # --- 5. 执行批量推理 ---
    print(f"开始对 {len(prompts)} 条样本进行推理 (vLLM 会自动处理批处理)...")
    # vLLM 一次性处理所有请求，内部自动调度
    outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)
    print("推理完成，正在处理输出...")

    # --- 6. 处理并保存结果 ---
    with open(args.output_file, 'w', encoding='utf-8') as f_out:
        # vLLM 的输出与输入顺序一致
        for i, request_output in enumerate(tqdm(outputs, desc="处理并保存结果")):
            original_item = test_data[i]
            generated_content = request_output.outputs[0].text.strip()
            
            # 使用 vLLM 返回的精确 token 数量
            prompt_tokens = len(request_output.prompt_token_ids)
            completion_tokens = len(request_output.outputs[0].token_ids)
            total_tokens = prompt_tokens + completion_tokens

            usage = {
                "completion_tokens": completion_tokens,
                "prompt_tokens": prompt_tokens,
                "total_tokens": total_tokens
            }
            
            result = {
                "custom_id": original_item.get("custom_id"),
                "question": original_item.get("question"),
                "usage": usage,
                "model": f"vllm_{'lora_' if use_lora_adapter else 'base_'}{os.path.basename(args.model_path)}",
                "message": {"role": "assistant", "content": generated_content}
            }
            
            f_out.write(json.dumps(result, ensure_ascii=False) + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="为基础模型或微调后的LoRA模型进行批量推理 (使用 vLLM)")
    # --- 模型和数据路径参数 ---
    parser.add_argument("--model_path", type=str, default="/root/autodl-tmp/openr1/open-r1/merged-model", help="基础预训练模型的路径")
    parser.add_argument("--lora_path", type=str, default="/root/autodl-tmp/openr1/open-r1/data/Qwen2.5-7B-GRPO-LoRA/checkpoint-600", help="训练好的LoRA适配器目录的路径")
    parser.add_argument("--test_file", type=str, default="/root/autodl-tmp/dataset/math_dataset_jsonl/test.jsonl", help="输入的测试集文件路径 (JSONL格式)")
    parser.add_argument("--output_file", type=str, default="predictions_vllm_qwen-max_3parts_Qwen2.5-7B-GRPO-LoRA_math500_4096.jsonl", help="用于保存预测结果的文件路径的基本名称")
    
    # --- 推理控制参数 ---
    parser.add_argument("--max_new_tokens", type=int, default=4096, help="每个回复生成的最大新token数量")
    # parser.add_argument("--max-model-len", type=int, default=16384, help="每个回复生成的最大新token数量")
    parser.add_argument("--use_lora", action="store_true", help="加载LoRA适配器进行推理。如果未提供此项，则使用基础模型。")
    
    # --- vLLM 特定参数 ---
    parser.add_argument("--tensor-parallel-size", "-tp", type=int, default=1, help="vLLM的张量并行大小")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8, help="vLLM可使用的GPU内存比例")

    parser.add_argument("--max-lora-rank", type=int, default=32, help="允许的最大LoRA rank。如果你的LoRA rank大于16，请设置此项。")

    # --- 移除旧的 batch_size 参数，因为 vLLM 会自动处理 ---
    # parser.add_argument("--batch_size", type=int, default=32, help="用于推理的批量大小")

    args = parser.parse_args()

    # 根据是否使用 LoRA 调整输出文件名并执行推理
    if args.use_lora:
        base, ext = os.path.splitext(args.output_file)
        args.output_file = f"{base}_lora{ext}"
        vllm_batch_inference(args, use_lora_adapter=True)
    else:
        base, ext = os.path.splitext(args.output_file)
        args.output_file = f"{base}_base{ext}"
        vllm_batch_inference(args, use_lora_adapter=False)

    print(f"\n推理全部完成。结果已保存至 {args.output_file}")

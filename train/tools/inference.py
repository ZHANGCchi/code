import json
import torch
import argparse
import os
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# --- 提示词和系统提示保持不变 ---
PROMPT_DICT = {
    "prompt_input": """<|im_start|>system
{instruction}<|im_end|>
<|im_start|>user
{input}<|im_end|>
<|im_start|>assistant
""",
}
SYSTEM_PROMPT = (
    "You are a helpful math assistant. Solve the problem step by step.\n"
    "At the end, output the final answer in the following format:\n"
    "**Answer:** \\boxed{your_final_numeric_answer}\n"
    "Do NOT include any text after the boxed answer."
)

def batch_inference(args, use_lora_adapter=False):
    # --- 模型加载逻辑与上一版相同 ---
    if use_lora_adapter:
        model_identifier = args.lora_path
        print(f"模式: 使用 LoRA 适配器 ('{args.lora_path}') 进行推理")
    else:
        model_identifier = args.model_path
        print(f"模式: 使用基础模型 ('{args.model_path}') 进行推理")
    print("正在加载基础模型...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    if use_lora_adapter:
        if not args.lora_path: raise ValueError("使用 --use_lora 时必须提供 --lora_path。")
        print(f"正在从 {args.lora_path} 加载LoRA适配器...")
        model = PeftModel.from_pretrained(model, args.lora_path)
        print("正在合并LoRA权重以提升速度...")
        model = model.merge_and_unload()
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True, padding_side='left'
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    test_data = []
    with open(args.test_file, 'r', encoding='utf-8') as f:
        for line in f:
            test_data.append(json.loads(line))
    
    print(f"开始对 {len(test_data)} 条样本进行推理...")
    with open(args.output_file, 'w', encoding='utf-8') as f_out:
        for i in tqdm(range(0, len(test_data), args.batch_size), desc="批量推理中"):
            batch_items = test_data[i:i + args.batch_size]
            questions = [item['question'] for item in batch_items]
            prompts = [PROMPT_DICT["prompt_input"].format(instruction=SYSTEM_PROMPT, input=q) for q in questions]
            
            inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=False).to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
                    eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id
                )
            
            # --- vvvvvv 最终的 Token 计算逻辑修正 vvvvvv ---
            # 解码所有生成文本和提示文本，用于后续处理
            decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            decoded_prompts = tokenizer.batch_decode(inputs.input_ids, skip_special_tokens=True)

            for j, item in enumerate(batch_items):
                # 1. 精确计算 prompt_tokens
                # 使用 attention_mask 对1求和，得到真实、未被填充的prompt长度
                actual_prompt_len = int(inputs.attention_mask[j].sum())
                
                # 2. 精确计算 completion_tokens
                # 从完整输出中移除prompt部分，得到纯粹的生成内容
                generated_content = decoded_outputs[j][len(decoded_prompts[j]):].strip()
                
                # 对纯粹的生成内容重新编码，以获得准确的token数
                completion_tokens = len(tokenizer.encode(generated_content, add_special_tokens=False))

                # 3. 计算总token数
                total_tokens = actual_prompt_len + completion_tokens

                # 4. 构建包含正确token计数的usage字典
                usage = {
                    "completion_tokens": completion_tokens,
                    "prompt_tokens": actual_prompt_len, # 使用修正后的真实长度
                    "total_tokens": total_tokens
                }
                
                # 5. 构建最终结果
                result = {
                    "custom_id": item.get("custom_id"),
                    "question": item.get("question"),
                    "usage": usage, # 使用包含精确计数的usage字典
                    "model": model_identifier,
                    "message": {"role": "assistant", "content": generated_content}
                }
                # --- ^^^^^^ Token 计算逻辑修正完毕 ^^^^^^ ---
                
                f_out.write(json.dumps(result, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="为基础模型或微调后的LoRA模型进行批量推理")
    parser.add_argument("--model_path", type=str, default="./qwen/Qwen2___5-7B-Instruct", help="基础预训练模型的路径 (请务必确保路径正确!)")
    parser.add_argument("--lora_path", type=str, default="./temp/results_qwen_finetuned/checkpoint-1405", help="训练好的LoRA适配器目录的路径")
    parser.add_argument("--test_file", type=str, default="./processed_data/test_data.jsonl", help="输入的测试集文件路径 (JSONL格式)")
    parser.add_argument("--output_file", type=str, default="predictions_long.jsonl", help="用于保存预测结果的文件路径的基本名称")
    parser.add_argument("--batch_size", type=int, default=32, help="用于推理的批量大小")
    parser.add_argument("--max_new_tokens", type=int, default=8192, help="每个回复生成的最大新token数量")
    parser.add_argument("--use_lora", action="store_true", help="加载LoRA适配器进行推理。如果未提供此项，则使用基础模型。")
    
    args = parser.parse_args()

    if args.use_lora:
        base, ext = os.path.splitext(args.output_file)
        args.output_file = f"{base}_lora{ext}"
        batch_inference(args, use_lora_adapter=True)
    else:
        base, ext = os.path.splitext(args.output_file)
        args.output_file = f"{base}_base{ext}"
        batch_inference(args, use_lora_adapter=False)

    print(f"\n推理完成。结果已保存至 {args.output_file}")
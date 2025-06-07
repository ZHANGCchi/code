from vllm import LLM, SamplingParams
from datasets import load_dataset
import re
import csv
from tqdm import tqdm
import logging

# ===== 日志配置 =====
logging.basicConfig(
    filename="deepseek_infer_log.txt",
    filemode="a",
    format="%(asctime)s - 样本 %(message)s",
    level=logging.INFO
)

# ===== 模型路径 =====
model_path = "/root/autodl-tmp/models/deepseek-r1-distill-qwen-7b/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"  # 替换为你本地模型路径
llm = LLM(model=model_path, dtype="float16")

# ===== 推理配置 =====
params = SamplingParams(
    temperature=0.7,
    max_tokens=4096,
    n=10,
    stop=["</s>"]
)

# ===== 数据集加载 =====
dataset = load_dataset("json", data_files="/root/gsm8k_train.jsonl", split="train")
dataset = dataset.select(range(1000))  # 测试用，实际推理全量可去掉 select
print(f"✅ 加载 {len(dataset)} 条样本")

# ===== 工具函数 =====
def extract_answer(text):
    match = re.search(r"\\boxed\{(-?\d+(?:\.\d+)?)\}", text)
    return match.group(1) if match else None

def safe_float(s):
    try:
        return round(float(s), 5)
    except:
        return None

def build_prompt(q):
    return (
        "<|im_start|>system\n"
        "You are a helpful math assistant. Solve the problem step by step. "
        "At the end, output the final answer in the following format:\n"
        "**Answer:** \\boxed{your_final_numeric_answer}\n"
        "Do NOT include any text after the boxed answer.\n"
        "<|im_end|>\n"
        f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"
    )

# ===== 推理主循环 =====
batch_size = 64
csv_path = "deepseek_pass10_vllm.csv"

with open(csv_path, "w", newline='', encoding="utf-8") as f_csv:
    writer = csv.DictWriter(f_csv, fieldnames=["question", "ground_truth", "pass@10", "predictions"])
    writer.writeheader()

    for start in tqdm(range(0, len(dataset), batch_size)):
        batch = dataset.select(range(start, min(start + batch_size, len(dataset))))
        questions = [ex["question"] for ex in batch]
        prompts = [build_prompt(q) for q in questions]
        gt_vals = [
            safe_float(re.sub(r"[^\d\.\-]+", "", ex["answer"].strip().split("####")[-1].strip()))
            for ex in batch
        ]

        outputs = llm.generate(prompts, sampling_params=params)

        for i, output in enumerate(outputs):
            completions = [cand.text for cand in output.outputs]
            predictions = [extract_answer(c) for c in completions]
            pass_rate = sum([safe_float(p) == gt_vals[i] for p in predictions]) / 10

            # ✅ 写入日志（包含完整上下文）
            for j, c in enumerate(completions):
                full_text = prompts[i] + c.strip()
                logging.info("%d 模型输出：%s", start + i, full_text)

            # ✅ 写入 CSV
            writer.writerow({
                "question": questions[i],
                "ground_truth": gt_vals[i],
                "pass@10": pass_rate,
                "predictions": predictions
            })

print(f"✅ 推理完成，结果保存到 {csv_path}，日志写入 deepseek_infer_log.txt")

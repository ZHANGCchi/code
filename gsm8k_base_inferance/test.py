from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm
import torch
import re
import logging
import csv

# ===== 日志设置 =====
logging.basicConfig(
    filename="infer_log.txt",
    filemode="a",
    format="%(asctime)s - 样本 %(message)s",
    level=logging.INFO
)

# ===== 模型路径 =====
model_id = "/root/models/qwen2.5/Qwen/Qwen2___5-7B-Instruct"  # 可替换为你本地的路径

# ===== 模型加载 =====
tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    trust_remote_code=True,
    padding_side="left"
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map={"": 0},
    torch_dtype=torch.float16,
    trust_remote_code=True
)
model.eval()

# ===== 加载 GSM8K 数据集 =====
dataset = load_dataset("json", data_files="/root/gsm8k_train.jsonl", split="train")
# dataset = dataset.select(range(17))  # 测试用，实际推理全量可去掉 select
print(f"✅ 数据集加载完成，共 {len(dataset)} 条数据。")

# ===== CSV 输出文件初始化 =====
with open("gsm8k_pass10.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["question", "ground_truth", "pass@10", "predictions"])
    writer.writeheader()

# ===== Prompt 构造函数 =====
def build_messages(question):
    return [
        {"role": "system", "content": "You are a well-trained mathematical problem-solving assistant. For every question posed by the user, you need to internally complete all reasoning steps, but do not display these steps. Only output the final answer, and format the answer using LaTeX syntax in \\boxed{}. Except for the content inside \\boxed{}, you are prohibited from outputting any other text, punctuation, or explanatory remarks."},
        {"role": "user", "content": question}
    ]

# ===== 答案提取与处理 =====
def extract_answer(output):
    match = re.search(r"\\boxed\{(-?\d+(?:\.\d+)?)\}", output)
    return match.group(1) if match else None

def safe_float(s):
    try:
        return round(float(s), 5)
    except:
        return None

# ===== 批量推理主循环 =====
batch_size = 96

for start in tqdm(range(0, len(dataset), batch_size)):
    batch = dataset.select(range(start, min(start + batch_size, len(dataset))))
    questions = [ex["question"] for ex in batch]
    gt_vals = [
        safe_float(re.sub(r"[^\d\.\-]+", "", ex["answer"].strip().split("####")[-1].strip()))
        for ex in batch
    ]

    messages_batch = [build_messages(q) for q in questions]
    texts = [
        tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        for msg in messages_batch
    ]
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    batch_outputs = [[] for _ in range(len(batch))]

    for _ in range(10):
        out = model.generate(**inputs, max_new_tokens=512, temperature=0.8, do_sample=True)
        decoded = tokenizer.batch_decode(out, skip_special_tokens=True)
        for i, output in enumerate(decoded):
            pred = extract_answer(output)
            batch_outputs[i].append(pred)
            logging.info("%d 模型输出：%s", start + i, output)

    for i in range(len(batch)):
        pass_rate = sum([safe_float(p) == gt_vals[i] for p in batch_outputs[i]]) / 10
        print(f"\n🎯 问题: {questions[i][:60].strip()}...")
        print(f"✅ GT: {gt_vals[i]}, pass@10 = {pass_rate:.2f}")
        print(f"📌 预测: {batch_outputs[i]}")

        with open("gsm8k_pass10.csv", "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["question", "ground_truth", "pass@10", "predictions"])
            writer.writerow({
                "question": questions[i],
                "ground_truth": gt_vals[i],
                "pass@10": pass_rate,
                "predictions": batch_outputs[i]
            })

print("✅ 推理完成，结果保存在 gsm8k_pass10.csv")

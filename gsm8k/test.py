# from transformers import AutoTokenizer, AutoModelForCausalLM
# from datasets import load_dataset
# from tqdm import tqdm
# import torch
# import pandas as pd
# import re
# import logging

# # 设置日志格式和输出文件
# logging.basicConfig(
#     filename="infer_log.txt",       # 日志保存的文件
#     filemode="a",                   # 追加写入
#     format="%(asctime)s - %(message)s",
#     level=logging.INFO              # 日志级别：INFO 以上的都会被记录
# )

# # 加载模型和 tokenizer（本地路径）
# model_id = "/root/models/qwen2.5/Qwen/Qwen2___5-Math-7B-Instruct"
# tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
# model = AutoModelForCausalLM.from_pretrained(
#     model_id,
#     device_map="auto",
#     torch_dtype=torch.float16,
#     trust_remote_code=True
# )
# model.eval()

# # 加载 GSM8K 数据
# dataset = load_dataset("json", data_files="/root/gsm8k_train.jsonl", split="train")
# print(f"✅ 数据集加载完成，共 {len(dataset)} 条数据。")
# dataset = dataset.select(range(len(dataset)))  # 可改成全部：.select(range(len(dataset)))

# # Chat 模板提示
# def build_messages(question):
#     return [
#         {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
#         {"role": "user", "content": f"{question}"}
#     ]

# # 提取 \boxed{...} 中的浮点答案
# def extract_answer(output):
#     match = re.search(r"\\boxed\{(-?\d+(?:\.\d+)?)\}", output)
#     return match.group(1) if match else None

# # 安全浮点转换 + 四舍五入
# def safe_float(s):
#     try:
#         return round(float(s), 5)
#     except:
#         return None

# results = []

# for example in tqdm(dataset, total=len(dataset)):
#     question = example["question"]
#     gt_answer_raw = example["answer"].strip().split("####")[-1].strip()
#     gt_val = safe_float(re.sub(r"[^\d\.\-]+", "", gt_answer_raw))  # 只提取数字部分

#     messages = build_messages(question)
#     text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
#     inputs = tokenizer([text], return_tensors="pt").to(model.device)

#     outputs = []
#     for _ in range(10):
#         out = model.generate(**inputs, max_new_tokens=512, temperature=0.8, do_sample=True)
#         decoded = tokenizer.decode(out[0], skip_special_tokens=True)
#         pred = extract_answer(decoded)
#         outputs.append(pred)
#         logging.info("模型输出：%s", text)

#     # 计算 pass@10
#     pass_rate = sum([safe_float(p) == gt_val for p in outputs]) / 10

#     # 输出
#     print(f"\n🎯 问题: {question[:60].strip()}...")
#     print(f"✅ GT: {gt_val}, pass@10 = {pass_rate:.2f}")
#     print(f"📌 预测: {outputs}")

#     results.append({
#         "question": question,
#         "ground_truth": gt_val,
#         "pass@10": pass_rate,
#         "predictions": outputs
#     })

# # 保存为 CSV
# df = pd.DataFrame(results)
# df.to_csv("gsm8k_pass10.csv", index=False)
# print("✅ 推理完成，结果保存在 gsm8k_pass10.csv")

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
    format="%(asctime)s - %(message)s",
    level=logging.INFO
)

# ===== 只用 GPU，强制全模型在 0 号卡 =====
model_id = "/root/models/qwen2.5/Qwen/Qwen2___5-Math-7B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    trust_remote_code=True,
    padding_side="left"
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map={"": 0},  # 强制全卡，避免自动分层和 offload
    torch_dtype=torch.float16,
    trust_remote_code=True,
)
model.eval()

# ===== 数据集加载 =====
dataset = load_dataset("json", data_files="/root/gsm8k_train.jsonl", split="train")
# dataset = dataset.select(range(17))  # 测试用，实际推理全量可去掉 select

print(f"✅ 数据集加载完成，共 {len(dataset)} 条数据。")

batch_size = 64

with open("gsm8k_pass10.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["question", "ground_truth", "pass@10", "predictions"])
    writer.writeheader()

# ===== Prompt 构建与答案提取 =====
def build_messages(question):
    return [
        {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
        {"role": "user", "content": f"{question}"}
    ]

def extract_answer(output):
    match = re.search(r"\\boxed\{(-?\d+(?:\.\d+)?)\}", output)
    return match.group(1) if match else None

def safe_float(s):
    try:
        return round(float(s), 5)
    except:
        return None

# ===== 主循环 =====
for start in tqdm(range(0, len(dataset), batch_size)):
    batch = dataset.select(range(start, min(start + batch_size, len(dataset))))
    questions = [ex["question"] for ex in batch]
    gt_vals = [safe_float(re.sub(r"[^\d\.\-]+", "", ex["answer"].strip().split("####")[-1].strip())) for ex in batch]

    messages_batch = [build_messages(q) for q in questions]
    texts = [tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages_batch]
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}  # 全量显式to GPU

    batch_outputs = [[] for _ in range(len(batch))]

    for _ in range(10):
        out = model.generate(**inputs, max_new_tokens=512, temperature=0.8, do_sample=True)
        decoded = tokenizer.batch_decode(out, skip_special_tokens=True)
        for i, output in enumerate(decoded):
            pred = extract_answer(output)
            batch_outputs[i].append(pred)
            logging.info("样本 %d 模型输出：%s", start + i, output)

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

print("✅ 全部推理完成，增量写入已保存到 gsm8k_pass10.csv")




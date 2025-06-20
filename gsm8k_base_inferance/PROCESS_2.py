# import pandas as pd
# import re

# # 路径：请修改为你的实际路径
# index_csv_path = "filtered_pass@10_equals_1.csv"
# log_file_path = "infer_log.txt"
# output_csv_path = "filtered_qa_split.csv"

# # 加载需要保留的样本索引及问题
# df = pd.read_csv(index_csv_path)
# valid_indices = set(df["gsm8k_index"].tolist())

# # 正则匹配每个日志的开始
# pattern = re.compile(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+ - 样本 (\d+) 模型输出：")

# # 读取日志
# with open(log_file_path, "r", encoding="utf-8") as f:
#     lines = f.readlines()

# samples = {}
# current_idx = None
# current_content = []

# for line in lines:
#     match = pattern.match(line)
#     if match:
#         if current_idx is not None and current_idx in valid_indices:
#             full_text = "\n".join(current_content).strip()
#             if "assistant" in full_text:
#                 q_part, a_part = full_text.split("assistant", 1)
#                 samples[current_idx] = {
#                     "question": q_part.strip(),
#                     "answer": a_part.strip()
#                 }
#         current_idx = int(match.group(1))
#         current_content = []
#     else:
#         if current_idx is not None:
#             current_content.append(line.strip())

# # 保存最后一段（若存在）
# if current_idx is not None and current_idx in valid_indices:
#     full_text = "\n".join(current_content).strip()
#     if "assistant" in full_text:
#         q_part, a_part = full_text.split("assistant", 1)
#         samples[current_idx] = {
#             "question": q_part.strip(),
#             "answer": a_part.strip()
#         }

# # 构建 DataFrame 并导出
# output_rows = []
# for idx in samples:
#     output_rows.append({
#         "gsm8k_index": idx,
#         "question": samples[idx]["question"],
#         "answer": samples[idx]["answer"]
#     })

# df_out = pd.DataFrame(output_rows)
# df_out.to_csv(output_csv_path, index=False)
# print(f"✅ 已保存为 {output_csv_path}")
import pandas as pd
import re

# ====== 文件路径（请根据实际修改）======
index_csv_path = "filtered_pass@10_equals_1.csv"
log_file_path = "infer_log.txt"
output_csv_path = "multi_answers_per_sample.csv"

# ====== 加载需要保留的样本 index ======
df = pd.read_csv(index_csv_path)
valid_indices = set(df["gsm8k_index"].tolist())

# 建立索引到question的映射（可选）
question_map = dict(zip(df["gsm8k_index"], df["question"]))

# ====== 正则提取样本起始 ======
pattern = re.compile(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+ - 样本 (\d+) 模型输出：")

samples = []
current_idx = None
current_content = []

with open(log_file_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

# 遍历每一行，分块处理每个样本回答
for line in lines:
    match = pattern.match(line)
    if match:
        # 若上一段合法，存入样本列表
        if current_idx is not None and current_idx in valid_indices:
            full_text = "\n".join(current_content).strip()
            if "assistant" in full_text:
                q_part, a_part = full_text.split("assistant", 1)
                samples.append({
                    "gsm8k_index": current_idx,
                    "question": q_part.strip(),
                    "answer": a_part.strip()
                })
        # 更新新样本段落
        current_idx = int(match.group(1))
        current_content = []
    else:
        if current_idx is not None:
            current_content.append(line.strip())

# 处理最后一段
if current_idx is not None and current_idx in valid_indices:
    full_text = "\n".join(current_content).strip()
    if "assistant" in full_text:
        q_part, a_part = full_text.split("assistant", 1)
        samples.append({
            "gsm8k_index": current_idx,
            "question": q_part.strip(),
            "answer": a_part.strip()
        })

# ====== 输出为 DataFrame + 保存 CSV ======
df_out = pd.DataFrame(samples)
df_out.to_csv(output_csv_path, index=False)
print(f"✅ 已成功保存：{output_csv_path}")

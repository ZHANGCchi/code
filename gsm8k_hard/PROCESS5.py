import pandas as pd
import json

# 读取两个CSV文件
gsm8k_df = pd.read_csv('h:/CoT剪枝/code/gsm8k/filtered_pass@10_equals_1.csv')
gsm8k_7b_df = pd.read_csv('h:/CoT剪枝/code/gsm8k_7b/filtered_pass@10_equals_1.csv')

# 提取两个文件中的gsm8k_index列
gsm8k_indices = set(gsm8k_df['gsm8k_index'])
gsm8k_7b_indices = set(gsm8k_7b_df['gsm8k_index'])

# 找出两个文件中gsm8k_index的并集
union_indices = gsm8k_indices.union(gsm8k_7b_indices)

# 读取JSONL文件
jsonl_data = []
with open('h:/CoT剪枝/code/gsm8k/gsm8k_train.jsonl', 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        jsonl_data.append((i, json.loads(line)))

# 从JSONL数据中过滤掉并集中的gsm8k_index
filtered_jsonl = [(idx, item) for idx, item in jsonl_data if idx not in union_indices]

# 将过滤后的数据保存到新的JSONL文件中
with open('h:/CoT剪枝/code/gsm8k_hard/filtered_gsm8k_train.jsonl', 'w', encoding='utf-8') as f:
    for _, item in filtered_jsonl:
        f.write(json.dumps(item) + '\n')

# 打印统计信息
print(f"原始JSONL文件中的样本数: {len(jsonl_data)}")
print(f"两个CSV文件中的并集样本数: {len(union_indices)}")
print(f"过滤后的JSONL样本数: {len(filtered_jsonl)}")
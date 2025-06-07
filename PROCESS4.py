# base与short的交集
import pandas as pd

# 读取两个CSV文件
gsm8k_df = pd.read_csv('h:/CoT剪枝/code/gsm8k/filtered_pass@10_equals_1.csv')
gsm8k_7b_df = pd.read_csv('h:/CoT剪枝/code/gsm8k_7b/filtered_pass@10_equals_1.csv')

# 提取两个文件中的gsm8k_index列
gsm8k_indices = set(gsm8k_df['gsm8k_index'])
gsm8k_7b_indices = set(gsm8k_7b_df['gsm8k_index'])

# 找出两个文件中gsm8k_index的交集
common_indices = gsm8k_indices.intersection(gsm8k_7b_indices)

# 获取交集中的样本（从gsm8k文件中提取）
intersection_df = gsm8k_df[gsm8k_df['gsm8k_index'].isin(common_indices)]

# 将交集数据保存到新的CSV文件中
intersection_df.to_csv('h:/CoT剪枝/code/easy_pass@10_equals_1.csv', index=False)

# 打印统计信息
print(f"原始gsm8k文件中的样本数: {len(gsm8k_df)}")
print(f"gsm8k_7b文件中的样本数: {len(gsm8k_7b_df)}")
print(f"两个文件中重复的样本数: {len(common_indices)}")
print(f"交集中的样本数: {len(intersection_df)}")
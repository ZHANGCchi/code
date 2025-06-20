## This script filters a CSV file to keep only rows where the column "pass@10" equals 1
import pandas as pd

# 读取原始CSV
df = pd.read_csv("gsm8k_pass10.csv")  # 请替换为你的实际文件路径

# 筛选出 pass@10 == 1 的行
filtered_df = df[df["pass@10"] == 1].copy()

# 加入在原始文件中的索引（对应 GSM8K 的 index）
filtered_df.reset_index(inplace=True)
filtered_df.rename(columns={"index": "gsm8k_index"}, inplace=True)

# 保存为新的CSV文件
filtered_df.to_csv("filtered_pass@10_equals_1.csv", index=False)

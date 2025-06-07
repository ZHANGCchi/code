import pandas as pd
import json

# 输入输出文件路径
input_csv = 'multi_answers_per_sample_for_easy_train.csv'  # 请替换为您的CSV文件路径
output_jsonl = 'gsm8k_easy_data.jsonl'  # 输出的JSONL文件路径

# 读取CSV文件
df = pd.read_csv(input_csv)

# 转换为JSONL格式并处理换行符
with open(output_jsonl, 'w', encoding='utf-8') as f:
    for _, row in df.iterrows():
        # 创建JSON对象
        json_obj = {
            "gsm8k_index": row['gsm8k_index'],
            "question": row['question'],
            "answer": str(row['answer']) # dumps会自动处理字符串中的换行符
        }
        
        # 写入JSONL文件
        f.write(json.dumps(json_obj, ensure_ascii=False) + '\n')

print(f"转换完成！CSV文件已转换为JSONL格式并保存到 {output_jsonl}")
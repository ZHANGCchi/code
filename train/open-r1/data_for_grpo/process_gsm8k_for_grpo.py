import csv
import json
import argparse
from tqdm import tqdm

def process_gsm8k_for_grpo(input_csv, output_jsonl):
    """
    将GSM8K数据从CSV格式转换为JSONL格式，
    只保留问题和参考答案字段，并重命名为'problem'和'reference'
    
    Args:
        input_csv (str): 输入CSV文件路径
        output_jsonl (str): 输出JSONL文件路径
    """
    # 读取CSV文件
    data = []
    with open(input_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
    
    print(f"从CSV文件中读取了 {len(data)} 条记录")
    
    # 转换为所需格式
    processed_data = []
    for item in tqdm(data, desc="处理数据"):
        processed_item = {
            "problem": item["question"],
            "reference": item["ground_truth"]
        }
        processed_data.append(processed_item)
    
    # 写入JSONL文件
    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for item in processed_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"成功将数据转换为JSONL格式并保存至 {output_jsonl}")
    print(f"共处理 {len(processed_data)} 条记录")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='将GSM8K数据从CSV转换为GRPO所需的JSONL格式')
    parser.add_argument('--input', '-i', type=str, 
                        default="gsm8k_easy_train_pass@10_equals_1.csv", 
                        help='输入CSV文件路径')
    parser.add_argument('--output', '-o', type=str, 
                        default="gsm8k_easy_train_for_grpo.jsonl", 
                        help='输出JSONL文件路径')
    
    args = parser.parse_args()
    
    process_gsm8k_for_grpo(args.input, args.output)
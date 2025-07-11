import json
import re
import argparse
import glob
from tqdm import tqdm
import os

def extract_boxed_content(text):
    """从文本中提取\boxed{}中的内容"""
    if not text:
        return None
    
    # 匹配\boxed{...}格式
    match = re.search(r'\\boxed\{([^{}]+(?:\{[^{}]*\}[^{}]*)*)\}', text)
    if match:
        return match.group(1).strip()
    return None

def process_math_jsonl(input_files, output_file):
    """
    将原始JSONL格式转换为只包含problem和reference的JSONL格式
    
    Args:
        input_files (list): 输入JSONL文件路径列表
        output_file (str): 输出JSONL文件路径
    """
    # 计数器
    total_count = 0
    success_count = 0
    
    # 处理文件
    with open(output_file, 'w', encoding='utf-8') as fout:
        # 处理每个输入文件
        for input_file in input_files:
            print(f"处理文件: {input_file}")
            try:
                with open(input_file, 'r', encoding='utf-8') as fin:
                    lines = fin.readlines()
                    
                    for line_num, line in enumerate(tqdm(lines, desc=f"处理 {os.path.basename(input_file)}")):
                        try:
                            total_count += 1
                            data = json.loads(line.strip())
                            
                            # 提取problem
                            problem = data.get("question", "")
                            
                            # 提取reference (从boxed内容)
                            content = data.get("message", {}).get("content", "")
                            reference = extract_boxed_content(content)
                            
                            if not problem or not reference:
                                print(f"警告: {input_file} 行 {line_num+1} 缺少问题或答案，已跳过")
                                continue
                            
                            # 创建新的数据格式
                            new_data = {
                                "problem": problem,
                                "solution": reference
                            }
                            
                            # 写入新文件
                            fout.write(json.dumps(new_data, ensure_ascii=False) + '\n')
                            success_count += 1
                            
                        except json.JSONDecodeError:
                            print(f"错误: {input_file} 行 {line_num+1} JSON格式错误，已跳过")
                        except Exception as e:
                            print(f"错误: 处理 {input_file} 行 {line_num+1} 时出现异常: {e}")
            except Exception as e:
                print(f"错误: 无法处理文件 {input_file}: {e}")
    
    # 打印统计信息
    print(f"处理完成！")
    print(f"总行数: {total_count}")
    print(f"成功转换: {success_count}")
    print(f"未能转换: {total_count - success_count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='将原始JSONL转换为仅包含problem和reference的JSONL格式')
    parser.add_argument('--input', '-i', type=str, default="H:/CoT剪枝/code/train/processed_data/*.jsonl", 
                        help='输入JSONL文件路径，支持通配符，如 "folder/*.jsonl"')
    parser.add_argument('--output', '-o', type=str, default="hard_data_for_grpo.jsonl", 
                        help='输出JSONL文件路径')
    
    args = parser.parse_args()
    
    # 使用glob获取所有匹配的文件
    input_files = glob.glob(args.input)
    
    if not input_files:
        print(f"错误: 未找到匹配的文件: {args.input}")
    else:
        print(f"找到 {len(input_files)} 个匹配的文件")
        process_math_jsonl(input_files, args.output)
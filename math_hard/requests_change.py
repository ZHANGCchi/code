import json
import os

# 输入和输出文件路径
input_file = 'math_hard.jsonl'  # 修改输入文件
output_file = 'requests_format.jsonl'

# 系统提示信息
system_prompt = "You are a helpful math assistant. Solve the problem step by step.\nAt the end, output the final answer in the following format:\n**Answer:** \\boxed{your_final_numeric_answer}\nDo NOT include any text after the boxed answer.\n"

# 读取JSONL文件（每行一个JSON对象）并转换格式
data_list = []
with open(input_file, 'r', encoding='utf-8') as f:
    for line in f:
        data_list.append(json.loads(line.strip()))

with open(output_file, 'w', encoding='utf-8') as f_out:
    for i, item in enumerate(data_list):
        try:
            # 检查项目是否包含instruction字段
            if 'instruction' in item:
                # 创建新的请求格式
                request = {
                    "custom_id": f"request-{i+1}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": "deepseek-r1",
                        "messages": [
                            {
                                "role": "system",
                                "content": system_prompt
                            },
                            {
                                "role": "user",
                                "content": item["instruction"]
                            }
                        ]
                    }
                }
                
                # 写入新格式
                f_out.write(json.dumps(request, ensure_ascii=False) + '\n')
                
        except Exception as e:
            print(f"处理项 {i+1} 时出错: {e}")

print(f"转换完成! 已将请求格式保存至 {output_file}")
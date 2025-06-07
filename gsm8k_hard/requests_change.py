import json
import os

# 输入和输出文件路径
input_file = 'filtered_gsm8k_train.jsonl'
output_file = 'requests_format.jsonl'

# 系统提示信息
system_prompt = "You are a helpful math assistant. Solve the problem step by step.\nAt the end, output the final answer in the following format:\n**Answer:** \\boxed{your_final_numeric_answer}\nDo NOT include any text after the boxed answer.\n"

# 读取JSONL文件并转换格式
with open(input_file, 'r', encoding='utf-8') as f_in, open(output_file, 'w', encoding='utf-8') as f_out:
    for i, line in enumerate(f_in):
        try:
            data = json.loads(line.strip())
            
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
                            "content": data["question"]
                        }
                    ]
                }
            }
            
            # 写入新格式
            f_out.write(json.dumps(request, ensure_ascii=False) + '\n')
            
        except json.JSONDecodeError:
            print(f"Error parsing line {i+1}: {line}")
        except KeyError:
            print(f"Missing 'question' key in line {i+1}")

print(f"转换完成! 已将请求格式保存至 {output_file}")
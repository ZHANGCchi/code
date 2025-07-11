import os
import json
from openai import OpenAI
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
import glob

# 配置信息
QWEN_MAX_API_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
QWEN_MAX_API_KEY = "sk-6ade11e6ddb14c49bc0734374a1739bc"
QWEN_MAX_MODEL = "qwen-vl-max-latest"

# 线程数配置
NUM_WORKERS = 8  # 可根据机器配置调整

def process_request(args):
    """处理单个请求"""
    idx, record, system_prompt = args
    
    # 只获取messages字段
    messages = record.get('body', {}).get('messages', [])
    user_content = None
    
    # 从messages中找到用户内容
    for message in messages:
        if message.get('role') == 'user':
            user_content = message.get('content')
            break
    
    if not user_content:
        return idx, {
            'user_content': None,
            'qwen_response': None,
            'error': 'No user message found'
        }
    
    # 准备调用QwenMax API
    client = OpenAI(api_key=QWEN_MAX_API_KEY, base_url=QWEN_MAX_API_URL)
    chat_memory = []
    
    # 添加系统提示(如果有)
    if system_prompt:
        chat_memory.append({"role": "system", "content": system_prompt})
    
    # 添加用户消息
    chat_memory.append({"role": "user", "content": user_content})
    
    try:
        response = client.chat.completions.create(
            model=QWEN_MAX_MODEL,
            messages=chat_memory,
            temperature=0.1,
            top_p=0.3
        )
        
        response_text = response.choices[0].message.content
        
        # 只返回用户内容和响应
        return idx, {
            'user_content': user_content,
            'qwen_response': response_text
        }
    except Exception as e:
        return idx, {
            'user_content': user_content,
            'qwen_response': None,
            'error': str(e)
        }

def main():
    # 参数解析 - 修改为支持多个输入文件
    parser = argparse.ArgumentParser(description='批量处理JSONL数据使用QwenMax API')
    parser.add_argument('--input', required=True, nargs='+', help='输入JSONL文件路径(支持多个文件或通配符)')
    parser.add_argument('--output', required=True, help='输出JSONL文件路径')
    parser.add_argument('--system_prompt', default='Please reason step by step, and put your final answer within \\boxed{}.', help='系统提示(可选)')
    parser.add_argument('--workers', type=int, default=NUM_WORKERS, help='并行工作进程数量')
    args = parser.parse_args()
    
    # 处理输入路径(支持通配符)
    input_files = []
    for input_path in args.input:
        # 扩展通配符路径
        expanded_paths = glob.glob(input_path)
        if expanded_paths:
            input_files.extend(expanded_paths)
        else:
            input_files.append(input_path)  # 保留原路径，即使它可能不存在
    
    # 加载输入数据
    data = []
    for input_file in input_files:
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                print(f"正在读取文件: {input_file}")
                file_count = 0
                for line in f:
                    if line.strip():
                        try:
                            record = json.loads(line)
                            data.append(record)
                            file_count += 1
                        except json.JSONDecodeError:
                            print(f"警告: 跳过无效的JSON行: {line[:50]}...")
                print(f"从 {input_file} 读取了 {file_count} 条记录")
        except FileNotFoundError:
            print(f"错误: 找不到文件 {input_file}")
    
    print(f"总共加载了 {len(data)} 条记录，来自 {len(input_files)} 个文件")
    
    if not data:
        print("没有找到有效数据，程序退出")
        return
    
    # 创建输出目录（如果需要）
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # 多进程处理
    results = [None] * len(data)  # 预分配结果列表
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        tasks = [(i, record, args.system_prompt) for i, record in enumerate(data)]
        futures = [executor.submit(process_request, t) for t in tasks]
        
        for fut in tqdm(as_completed(futures), total=len(futures), desc='处理请求'):
            idx, result = fut.result()
            results[idx] = result
    
    # 过滤掉None值（如果有）
    results = [r for r in results if r is not None]
    
    # 保存结果为JSONL格式
    with open(args.output, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    # 统计结果
    success_count = sum(1 for r in results if 'error' not in r)
    
    print(f"处理完成。结果已保存到 {args.output}")
    print(f"成功处理的记录: {success_count} / {len(results)}")

if __name__ == '__main__':
    main()
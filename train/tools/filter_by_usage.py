import json
import argparse
import logging
import os
from tqdm import tqdm

def filter_files_by_usage(args):
    """
    根据usage字段中的token数，批量过滤一个或多个JSONL文件。
    """
    # 1. 解析逗号分隔的路径字符串为文件列表
    input_paths = args.data_paths.split(',')

    print(f"开始处理 {len(input_paths)} 个文件...")
    print(f"过滤标准: '{args.filter_key}' <= {args.max_length}")

    # 2. 循环处理每一个输入文件
    for input_path in input_paths:
        input_path = input_path.strip()
        if not os.path.exists(input_path):
            logging.error(f"\n文件未找到: {input_path}，已跳过。")
            continue

        # 自动生成输出文件名，例如 a.jsonl -> a_filtered.jsonl
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_filtered{ext}"

        # 初始化当前文件的计数器和数据列表
        filtered_data = []
        total_samples = 0
        kept_samples = 0
        discarded_samples = 0

        print(f"\n--- 正在处理文件: {input_path} ---")

        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                # 统计文件总行数以用于tqdm
                num_lines = sum(1 for line in f)
                f.seek(0) # 重置文件指针

                for line in tqdm(f, total=num_lines, desc="过滤进度"):
                    total_samples += 1
                    try:
                        item = json.loads(line.strip())
                        
                        # 根据 --filter_key 参数获取要比较的token数量
                        token_count = item.get("usage", {}).get(args.filter_key, 0)
                        
                        if token_count > 0 and token_count <= args.max_length:
                            filtered_data.append(item)
                            kept_samples += 1
                        else:
                            discarded_samples += 1
                    
                    except json.JSONDecodeError:
                        logging.warning(f"\n发现无效的JSON行，已跳过。")
                    except KeyError:
                        logging.warning(f"\n某行缺少 'usage' 或 '{args.filter_key}' 字段，已跳过。")

            # 保存当前文件过滤后的数据
            if filtered_data:
                with open(output_path, 'w', encoding='utf-8') as f_out:
                    for item in filtered_data:
                        f_out.write(json.dumps(item, ensure_ascii=False) + '\n')
            
            # 打印当前文件的过滤信息
            print(f"处理完成: {input_path}")
            print(f"  - 总计样本数: {total_samples}")
            print(f"  - 保留的样本数: {kept_samples}")
            print(f"  - 丢弃的样本数: {discarded_samples}")
            print(f"  - 过滤后的文件已保存至: {output_path}")

        except Exception as e:
            logging.error(f"\n处理文件 {input_path} 时发生错误: {e}")

    print("\n所有文件处理完毕！")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    parser = argparse.ArgumentParser(description="根据usage中的token数批量过滤JSONL数据集。")
    parser.add_argument(
        "--data_paths",
        type=str,
        default="./hard_train_gsm8k.jsonl, ./hard_train_math.jsonl, ./hard_train_prm.jsonl",
        help="一个或多个待过滤数据文件的路径，用逗号分隔。"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=4096,
        help="允许的最大token长度。"
    )
    parser.add_argument(
        "--filter_key",
        type=str,
        default="total_tokens",
        choices=["total_tokens", "completion_tokens", "prompt_tokens"],
        help="在 'usage' 对象中用作过滤依据的键。默认为 'total_tokens'。"
    )
    
    args = parser.parse_args()
    filter_files_by_usage(args)
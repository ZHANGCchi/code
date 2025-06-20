import json
import re
import argparse
import logging

# --- 辅助函数部分保持不变 ---

def normalize_number(number_str):
    """将字符串转换为数值并进行规范化比较"""
    try:
        num = float(number_str)
        if num.is_integer():
            return int(num)
        return num
    except (ValueError, TypeError):
        return number_str

def extract_boxed_content(text):
    """
    健壮地提取所有 \\boxed{...} 中的内容，正确处理嵌套花括号。
    """
    results = []
    i = 0
    while i < len(text):
        boxed_start = text.find('\\boxed{', i)
        if boxed_start == -1:
            break
            
        start_idx = boxed_start + len('\\boxed{')
        bracket_count = 1
        end_idx = start_idx
        
        while end_idx < len(text) and bracket_count > 0:
            if text[end_idx] == '{':
                bracket_count += 1
            elif text[end_idx] == '}':
                bracket_count -= 1
            end_idx += 1
        
        if bracket_count == 0:
            results.append(text[start_idx:end_idx-1])
        
        i = end_idx
    
    return results

def values_equal(val1, val2, question=None):
    """
    使用您定义的标准比较两个值是否相等。
    """
    try:
        s_val1 = re.sub(r'\\dfrac{(.*?)}{(.*?)}', r'\\frac{\1}{\2}', str(val1))
        s_val2 = re.sub(r'\\dfrac{(.*?)}{(.*?)}', r'\\frac{\1}{\2}', str(val2))
        
        frac_match1 = re.match(r'\\frac{(\d+)}{(\d+)}', s_val1)
        frac_match2 = re.match(r'\\frac{(\d+)}{(\d+)}', s_val2)
        
        if frac_match1:
            s_val1 = float(frac_match1.group(1)) / float(frac_match1.group(2))
        if frac_match2:
            s_val2 = float(frac_match2.group(1)) / float(frac_match2.group(2))
        
        num1 = normalize_number(s_val1)
        num2 = normalize_number(s_val2)
        
        if isinstance(num1, (int, float)) and isinstance(num2, (int, float)):
            return abs(float(num1) - float(num2)) < 1e-6
        else:
            return str(val1).strip() == str(val2).strip()
    except Exception as e:
        logging.error(f"比较答案时出错: val1='{val1}', val2='{val2}'. 错误: {e}")
        return False

# --- 评估主逻辑 ---

def evaluate(predictions_file, ground_truth_file):
    """
    主评估函数
    :param predictions_file: 模型生成的预测结果文件路径 (jsonl格式)
    :param ground_truth_file: 包含正确答案的测试集文件路径 (jsonl格式)
    """
    # 1. 加载标准答案 (ground truth)
    ground_truths = {}
    try:
        with open(ground_truth_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                try:
                    item = json.loads(line.strip())
                    question = item.get("question")
                    
                    # VVVVVVVVVV  修改部分 VVVVVVVVVV
                    # 从 message.content 中提取 \boxed{} 内的答案作为标准答案
                    content = item.get("message", {}).get("content", "")
                    
                    if not question or not content:
                        logging.warning(f"标准答案文件第 {i+1} 行缺少 'question' 或 'message.content'，已跳过。")
                        continue
                    
                    boxed_answers = extract_boxed_content(content)
                    
                    if not boxed_answers:
                        logging.warning(f"标准答案文件第 {i+1} 行的内容中未找到 \\boxed{{}} 答案，已跳过。")
                        continue
                    
                    # 使用最后一个\boxed{}作为标准答案
                    ground_truth_answer = boxed_answers[-1]
                    ground_truths[question] = ground_truth_answer
                    # ^^^^^^^^^^  修改结束 ^^^^^^^^^^
                    
                except json.JSONDecodeError:
                    logging.error(f"标准答案文件第 {i+1} 行的JSON格式错误，已跳过。")
                except KeyError as e:
                    logging.error(f"标准答案文件第 {i+1} 行缺少关键字段: {e}，已跳过。")

    except FileNotFoundError:
        logging.error(f"标准答案文件未找到: {ground_truth_file}")
        return
    
    if not ground_truths:
        logging.error("标准答案文件为空或未能成功加载任何标准答案。")
        return

    # 2. 遍历预测文件并进行比较
    correct_count = 0
    total_count = 0
    
    try:
        with open(predictions_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                try:
                    prediction_data = json.loads(line.strip())
                    total_count += 1

                    question = prediction_data.get("question")
                    model_output = prediction_data.get("message", {}).get("content", "")

                    if not question or question not in ground_truths:
                        logging.warning(f"第 {i+1} 行的预测无法在标准答案中找到对应问题，已跳过。")
                        continue
                    
                    ground_truth_answer = ground_truths[question]
                    predicted_answers = extract_boxed_content(model_output)
                    
                    if not predicted_answers:
                        logging.warning(f"问题 '{question[:50]}...' 的模型输出中未找到\\boxed{{}}答案。")
                        continue

                    final_predicted_answer = predicted_answers[-1]
                    
                    if values_equal(final_predicted_answer, ground_truth_answer, question):
                        correct_count += 1

                except json.JSONDecodeError:
                    logging.error(f"预测文件第 {i+1} 行的JSON格式错误，已跳过。")
                except KeyError as e:
                    logging.error(f"预测文件第 {i+1} 行缺少关键字段: {e}，已跳过。")

    except FileNotFoundError:
        logging.error(f"预测结果文件未找到: {predictions_file}")
        return

    # 3. 计算并打印最终结果
    if total_count > 0:
        accuracy = (correct_count / total_count) * 100
        print("\n--- 评估结果 ---")
        print(f"总计样本数: {total_count}")
        print(f"回答正确数: {correct_count}")
        print(f"准确率: {accuracy:.2f}%")
        print("----------------")
    else:
        print("没有找到可供评估的样本。")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    parser = argparse.ArgumentParser(description="评估模型在数学问题上的表现")
    parser.add_argument(
        "--predictions_file", 
        type=str, 
        default="./predictions_base.jsonl", 
        help="模型生成的预测结果文件路径 (jsonl格式)"
    )
    parser.add_argument(
        "--ground_truth_file", 
        type=str, 
        default="./processed_data/test_data.jsonl", 
        help="包含正确答案的测试集文件路径 (jsonl格式)"
    )
    
    args = parser.parse_args()
    
    evaluate(args.predictions_file, args.ground_truth_file)
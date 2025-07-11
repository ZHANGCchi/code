# 评估模型在数学问题上的表现，使用高级数学验证
# 该脚本将尝试使用math_verify库进行数学验证，如果不可用则回退到简单提取boxed以进行比较。
# 该脚本支持jsonl格式的"reference" 和 "answer"两个字段提取gt答案，
# 以及模型输出的"message"字段提取预测答案。
import json
import re
import argparse
import logging

# 添加必要的数学验证库
try:
    from latex2sympy2_extended import NormalizationConfig
    from math_verify import LatexExtractionConfig, parse, verify
    MATH_VERIFY_AVAILABLE = True
except ImportError:
    logging.warning("未安装math_verify和latex2sympy2_extended库，将使用简单匹配。")
    MATH_VERIFY_AVAILABLE = False
    logging.warning("请使用以下命令安装依赖：")
    logging.warning("pip install git+https://github.com/philgiese/latex2sympy2-extended")
    logging.warning("pip install git+https://github.com/philgiese/math-verify")

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

def values_equal(val1, val2, sample_idx=None):
    """增强版值比较函数，处理更多情况"""
    try:
        # 预处理：清理两侧空白和引号
        s_val1 = str(val1).strip().strip('"\'')
        s_val2 = str(val2).strip().strip('"\'')
        
        # 标准化格式
        s_val1 = re.sub(r'\\dfrac{(.*?)}{(.*?)}', r'\\frac{\1}{\2}', s_val1)
        s_val2 = re.sub(r'\\dfrac{(.*?)}{(.*?)}', r'\\frac{\1}{\2}', s_val2)
        
        # 尝试提取boxed内容
        boxed1 = extract_boxed_content(s_val1)
        boxed2 = extract_boxed_content(s_val2)
        
        if boxed1 and boxed2:
            s_val1 = boxed1[-1]
            s_val2 = boxed2[-1]
        
        # 处理分数格式 - 标准格式
        frac_match1 = re.match(r'\\frac{(\d+)}{(\d+)}', s_val1)
        frac_match2 = re.match(r'\\frac{(\d+)}{(\d+)}', s_val2)
        
        # 转换分数为数值
        if frac_match1:
            num1 = float(frac_match1.group(1)) / float(frac_match1.group(2))
        else:
            num1 = normalize_number(s_val1)
            
        if frac_match2:
            num2 = float(frac_match2.group(1)) / float(frac_match2.group(2))
        else:
            num2 = normalize_number(s_val2)
        
        # 数值比较
        if isinstance(num1, (int, float)) and isinstance(num2, (int, float)):
            return abs(float(num1) - float(num2)) < 1e-6
        else:
            # 确保用于字符串比较的是字符串
            str_val1 = str(s_val1)
            str_val2 = str(s_val2)
            # 去除所有空格和不可见字符再比较
            cleaned1 = re.sub(r'\s+', '', str_val1)
            cleaned2 = re.sub(r'\s+', '', str_val2)
            return cleaned1 == cleaned2
    except Exception as e:
        idx_info = f"样本索引[{sample_idx}]" if sample_idx is not None else ""
        logging.error(f"{idx_info} 比较答案时出错: val1='{val1}', val2='{val2}'. 错误: {e}")
        return False

# --- 新增使用accuracy_reward风格的验证函数 ---

def verify_with_math(predicted_answer, ground_truth_answer, sample_idx=None):
    """
    使用与accuracy_reward相同的方法验证答案。
    
    返回:
        - True: 答案正确
        - False: 答案错误
        - None: 无法验证（解析失败）
    """
    if not MATH_VERIFY_AVAILABLE:
        return None
        
    try:
        # 解析参考答案
        gold_parsed = parse(
            ground_truth_answer,
            extraction_mode="first_match",
        )
        
        if len(gold_parsed) == 0:
            idx_info = f"样本索引[{sample_idx}]" if sample_idx is not None else ""
            logging.warning(f"{idx_info} 无法解析参考答案: {ground_truth_answer}")
            return None
            
        # 解析预测答案，使用与accuracy_reward相同的配置
        answer_parsed = parse(
            predicted_answer,
            extraction_config=[
                LatexExtractionConfig(
                    normalization_config=NormalizationConfig(
                        nits=False,
                        malformed_operators=False,
                        basic_latex=True,
                        equations=True,
                        boxed="all",
                        units=True,
                    ),
                    boxed_match_priority=0,
                    try_extract_without_anchor=False,
                )
            ],
            extraction_mode="first_match",
        )
        
        # 验证答案
        is_correct = float(verify(gold_parsed, answer_parsed))
        return bool(is_correct)
    except Exception as e:
        idx_info = f"样本索引[{sample_idx}]" if sample_idx is not None else ""
        logging.warning(f"{idx_info} 使用数学验证时出错: {e}")
        return None

# --- 更新后的评估主逻辑 ---

def evaluate(predictions_file, ground_truth_file):
    """
    主评估函数 - 优先使用数学验证
    :param predictions_file: 模型生成的预测结果文件路径 (jsonl格式)
    :param ground_truth_file: 包含正确答案的测试集文件路径 (jsonl格式)
    """
    # 1. 加载所有标准答案和预测结果到列表中
    ground_truths = []
    predictions = []
    
    try:
        with open(ground_truth_file, 'r', encoding='utf-8') as f:
            for line in f:
                ground_truths.append(json.loads(line.strip()))
    except FileNotFoundError:
        logging.error(f"标准答案文件未找到: {ground_truth_file}")
        return
    except json.JSONDecodeError as e:
        logging.error(f"解析标准答案文件时出错: {e}")
        return

    try:
        with open(predictions_file, 'r', encoding='utf-8') as f:
            for line in f:
                predictions.append(json.loads(line.strip()))
    except FileNotFoundError:
        logging.error(f"预测结果文件未找到: {predictions_file}")
        return
    except json.JSONDecodeError as e:
        logging.error(f"解析预测文件时出错: {e}")
        return

    # 2. 检查文件行数是否一致
    if len(ground_truths) != len(predictions):
        logging.warning(
            f"文件行数不匹配! 标准答案文件有 {len(ground_truths)} 行, "
            f"预测文件有 {len(predictions)} 行。将按最短的文件长度进行评估。"
        )
    
    # 3. 按顺序遍历并进行比较
    correct_count = 0
    total_count = 0
    math_verify_used = 0
    simple_verify_used = 0
    error_indices = []  # 存储错误样本的索引
    skipped_indices = []  # 存储跳过的样本索引
    
    for i in range(min(len(ground_truths), len(predictions))):
        gt_item = ground_truths[i]
        pred_item = predictions[i]

        try:
            # 从标准答案文件中获取答案
            ground_truth_answer = str(gt_item.get("reference", gt_item.get("answer", "")))
            
            if not ground_truth_answer:
                logging.warning(f"样本索引[{i}] 第 {i+1} 行的标准答案中缺少答案字段")
                skipped_indices.append(i)  # 记录跳过的样本
                continue
                
            # 从预测文件中获取内容
            model_output = pred_item.get("message", {}).get("content", "")
            
            # 首先尝试直接用数学验证整个输出
            if MATH_VERIFY_AVAILABLE:
                math_result = verify_with_math(model_output, ground_truth_answer, sample_idx=i)
                if math_result is not None:
                    # 成功使用数学验证
                    total_count += 1
                    if math_result:
                        correct_count += 1
                    else:
                        error_indices.append(i)  # 记录错误样本
                    math_verify_used += 1
                    continue
            
            # 如果数学验证失败或不可用，回退到提取boxed内容再验证
            predicted_answers = extract_boxed_content(model_output)
            
            if not predicted_answers:
                logging.warning(f"样本索引[{i}] 第 {i+1} 行的预测输出中未找到 \\boxed{{}} 答案。")
                skipped_indices.append(i)  # 记录跳过的样本
                continue
                
            final_predicted_answer = predicted_answers[-1]
            total_count += 1
            
            # 先尝试使用数学验证
            if MATH_VERIFY_AVAILABLE:
                math_result = verify_with_math(final_predicted_answer, ground_truth_answer, sample_idx=i)
                if math_result is not None:
                    if math_result:
                        correct_count += 1
                    else:
                        error_indices.append(i)  # 记录错误样本
                    math_verify_used += 1
                    continue
            
            # 回退到简单比较
            if values_equal(final_predicted_answer, ground_truth_answer, sample_idx=i):
                correct_count += 1
            else:
                error_indices.append(i)  # 记录错误样本
            simple_verify_used += 1
            
        except (KeyError, AttributeError) as e:
            logging.error(f"样本索引[{i}] 处理第 {i+1} 行数据时出错，缺少关键字段: {e}，已跳过。")
            skipped_indices.append(i)  # 记录跳过的样本
            continue

    # 4. 计算并打印最终结果
    if total_count == 0:
        print("没有找到可供评估的样本。")
        return
        
    accuracy = (correct_count / total_count) * 100
    print("\n--- 评估结果 ---")
    print(f"总计评估样本数: {total_count}")
    print(f"回答正确数: {correct_count}")
    print(f"准确率: {accuracy:.2f}%")
    
    if MATH_VERIFY_AVAILABLE:
        print(f"使用数学验证次数: {math_verify_used} ({math_verify_used/total_count*100:.1f}%)")
        print(f"使用简单验证次数: {simple_verify_used} ({simple_verify_used/total_count*100:.1f}%)")
    
    # 打印错误样本索引
    print(f"\n错误样本数量: {len(error_indices)}")
    if len(error_indices) > 0:
        print(f"错误样本索引: {error_indices[:20]}{'...' if len(error_indices) > 20 else ''}")
        if len(error_indices) > 20:
            print(f"(共 {len(error_indices)} 个错误样本，只显示前20个)")
    
    # 打印跳过样本索引
    print(f"\n跳过样本数量: {len(skipped_indices)}")
    if len(skipped_indices) > 0:
        print(f"跳过样本索引: {skipped_indices[:20]}{'...' if len(skipped_indices) > 20 else ''}")
        if len(skipped_indices) > 20:
            print(f"(共 {len(skipped_indices)} 个跳过样本，只显示前20个)")
    
    print("----------------")
    
    # 返回错误样本索引，以便调用者进一步分析
    return error_indices, skipped_indices


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    parser = argparse.ArgumentParser(description="评估模型在数学问题上的表现 (使用高级数学验证)")
    parser.add_argument(
        "--predictions_file", 
        type=str, 
        default="./predictions_vllm_qwen-max_3parts_Qwen2.5-7B-GRPO-LoRA_math500_4096_base.jsonl", 
        help="模型生成的预测结果文件路径 (jsonl格式)"
    )
    parser.add_argument(
        "--ground_truth_file", 
        type=str, 
        default="./dataset/math_dataset_jsonl/test.jsonl", 
        help="包含正确答案的测试集文件路径 (jsonl格式)"
    )
    
    args = parser.parse_args()
    
    evaluate(args.predictions_file, args.ground_truth_file)
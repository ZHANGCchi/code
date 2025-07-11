import json
import re
import math

# 辅助函数：将字符串转换为数值并进行规范化比较
def normalize_number(number_str):
    try:
        # 转换为浮点数
        num = float(number_str)
        # 如果是整数值的浮点数，转回整数
        if num.is_integer():
            return int(num)
        return num
    except:
        # 如果转换失败，返回原字符串
        return number_str

# 辅助函数：比较两个数值是否相等（考虑数值类型）
def values_equal(val1, val2):
    try:
        num1 = normalize_number(val1)
        num2 = normalize_number(val2)
        
        # 比较数值
        if isinstance(num1, (int, float)) and isinstance(num2, (int, float)):
            # 使用接近比较，允许微小的浮点误差
            if isinstance(num1, float) or isinstance(num2, float):
                return abs(float(num1) - float(num2)) < 1e-6
            else:
                return num1 == num2
        else:
            # 如果不是数值，按字符串比较
            return str(val1).strip() == str(val2).strip()
    except:
        # 出错时默认为不相等
        return False

# 辅助函数：提取所有boxed内容
def extract_all_boxed_content(text):
    if not text:
        return []
    # 匹配\boxed{...}格式，更健壮的正则表达式
    boxed_matches = re.findall(r'\\boxed\{([^{}]+(?:\{[^{}]*\}[^{}]*)*)\}', text)
    return boxed_matches

# 辅助函数：从文本中提取数字
def extract_numbers(text):
    if not text:
        return []
    return re.findall(r"[-+]?\d*\.\d+|\d+", text)

# 辅助函数：多方式验证答案
def verify_answer(prediction, ground_truth):
    # 初始设置为不接受
    accept = False
    
    # 方法1: 直接比较（使用values_equal函数）
    if values_equal(prediction, ground_truth):
        return True
        
    # 方法2: 尝试提取数值并比较
    try:
        num_pred = extract_numbers(prediction)
        num_gt = extract_numbers(ground_truth)
        
        # 如果都只有一个数字，比较这两个数字
        if len(num_pred) == 1 and len(num_gt) == 1:
            if abs(float(num_pred[0]) - float(num_gt[0])) < 1e-6:
                return True
                
        # # 如果有多个数字，尝试匹配第一个
        # elif num_pred and num_gt:
        #     if abs(float(num_pred[0]) - float(num_gt[0])) < 1e-6:
        #         return True
                # 如果数量相同，检查是否所有数字都匹配
        if len(num_pred) == len(num_gt):
            all_match = True
            for i in range(len(num_pred)):
                if abs(float(num_pred[i]) - float(num_gt[i])) > 1e-6:
                    all_match = False
                    break
            if all_match:
                return True
    except:
        pass
    
    # 方法3: 尝试去除所有空格后比较
    try:
        clean_pred = re.sub(r'\s+', '', prediction)
        clean_gt = re.sub(r'\s+', '', ground_truth)
        if clean_pred == clean_gt:
            return True
    except:
        pass
    
    # 所有方法都失败，返回False
    return False

# 步骤1: 从filtered_gsm8k_train.jsonl提取问题和ground truth
questions_and_gt = []
with open('./困难数据含答案/filtered_gsm8k_train.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line.strip())
        question = data['question']
        answer = data['answer']
        
        # 提取ground truth (#### 后面的数字)
        gt_match = re.search(r'####\s*(\d+(?:\.\d+)?)', answer)
        if gt_match:
            ground_truth = gt_match.group(1).strip()
            questions_and_gt.append((question, ground_truth))

# 步骤2: 从响应文件中提取boxed答案并与ground truth比较
correct_results = []
responses = []

# 加载所有响应
with open('./requests_format_gsm8k_hard_cot.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        try:
            response_data = json.loads(line.strip())
            responses.append(response_data)
        except json.JSONDecodeError:
            print(f"Error decoding JSON line")
            continue

# 如果问题和响应数量不一致，给出警告
if len(questions_and_gt) != len(responses):
    print(f"警告：问题数量({len(questions_and_gt)})与响应数量({len(responses)})不一致")

# 按照顺序匹配问题和响应
for i, (question, ground_truth) in enumerate(questions_and_gt):
    if i >= len(responses):
        break
        
    response_data = responses[i]
    
    try:
        # 从响应中提取boxed答案
        qwen_response = response_data.get('qwen_response', '')
        
        # 尝试提取所有boxed内容
        boxed_matches = extract_all_boxed_content(qwen_response)
        
        # 设置默认不接受
        accept = False
        
        if boxed_matches:
            # 取最后一个boxed值
            boxed_answer = boxed_matches[-1].strip()
            
            # 去掉可能残留的\boxed{}标记
            if '\\boxed' in boxed_answer:
                boxed_answer = boxed_answer.replace('\\boxed{', '').rstrip('}')
            
            # 使用增强的验证方法
            accept = verify_answer(boxed_answer, ground_truth)
            
        # 如果boxed内容验证失败，尝试直接从最后部分中提取数字
        if not accept and qwen_response:
            # # 从响应的最后部分尝试提取数字
            # last_part = qwen_response[-200:] if len(qwen_response) > 200 else qwen_response
            # numbers = extract_numbers(last_part)
            
            # if numbers:
            #     # 使用最后一个数字尝试匹配
            #     accept = verify_answer(numbers[-1], ground_truth)
            pass
        
        # 如果验证通过，添加到正确结果中
        if accept:
            correct_results.append(response_data)
            
    except Exception as e:
        print(f"Error processing response {i}: {e}")
        continue

# 步骤3: 输出正确的结果到新的jsonl文件
with open('./gsm8k_hard_train_with_qwen-max.jsonl', 'w', encoding='utf-8') as f:
    for result in correct_results:
        # 直接写入匹配正确的原始数据
        f.write(json.dumps(result, ensure_ascii=False) + '\n')

print(f"处理完毕，找到 {len(correct_results)} 个答案正确的样本")
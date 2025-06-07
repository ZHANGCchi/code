import json
import re

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

# 步骤1: 从gsm8k_train.jsonl提取问题和ground truth
questions_to_gt = {}
with open('h:/CoT剪枝/code/gsm8k/gsm8k_train.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line.strip())
        question = data['question']
        answer = data['answer']
        
        # 提取ground truth (#### 后面的数字)
        gt_match = re.search(r'####\s*(\d+(?:\.\d+)?)', answer)
        if gt_match:
            ground_truth = gt_match.group(1).strip()
            questions_to_gt[question] = ground_truth

# 步骤1.5: 从requests_format.jsonl中建立custom_id到问题的映射和保存完整请求数据
custom_id_to_question = {}
with open('h:/CoT剪枝/code/gsm8k_hard/requests_format.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        request_data = json.loads(line.strip())
        custom_id = request_data['custom_id']
        
        # 提取问题内容
        for message in request_data['body']['messages']:
            if message['role'] == 'user':
                user_message = message['content']
                custom_id_to_question[custom_id] = user_message
                break

# 步骤2: 从响应文件中提取boxed答案并与ground truth比较
correct_results = []
with open('h:/CoT剪枝/code/gsm8k_hard/hard_train_no_filtered.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        try:
            response_data = json.loads(line.strip())
            
            # 获取custom_id
            custom_id = response_data.get('custom_id')
            if not custom_id or custom_id not in custom_id_to_question:
                continue
                
            question = custom_id_to_question[custom_id]
            
            # 提取响应中的boxed答案 - 适应示例中的JSON结构
            try:
                content = response_data['response']['body']['choices'][0]['message']['content']
                boxed_matches = re.findall(r'\\boxed\{([^}]+)\}', content)
                
                if boxed_matches and question in questions_to_gt:
                    # 取最后一个boxed值
                    boxed_answer = boxed_matches[-1].strip()
                    ground_truth = questions_to_gt[question]
                    
                    # 比较答案（使用数值比较）
                    if values_equal(boxed_answer, ground_truth):
                        correct_results.append(response_data)
            except (KeyError, IndexError) as e:
                print(f"Error extracting answer: {e}")
                continue
        except json.JSONDecodeError:
            print(f"Error decoding JSON line")
            continue

# 步骤3: 输出正确的结果到新的jsonl文件（按指定格式保存）
with open('h:/CoT剪枝/code/gsm8k_hard/hard_train.jsonl', 'w', encoding='utf-8') as f:
    for result in correct_results:
        custom_id = result['custom_id']
        
        # 创建简化的输出对象，只包含指定字段
        simplified_result = {
            "custom_id": custom_id,
            "question": custom_id_to_question[custom_id],  # 添加问题内容
            "usage": result['response']['body']['usage'],
            "model": result['response']['body']['model'],
            "message": result['response']['body']['choices'][0]['message']
        }
        
        # 写入简化后的结果
        f.write(json.dumps(simplified_result, ensure_ascii=False) + '\n')

print(f"处理完毕，找到 {len(correct_results)} 个答案正确的样本")
# import json
# import re

# # 辅助函数：将字符串转换为数值并进行规范化比较
# def normalize_number(number_str):
#     try:
#         # 转换为浮点数
#         num = float(number_str)
#         # 如果是整数值的浮点数，转回整数
#         if num.is_integer():
#             return int(num)
#         return num
#     except:
#         # 如果转换失败，返回原字符串
#         return number_str

# # 辅助函数：比较两个数值是否相等（考虑数值类型）
# def values_equal(val1, val2):
#     try:
#         num1 = normalize_number(val1)
#         num2 = normalize_number(val2)
        
#         # 比较数值
#         if isinstance(num1, (int, float)) and isinstance(num2, (int, float)):
#             # 使用接近比较，允许微小的浮点误差
#             if isinstance(num1, float) or isinstance(num2, float):
#                 return abs(float(num1) - float(num2)) < 1e-6
#             else:
#                 return num1 == num2
#         else:
#             # 如果不是数值，按字符串比较
#             return str(val1).strip() == str(val2).strip()
#     except:
#         # 出错时默认为不相等
#         return False

# # 步骤1: 从prm_less_than_10_pass_questions.json提取问题和ground truth
# questions_to_gt = {}
# with open('h:/CoT剪枝/code/prm800k_hard/prm_less_than_10_pass_questions.json', 'r', encoding='utf-8') as f:
#     data_list = json.load(f)  # 加载整个JSON数组
    
#     for item in data_list:
#         # 检查数据项是否同时包含instruction和answer字段
#         if 'instruction' in item and 'answer' in item:
#             instruction = item['instruction']
#             answer = item['answer']
            
#             # 直接将问题和答案存入字典，不需要正则表达式提取
#             questions_to_gt[instruction] = answer

# # 步骤1.5: 从requests_format.jsonl中建立custom_id到问题的映射和保存完整请求数据
# custom_id_to_question = {}
# with open('h:/CoT剪枝/code/prm800k_hard/requests_format.jsonl', 'r', encoding='utf-8') as f:
#     for line in f:
#         request_data = json.loads(line.strip())
#         custom_id = request_data['custom_id']
        
#         # 提取问题内容
#         for message in request_data['body']['messages']:
#             if message['role'] == 'user':
#                 user_message = message['content']
#                 custom_id_to_question[custom_id] = user_message
#                 break

# # 步骤2: 从响应文件中提取boxed答案并与ground truth比较
# correct_results = []
# with open('h:/CoT剪枝/code/prm800k_hard/hard_train_no_filtered.jsonl', 'r', encoding='utf-8') as f:
#     for line in f:
#         try:
#             response_data = json.loads(line.strip())
            
#             # 获取custom_id
#             custom_id = response_data.get('custom_id')
#             if not custom_id or custom_id not in custom_id_to_question:
#                 continue
                
#             question = custom_id_to_question[custom_id]
            
#             # 提取响应中的boxed答案 - 适应示例中的JSON结构
#             try:
#                 content = response_data['response']['body']['choices'][0]['message']['content']
#                 boxed_matches = re.findall(r'\\boxed\{([^}]+)\}', content)
                
#                 if boxed_matches and question in questions_to_gt:
#                     # 取最后一个boxed值
#                     boxed_answer = boxed_matches[-1].strip()
#                     ground_truth = questions_to_gt[question]
                    
#                     # 比较答案（使用数值比较）
#                     if values_equal(boxed_answer, ground_truth):
#                         correct_results.append(response_data)
#             except (KeyError, IndexError) as e:
#                 print(f"Error extracting answer: {e}")
#                 continue
#         except json.JSONDecodeError:
#             print(f"Error decoding JSON line")
#             continue

# # 步骤3: 输出正确的结果到新的jsonl文件（按指定格式保存）
# with open('h:/CoT剪枝/code/prm800k_hard/hard_train.jsonl', 'w', encoding='utf-8') as f:
#     for result in correct_results:
#         custom_id = result['custom_id']
        
#         # 创建简化的输出对象，只包含指定字段
#         simplified_result = {
#             "custom_id": custom_id,
#             "question": custom_id_to_question[custom_id],  # 添加问题内容
#             "usage": result['response']['body']['usage'],
#             "model": result['response']['body']['model'],
#             "message": result['response']['body']['choices'][0]['message']
#         }
        
#         # 写入简化后的结果
#         f.write(json.dumps(simplified_result, ensure_ascii=False) + '\n')

# print(f"处理完毕，找到 {len(correct_results)} 个答案正确的样本")
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

# 辅助函数：处理选择题答案和数学表达式格式问题
def values_equal(val1, val2, question=None):
    try:
        # # 先尝试检查是否为选择题答案（单个字母A-E）
        # if re.match(r'^[A-E]$', val1):
        #     # 从问题中提取选项及其对应值
        #     option_matches = None
        #     if question:
        #         # 尝试匹配 "(A) value" 格式的选项
        #         option_matches = re.findall(r'\\textbf{\(([A-E])\) }\s*(\d+|[^\\]+?)(?=\\|\s*\\textbf{|$)', question)
        #         if not option_matches:
        #             # 尝试匹配其他可能的选项格式
        #             option_matches = re.findall(r'\(([A-E])\)\s*(\d+|[^\\]+?)(?=\\|\s*\(|$)', question)

        #         # 如果找到了选项，创建选项到值的映射并清理值
        #         if option_matches:
        #             options_map = {opt: val.strip() for opt, val in option_matches}
        #             if val1 in options_map:
        #                 val1 = options_map[val1]
                    
        # 标准化分数表示 - 处理\dfrac和\frac
        val1 = re.sub(r'\\dfrac{(.*?)}{(.*?)}', r'\\frac{\1}{\2}', str(val1))
        val2 = re.sub(r'\\dfrac{(.*?)}{(.*?)}', r'\\frac{\1}{\2}', str(val2))
        
        # 尝试从\frac{}{}提取分数值
        frac_match1 = re.match(r'\\frac{(\d+)}{(\d+)}', val1)
        frac_match2 = re.match(r'\\frac{(\d+)}{(\d+)}', val2)
        
        if frac_match1:
            val1 = float(frac_match1.group(1)) / float(frac_match1.group(2))
        if frac_match2:
            val2 = float(frac_match2.group(1)) / float(frac_match2.group(2))
        
        # 数值比较
        num1 = normalize_number(val1)
        num2 = normalize_number(val2)
        
        if isinstance(num1, (int, float)) and isinstance(num2, (int, float)):
            # 使用接近比较，允许微小的浮点误差
            if isinstance(num1, float) or isinstance(num2, float):
                return abs(float(num1) - float(num2)) < 1e-6
            else:
                return num1 == num2
        else:
            # 如果不是数值，按字符串比较
            return str(val1).strip() == str(val2).strip()
    except Exception as e:
        print(f"比较错误: {e}")
        # 出错时默认为不相等
        return False

# 步骤1: 从math_hard.jsonl提取问题和ground truth
questions_to_gt = {}
with open('h:/CoT剪枝/code/math_hard/math_hard.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        try:
            item = json.loads(line.strip())
            # 检查数据项是否同时包含instruction和answer字段
            if 'instruction' in item and 'answer' in item:
                instruction = item['instruction']
                answer = item['answer']
                
                # 直接将问题和答案存入字典，不需要正则表达式提取
                questions_to_gt[instruction] = answer
        except json.JSONDecodeError:
            continue

# 步骤1.5: 从requests_format.jsonl中建立custom_id到问题的映射和保存完整请求数据
custom_id_to_question = {}
with open('h:/CoT剪枝/code/math_hard/requests_format.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        request_data = json.loads(line.strip())
        custom_id = request_data['custom_id']
        
        # 提取问题内容
        for message in request_data['body']['messages']:
            if message['role'] == 'user':
                user_message = message['content']
                custom_id_to_question[custom_id] = user_message
                break

def extract_boxed_content(text):
    results = []
    i = 0
    while i < len(text):
        # 查找下一个\boxed{
        boxed_start = text.find('\\boxed{', i)
        if boxed_start == -1:
            break
            
        # 从boxed命令后的第一个位置开始
        start_idx = boxed_start + 7  # len('\\boxed{')
        bracket_count = 1  # 已经遇到一个左花括号
        end_idx = start_idx
        
        # 循环直到找到匹配的右花括号
        while end_idx < len(text) and bracket_count > 0:
            if text[end_idx] == '{':
                bracket_count += 1
            elif text[end_idx] == '}':
                bracket_count -= 1
            end_idx += 1
        
        if bracket_count == 0:  # 找到了匹配的结束括号
            results.append(text[start_idx:end_idx-1])
        
        i = end_idx
    
    return results

# 步骤2: 从响应文件中提取boxed答案并与ground truth比较
correct_results = []
with open('h:/CoT剪枝/code/math_hard/hard_train_no_filtered.jsonl', 'r', encoding='utf-8') as f:
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
                boxed_matches = extract_boxed_content(content)
                
                if boxed_matches and question in questions_to_gt:
                    # 取最后一个boxed值
                    boxed_answer = boxed_matches[-1].strip()
                    ground_truth = questions_to_gt[question]
                    
                    # 增强比较 - 传递问题内容以处理选择题
                    if values_equal(boxed_answer, ground_truth, question):
                        correct_results.append(response_data)
            except (KeyError, IndexError) as e:
                print(f"Error extracting answer: {e}")
                continue
        except json.JSONDecodeError:
            print(f"Error decoding JSON line")
            continue

# 步骤3: 输出正确的结果到新的jsonl文件（按指定格式保存）
with open('h:/CoT剪枝/code/math_hard/hard_train.jsonl', 'w', encoding='utf-8') as f:
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
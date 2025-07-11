# coding=utf-8
# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""GRPO训练的奖励函数。"""

import asyncio
import json
import math
import re
from functools import partial, update_wrapper
from typing import Callable, Dict, Literal, Optional

from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify

# from .utils.code_providers import get_provider
# from .utils.competitive_programming import (
#     SubtaskResult,
#     add_includes,
#     get_morph_client_from_env,
#     get_piston_client_from_env,
# )
# from .utils.competitive_programming import patch_code as cf_patch_code
# from .utils.competitive_programming import score_submission as cf_score_submission
# from .utils.competitive_programming import score_subtask


# def accuracy_reward(completions: list[list[dict[str, str]]], solution: list[str], **kwargs) -> list[Optional[float]]:
#     """检查模型生成内容是否与参考答案一致的奖励函数。"""
#     contents = [completion[0]["content"] for completion in completions]
#     rewards = []
#     for content, sol in zip(contents, solution):
#         gold_parsed = parse(
#             sol,
#             extraction_mode="first_match",
#         )
#         if len(gold_parsed) != 0:
#             # 要求答案使用正确的latex格式提供（无格式错误的运算符）
#             answer_parsed = parse(
#                 content,
#                 extraction_config=[
#                     LatexExtractionConfig(
#                         normalization_config=NormalizationConfig(
#                             nits=False,
#                             malformed_operators=False,
#                             basic_latex=True,
#                             equations=True,
#                             boxed="all",
#                             units=True,
#                         ),
#                         # 确保优先尝试匹配boxed内容
#                         boxed_match_priority=0,
#                         try_extract_without_anchor=False,
#                     )
#                 ],
#                 extraction_mode="first_match",
#             )
#             # 如果可验证则计算二元奖励，否则返回`None`以跳过此示例
#             try:
#                 reward = float(verify(gold_parsed, answer_parsed))
#             except Exception as e:
#                 print(f"验证失败: {e}, 答案: {answer_parsed}, 参考: {gold_parsed}")
#                 reward = None
#         else:
#             # 如果参考答案无法解析，我们赋值`None`以跳过此示例
#             reward = None
#             print("无法解析参考答案: ", sol)
#         rewards.append(reward)

#     return rewards

def accuracy_reward(completions: list[list[dict[str, str]]], solution: list[str], **kwargs) -> list[Optional[float]]:
    """检查模型生成内容是否与参考答案一致的奖励函数。"""
    def extract_boxed_content(text):
        """健壮地提取所有 \\boxed{...} 中的内容，正确处理嵌套花括号。"""
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
    
    def normalize_number(number_str):
        """将字符串转换为数值并进行规范化比较"""
        try:
            num = float(number_str)
            if num.is_integer():
                return int(num)
            return num
        except (ValueError, TypeError):
            return number_str
    
    def values_equal(val1, val2):
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
            print(f"比较答案时出错: val1='{val1}', val2='{val2}'. 错误: {e}")
            return False
    
    # 主处理逻辑
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    
    for content, sol in zip(contents, solution):
        try:
            # 第一步：尝试使用math_verify进行验证
            gold_parsed = parse(
                sol,
                extraction_mode="first_match",
            )
            
            if len(gold_parsed) != 0:
                # 要求答案使用正确的latex格式提供（无格式错误的运算符）
                answer_parsed = parse(
                    content,
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
                            # 确保优先尝试匹配boxed内容
                            boxed_match_priority=0,
                            try_extract_without_anchor=False,
                        )
                    ],
                    extraction_mode="first_match",
                )
                
                # 尝试数学验证
                try:
                    reward = float(verify(gold_parsed, answer_parsed))
                    rewards.append(reward)
                    continue
                except Exception as e:
                    print(f"验证失败: {e}, 答案: {answer_parsed}, 参考: {gold_parsed}")
                    # 不立即返回None，继续尝试简单验证
            
            # 第二步：如果数学验证失败，尝试提取boxed内容并使用简单比较
            predicted_answers = extract_boxed_content(content)
            
            if predicted_answers:
                final_predicted_answer = predicted_answers[-1]
                
                # 再次尝试使用math_verify（仅针对boxed内容）
                try:
                    boxed_parsed = parse(
                        final_predicted_answer,
                        extraction_mode="first_match",
                    )
                    gold_parsed = parse(
                        sol,
                        extraction_mode="first_match",
                    )
                    
                    if len(boxed_parsed) > 0 and len(gold_parsed) > 0:
                        reward = float(verify(gold_parsed, boxed_parsed))
                        rewards.append(reward)
                        continue
                except Exception:
                    pass  # 如果解析失败，继续尝试简单比较
                
                # 最后回退到简单比较
                if values_equal(final_predicted_answer, sol):
                    rewards.append(1.0)
                else:
                    rewards.append(0.0)
            else:
                # 如果所有验证方法都失败，返回None跳过此示例
                rewards.append(None)
                print(f"无法验证答案: 未找到boxed内容，内容: {content}, 参考: {sol}")
        except Exception as e:
            # 如果处理过程中出现任何错误，返回None跳过此示例
            rewards.append(None)
            print(f"处理样本时出错: {e}, 内容: {content}, 参考: {sol}")

    return rewards

def format_reward(completions, **kwargs):
    """检查推理过程是否包含在<think>和</think>标签内，而最终答案是否包含在<answer>和</answer>标签内的奖励函数。"""
    pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>$"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]


def tag_count_reward(completions, **kwargs) -> list[float]:
    """检查是否生成了与`format_reward()`相关的期望数量的think和answer标签的奖励函数。
    
    改编自：https://gist.github.com/willccbb/4676755236bb08cab5f4e54a0475d6fb#file-grpo_demo-py-L90
    """

    def count_tags(text: str) -> float:
        count = 0.0
        if text.count("<think>\n") == 1:
            count += 0.25
        if text.count("\n</think>\n") == 1:
            count += 0.25
        if text.count("\n<answer>\n") == 1:
            count += 0.25
        if text.count("\n</answer>") == 1:
            count += 0.25
        return count

    contents = [completion[0]["content"] for completion in completions]
    return [count_tags(c) for c in contents]


def reasoning_steps_reward(completions, **kwargs):
    r"""检查是否有清晰的逐步推理的奖励函数。
    正则表达式模式：
        Step \d+: - 匹配"Step 1:", "Step 2:"等
        ^\d+\. - 匹配行首的编号列表，如"1.", "2."等
        \n- - 匹配连字符项目符号
        \n\* - 匹配星号项目符号
        First,|Second,|Next,|Finally, - 匹配过渡词
    """
    pattern = r"(Step \d+:|^\d+\.|\n-|\n\*|First,|Second,|Next,|Finally,)"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [len(re.findall(pattern, content)) for content in completion_contents]

    # 神奇数字3鼓励3个或更多步骤，否则给予部分奖励
    return [min(1.0, count / 3) for count in matches]


def len_reward(completions: list[Dict[str, str]], solution: list[str], **kwargs) -> float:
    """计算基于长度的奖励，以防止过度思考并促进令牌效率。

    取自Kimi 1.5技术报告：https://huggingface.co/papers/2501.12599

    参数：
        completions: 模型完成列表
        solution: 参考答案列表

    返回：
        奖励列表，其中：
        - 对于正确答案：reward = 0.5 - (len - min_len)/(max_len - min_len)
        - 对于错误答案：reward = min(0, 0.5 - (len - min_len)/(max_len - min_len))
    """
    contents = [completion[0]["content"] for completion in completions]

    # 首先检查答案的正确性
    correctness = []
    for content, sol in zip(contents, solution):
        gold_parsed = parse(
            sol,
            extraction_mode="first_match",
            extraction_config=[LatexExtractionConfig()],
        )
        if len(gold_parsed) == 0:
            # 跳过无法解析的示例
            correctness.append(True)  # 视为正确以避免惩罚
            print("无法解析参考答案: ", sol)
            continue

        answer_parsed = parse(
            content,
            extraction_config=[
                LatexExtractionConfig(
                    normalization_config=NormalizationConfig(
                        nits=False,
                        malformed_operators=False,
                        basic_latex=True,
                        equations=True,
                        boxed=True,
                        units=True,
                    ),
                    boxed_match_priority=0,
                    try_extract_without_anchor=False,
                )
            ],
            extraction_mode="first_match",
        )
        correctness.append(verify(answer_parsed, gold_parsed))

    # 计算长度
    lengths = [len(content) for content in contents]
    min_len = min(lengths)
    max_len = max(lengths)

    # 如果所有响应长度相同，返回零奖励
    if max_len == min_len:
        return [0.0] * len(completions)

    rewards = []
    for length, is_correct in zip(lengths, correctness):
        lambda_val = 0.5 - (length - min_len) / (max_len - min_len)

        if is_correct:
            reward = lambda_val
        else:
            reward = min(0, lambda_val)

        rewards.append(float(reward))

    return rewards


def get_cosine_scaled_reward(
    min_value_wrong: float = -1.0,
    max_value_wrong: float = -0.5,
    min_value_correct: float = 0.5,
    max_value_correct: float = 1.0,
    max_len: int = 1000,
):
    def cosine_scaled_reward(completions, solution, **kwargs):
        """基于完成长度使用余弦调度进行缩放的奖励函数。

        较短的正确解决方案获得的奖励比较长的解决方案更多。
        较长的错误解决方案受到的惩罚比较短的解决方案更少。

        参数：
            completions: 模型完成列表
            solution: 参考答案列表

        此函数由以下参数参数化：
            min_value_wrong: 错误答案的最小奖励
            max_value_wrong: 错误答案的最大奖励
            min_value_correct: 正确答案的最小奖励
            max_value_correct: 正确答案的最大奖励
            max_len: 缩放的最大长度
        """
        contents = [completion[0]["content"] for completion in completions]
        rewards = []

        for content, sol in zip(contents, solution):
            gold_parsed = parse(
                sol,
                extraction_mode="first_match",
                extraction_config=[LatexExtractionConfig()],
            )
            if len(gold_parsed) == 0:
                rewards.append(1.0)  # 跳过无法解析的示例
                print("无法解析参考答案: ", sol)
                continue

            answer_parsed = parse(
                content,
                extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(
                            nits=False,
                            malformed_operators=False,
                            basic_latex=True,
                            equations=True,
                            boxed=True,
                            units=True,
                        ),
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    )
                ],
                extraction_mode="first_match",
            )

            is_correct = verify(answer_parsed, gold_parsed)
            gen_len = len(content)

            # 基于长度应用余弦缩放
            progress = gen_len / max_len
            cosine = math.cos(progress * math.pi)

            if is_correct:
                min_value = min_value_correct
                max_value = max_value_correct
            else:
                # 对于错误答案交换最小/最大值
                min_value = max_value_wrong
                max_value = min_value_wrong

            reward = min_value + 0.5 * (max_value - min_value) * (1.0 + cosine)
            rewards.append(float(reward))

        return rewards

    return cosine_scaled_reward


def get_repetition_penalty_reward(ngram_size: int, max_penalty: float, language: str = "en"):
    """
    计算N-gram重复惩罚，如https://huggingface.co/papers/2502.03373附录C.2所述。
    参考实现来自：https://github.com/eddycmu/demystify-long-cot/blob/release/openrlhf/openrlhf/reward/repetition.py

    参数：
    ngram_size: n-gram的大小
    max_penalty: 错误答案的最大（负）惩罚
    language: 文本语言，默认为`en`。用于选择将文本分割为n-gram的方式。
    """
    if max_penalty > 0:
        raise ValueError(f"max_penalty {max_penalty} 不应为正数")

    if language == "en":

        def zipngram(text: str, ngram_size: int):
            words = text.lower().split()
            return zip(*[words[i:] for i in range(ngram_size)]), words

    # elif language == "zh":
    #     from transformers.utils.import_utils import _is_package_available

    #     if not _is_package_available("jieba"):
    #         raise ValueError("请安装jieba以使用中文语言")

    #     def zipngram(text: str, ngram_size: int):
    #         import jieba

    #         seg_list = list(jieba.cut(text))
    #         return zip(*[seg_list[i:] for i in range(ngram_size)]), seg_list

    else:
        raise ValueError(
            f"尚未实现语言`{language}`的单词分割。请实现您自己的zip-ngram函数。"
        )

    def repetition_penalty_reward(completions, **kwargs) -> float:
        """
        惩罚重复的奖励函数
        参考实现：https://github.com/eddycmu/demystify-long-cot/blob/release/openrlhf/openrlhf/reward/repetition.py

        参数：
            completions: 模型完成列表
        """

        contents = [completion[0]["content"] for completion in completions]
        rewards = []
        for completion in contents:
            if completion == "":
                rewards.append(0.0)
                continue

            ngrams = set()
            total = 0
            ngram_array, words = zipngram(completion, ngram_size)

            if len(words) < ngram_size:
                rewards.append(0.0)
                continue

            for ng in ngram_array:
                ngrams.add(ng)
                total += 1

            scaling = 1 - len(ngrams) / total
            reward = scaling * max_penalty
            rewards.append(reward)
        return rewards

    return repetition_penalty_reward


def _init_event_loop():
    """初始化或获取当前事件循环。"""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop


# def ioi_code_reward(completions, test_batch_size: int = 1, provider_type: str = "piston", **kwargs) -> list[float]:
#     """使用指定执行客户端评估IOI问题的奖励函数。

#     假设数据集格式与hf.co/datasets/open-r1/ioi相同

#     参数：
#         completions: 要评估的模型完成列表
#         test_batch_size: 并行评估这么多测试用例，然后检查它们中的任何一个是否失败（0分）：
#                        如果失败则停止评估；否则继续下一批测试用例。
#         provider_type: 要使用的执行提供者（默认："piston"）。支持的值："piston"，"morph"
#         **kwargs: 从数据集传递的额外参数
#     """
#     # 根据provider_type获取适当的客户端
#     if provider_type == "morph":
#         execution_client = get_morph_client_from_env()
#     else:
#         # 有关设置piston工作程序的信息，请参见slurm/piston/README.md
#         execution_client = get_piston_client_from_env()

#     code_snippets = [
#         # 注意：如果没有提取到代码，评分会自动跳过
#         add_includes(extract_code(completion[-1]["content"], "cpp"), problem_id)
#         for completion, problem_id in zip(completions, kwargs["id"])
#     ]

#     async def run_catch_exceptions(task):
#         try:
#             return await task
#         except Exception as e:
#             print(f"来自{provider_type}工作程序的错误：{e}")
#             return SubtaskResult()

#     problems_data = [dict(zip(kwargs.keys(), values)) for values in zip(*kwargs.values())]

#     loop = _init_event_loop()
#     evals = [
#         loop.create_task(
#             run_catch_exceptions(
#                 score_subtask(
#                     execution_client,
#                     problem_data,
#                     code,
#                     test_batch_size=test_batch_size,
#                 )
#             )
#         )
#         for problem_data, code in zip(problems_data, code_snippets)
#     ]
#     results = loop.run_until_complete(asyncio.gather(*evals))

#     return [result.score for result in results]


# def cf_code_reward(
#     completions,
#     test_batch_size: int = 1,
#     patch_code: bool = False,
#     scoring_mode: Literal["pass_fail", "partial", "weighted_sum"] = "weighted_sum",
#     **kwargs,
# ) -> list[float]:
#     """使用Piston+我们的CF包评估Codeforces问题的奖励函数。

#     假设数据集格式与hf.co/datasets/open-r1/codeforces（verifiable-prompts子集）相同

#     test_batch_size: 并行评估这么多测试用例，然后检查它们中的任何一个是否失败（0分）：如果失败则停止评估；否则继续下一批测试用例。
#     """
#     # 有关设置piston工作程序的信息，请参见slurm/piston/README.md
#     piston_client = get_piston_client_from_env()

#     languages = kwargs["language"] if "language" in kwargs else [None] * len(completions)
#     code_snippets = [
#         # 注意：如果问题没有测试，评分会自动跳过
#         cf_patch_code(extract_code(completion[-1]["content"], language), language)
#         if patch_code
#         else extract_code(completion[-1]["content"], language)
#         for completion, language in zip(completions, languages)
#     ]

#     async def run_catch_exceptions(task):
#         try:
#             return await task
#         except Exception as e:
#             print(f"来自Piston工作程序的错误：{e}")
#             return None

#     # 加载问题数据。撤销按列分离kwargs
#     problems_data = [dict(zip(kwargs.keys(), values)) for values in zip(*kwargs.values())]

#     loop = _init_event_loop()
#     evals = [
#         loop.create_task(
#             run_catch_exceptions(
#                 cf_score_submission(
#                     piston_client,
#                     problem_data,
#                     code,
#                     test_batch_size=test_batch_size,
#                     scoring_mode=scoring_mode,
#                     submission_language=problem_data.get("language", None),
#                 )
#             )
#         )
#         for problem_data, code in zip(problems_data, code_snippets)
#     ]
#     results = loop.run_until_complete(asyncio.gather(*evals))

#     return results


# def extract_code(completion: str, language: str | None = "python") -> str:
#     """从完成内容中提取代码片段。"""
#     if language is None:
#         return ""
#     pattern = re.compile(rf"```{language}\n(.*?)```", re.DOTALL)
#     matches = pattern.findall(completion)
#     extracted_answer = matches[-1] if len(matches) >= 1 else ""
#     return extracted_answer


# def binary_code_reward(
#     completions,
#     num_parallel: int = 2,
#     provider_type: str = "e2b",
#     enforce_same_language: bool = False,
#     **kwargs,
# ) -> list[float]:
#     """将代码奖励转换为二元奖励（0或1）的函数。"""
#     rewards = code_reward(
#         completions,
#         num_parallel=num_parallel,
#         provider_type=provider_type,
#         enforce_same_language=enforce_same_language,
#         **kwargs,
#     )
#     BINARY_THRESHOLD = 0.99

#     output = []
#     for reward in rewards:
#         if reward is None:
#             output.append(None)
#         else:
#             output.append(1.0 if reward > BINARY_THRESHOLD else 0.0)

#     return output


# def code_reward(
#     completions,
#     num_parallel: int = 2,
#     provider_type: str = "e2b",
#     enforce_same_language: bool = False,
#     **kwargs,
# ) -> list[float]:
#     """使用代码执行提供者评估代码片段的奖励函数。

#     假设数据集包含带有测试用例的`verification_info`列。

#     参数：
#         completions: 要评估的模型完成列表
#         num_parallel: 并行代码执行数量（默认：2）
#         provider_type: 使用哪个代码执行提供者（默认："e2b"）
#         enforce_same_language: 如果为True，验证所有问题使用相同的语言（默认：False）
#         **kwargs: 传递给验证的额外参数
#     """
#     evaluation_script_template = """
#     import subprocess
#     import json

#     def evaluate_code(code, test_cases):
#         passed = 0
#         total = len(test_cases)
#         exec_timeout = 5

#         for case in test_cases:
#             process = subprocess.run(
#                 ["python3", "-c", code],
#                 input=case["input"],
#                 text=True,
#                 capture_output=True,
#                 timeout=exec_timeout
#             )

#             if process.returncode != 0:  # 执行错误
#                 continue

#             output = process.stdout.strip()

#             # TODO: 实现一个适当的验证器与地面真相比较。目前我们只检查stdout每行的精确字符串匹配。
#             all_correct = True
#             for line1, line2 in zip(output.split('\\n'), case['output'].split('\\n')):
#                 all_correct = all_correct and line1.strip() == line2.strip()

#             if all_correct:
#                 passed += 1

#         success_rate = (passed / total)
#         return success_rate

#     code_snippet = {code}
#     test_cases = json.loads({test_cases})

#     evaluate_code(code_snippet, test_cases)
#     """

#     code_snippets = [extract_code(completion[-1]["content"]) for completion in completions]
#     verification_info = kwargs["verification_info"]

#     template = evaluation_script_template

#     scripts = [
#         template.format(code=json.dumps(code), test_cases=json.dumps(json.dumps(info["test_cases"])))
#         for code, info in zip(code_snippets, verification_info)
#     ]

#     language = verification_info[0]["language"]

#     if enforce_same_language:
#         all_same_language = all(v["language"] == language for v in verification_info)
#         if not all_same_language:
#             raise ValueError("所有verification_info必须有相同的语言", verification_info)

#     execution_provider = get_provider(
#         provider_type=provider_type,
#         num_parallel=num_parallel,
#         **kwargs,
#     )

#     return execution_provider.execute_scripts(scripts, ["python"] * len(scripts))


# def get_code_format_reward(language: str = "python"):
#     """专门针对代码响应的格式奖励函数。

#     参数：
#         language: E2B支持的编程语言 https://e2b.dev/docs/code-interpreting/supported-languages
#     """

#     def code_format_reward(completions, **kwargs):
#         # 如果有language字段，使用它代替默认语言。这样我们可以进行混合语言训练。
#         languages = kwargs["language"] if "language" in kwargs else [language] * len(completions)

#         completion_contents = [completion[0]["content"] for completion in completions]
#         matches = [
#             re.match(
#                 rf"^<think>\n.*?\n</think>\n<answer>\n.*?```{sample_language}.*?```.*?\n</answer>$",
#                 content,
#                 re.DOTALL | re.MULTILINE,
#             )
#             for content, sample_language in zip(completion_contents, languages)
#         ]
#         return [1.0 if match else 0.0 for match in matches]

#     return code_format_reward


def get_soft_overlong_punishment(max_completion_len, soft_punish_cache):
    """
    惩罚过长完成的奖励函数。用于惩罚过长的完成，但不奖励较短的完成。
    参考：DAPO论文中的等式(13)（https://huggingface.co/papers/2503.14476）

    参数：
        max_completion_len: 完成的最大长度
        soft_punish_cache: 完成的最小长度。如果设置为0，则不应用最小长度。
    """

    def soft_overlong_punishment_reward(completion_ids: list[list[int]], **kwargs) -> list[float]:
        """惩罚过长完成的奖励函数。"""
        rewards = []
        for ids in completion_ids:
            completion_length = len(ids)
            if completion_length <= max_completion_len - soft_punish_cache:
                rewards.append(0.0)
            elif max_completion_len - soft_punish_cache < completion_length <= max_completion_len:
                rewards.append((max_completion_len - soft_punish_cache - completion_length) / soft_punish_cache)
            else:
                rewards.append(-1.0)
        return rewards

    return soft_overlong_punishment_reward


def get_reward_funcs(script_args) -> list[Callable]:
    """根据脚本参数获取奖励函数列表。"""
    REWARD_FUNCS_REGISTRY = {
        "accuracy": accuracy_reward,
        "format": format_reward,
        "reasoning_steps": reasoning_steps_reward,
        "cosine": get_cosine_scaled_reward(
            min_value_wrong=script_args.cosine_min_value_wrong,
            max_value_wrong=script_args.cosine_max_value_wrong,
            min_value_correct=script_args.cosine_min_value_correct,
            max_value_correct=script_args.cosine_max_value_correct,
            max_len=script_args.cosine_max_len,
        ),
        "repetition_penalty": get_repetition_penalty_reward(
            ngram_size=script_args.repetition_n_grams,
            max_penalty=script_args.repetition_max_penalty,
        ),
        "length": len_reward,
        "code": update_wrapper(
            partial(
                # code_reward,
                num_parallel=script_args.parallel_code_exec_per_proc,
                provider_type=script_args.code_provider,
                enforce_same_language=getattr(script_args, "enforce_same_language", False),
            ),
            # code_reward,
        ),
        # "binary_code": update_wrapper(
        #     partial(
        #         binary_code_reward,
        #         num_parallel=script_args.parallel_code_exec_per_proc,
        #         provider_type=script_args.code_provider,
        #         enforce_same_language=getattr(script_args, "enforce_same_language", False),
        #     ),
        #     binary_code_reward,
        # ),
        # "ioi_code": update_wrapper(
        #     partial(
        #         ioi_code_reward,
        #         test_batch_size=script_args.code_eval_test_batch_size,
        #         provider_type=getattr(script_args, "ioi_provider", "piston"),
        #     ),
        #     ioi_code_reward,
        # ),
        # "cf_code": update_wrapper(
        #     partial(
        #         cf_code_reward,
        #         test_batch_size=script_args.code_eval_test_batch_size,
        #         scoring_mode=script_args.code_eval_scoring_mode,
        #     ),
        #     cf_code_reward,
        # ),
        # "code_format": get_code_format_reward(language=script_args.code_language),
        "tag_count": tag_count_reward,
        "soft_overlong_punishment": get_soft_overlong_punishment(
            max_completion_len=script_args.max_completion_len,
            soft_punish_cache=script_args.soft_punish_cache,
        ),
    }
    reward_funcs = [REWARD_FUNCS_REGISTRY[func] for func in script_args.reward_funcs]

    return reward_funcs
# -*- coding: utf-8 -*-
# API配置
API_SECRET_KEY = "sk-cae49465bd51454ab5ead6933cdb19da"
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

import os
from datetime import datetime
from openai import OpenAI
from transformers import AutoTokenizer

# 可配置的模型名称 → 用于选择 tokenizer
TOKENIZER_MAP = {
    "deepseek-r1-distill-qwen-7b": "deepseek-ai/Deepseek-R1-Distill-Qwen-7B",
    "qwen2.5-math-7b-instruct": "Qwen/Qwen2.5-Math-7B-Instruct",
    "qwen2.5-7b-instruct": "Qwen/Qwen2.5-7B-Instruct"
}

def prompt_from_files(system_file="system_prompt.txt", user_file="user_prompt.txt", model_name="qwen2.5-math-7b-instruct"):
    # 读取提示词
    with open(system_file, 'r', encoding='utf-8') as f:
        system_content = f.read().strip()
    with open(user_file, 'r', encoding='utf-8') as f:
        user_content = f.read().strip()

    # 初始化客户端
    client = OpenAI(api_key=API_SECRET_KEY, base_url=BASE_URL)

    reasoning_content = ""
    answer_content = ""
    is_answering = False

    # 构造消息
    messages = []
    if system_content:
        messages.append({"role": "system", "content": system_content})
    messages.append({"role": "user", "content": user_content})

    try:
        print("=== 构造的消息 ===")
        print(f"系统角色: {system_content[:50]}..." if len(system_content) > 50 else f"系统角色: {system_content}")
        print(f"用户提示: {user_content[:50]}..." if len(user_content) > 50 else f"用户提示: {user_content}")
        print("\nAI思考中...")

        completion = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.7,
            stream=True
        )

        print("\n" + "=" * 20 + "思考过程" + "=" * 20 + "\n")

        for chunk in completion:
            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta
            delta_str = ""

            if hasattr(delta, 'reasoning_content') and delta.reasoning_content:
                delta_str = delta.reasoning_content
                reasoning_content += delta_str
            elif hasattr(delta, 'content'):
                delta_str = delta.content or ""
                answer_content += delta_str

                if not is_answering:
                    print("\n" + "=" * 20 + "完整回复" + "=" * 20 + "\n")
                    is_answering = True

            print(delta_str, end='', flush=True)

        # 获取 tokenizer 并统计 token 数
        tokenizer_model = TOKENIZER_MAP.get(model_name, None)
        if tokenizer_model is None:
            print(f"\n⚠️ 未知模型 {model_name}，无法加载 tokenizer 统计 token")
            prompt_tokens = completion_tokens = total_tokens = -1
        else:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_model, trust_remote_code=True)
            prompt_text = system_content + "\n" + user_content
            completion_text = reasoning_content + answer_content
            prompt_tokens = len(tokenizer.encode(prompt_text))
            completion_tokens = len(tokenizer.encode(completion_text))
            total_tokens = prompt_tokens + completion_tokens

            print(f"\n📊 Token 使用统计：")
            print(f"- Prompt tokens: {prompt_tokens}")
            print(f"- Completion tokens: {completion_tokens}")
            print(f"- Total tokens: {total_tokens}")

        # 保存文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"ai_response_{model_name}_{timestamp}.txt"
        with open(filename, "w", encoding="utf-8") as f:
            if reasoning_content:
                f.write("=" * 20 + "思考过程" + "=" * 20 + "\n")
                f.write(reasoning_content)
                f.write("\n\n")
            f.write("=" * 20 + "完整回复" + "=" * 20 + "\n")
            f.write(answer_content)
            f.write("\n\n" + "=" * 20 + "Token 使用统计" + "=" * 20 + "\n")
            f.write(f"Prompt tokens: {prompt_tokens}\n")
            f.write(f"Completion tokens: {completion_tokens}\n")
            f.write(f"Total tokens: {total_tokens}\n")

        print(f"\n✅ 回复已保存到 {filename}")

    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
if __name__ == "__main__":
    prompt_from_files(model_name="deepseek-r1-distill-qwen-7b")  # 可以修改为其他模型名称，如 model_name="deepseek-r1-distill-qwen-7b"
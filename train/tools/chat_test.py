import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel

# --- 1. 定义需要对比的两个模型路径 ---

# (1) 原始的基础模型路径
BASE_MODEL_PATH = "./qwen/Qwen2___5-7B-Instruct" 
TOKENIZER_PATH = BASE_MODEL_PATH # 分词器路径通常与基础模型一致

# (2) 您微调好的LoRA模型路径
# 请确保这个路径是包含 adapter_model.safetensors 的【文件夹路径】
LORA_MODEL_PATH = "./results_qwen_finetuned"

# --- 2. 准备加载模型 ---

pipes = {}
# 为两个模型在界面上起个清晰的名字
model_names = ["原始模型 (Base)", "我的CoT模型 (LoRA)"]

print(">>> 正在加载基础模型 (这是所有模型共用的)...")
# 先加载一次基础模型到GPU
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map={"": 0}, # 直接将模型加载到 GPU 0
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
print("✅ 基础模型加载完毕。")

# --- 3. 创建两个模型的推理管道(Pipeline) ---

# (1) 为【原始模型】创建推理管道
print(f">>> 正在为 '{model_names[0]}' 创建推理管道...")
pipes[model_names[0]] = pipeline(
    "text-generation",
    model=base_model,
    tokenizer=tokenizer
    # device 已在加载时通过 device_map 指定
)
print(f"✅ '{model_names[0]}' 的推理管道创建完毕。")


# (2) 为【您的LoRA模型】创建推理管道
print(f">>> 正在加载LoRA适配器: '{model_names[1]}'")
# 在基础模型之上加载LoRA适配器
lora_model = PeftModel.from_pretrained(base_model, LORA_MODEL_PATH)
print("✅ LoRA适配器加载完毕。")

print(f">>> 正在为 '{model_names[1]}' 创建推理管道...")
pipes[model_names[1]] = pipeline(
    "text-generation",
    model=lora_model, # 注意这里用的是加载了LoRA之后的新模型
    tokenizer=tokenizer
)
print(f"✅ '{model_names[1]}' 的推理管道创建完毕。")


# --- 4. Gradio 的后端函数和界面 ---

def generate_response(instruction, user_input):
    """根据输入，调用两个模型生成回复"""
    # 使用Qwen2的对话模板
    messages = [
        {"role": "system", "content": instruction if instruction else "You are a helpful assistant."},
        {"role": "user", "content": user_input},
    ]
    
    prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )

    outputs = []
    # 依次调用两个模型的 pipeline
    for name in model_names:
        print(f"--- 正在使用 '{name}' 生成回复 ---")
        result = pipes[name](prompt, max_new_tokens=4096, do_sample=True, temperature=0.3, top_p=0.9)
        # 从完整输出中提取助手的回复部分
        assistant_reply = result[0]["generated_text"].split("<|im_start|>assistant\n")[-1]
        outputs.append(assistant_reply)
    
    return tuple(outputs)


# 创建 Gradio 界面
demo = gr.Interface(
    fn=generate_response,
    inputs=[
        gr.Textbox(label="指令 (Instruction / System Prompt)", value="You are a helpful assistant."),
        gr.Textbox(label="您的输入 (Input / User Prompt)"),
    ],
    # 自动根据 model_names 列表创建对应数量和标签的输出框
    outputs=[gr.Textbox(label=name, lines=15) for name in model_names],
    title="Qwen2.5 7B 微调效果对比",
    description="对比原始的 Qwen2.5-7B-Instruct 模型和经过CoT数据微调后的LoRA模型的效果。"
)

if __name__ == "__main__":
    # share=True 会创建一个72小时有效的公开链接，方便分享
    demo.launch(share=True)
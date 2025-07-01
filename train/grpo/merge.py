# 合并LoRA适配器到基础模型
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

base_model_path = "../qwen/Qwen2___5-7B-Instruct"
lora_adapter_path = "../checkpoint-1405"
merged_model_path = "./merged-model"

print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.bfloat16, # 根据你的硬件选择合适的dtype
    device_map='cpu' # 先加载到CPU
)

print("Loading LoRA adapter...")
model_to_merge = PeftModel.from_pretrained(base_model, lora_adapter_path)

print("Merging model...")
merged_model = model_to_merge.merge_and_unload()

print(f"Saving merged model to {merged_model_path}...")
merged_model.save_pretrained(merged_model_path)

# 别忘了保存tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_path)
tokenizer.save_pretrained(merged_model_path)

print("Done!")
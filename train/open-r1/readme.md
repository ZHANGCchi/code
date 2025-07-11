用于设置 Hugging Face 镜像地址，否则无法搜索本地缓存路径，只能用本地目录的形式
export HF_ENDPOINT=https://hf-mirror.com

设置huggingface的缓存目录，否则自动下在/root/.cache里
export HF_HOME=./data_temp/

# sft命令
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file ./recipes/accelerate_configs/zero2.yaml ./src/open_r1/sft.py --config ./recipes/OpenR1-Distill-7B/sft/config_distill.yaml

# grpo命令
CUDA_VISIBLE_DEVICES=0 trl vllm-serve  --model ./merged-model --dtype auto  --gpu_memory_utilization 0.8
CUDA_VISIBLE_DEVICES=1 accelerate launch --config_file recipes/accelerate_configs/zero2.yaml src/open_r1/grpo.py --config recipes/DeepSeek-R1-Distill-Qwen-1.5B/grpo/config_demo.yaml
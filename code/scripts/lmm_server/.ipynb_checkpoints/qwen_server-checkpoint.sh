gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"
port=$((8000 + ${GPULIST[0]}))

vllm serve /data/yfli/code/DyFo_CVPR2025/model/Qwen2-VL-7B-Instruct \
    --port $port \
    --max-model-len 4096 \
    --trust-remote-code \
    --served-model-name Qwen/Qwen2-VL-7B-Instruct

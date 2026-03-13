gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"
port=$((8000 + ${GPULIST[0]}))

vllm serve  llava-hf/llava-1.5-7b-hf  --port $port --chat-template $(dirname "$0")/template_llava.jinja
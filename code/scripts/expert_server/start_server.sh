#!/bin/bash

# If CUDA_VISIBLE_DEVICES is not set, default to 0
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    CUDA_VISIBLE_DEVICES="0"
fi

# Get CUDA_VISIBLE_DEVICES environment variable
IFS=',' read -ra GPUS <<< "$CUDA_VISIBLE_DEVICES"

cd "$(dirname "$0")"

# Start a service for each GPU
for i in "${!GPUS[@]}"; do
    # Calculate port number
    PORT=$((8000 + ${GPUS[$i]}))
    echo "Starting server on port $PORT"
    # Set CUDA device and start service
    CUDA_VISIBLE_DEVICES=${GPUS[$i]} python model_service.py --port $PORT --max_batch_size 12 &
done

# Wait for all background processes to complete
wait
#!/bin/bash

if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    CUDA_VISIBLE_DEVICES="0"
fi

# Get CUDA_VISIBLE_DEVICES environment variable
IFS=',' read -ra GPUS <<< "$CUDA_VISIBLE_DEVICES"

cd "$(dirname "$0")"

# Start a service for each GPU
for i in "${!GPUS[@]}"; do
    # Calculate port number
    # standard 8000 +
    PORT=$((8200 + ${GPUS[$i]}))
    echo "Starting server on port $PORT"
    # Set CUDA device and start service
    CUDA_VISIBLE_DEVICES=${GPUS[$i]} python sam2_service.py --port $PORT &
done

# Wait for all background processes to complete
wait
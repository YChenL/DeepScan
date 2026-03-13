import os
import base64
import io

from openai import OpenAI, AsyncOpenAI
from PIL import Image
import math


# Configure OpenAI client load balancing
def get_openai_clients_and_models(model_name="Qwen/Qwen2-VL-7B-Instruct"):
    api_key = "EMPTY"
    gpu_ids = [int(x) for x in os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(',')]
    clients = []
    models = []
    
    # Create client and model for each GPU
    for gpu_id in gpu_ids:
        api_base = f"http://localhost:{8000 + gpu_id}/v1"
        client = AsyncOpenAI(
            api_key=api_key,
            base_url=api_base,
            timeout=3600
        )
        clients.append(client)
        models.append(model_name)
        # print(f"client {gpu_id} model {models[-1]}")
        
    return clients, models

# # Get all clients and models
# clients, models = get_openai_clients_and_models()

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def is_none(value):
    if value is None:
        return True
    if type(value) is float and math.isnan(value):
        return True
    if type(value) is str and value.lower() == 'nan':
        return True
    if type(value) is str and value.lower() == 'none':
        return True
    return False

def get_options(row, options):
    parsed_options = []
    for option in options:
        option_value = row[option]
        if is_none(option_value):
            break
        parsed_options.append(option_value)
    return parsed_options

def load_image_from_base64(base64_str):
    image_bytes = base64.b64decode(base64_str)
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    return image

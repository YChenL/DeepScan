import os
import asyncio, io, base64
from typing import Optional
from PIL import Image
import torch
from transformers import AutoProcessor
from transformers import Qwen2_5_VLForConditionalGeneration as QwenVL
from qwen_vl_utils import process_vision_info

def _to_dtype(s: str):
    if s == "bfloat16": return torch.bfloat16
    if s == "float16":  return torch.float16
    return "auto"

class QwenVLRuntime:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        attn_impl: Optional[str]  = None,
        dtype: str = "auto",
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        max_concurrency: int = 1,            
    ):
        self.model = QwenVL.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        if min_pixels and max_pixels:
            self.processor = AutoProcessor.from_pretrained(
                model_name, min_pixels=int(min_pixels), max_pixels=int(max_pixels)
            )
        else:
            self.processor = AutoProcessor.from_pretrained(model_name)

        self.sem = asyncio.Semaphore(max_concurrency)

    async def generate(self, prompt: str, image_b64: str,
                       max_tokens: int = 1024, temperature: float = 0.0) -> str:
    
        async with self.sem:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None, self._blocking_generate, prompt, image_b64, max_tokens, temperature
            )


    def _blocking_generate(self, prompt: str, image_b64: str,
                           max_tokens: int, temperature: float) -> str:
        
        messages = [
            {"role": "system", "content": "You are an advanced image understanding assistant. You will be given an image and a question about it."},
            {"role": "user", "content": [
                {"type": "image", "image": f"data:image;base64,{image_b64}"},
                {"type": "text",  "text": prompt},
            ],
        }]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)
        gen_kwargs = {
            "max_new_tokens": int(max_tokens),
            "do_sample": temperature > 0,
            "temperature": float(temperature) if temperature > 0 else 1.0,
        }

        with torch.inference_mode():
            out = self.model.generate(**inputs, **gen_kwargs)

        new_tokens = out[0, inputs.input_ids.shape[1]:]
        return self.processor.decode(
            new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )


# ----------- Lazy loading -----------
_singleton: Optional[QwenVLRuntime] = None
_singleton_lock = asyncio.Lock()

async def get_qwen_runtime(
    model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
    attn_impl: Optional[str] = None,
    dtype: str = "auto",
    min_pixels: Optional[int] = None,
    max_pixels: Optional[int] = None,
    max_concurrency: int = 1,
) -> QwenVLRuntime:
    
    global _singleton
    if _singleton is None:
        async with _singleton_lock:
            if _singleton is None:
                _singleton = QwenVLRuntime(
                    model_name=model_name,
                    attn_impl=attn_impl,
                    dtype=dtype,
                    min_pixels=min_pixels,
                    max_pixels=max_pixels,
                    max_concurrency=max_concurrency,
                )
    return _singleton
from abc import ABC, abstractmethod
from utils import get_options, get_openai_clients_and_models
from qwen_runtime import get_qwen_runtime

import shortuuid
import base64
import io, os, time
import asyncio 
from PIL import Image
from openai import AsyncOpenAI
from aiolimiter import AsyncLimiter


class QuestionSample(ABC):
    def __init__(self, row, args, round_idx=0, ):
        self.row = row
        self.args = args
        self.round_idx = round_idx
        self.image = row['image']  # base64
        self.options = get_options(row, ['A', 'B', 'C', 'D'])
        self.cur_option_char = ['A', 'B', 'C', 'D'][:len(self.options)]
        self.clients, self.models = get_openai_clients_and_models(self.args.model_path)
        self.current_client_idx = 0
        self.category = row['category']

    async def generate(self, prompt, image, max_tokens=1024):
        client = self.clients[self.current_client_idx]
        model = self.models[self.current_client_idx]
        self.current_client_idx = (self.current_client_idx + 1) % len(self.clients)
        image_bytes = base64.b64decode(image)
        img = Image.open(io.BytesIO(image_bytes))
        target_size = (self.args.image_size, self.args.image_size)

        if img.width > target_size[0] or img.height > target_size[1]:
            ratio = min(target_size[0]/img.width, target_size[1]/img.height)
            new_size = (int(img.width*ratio), int(img.height*ratio))
            img = img.resize(new_size, Image.Resampling.BILINEAR)
            buffered = io.BytesIO()
            img.save(buffered, format="JPEG")
            processed_image = base64.b64encode(buffered.getvalue()).decode()
        else:
            processed_image = image

        chat_completion = await client.chat.completions.create(
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "text", 
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{processed_image}"
                        }
                    }
                ]
            }],
            model=model,
            max_tokens=max_tokens,
            temperature=self.args.temperature if self.args.temperature > 0 else 0.0,
        )
        
        result = chat_completion.choices[0].message.content
        return result

    async def generate_local(self, prompt, image, max_tokens=1024):
        processed_image = image
        runtime = await get_qwen_runtime(
            model_name=self.args.model_path,                      
            attn_impl=("flash_attention_2" if getattr(self.args,"use_fa2",False) else None),
            dtype="auto",
            max_concurrency=getattr(self.args, "max_concurrency", 1),
        )
        temperature = self.args.temperature if self.args.temperature > 0 else 0.0
        return await runtime.generate(prompt, processed_image, max_tokens=max_tokens, temperature=temperature)

    async def process(self):
        try:
            return await self._process()
        except Exception as e:
            import traceback
            print(f"Error processing sample: {e}")
            print(f"Error stack:\n{traceback.format_exc()}")
            return {
                "question_id": self.row['index'],
                "round_id": self.round_idx,
                "prompt": "",
                "text": "A",  # Default return A
                "options": self.options,
                "option_char": self.cur_option_char,
                "answer_id": shortuuid.uuid(),
                "model_id": self.args.model_path,
                "answer": self.row['answer'],
                "metadata": {"error": str(e), "traceback": traceback.format_exc()}
            }

    @abstractmethod
    async def _process(self):
        """Abstract method that must be implemented by subclasses"""
        pass
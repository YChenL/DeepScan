from abc import ABC, abstractmethod
from utils import get_options, get_openai_clients_and_models
import shortuuid
import base64
import io
from PIL import Image


class QuestionSample(ABC):
    def __init__(self, row, args, round_idx=0, ):
        self.row = row
        self.args = args
        self.round_idx = round_idx
        self.image = row['image'] # base64
        self.options = get_options(row, ['A', 'B', 'C', 'D'])
        self.cur_option_char = ['A', 'B', 'C', 'D'][:len(self.options)]
        # Get clients and models list
        self.clients, self.models = get_openai_clients_and_models(self.args.model_path)
        # Counter for round-robin calls
        self.current_client_idx = 0
        
        # api 
        self.model = getattr(args, "qwen_model", "qwen2.5-vl-7b-instruct")
        base_url = getattr(args, "qwen_base_url",
                           "https://dashscope-intl.aliyuncs.com/compatible-mode/v1")
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise RuntimeError("Missing DASHSCOPE_API_KEY in environment.")

        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    
    async def generate(self, prompt, image, max_tokens=1024):
        # Round-robin client selection
        client = self.clients[self.current_client_idx]
        model = self.models[self.current_client_idx]
        # Update index for round-robin
        self.current_client_idx = (self.current_client_idx + 1) % len(self.clients)

        # Process image scaling
        image_bytes = base64.b64decode(image)
        img = Image.open(io.BytesIO(image_bytes))
        
        # Get target size from args
        target_size = (self.args.image_size, self.args.image_size)
        
        # Only scale if image is larger than target size
        if img.width > target_size[0] or img.height > target_size[1]:
            # Calculate scaling ratio
            ratio = min(target_size[0]/img.width, target_size[1]/img.height)
            new_size = (int(img.width*ratio), int(img.height*ratio))
            
            # Use bilinear interpolation for scaling
            img = img.resize(new_size, Image.Resampling.BILINEAR)
            
            # Convert back to base64
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


    async def generate_api(self, prompt: str, image_b64: str, max_tokens: int = 1024) -> str:
        """
        与你现有 generate() 行为一致：
        - 如图像大于 target_size，按比例缩放到不超过 (S,S)（双线性）
        - 以 base64 JPEG 走 OpenAI 兼容 chat.completions 的多模态消息
        - 返回 assistant 文本
        """
        # 1) 读取并按需缩放
        image_bytes = base64.b64decode(image_b64)
        img = Image.open(io.BytesIO(image_bytes))
        target = (self.args.image_size, self.args.image_size)

        if img.width > target[0] or img.height > target[1]:
            ratio = min(target[0] / img.width, target[1] / img.height)
            new_size = (max(1, int(img.width * ratio)),
                        max(1, int(img.height * ratio)))
            img = img.resize(new_size, Image.Resampling.BILINEAR)

            buf = io.BytesIO()
            img.save(buf, format="JPEG")
            processed_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        else:
            processed_b64 = image_b64  # 保持原图

        # 2) 组装多模态消息并请求
        # OpenAI 兼容：content 允许 text + image_url（可用 data: URL 传本地 base64）
        # 参考阿里云文档的示例写法。
        t = self.args.temperature if getattr(self.args, "temperature", 0) > 0 else 0.0
        resp = await self.client.chat.completions.create(
            model=self.model,                      # 例如 'qwen2.5-vl-7b-instruct'
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{processed_b64}"
                        }
                    }
                ]
            }],
            max_tokens=max_tokens,
            temperature=t,
            # 可选：透传 Qwen 特定参数（如高分辨率图像处理/惩罚项等）
            # extra_body={"vl_high_resolution_images": True, "repetition_penalty": 1.05}
        )

        return resp.choices[0].message.content


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
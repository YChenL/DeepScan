#!/usr/bin/env python3
import argparse
import base64
import io
import math

import cv2
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI
from PIL import Image
from pydantic import BaseModel
from torchvision import transforms
from transformers import BertTokenizer

from lavis.common.gradcam import getAttMap
from lavis.models import load_model_and_preprocess
from lavis.models.blip_models.blip import BlipBase
from lavis.models.blip_models.blip_image_text_matching import compute_gradcam


# ----------------------------- tokenizer monkey-patch ----------------------------- #
LOCAL_TOKENIZER_PATH = "replace/with/your/model_path"
original_init_tokenizer = BlipBase.init_tokenizer


@classmethod
def patched_init_tokenizer(cls):
    print(f"--- Loading tokenizer from local path: {LOCAL_TOKENIZER_PATH} ---")
    tokenizer = BertTokenizer.from_pretrained(LOCAL_TOKENIZER_PATH)
    tokenizer.add_special_tokens({"bos_token": "[DEC]"})
    tokenizer.add_special_tokens({"additional_special_tokens": ["[ENC]"]})
    return tokenizer


BlipBase.init_tokenizer = patched_init_tokenizer  # type: ignore[misc]


# ----------------------------- model & preprocessors ----------------------------- #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_itm, vis_processors, text_processors = load_model_and_preprocess(
    "blip_image_text_matching",
    "large",
    device=device,
    is_eval=True,
)
loader = transforms.Compose([transforms.ToTensor()])


# ------------------------------------- API -------------------------------------- #
app = FastAPI(title="Search Expert Service")


class AttentionRequest(BaseModel):
    image: str  # base64-encoded RGB image
    question: str
    block: int


# ----------------------------------- helpers ------------------------------------ #
def resize_to_multiple(img: Image.Image, block: int) -> Image.Image:
    """Resize PIL image so that width/height are integer multiples of `block`."""
    w, h = img.size
    new_w = math.ceil(w / block) * block
    new_h = math.ceil(h / block) * block
    return img.resize((new_w, new_h), Image.BICUBIC)


def genAttnMap(
    image: torch.Tensor,
    question: str,
    tensor_image: torch.Tensor,
    model: torch.nn.Module,
    tokenized_text: dict,
    raw_image: Image.Image,
) -> np.ndarray:
    """
    Compute Grad-CAM for BLIP-ITM and render an attention map aligned to the input.
    Returns a float numpy array in [0, 1] with the same spatial size as `raw_image`.
    """
    with torch.set_grad_enabled(True):
        gradcams, _ = compute_gradcam(
            model=model,
            visual_input=image,
            text_input=question,
            tokenized_text=tokenized_text,
            block_num=6,
        )

    # Each element in gradcams is (token, cam); we take the cam part and aggregate.
    cams = [cam_tuple[1] for cam_tuple in gradcams]
    cams_stacked = torch.stack(cams).reshape(image.size(0), -1)  # (B, N)
    cam_2d = cams_stacked.reshape(24, 24)  # BLIP-ITM large ViT feature grid

    resized = raw_image.resize((384, 384))
    norm_img = np.float32(resized) / 255.0
    attn_map = getAttMap(norm_img, cam_2d.detach().cpu().numpy(), blur=True, overlap=False)
    return attn_map  # HxW float array


def compute_attention(raw_image_b64: str, question: str) -> tuple[str, np.ndarray]:
    """
    Compute attention map for a single image and return:
      - image base64 (PNG)
      - attention map as a numpy array (float)
    """
    img_bytes = base64.b64decode(raw_image_b64)
    raw_image_pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    tensor_image = loader(raw_image_pil.resize((384, 384)))
    image = vis_processors["eval"](raw_image_pil).unsqueeze(0).to(device)
    question_tok = text_processors["eval"](question)
    tokenized_text = model_itm.tokenizer(
        question_tok, padding="longest", truncation=True, return_tensors="pt"
    ).to(device)

    heat = genAttnMap(image, question_tok, tensor_image, model_itm, tokenized_text, raw_image_pil)

    buf_img = io.BytesIO()
    raw_image_pil.save(buf_img, format="PNG")
    img_b64 = base64.b64encode(buf_img.getvalue()).decode("utf-8")

    return img_b64, heat


def compute_full_attention(raw_image_b64: str, question: str, block: int) -> tuple[str, np.ndarray]:
    """
    Tiled attention over the whole image:
      - Resize the image to (H, W) multiples of `block`.
      - For each (block x block) tile, compute attention at 384x384 then resize back.
      - Stitch tiles into a full-resolution attention map.

    Returns:
        resized_img_b64: base64-encoded PNG for the resized image
        full_heatmap: float32 heatmap aligned with the resized image (H, W)
    """
    img_bytes = base64.b64decode(raw_image_b64)
    raw_image_pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    resized_img = resize_to_multiple(raw_image_pil, block)
    W, H = resized_img.size
    n_cols, n_rows = W // block, H // block

    full_heatmap = np.zeros((H, W), dtype=np.float32)

    for r in range(n_rows):
        for c in range(n_cols):
            x0, y0 = c * block, r * block
            patch = resized_img.crop((x0, y0, x0 + block, y0 + block)).resize((384, 384), Image.BICUBIC)

            tensor_image = loader(patch)
            image = vis_processors["eval"](patch).unsqueeze(0).to(device)
            q_tensor = text_processors["eval"](question)
            tokenized_text = model_itm.tokenizer(
                q_tensor, padding="longest", truncation=True, return_tensors="pt"
            ).to(device)

            heat_patch = genAttnMap(image, q_tensor, tensor_image, model_itm, tokenized_text, patch)
            if not isinstance(heat_patch, np.ndarray):
                heat_patch = heat_patch.cpu().numpy()

            heat_patch = cv2.resize(heat_patch, (block, block), interpolation=cv2.INTER_LINEAR)
            full_heatmap[y0 : y0 + block, x0 : x0 + block] = heat_patch

    buf_resized_img = io.BytesIO()
    resized_img.save(buf_resized_img, format="PNG")
    resized_img_b64 = base64.b64encode(buf_resized_img.getvalue()).decode("utf-8")

    return resized_img_b64, full_heatmap


# ------------------------------------ routes ------------------------------------ #
parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, default=8100, help="Service port")
args = parser.parse_args()


@app.post("/attention_map")
async def attention_map(request: AttentionRequest):
    """
    Request body:
        {
            "image": "<base64 PNG/JPEG RGB>",
            "question": "<text>",
            "block": <int>
        }
    Response:
        {
            "resized_img": "<base64 PNG>",
            "heatmap": {
                "data_b64": "<base64 raw bytes>",
                "shape": [H, W],
                "dtype": "float32"
            }
        }
    """
    resized_img_b64, heatmap_np = compute_full_attention(request.image, request.question, request.block)
    heatmap_b64 = base64.b64encode(heatmap_np.tobytes()).decode("utf-8")

    return {
        "resized_img": resized_img_b64,
        "heatmap": {"data_b64": heatmap_b64, "shape": heatmap_np.shape, "dtype": str(heatmap_np.dtype)},
    }


if __name__ == "__main__":
    uvicorn.run(
        "blip_service:app",
        host="0.0.0.0",
        port=args.port,
        limit_concurrency=10000,
        backlog=10000,
        log_level="debug",
    )

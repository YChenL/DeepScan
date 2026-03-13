import base64, io, requests, cv2
import numpy as np
from PIL import Image
from typing import List, Tuple


def get_heatmap(raw_image: str, 
                question: str,
                endpoint: str = "http://localhost:8101/attention_map",
                block: int = 786):

    # prepare for JSON request
    payload = {
        "image": raw_image,
        "question": question,
        "block": block
    }
    resp = requests.post(
        endpoint,
        json=payload,
        timeout=60,
    )

    resp.raise_for_status()
    data = resp.json()
    resized_img_b64 = data["resized_img"]
    heatmap_info = data["heatmap"]
    decoded_bytes = base64.b64decode(heatmap_info["data_b64"])
    heatmap_np = np.frombuffer(
        decoded_bytes,
        dtype=np.dtype(heatmap_info["dtype"])
    ).reshape(heatmap_info["shape"])

    return resized_img_b64, heatmap_np


def get_mask_point(
    image_b64: str,
    positive_point: Tuple[int, int],
    endpoint: str = "http://127.0.0.1:8201/sam2/point_predict"
) -> np.ndarray:
   
    # prepare for JSON request
    payload = {
        "image_b64": image_b64,
        "pos": [list(positive_point)],
        "neg": None,
        "multimask": False
    }

    try:
        resp = requests.post(
            endpoint,
            json=payload,
            timeout=300, 
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.RequestException as e:
        print(f"request error: {e}")
        raise

    mask_info = data["mask"]
    decoded_bytes = base64.b64decode(mask_info["data_b64"])
    mask_np = np.frombuffer(
        decoded_bytes,
        dtype=np.dtype(mask_info["dtype"])
    ).reshape(mask_info["shape"])


    kernel = np.ones((3, 3), np.uint8)
    mask_dilated = cv2.dilate(mask_np, kernel, iterations=40)

    # ys, xs = np.where(mask_dilated > 0)
    # if len(xs) == 0:                       # 若掩码为空，直接返回全零
    #     return np.zeros_like(mask_dilated, dtype=np.uint8)

    # x1, x2 = xs.min(), xs.max()
    # y1, y2 = ys.min(), ys.max()

    # bbox_mask = np.zeros_like(mask_dilated, dtype=np.uint8)
    # bbox_mask[y1:y2 + 1, x1:x2 + 1] = 255  # +1 因为切片上界是开区间

    return mask_dilated   


def get_mask_group(
    image_b64: str,
    positive_points: List[Tuple[int, int]],
    endpoint: str = "http://127.0.0.1:8201/sam2/point_predict"
) -> np.ndarray:
    
    # prepare for JSON request
    payload = {
        "image_b64": image_b64,
        "pos": positive_points,
        "neg": None,
        "multimask": False
    }

    try:
        resp = requests.post(
            endpoint,
            json=payload,
            timeout=300,  
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.RequestException as e:
        print(f"request error: {e}")
        raise

    mask_info = data["mask"]
    decoded_bytes = base64.b64decode(mask_info["data_b64"])
    mask_np = np.frombuffer(
        decoded_bytes,
        dtype=np.dtype(mask_info["dtype"])
    ).reshape(mask_info["shape"])

    return mask_np
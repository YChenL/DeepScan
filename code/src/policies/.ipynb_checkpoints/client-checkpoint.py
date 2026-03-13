import base64, io, requests, cv2
import numpy as np
from PIL import Image
from typing import List, Tuple


def get_heatmap(raw_image: str, question: str,
                endpoint: str = "http://localhost:8100/attention_map"):
    """
    修改: 与服务器端的JSON API进行通信。

    Args:
        raw_image (str): 输入图像的Base64编码字符串。
        question (str): 相关的问题文本。
        endpoint (str): 服务器API的地址。

    Returns:
        tuple[str, np.ndarray]:
            - resized_img (str): 服务器返回的、经过尺寸调整后的图像的Base64字符串。
            - heat (np.ndarray): 与resized_img同样大小的单通道热力图Numpy数组。
    """
    # 1. 准备JSON请求体
    payload = {
        "image": raw_image,
        "question": question
    }

    # 2. 发送POST请求，数据为JSON格式
    resp = requests.post(
        endpoint,
        json=payload, # 使用 json=... 来发送 application/json 请求
        timeout=60,
    )
    # 如果服务器返回错误状态码 (如 4xx, 5xx), 将会抛出异常
    resp.raise_for_status()
    data = resp.json()

    # 3. 解析服务器返回的数据
    # resized_img已经是Base64字符串，直接获取
    resized_img_b64 = data["resized_img"]
    # heatmap是Base64字符串，需要解码并转换为Numpy数组
    heatmap_info = data["heatmap"]
    # 解码 Base64 得到原始字节
    decoded_bytes = base64.b64decode(heatmap_info["data_b64"])
    # 使用 np.frombuffer 从字节和元数据重建数组
    heatmap_np = np.frombuffer(
        decoded_bytes,
        dtype=np.dtype(heatmap_info["dtype"])
    ).reshape(heatmap_info["shape"])

    return resized_img_b64, heatmap_np


def get_mask_point(
    image_b64: str,
    positive_point: Tuple[int, int],
    endpoint: str = "http://127.0.0.1:8000/sam2/point_predict"
) -> np.ndarray:
    """
    通过调用 SAM-2 服务 API 获取分割掩码。

    Args:
        image_b64 (str): 输入图像的 Base64 编码字符串。
        positive_point (Tuple[int, int]): 一个表示正向提示点的 (x, y) 坐标元组。
        endpoint (str): SAM-2 服务 API 的地址。

    Returns:
        np.ndarray: 解码后的单通道分割掩码 NumPy 数组。
    """
    # 1. 准备 JSON 请求体
    #    服务器需要一个坐标点列表，所以我们将单个元组转换为 [[x, y]] 的格式。
    payload = {
        "image_b64": image_b64,
        "pos": [list(positive_point)],
        "neg": None,
        "multimask": False
    }

    # 2. 发送 POST 请求
    try:
        resp = requests.post(
            endpoint,
            json=payload,
            timeout=300,  # 设置一个较长的超时时间，以防模型推理过慢
        )
        # 如果服务器返回错误状态码 (如 4xx, 5xx), 将会抛出 HTTPError 异常
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.RequestException as e:
        print(f"❌ 请求 API 时发生错误: {e}")
        raise

    # 3. 解析并解码服务器返回的掩码数据
    mask_info = data["mask"]
    
    # 从 Base64 字符串解码为原始字节
    decoded_bytes = base64.b64decode(mask_info["data_b64"])
    
    # 使用 np.frombuffer 从字节流和元数据（形状、数据类型）重建 NumPy 数组
    mask_np = np.frombuffer(
        decoded_bytes,
        dtype=np.dtype(mask_info["dtype"])
    ).reshape(mask_info["shape"])

    # 4. 20 px 膨胀
    #    先转成 0/1 uint8，再用 3×3 核连续迭代 20 次 (≈ 膨胀 20 px)
    kernel = np.ones((3, 3), np.uint8)
    mask_dilated = cv2.dilate(mask_np, kernel, iterations=40)

    # 5. 计算外接矩形 ----------
    # ys, xs = np.where(mask_dilated > 0)
    # if len(xs) == 0:                       # 若掩码为空，直接返回全零
    #     return np.zeros_like(mask_dilated, dtype=np.uint8)

    # x1, x2 = xs.min(), xs.max()
    # y1, y2 = ys.min(), ys.max()

    # # 6. 构造“矩形内=255”掩码 ----------
    # bbox_mask = np.zeros_like(mask_dilated, dtype=np.uint8)
    # bbox_mask[y1:y2 + 1, x1:x2 + 1] = 255  # +1 因为切片上界是开区间

    return mask_dilated   


def get_mask_group(
    image_b64: str,
    positive_points: List[Tuple[int, int]],
    endpoint: str = "http://127.0.0.1:8000/sam2/point_predict"
) -> np.ndarray:
    """
    通过调用 SAM-2 服务 API 获取分割掩码。

    Args:
        image_b64 (str): 输入图像的 Base64 编码字符串。
        positive_point List[Tuple[int, int]]: 一个包含多个 (x, y) 坐标元组的列表，作为正向提示点。
        endpoint (str): SAM-2 服务 API 的地址。

    Returns:
        np.ndarray: 解码后的单通道分割掩码 NumPy 数组。
    """
    # 1. 准备 JSON 请求体
    #    服务器需要一个坐标点列表，所以我们将单个元组转换为 [[x, y]] 的格式。
    payload = {
        "image_b64": image_b64,
        "pos": positive_points,
        "neg": None,
        "multimask": False
    }

    # 2. 发送 POST 请求
    try:
        resp = requests.post(
            endpoint,
            json=payload,
            timeout=300,  # 设置一个较长的超时时间，以防模型推理过慢
        )
        # 如果服务器返回错误状态码 (如 4xx, 5xx), 将会抛出 HTTPError 异常
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.RequestException as e:
        print(f"❌ 请求 API 时发生错误: {e}")
        raise

    # 3. 解析并解码服务器返回的掩码数据
    mask_info = data["mask"]
    
    # 从 Base64 字符串解码为原始字节
    decoded_bytes = base64.b64decode(mask_info["data_b64"])
    
    # 使用 np.frombuffer 从字节流和元数据（形状、数据类型）重建 NumPy 数组
    mask_np = np.frombuffer(
        decoded_bytes,
        dtype=np.dtype(mask_info["dtype"])
    ).reshape(mask_info["shape"])

    return mask_np
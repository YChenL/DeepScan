import base64, math, io, cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from typing import List, Tuple, Dict, Any
from .client import get_heatmap, get_mask_point, get_mask_group
from .control_point_sam import filter_heatmap_and_find_centroids, visualize_highlighted_regions, filter_heatmap_and_find_control_points, cluster_centroids_for_prompts


def filter_points_by_mask(
    points_to_filter: List[Tuple[int, int]], 
    existing_mask: np.ndarray
) -> List[Tuple[int, int]]:
    """
    使用一个已有的掩码来过滤控制点列表，返回所有不在掩码区域内的点。
    """
    kept_points = []
    if existing_mask is None or existing_mask.size == 0:
        return points_to_filter

    for point in points_to_filter:
        x, y = point
        h, w = existing_mask.shape[:2]
        # 核心检查逻辑: NumPy索引是[y, x]
        # 假设掩码中非零值为物体区域
        if 0 <= y < h and 0 <= x < w and existing_mask[y, x] == 0:
            kept_points.append(point)
            
    return kept_points


def filter_groups_by_mask(
    groups_to_filter: List[List[Tuple[int, int]]],
    existing_mask: np.ndarray
) -> List[List[Tuple[int, int]]]:
    """
    使用一个已有的掩码来过滤一个点组列表。

    如果一个点组的代表点（我们约定为列表中的第一个点，即质心）
    位于掩码内，则整个组被过滤掉。

    Args:
        groups_to_filter (List[List[Tuple]]): 待过滤的点组列表。
        existing_mask (np.ndarray): 用于过滤的二值掩码。

    Returns:
        List[List[Tuple]]: 过滤后剩下的点组列表。
    """
    kept_groups = []
    if existing_mask is None or existing_mask.size == 0:
        return groups_to_filter

    for group in groups_to_filter:
        if not group:  # 跳过空组
            continue
        
        # 使用组内的第一个点（即质心）作为代表点进行检查
        centroid = group[0]
        x, y = centroid
        h, w = existing_mask.shape[:2]
        # 如果质心在掩码之外，则保留整个组
        if 0 <= y < h and 0 <= x < w and existing_mask[y, x] == 0:
            kept_groups.append(group)

    return kept_groups


def calculate_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """计算两个掩码的交并比 (IoU)。"""
    # 将掩码转换为布尔类型以进行逻辑运算
    mask1_bool = mask1.astype(bool)
    mask2_bool = mask2.astype(bool)
    
    # 计算交集和并集
    intersection = np.logical_and(mask1_bool, mask2_bool).sum()
    union = np.logical_or(mask1_bool, mask2_bool).sum()
    
    # 避免除以零
    if union == 0:
        return 0.0
        
    return intersection / union


def get_bbox_from_mask_numpy(mask):
    """
    使用 NumPy 从二值蒙版中提取边界框。
    
    参数:
        mask (np.ndarray): 一个 2D 的 NumPy 数组 (0 或 255)。

    返回:
        tuple or None: 一个包含 (x_min, y_min, x_max, y_max) 的元组，
                       如果蒙版为空则返回 None。
    """
    # 找到所有非零（即值为255）像素的行和列索引
    rows, cols = np.where(mask > 0)
    
    # 如果没有找到任何非零像素，说明蒙版是空的
    if rows.size == 0 or cols.size == 0:
        return None
        
    # 计算边界
    x_min = np.min(cols)
    y_min = np.min(rows)
    x_max = np.max(cols)
    y_max = np.max(rows)
    
    return (int(x_min), int(y_min), int(x_max), int(y_max))


def merge_overlapping_masks(
    found_objects: List[Dict[str, Any]], 
    iou_threshold: float = 0.75
) -> List[Dict[str, Any]]:
    """
    合并 found_objects 列表中高度重叠的掩码。

    Args:
        found_objects (List[Dict[str, Any]]): 原始分割结果列表。
        iou_threshold (float): IoU阈值，超过此值则合并。

    Returns:
        List[Dict[str, Any]]: 合并后的、去重的结果列表。
    """
    if not found_objects:
        return []

    # 创建一个可修改的副本进行处理
    objects_to_process = found_objects.copy()
    merged_objects = []
    
    # 当还有对象未被分配到合并组时，持续循环
    while objects_to_process:
        # 1. 取出第一个对象作为当前合并组的“基础”
        base_object = objects_to_process.pop(0)
        base_mask = base_object['mask']
        
        i = 0
        # 2. 遍历剩余的待处理对象，看是否能合并到当前组
        while i < len(objects_to_process):
            other_object = objects_to_process[i]
            other_mask = other_object['mask']
            
            # 3. 计算IoU，判断是否需要合并
            iou = calculate_iou(base_mask, other_mask)
            
            if iou >= iou_threshold:
                # print(f"[*] 合并对象: IoU={iou:.2f} >= {iou_threshold}。")
                # 合并掩码 (使用逻辑或)
                base_mask = np.logical_or(base_mask, other_mask)
                
                # 从待处理列表中移除已被合并的对象
                objects_to_process.pop(i)
                # 注意：不增加 i，因为列表缩短了，新的元素移动到了当前索引
            else:
                # 如果不合并，则继续检查下一个
                i += 1
        
        # 4. (可选但推荐) 对合并后的大掩码进行形态学闭操作，填充内部小洞
        kernel = np.ones((5, 5), np.uint8)
        final_merged_mask = cv2.morphologyEx((base_mask * 255).astype(np.uint8), cv2.MORPH_CLOSE, kernel)

        # 5. 将合并后的最终对象存入结果列表
        base_object['mask'] = final_merged_mask
        merged_objects.append(base_object)
        
    return merged_objects


def iterative_segmentation_from_heatmap(
    initial_points: List[Tuple], 
    image_b64: str, # 确认 get_mask 直接接收这个 base64 字符串
    sam_endpoint: str = "http://127.0.0.1:8201/sam2/point_predict"
) -> List[Dict[str, Any]]:
    """
    通过迭代调用SAM服务和过滤控制点来分割所有物体。
    此版本直接将 base64 图像字符串传递给 get_mask 函数。
    """
    found_objects = []
    iteration_count = 0
    
    # 当还有控制点需要处理时，持续循环
    while initial_points:
        iteration_count += 1
        # print(f"\n--- [迭代 {iteration_count}] 开始 ---")
        
        # 从列表中取出一个点进行处理
        current_point = initial_points.pop(0)
        # print(f"[*] 正在处理控制点: {current_point}, 剩余 {len(initial_points)} 个待定点。")
        
        # =============================================================
        # 核心修正如下:
        # 1. 直接传递 base64 字符串 `image_b64`
        # 2. 使用动态的 `current_point`
        # =============================================================
        mask_np = get_mask_point(image_b64, current_point, endpoint=sam_endpoint)
        if mask_np is None or mask_np.size == 0 or np.all(mask_np == 0):
            print(f"[!] 警告: 点 {current_point} 未能生成有效掩码（或为空掩码），跳过。")
            continue
            
        # 保存本次找到的物体
        found_objects.append({
            "mask": mask_np,
            "source_point": current_point
        })
        
        # 使用新生成的掩码，过滤掉剩余的控制点
        initial_points = filter_points_by_mask(initial_points, mask_np)
        # print(f"    -> 过滤完成。现在还剩 {len(initial_points)} 个控制点。")

    merged_objects = merge_overlapping_masks(found_objects, iou_threshold=0.3)
    # 构造“矩形内=255”掩码 
    for obj in merged_objects:
        mask = obj["mask"]
        ys, xs = np.where(mask > 0)
        x1, x2, y1, y2 = xs.min(), xs.max(), ys.min(), ys.max()
        bbox_mask = np.zeros_like(mask, dtype=np.uint8)
        bbox_mask[y1:y2 + 1, x1:x2 + 1] = 255
        obj["mask"] = bbox_mask
        obj["area"] = np.count_nonzero(bbox_mask)   # 记录面积供排序

    # —— 按面积升序排序 ——
    merged_objects.sort(key=lambda o: o["area"])

    # 不再需要面积字段的话，随手删除
    for obj in merged_objects:
        obj.pop("area", None)

    # print(f"\n--- 所有控制点处理完毕！共找到 {len(merged_objects)} 个独立物体。 ---")
    return merged_objects


def grounding(img: str, #base64str
              question: str,
              BLOCK: int):

    ## Image —> base64
    # buffer = io.BytesIO()
    # if img.mode != 'RGB':
    #     img = img.convert('RGB')
    # img.save(buffer, format="PNG")
    # img_bytes = buffer.getvalue()
    # base64_str = base64.b64encode(img_bytes).decode('utf-8')

    # gen heatmap
    resized_img, heatmap = get_heatmap(img, question, endpoint = "http://localhost:8101/attention_map", block=BLOCK)
    # get control points
    points, raw_binary, final_mask = filter_heatmap_and_find_centroids(heatmap)
    # get objects
    found_objects = iterative_segmentation_from_heatmap(points, resized_img)

    # base64 ——> np.ndarray
    pil_image = Image.open(io.BytesIO(base64.b64decode(resized_img)))
    resized_width, resized_height = pil_image.size
    # img_rgb   = np.array(pil_image)            # RGB, uint8
    # 循环处理每个找到的物体
    for obj in found_objects:
        mask = obj["mask"]  # 假设 mask 是一个 0/255 的灰度图
        # 提取蒙板bbox
        bbox = get_bbox_from_mask_numpy(mask) 
        x_min, y_min, x_max, y_max = bbox
        # 增加一个额外的安全检查，防止 crop 坐标无效
        if x_min >= x_max or y_min >= y_max:
            continue

        obj['bbox'] = bbox
        # 使用正确的坐标系 (+1) 来进行裁剪
        cropped_img = pil_image.crop((x_min, y_min, x_max + 1, y_max + 1))
        # img ——> base64str
        buffered = io.BytesIO()
        cropped_img.save(buffered, format="PNG")
        cropped_img_base64 = base64.b64encode(buffered.getvalue()).decode()
        obj['crop_img'] = cropped_img_base64

    return resized_img, resized_width, resized_height, found_objects
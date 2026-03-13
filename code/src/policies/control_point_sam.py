import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any
from sklearn.cluster import DBSCAN # <-- 导入DBSCAN

def sample_points_from_polygon(polygon: np.ndarray, num_points: int) -> List[Tuple[int, int]]:
    if num_points <= 0:
        return []

    # 计算多边形的总周长
    perimeter = cv2.arcLength(polygon, closed=True)
    
    # 如果周长为0或非常小，直接返回空列表
    if perimeter < 1.0:
        return []

    sampled_points = []
    distance_interval = perimeter / num_points # 计算每个采样点之间的距离
    
    current_distance = 0.0
    for i in range(len(polygon)):
        p1 = polygon[i][0]
        p2 = polygon[(i + 1) % len(polygon)][0]
        
        edge_vector = p2 - p1
        edge_length = np.linalg.norm(edge_vector)
        
        if edge_length == 0:
            continue
            
        # 在当前边上进行采样
        while current_distance <= edge_length:
            # 计算当前采样点在边上的比例
            ratio = current_distance / edge_length
            # 通过线性插值计算采样点坐标
            sp_x = int(p1[0] + ratio * edge_vector[0])
            sp_y = int(p1[1] + ratio * edge_vector[1])
            sampled_points.append((sp_x, sp_y))

            if len(sampled_points) == num_points:
                return sampled_points
            
            # 移动到下一个采样点的位置
            current_distance += distance_interval
        
        # 减去已走过的边长，为下一条边做准备
        current_distance -= edge_length
        
    return sampled_points


# ==============================================================================
# 步骤 1: 热力图滤波与质心提取
# ==============================================================================
def filter_heatmap_and_find_centroids(
    heatmap_np: np.ndarray, 
    min_area_threshold: int = 50
) -> tuple[list[tuple[int, int]], np.ndarray, np.ndarray]:
    """
    通过自适应阈值对热力图进行滤波，并找到高亮区域的质心。

    Args:
        heatmap_np (np.ndarray): 输入的单通道float32热力图。
        min_area_threshold (int): 忽略的最小连通区域面积，用于过滤噪点。

    Returns:
        tuple[list[tuple[int, int]], np.ndarray, np.ndarray]:
            - control_points (list): 包含所有质心坐标 (x, y) 的列表。
            - binary_mask (np.ndarray): Otsu阈值法生成的原始二值图。
            - filtered_mask (np.ndarray): **[新增]** 仅包含通过面积阈值过滤后的高亮区域的二值图。
    """
    if heatmap_np.dtype != np.float32:
        heatmap_np = heatmap_np.astype(np.float32)

    # 1. 归一化到 0-255 并转为 8-bit 整型
    heatmap_norm = cv2.normalize(heatmap_np, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # 2. 应用Otsu自适应阈值法
    threshold_value, binary_mask = cv2.threshold(
        heatmap_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    # print(f"[*] 自适应阈值 (Otsu) 计算值为: {threshold_value}")

    # 3. 寻找连通区域
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary_mask, connectivity=8, ltype=cv2.CV_32S
    )

    control_points = []
    # 新增: 创建一个空白图像来存储过滤后的高亮区域
    filtered_mask = np.zeros_like(binary_mask)

    # 从索引1开始循环，因为索引0是背景
    for i in range(1, num_labels):
        # 过滤掉面积过小的噪点区域
        if stats[i, cv2.CC_STAT_AREA] >= min_area_threshold:
            # 提取质心
            center_x, center_y = centroids[i]
            control_points.append((int(center_x), int(center_y)))
            
            # 新增: 将通过筛选的区域绘制到 filtered_mask 上
            # `labels == i` 会创建一个布尔掩码，标记出当前连通区域的所有像素
            filtered_mask[labels == i] = 255
            
    # print(f"[*] 从热力图中提取到 {len(control_points)} 个初始控制点。")
    # 这里binary_mask是带噪声的,未被过滤的mask区域; 对小于threshold的binary_mask进行过滤后得到filtered_mask;
    # 这里可以补充消融实验, 来看看不同的min_area_threshold对算法性能的影响; 观察两个变量的变化进行可视化
    return control_points, binary_mask, filtered_mask


def filter_heatmap_and_find_control_points(
    heatmap_np: np.ndarray, 
    min_area_threshold: int = 50,
    num_boundary_points: int = 4  # <-- 新增的超参数
) -> Tuple[List[List[Tuple[int, int]]], np.ndarray, np.ndarray]:
    """
    通过自适应阈值对热力图进行滤波，并为每个高亮区域找到一组控制点
    (质心 + 边界点)。

    Args:
        heatmap_np (np.ndarray): 输入的单通道float32热力图。
        min_area_threshold (int): 忽略的最小连通区域面积。
        num_boundary_points (int): 要在每个区域凸包边界上采样的点的数量。

    Returns:
        tuple:
            - control_points_groups (List[List[Tuple]]): 控制点组的列表。
              每个内部列表包含一个区域的质心和边界点。
            - binary_mask (np.ndarray): 原始二值图。
            - filtered_mask (np.ndarray): 过滤后的高亮区域图。
    """
    # ... (步骤 1 和 2 保持不变) ...
    if heatmap_np.dtype != np.float32:
        heatmap_np = heatmap_np.astype(np.float32)
    heatmap_norm = cv2.normalize(heatmap_np, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    threshold_value, binary_mask = cv2.threshold(
        heatmap_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary_mask, connectivity=8, ltype=cv2.CV_32S
    )

    control_points_groups = []
    filtered_mask = np.zeros_like(binary_mask)

    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area_threshold:
            # --- 这是主要的修改区域 ---

            # 1. 提取质心 (和以前一样)
            center_x, center_y = centroids[i]
            centroid_point = (int(center_x), int(center_y))

            # 2. 找到当前连通分量的所有像素点坐标
            # np.where返回(row, col)，需要堆叠并反转为(x, y)格式
            component_points = np.column_stack(np.where(labels == i)[::-1])
            
            # 如果点数太少，无法形成多边形，则只使用质心
            if len(component_points) < 3:
                control_points_groups.append([centroid_point])
                filtered_mask[labels == i] = 255
                continue

            # 3. 计算这些点的凸包 (Convex Hull)
            hull = cv2.convexHull(component_points)
            
            # 4. 从凸包边界采样点
            boundary_points = sample_points_from_polygon(hull, num_boundary_points)
            
            # 5. 将质心和边界点合并为一组控制点
            # 将质心放在列表的第一个位置
            prompt_group = [centroid_point] + boundary_points
            control_points_groups.append(prompt_group)

            # 更新 filtered_mask (和以前一样)
            filtered_mask[labels == i] = 255
            
    print(f"[*] 从热力图中提取到 {len(control_points_groups)} 组控制点。")
    return control_points_groups, binary_mask, filtered_mask


# ===============================
def cluster_centroids_for_prompts(
    heatmap_np: np.ndarray,
    min_area_threshold: int = 50,
    distance_threshold: int = 100  # <-- 新超参数: 聚类的最大距离 (像素)
) -> Tuple[List[List[Tuple[int, int]]], np.ndarray]:
    """
    先找到所有高亮区域的质心，然后使用DBSCAN对这些质心进行聚类，
    将每个簇作为一组独立的控制点。

    Args:
        heatmap_np (np.ndarray): 输入的单通道float32热力图。
        min_area_threshold (int): 忽略的最小连通区域面积。
        distance_threshold (int): DBSCAN算法的eps参数，即样本被视为相邻的最大距离。

    Returns:
        tuple:
            - prompt_groups (List[List[Tuple]]): 控制点组的列表。
              每个内部列表是一个由邻近质心组成的簇。
            - filtered_mask (np.ndarray): 包含所有有效高亮区域的掩码，用于可视化。
    """
    # 1. 提取所有满足条件的质心 (与最初版本类似)
    if heatmap_np.dtype != np.float32:
        heatmap_np = heatmap_np.astype(np.float32)
    heatmap_norm = cv2.normalize(heatmap_np, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    _, binary_mask = cv2.threshold(heatmap_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary_mask, connectivity=8, ltype=cv2.CV_32S
    )

    all_valid_centroids = []
    filtered_mask = np.zeros_like(binary_mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area_threshold:
            all_valid_centroids.append(centroids[i])
            filtered_mask[labels == i] = 255
    
    if not all_valid_centroids:
        print("[!] 未找到任何有效的质心。")
        return [], filtered_mask

    print(f"[*] 步骤1: 从热力图中提取到 {len(all_valid_centroids)} 个初始质心。")

    # 2. 使用DBSCAN对所有质心进行聚类
    # min_samples=1 确保即使是离群点也会自成一簇，符合您的要求。
    db = DBSCAN(eps=distance_threshold, min_samples=1).fit(all_valid_centroids)
    cluster_labels = db.labels_

    # 3. 根据聚类结果，将质心分组
    num_clusters = len(set(cluster_labels))
    print(f"[*] 步骤2: 将质心聚类成 {num_clusters} 个点组。")

    grouped_points = {}
    for i, label in enumerate(cluster_labels):
        if label not in grouped_points:
            grouped_points[label] = []
        # 将坐标转换为整数元组
        point = tuple(int(coord) for coord in all_valid_centroids[i])
        grouped_points[label].append(point)

    # 4. 将分组后的结果转换为所需的列表格式
    prompt_groups = list(grouped_points.values())
    
    return prompt_groups, filtered_mask


# ==============================================================================
# 步骤 2: SAM模型抽象
# ==============================================================================

def get_mask_and_label_from_sam(image: np.ndarray, point: tuple[int, int]) -> tuple[np.ndarray, str]:
    """
    [抽象函数] 模拟SAM模型的行为。
    
    在您的实际应用中，您需要在此处调用真实的SAM模型。
    输入一个控制点，SAM会返回它认为该点所属物体的掩码和可能的标签。

    Args:
        image (np.ndarray): BGR格式的图像。
        point (tuple[int, int]): 一个 (x, y) 格式的控制点。

    Returns:
        tuple[np.ndarray, str]:
            - mask: 与输入图像同尺寸的二值掩码 (0或255), uint8类型。
            - label: 对分割出物体的描述，这里用坐标代替。
    """
    print(f"    - [SAM Mock] 正在处理点 {point}...")
    
    # --- MOCK IMPLEMENTATION ---
    # 为了演示，我们在这里创建一个以控制点为中心、半径为80的圆形作为模拟掩码
    h, w, _ = image.shape
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, center=point, radius=80, color=255, thickness=-1)
    
    label = f"object_at_{point[0]}_{point[1]}"
    # --- END MOCK ---
    
    return mask, label


# ==============================================================================
# 步骤 3: 迭代式控制点过滤与对象提取
# ==============================================================================

def iterative_object_extraction(image: np.ndarray, control_points: list[tuple[int, int]]) -> list[dict]:
    """
    通过迭代循环，处理控制点并过滤已被分割区域中的点。

    Args:
        image (np.ndarray): 用于SAM分割的原始图像 (BGR格式)。
        control_points (list[tuple[int, int]]): 初始控制点列表。

    Returns:
        list[dict]: 包含所有分割出的物体的列表。
                    每个dict包含: {'mask': np.ndarray, 'label': str, 'source_point': tuple}
    """
    unprocessed_points = list(control_points) # 创建一个可修改的副本
    found_objects = []

    loop_count = 0
    while unprocessed_points:
        loop_count += 1
        print(f"\n--- 开始第 {loop_count} 轮迭代 ---")
        
        # 1. 从列表中取出一个控制点进行处理
        current_point = unprocessed_points.pop(0)
        print(f"[*] 正在处理控制点: {current_point}, 剩余 {len(unprocessed_points)} 个点。")

        # 2. 调用SAM模型获取掩码和标签
        mask, label = get_mask_and_label_from_sam(image, current_point)
        
        # 3. 将找到的物体存入结果列表
        found_objects.append({
            "mask": mask,
            "label": label,
            "source_point": current_point
        })

        # 4. **核心逻辑**: 利用新生成的掩码来过滤掉剩余的控制点
        points_to_keep = []
        for point in unprocessed_points:
            # 检查点 (x, y) 是否在掩码 (值为255) 的区域内
            # 注意numpy数组索引是 (row, col) 即 (y, x)
            y, x = point[1], point[0]
            if mask[y, x] == 0:
                # 如果掩码在该点的值为0，意味着这个点不在刚被分割的物体内，予以保留
                points_to_keep.append(point)
            else:
                # 否则，该点被新掩码“吞噬”，予以过滤
                print(f"    - [过滤] 点 {point} 已被 {label} 的掩码覆盖，移除。")
        
        # 更新待处理点列表
        unprocessed_points = points_to_keep
        
    print(f"\n--- 迭代完成！总共找到了 {len(found_objects)} 个独立物体。 ---")
    return found_objects


def visualize_highlighted_regions(
    original_image: Image.Image,
    heatmap: np.ndarray,
    filtered_mask: np.ndarray,
    control_points: list[tuple[int, int]]
):
    """
    可视化热力图处理过程，重点展示过滤后的高亮区域及其质心。

    Args:
        original_image (Image.Image): 原始的PIL图像。
        heatmap (np.ndarray): 输入的原始热力图。
        filtered_mask (np.ndarray): 仅包含有效高亮区域的二值图。
        control_points (list): 提取出的质心列表。
    """
    # 将PIL图像转为OpenCV BGR格式用于叠加操作
    image_bgr = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    # fig.suptitle("高亮区域与质心提取结果检查", fontsize=16)

    # Panel 1: 原始热力图
    ax = axes[0]
    im = ax.imshow(heatmap, cmap='hot')
    # ax.set_title("1. 原始热力图")
    ax.axis('off')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Panel 2: 过滤后的高亮区域
    ax = axes[1]
    ax.imshow(filtered_mask, cmap='gray')
    # ax.set_title(f"2. 过滤后的高亮区域 (面积 > 阈值)")
    ax.axis('off')

    # Panel 3: 结果叠加到原图
    ax = axes[2]
    # 创建一个彩色的叠加层 (例如，用红色高亮)
    overlay = image_bgr.copy()
    highlight_color = [0, 0, 255] # BGR for Red
    # 在mask为白色的地方，将overlay图片对应位置涂成红色
    overlay[filtered_mask == 255] = highlight_color
    
    # 混合原图和叠加层
    final_image = cv2.addWeighted(overlay, 0.5, image_bgr, 0.5, 0)

    # 绘制质心
    if control_points:
        for x, y in control_points:
            cv2.drawMarker(final_image, (x, y), (0, 255, 0), 
                           markerType=cv2.MARKER_CROSS, markerSize=15, thickness=2)
    
    ax.imshow(cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB))
    # ax.set_title("3. 高亮区(红)与质心(绿)叠加")
    ax.axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

 
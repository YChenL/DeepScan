# zoom_refine_mcts.py
import base64, io, asyncio
import numpy as np
from PIL import Image
from typing import Tuple
from .MCTS import MCTSQuestionSample, MCTSNode   # 你的原始类路径
from .prefilter_zoom_refine import zoom_refine_prefilter
# gen heatmap
from .client import get_heatmap
from .visual_grounding import grounding


class OursMCTSQuestionSample(MCTSQuestionSample):
    """在 MCTS 前跑一次 Zoom-Refine 式裁剪""" 
    # overriding get_final_answer
    async def get_final_answer(self, use_mcts=True):
        # 只在第一次进入时做预裁剪
        if not hasattr(self, "_prefilter_done"):
                self._prefilter_done = True
            # crop_b64, region_px = await zoom_refine_prefilter(
            #     self.row['question'],      # 原始问句
            #     self.image,                # 原始 base64 图
            #     self.clients, self.models  # 复用父类拿到的 client+model
            # )
            # if crop_b64 and region_px:
            #     print("llm定位成功", region_px)
            #     # 要对region_px做变换！外扩到384的整数倍，确保坐标一致！
            #     # 这里的resized_img是局部图！！
            #     resized_img, resized_width, resized_height, objects = grounding(crop_b64, self.row['question'], BLOCK=384)
            #     '''
            #      objects ={
            #       "mask": 01蒙板
            #       "source_point": 生成mask的控制点集
            #       "area": 蒙板区域大小
            #       "bbox": 蒙板对应bounding box, 已归一化 (x_min, y_min, x_max, y_max)
            #       "crop_img": 从原图中用蒙板crop出来的子图 (base64str)
            #      }
            #     '''
            #     flag, union_bbox = await self.justify(objects)
            #     # 成功定位
            #     if flag:      
            #         '''
            #          要把image替换为resized_img
            #          width和hight也得换
            #         '''
            #         # 相对坐标换算
            #         bbox_org  = self.convert_bbox_to_original_frame(region_px, resized_width, resized_height, union_bbox)                    
            #         img_bytes = base64.b64decode(self.image)
            #         groud_img = Image.open(io.BytesIO(img_bytes)).crop(bbox_org)
            #         buffered = io.BytesIO()
            #         groud_img.save(buffered, format="PNG")
            #         groud_img_b64 = base64.b64encode(buffered.getvalue()).decode()
                    
            #         # 更新初始化状态
            #         self.initial_state = {
            #             'depth': 0,
            #             'image': groud_img_b64,
            #             'action_history': [],
            #             'text': self.row['question'],  # Root node uses original question as text
            #             'image_width': self.image_width,
            #             'image_height': self.image_height,
            #             'region_coords': bbox_org
            #         }

            #     # 无法成功定位, 用llm自己的定位结果
            #     else:      
            #         # 更新初始化状态
            #         self.initial_state = {
            #             'depth': 0,
            #             'image': crop_b64,
            #             'action_history': [],
            #             'text': self.row['question'],  # Root node uses original question as text
            #             'image_width': self.image_width,
            #             'image_height': self.image_height,
            #             'region_coords': region_px
            #         }
                    
            # else:
                # print("定位失败!")
                # 无法初始定位，直接用grounding计算初始区域
                resized_img, resized_width, resized_height, objects = grounding(self.image, self.row['question'], BLOCK=768)
                flag, union_bbox = await self.justify(objects)
                if flag:      
                    # 换算坐标
                    bbox_org  = self.convert_bbox_to_original_frame((0, 0, self.image_width, self.image_height), resized_width, resized_height, union_bbox)
                    img_bytes = base64.b64decode(self.image)
                    groud_img = Image.open(io.BytesIO(img_bytes)).crop(bbox_org)
                    buffered = io.BytesIO()
                    groud_img.save(buffered, format="PNG")
                    groud_img_b64 = base64.b64encode(buffered.getvalue()).decode()
                    
                    # 更新初始化状态
                    self.initial_state = {
                        'depth': 0,
                        'image': groud_img_b64,
                        'action_history': [],
                        'text': self.row['question'],  # Root node uses original question as text
                        'image_width': self.image_width,
                        'image_height': self.image_height,
                        'region_coords': bbox_org
                    }
                    
                else:
                    # 退化为初始状态
                    self.initial_state = {
                        'depth': 0,
                        'image': self.image,
                        'action_history': [],
                        'text': self.row['question'],  # Root node uses original question as text
                        'image_width': self.image_width,
                        'image_height': self.image_height,
                        'region_coords': (0, 0, self.image_width, self.image_height)
                    }

        # 回调父类逻辑 (MCTS)
        # super.function (会自动搜索回调父类的function实现)
        if use_mcts:
            return await super().get_final_answer()
        else:
            return await super().get_final_answer_vanilla()


    async def single_run(self, root_state):
        """
         Overridden single_run to handle root node initialization dynamically.
         This version checks if the initial state contains a pre-cropped image.
         - If it's a crop, the root can explore all actions (including zoom_out).
         - If it's the full original image (due to localization failure), it 
         defaults to the parent's behavior of forcing a zoom-in first.
        """
        # --- DYNAMIC ROOT INITIALIZATION ---
        if not self.root:
            # Check if the initial state uses the original, full-frame image.
            # This occurs if pre-filtering and grounding fail to find a region.
            is_original_image = (root_state['image'] == self.image)

            if is_original_image:
                # CASE 1: Localization failed. Revert to original MCTS logic.
                # The first action must be a zoom-in ('repeat_question').
                temp_root = MCTSNode(root_state, available_actions=self.actions)
                # Execute repeat_question_action to get the real root node.
                self.root = await self.execute_repeat_question_action(temp_root)
                self.root.parent = None # This is the true root.
            else:
                # CASE 2: Localization succeeded. The initial image is a crop.
                # Initialize the root directly, making all actions available.
                self.root = MCTSNode(root_state, available_actions=self.actions.copy())
                # Calculate the initial valid_area_ratio for the pre-cropped image.
                x1, y1, x2, y2 = self.root.region_coords
                total_area = self.image_width * self.image_height
                self.root.valid_area_ratio = ((x2 - x1) * (y2 - y1)) / total_area

        # --- MCTS CORE LOOP (from parent class) ---
        # 1. Selection: Find the best node to expand.
        node = self.selection(self.root)
        
        # If max depth is reached during selection, this simulation run ends.
        if node.state['depth'] >= self.max_depth:
            return 0
            
        # 2. Expansion: Add a new child node to the selected node.
        node = await self.expansion(node)
            
        # 3. Simulation: Calculate a reward for the new node.
        reward = await self.simulation(node)
        
        # Update the reward value on the leaf node itself.
        node.leaf_reward = reward

        # 4. Backpropagation: Update the values of parent nodes.
        self.backpropagation(node, reward)
        
        return reward

        
    async def corse_search(self):
        # crop_b64, region_px = await zoom_refine_prefilter(
        #         self.row['question'],      # 原始问句
        #         self.image,                # 原始 base64 图
        #         self.clients, self.models  # 复用父类拿到的 client+model
        #     )
            self_img_bytes = base64.b64decode(self.image)
            self_img       = Image.open(io.BytesIO(self_img_bytes))
        
        # if crop_b64 and region_px:              
        #     # 这里的resized_img是局部图！！
        #     crop_img_bytes = base64.b64decode(crop_b64)
        #     crop_img       = Image.open(io.BytesIO(crop_img_bytes))

        #     resized_img, resized_width, resized_height, objects = grounding(crop_b64, self.row['question'], BLOCK=384)
        #     flag, union_bbox = await self.justify(objects)
        #     if flag:
        #         bbox_org  = self.convert_bbox_to_original_frame(region_px, resized_width, resized_height, union_bbox)    
        #         groud_img = self_img.crop(bbox_org)
                
        #         return self_img, groud_img, objects

        #     else:
        #         return self_img, crop_img, objects

        # else:
            resized_img, resized_width, resized_height, objects = grounding(self.image, self.row['question'], BLOCK=768)
            flag, union_bbox = await self.justify(objects)
            if flag:
                bbox_org  = self.convert_bbox_to_original_frame((0, 0, self.image_width, self.image_height), resized_width, resized_height, union_bbox)
                groud_img = self_img.crop(bbox_org)
                
                return self_img, groud_img, objects

            else:
                print('search failed')
                return self_img, self_img, objects

    
    async def justify(self, found_objs, TOP_K=10):
        """
         分析TOP_K个候选图像区域，找出所有与问题相关的对象，
         并将它们的边界框（bbox）合并成一个大的联合边界框。
        """
        def is_contained(box_a, box_b):
            """检查 box_a 是否完全被 box_b 包裹"""
            ax1, ay1, ax2, ay2 = box_a
            bx1, by1, bx2, by2 = box_b
            return ax1 >= bx1 and ay1 >= by1 and ax2 <= bx2 and ay2 <= by2
            
        # 用于存储所有被LLM确认为相关的对象的bbox
        confirmed_bboxes = []

        # 1. 修改Prompt以更精确地提问
        # 这个Prompt引导LLM专注于判断当前小图中的物体是否在给定的关键对象列表中
        # prompt = f"""**Role:** You are a precise visual reasoning assitant.
        # **Objective:** Identify the main object in the provided image and determine if it contains any of the objects in the "Key Object List" or the clues for answering the "Contextual Question".
        # **Key Object List:** {', '.join(self.key_objects)}
        # **Contextual Question:** {self.row['question']}
        # **Your Task:**
        # 1.  Identify the primary object depicted in the image.
        # 2.  Check if this identified object is present in the "Key Object List" above.
        # 3.  Answer with a a single word, either "Yes" or "No"."""
        prompt = f"""I will provide you an image and a **question** {self.row['question']}, \
please firstly determine wether the image contains the clues for answering the question or not (answer with **Yes** or **No**); \
then give the evidence of your decision."""
        
        # 2. 遍历所有TOP_K个候选对象，不再提前中断
        for obj in found_objs:
            image = obj['crop_img']
            bbox = obj['bbox']  # bbox格式: (x_min, y_min, x_max, y_max)

            # 调用LLM进行判断
            response = await self.generate(prompt, image, max_tokens=50)
            # print("待检测区域为: ", bbox, "\n", "判断结果为: ", response)
            # 检查LLM的回答，如果包含'yes'，则收集其bbox
            if 'yes' in response.lower():
                confirmed_bboxes.append(bbox)
            # 如果是'no'，则直接进入下一个循环 (continue)

        # 3. 在循环结束后，根据收集到的bbox进行决策
        # 如果列表为空，说明没有找到任何相关的对象
        if not confirmed_bboxes:
            return False, None

        # 如果只找到一个相关的bbox，直接返回它
        if len(confirmed_bboxes) == 1:
            return True, confirmed_bboxes[0]

        # 步骤 3: 如果有多个bbox，检查是否存在包裹关系
        # contained_boxes = []
        # # 寻找所有被其他bbox完全包裹的bbox
        # for i, box1 in enumerate(confirmed_bboxes):
        #     for j, box2 in enumerate(confirmed_bboxes):
        #         if i == j:
        #             continue
        #         # 如果 box1 被 box2 包裹，则将 box1 加入列表
        #         if is_contained(box1, box2):
        #             contained_boxes.append(box1)
        #             break  # 只要被一个包裹就算，无需再检查

        # # 步骤 4: 根据是否存在包裹关系进行决策
        # if contained_boxes:
        #     # 情况一: 找到了被包裹的bbox
        #     # 我们要返回其中面积最小的那个，作为最精确的定位
        #     # print("[*] 发现包裹关系，返回最精细区域。")
        
        #     # 计算所有被包裹box的面积
        #     areas = [(box[2] - box[0]) * (box[3] - box[1]) for box in contained_boxes]
        #     # 找到面积最小的box的索引
        #     min_area_index = np.argmin(areas)
        
        #     finest_bbox = contained_boxes[min_area_index]
        #     return True, finest_bbox
        
        # else:
        # 情况二: 没有发现任何完全包裹关系，取并集
        # print("[*] 未发现包裹关系，返回所有区域的并集。")
        
        bboxes_array = np.array(confirmed_bboxes)
        x_min = np.min(bboxes_array[:, 0])
        y_min = np.min(bboxes_array[:, 1])
        x_max = np.max(bboxes_array[:, 2])
        y_max = np.max(bboxes_array[:, 3])
        
        union_bbox = (x_min, y_min, x_max, y_max)
        return True, union_bbox


    def convert_bbox_to_original_frame(
        self,
        region_px: Tuple[int, int, int, int],
        resized_width: int,
        resized_height: int,
        union_bbox: Tuple[int, int, int, int]
    ) -> Tuple[int, int, int, int]:
        """
        将一个在经过缩放的局部图上的Bbox坐标，换算回原始大图上的绝对坐标。(高效版)

        Args:
            region_px (tuple): crop_img在原始大图上的坐标 (x1, y1, x2, y2)。
            resized_width (int): 缩放后的局部图(resized_img)的宽度。
            resized_height (int): 缩放后的局部图(resized_img)的高度。
            union_bbox (tuple): 在resized_img上的子区域坐标 (x_u, y_u, x2_u, y2_u)。

        Returns:
            tuple: union_bbox区域在原始大图(self.image)上的最终绝对坐标。
        """
        # --- 准备工作：直接从坐标计算尺寸 ---
        # 解包父区域坐标
        x_p, y_p, x2_p, y2_p = region_px
    
        # 从父区域坐标直接计算crop_img的原始尺寸
        w_crop = x2_p - x_p
        h_crop = y2_p - y_p

        # 直接使用传入的resized_img尺寸
        w_resized = resized_width
        h_resized = resized_height

        # 解包子区域坐标
        x_u, y_u, x2_u, y2_u = union_bbox

        # --- 步骤一：反算缩放 ---
        # 计算 crop_img 到 resized_img 的缩放比例
        # 防止因尺寸为0导致除零错误
        if w_crop == 0 or h_crop == 0:
            raise ValueError("crop_img 的计算尺寸不能为零，请检查 region_px。")
    
        scale_x = w_resized / w_crop
        scale_y = h_resized / h_crop
    
        # 将 union_bbox 的坐标从 resized_img 坐标系按比例换算回 crop_img 坐标系
        x_on_crop = x_u / scale_x
        y_on_crop = y_u / scale_y
        x2_on_crop = x2_u / scale_x
        y2_on_crop = y2_u / scale_y

        # --- 步骤二：反算裁剪 ---
        # 将在 crop_img 上的坐标加上 crop_img 本身在原图上的偏移量 (x_p, y_p)
        final_x1 = x_p + x_on_crop
        final_y1 = y_p + y_on_crop
        final_x2 = x_p + x2_on_crop
        final_y2 = y_p + y2_on_crop

        # 返回由整数组成的坐标元组
        return tuple(map(int, (final_x1, final_y1, final_x2, final_y2)))
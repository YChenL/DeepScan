import base64, io, asyncio
import numpy as np
from PIL import Image
from typing import Tuple
from .MCTS import MCTSQuestionSample, MCTSNode   # 你的原始类路径
# gen heatmap
from .client import get_heatmap
from .visual_grounding import grounding


class DeepScan(MCTSQuestionSample):
    async def get_final_answer(self, use_mcts=True):
        if not hasattr(self, "_prefilter_done"):
            # Hierarchical Scanning
            self._prefilter_done = True
            if len(self.key_objects) >= 1: # relative_position
                resized_img, resized_width, resized_height, objects_1 = grounding(self.image, self.row['question'], BLOCK=384)
            else:
                resized_img, resized_width, resized_height, objects_1 = grounding(self.image, self.row['question'], BLOCK=384)

            flag, union_bbox = await self.justify(objects_1)
            if flag:      
                bbox_org  = self.convert_bbox_to_original_frame((0, 0, self.image_width, self.image_height), resized_width, resized_height, union_bbox)
                img_bytes = base64.b64decode(self.image)
                groud_img = Image.open(io.BytesIO(img_bytes)).crop(bbox_org)
                buffered = io.BytesIO()
                groud_img.save(buffered, format="PNG")
                groud_img_b64 = base64.b64encode(buffered.getvalue()).decode()
     
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
                self.initial_state = {
                    'depth': 0,
                    'image': self.image,
                    'action_history': [],
                    'text': self.row['question'],  # Root node uses original question as text
                    'image_width': self.image_width,
                    'image_height': self.image_height,
                    'region_coords': (0, 0, self.image_width, self.image_height)
                }

        if use_mcts:
            return await super().get_final_answer()
        else:
            return await super().get_final_answer_vanilla()

        
    async def justify(self, found_objs, TOP_K=None):
        def is_contained(box_a, box_b):
            ax1, ay1, ax2, ay2 = box_a
            bx1, by1, bx2, by2 = box_b
            return ax1 >= bx1 and ay1 >= by1 and ax2 <= bx2 and ay2 <= by2

        confirmed_bboxes = []
        prompt = f"""I will provide you an image and a **question** {self.row['question']}, \
please firstly determine wether the image contains the clues for answering the question or not (answer with **Yes** or **No**); \
then give the evidence of your decision."""
        
        if TOP_K:
            for obj in found_objs[: TOP_K]:
                image = obj['crop_img']
                bbox = obj['bbox']  # (x_min, y_min, x_max, y_max)

                response = await self.generate_local(prompt, image, max_tokens=50)
                if 'yes' in response.lower():
                    confirmed_bboxes.append(bbox)
        else:
            for obj in found_objs:
                image = obj['crop_img']
                bbox = obj['bbox']  # (x_min, y_min, x_max, y_max)

                response = await self.generate_local(prompt, image, max_tokens=50)
                if 'yes' in response.lower():
                    confirmed_bboxes.append(bbox)
               
        if not confirmed_bboxes:
            return False, None

        if len(confirmed_bboxes) == 1:
            return True, confirmed_bboxes[0]

        # check candidates spatial relations
        contained_boxes = []
        for i, box1 in enumerate(confirmed_bboxes):
            for j, box2 in enumerate(confirmed_bboxes):
                if i == j:
                    continue
              
                if is_contained(box1, box2):
                    contained_boxes.append(box1)
                    break 
                    
        if contained_boxes:
            areas = [(box[2] - box[0]) * (box[3] - box[1]) for box in contained_boxes]
            min_area_index = np.argmin(areas)
            finest_bbox = contained_boxes[min_area_index]
            return True, finest_bbox
        
        else:
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
      
        x_p, y_p, x2_p, y2_p = region_px
        w_crop = x2_p - x_p
        h_crop = y2_p - y_p
    
        w_resized = resized_width
        h_resized = resized_height
    
        x_u, y_u, x2_u, y2_u = union_bbox
    
        if w_crop == 0 or h_crop == 0:
            raise ValueError(
                f"Invalid region_px={region_px}: computed crop dimensions must be non-zero "
                f"(got width={w_crop}, height={h_crop})."
            )
    
        scale_x = w_resized / w_crop
        scale_y = h_resized / h_crop
    
        x_on_crop = x_u / scale_x
        y_on_crop = y_u / scale_y
        x2_on_crop = x2_u / scale_x
        y2_on_crop = y2_u / scale_y
    
        final_x1 = x_p + x_on_crop
        final_y1 = y_p + y_on_crop
        final_x2 = x_p + x2_on_crop
        final_y2 = y_p + y2_on_crop
    
        return tuple(map(int, (final_x1, final_y1, final_x2, final_y2)))
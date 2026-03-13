from .policy import QuestionSample as BaseQuestionSample
from utils import is_none
import shortuuid
import base64
import io
from PIL import Image
import numpy as np
import random
import math
import aiohttp
import traceback
import re

class MCTSNode:
    """MCTS Tree Node Class"""
    def __init__(self, state, parent=None, available_actions=None):
        self.state = state  
        self.parent = parent  
        self.children = {} 
        self.visits = 0  
        self.value = 0 
        self.leaf_reward = 0  
        self.untried_actions = available_actions.copy() if available_actions else []
        self.expert_info = None
        self.valid_area_ratio = 1.0
        self.region_coords = state.get('region_coords', (0, 0, state['image_width'], state['image_height']))
        self.extra_info = {}


class MCTSQuestionSample(BaseQuestionSample):
    def __init__(self, row, args, round_idx=0):
        super().__init__(row, args, round_idx)
        # Get image dimensions
        image_bytes = base64.b64decode(self.image)
        img = Image.open(io.BytesIO(image_bytes))
        self.image_width, self.image_height = img.size
        
        # Create 32x32 blank image
        blank_image = Image.new('RGB', (32, 32), color='white')
        buffered = io.BytesIO()
        blank_image.save(buffered, format="PNG")
        self.blank_image = base64.b64encode(buffered.getvalue()).decode()
        
        # MCTS parameters
        self.max_depth = 1     
        self.c_puct = 1.0      
        self.n_simulations = 3
        self.use_ensemble = True
        
        # Define action space
        self.actions = [
            "repeat_question",
            "zoom_out"  # New zoom out action
        ]
        
        # Define action prompts
        self.action_prompts = {
            "repeat_question": "Repeat the question.",
            "zoom_out": "Zoom out the region by 1.5x"  # New zoom out action prompt
        }
        
        # Define action executor mapping
        self.action_executors = {
            "repeat_question": self.execute_repeat_question_action,
            "zoom_out": self.execute_zoom_out_action  # New zoom out action executor
        }
        
        # MCTS tree root node
        self.root = None
        
        # Visual expert API
        self.expert_ports = [1]  # Multiple expert ports, corresponding to port number +8000
        self.expert_ports = [port + 8000 for port in self.expert_ports]
        self.expert_base_url = "http://localhost:{}/predict"

    async def extract_key_objects(self):
        """Extract key objects from question"""
        if 'llava' in self.args.model_path
            # Preprocess question text
            question = self.row['question'].replace('?', '')
            stop_words = ['is', 'in the image', 'IS', 'THE', 'IMAGE','what','color of','there', 'a', 'an', 'How']
            for word in stop_word:
                question = re.sub(r'\b' + word + r'\b', '', question, flags=re.IGNORECASE)
            question = ' '.join(question.split())
            
            # Extract objects
            prompt = f"Task: List objects mentioned in text in List format.\nInput text: {question}\nAction: What objects are mentioned in original text? List separated by commas. For example, from \"person with white trousers on the left or right side of the person in blue\", output \"[\"person with white trousers\", \"person in blue\"]\"."
            response = await self.generate_local(prompt, self.blank_image, max_tokens=50)
            
            # Try to parse as list format
            try:
                objects = eval(response)
            except:
                response = response.replace('[', '').replace(']', '').replace('"', '')
                objects = response.split(',')
                
            # Filter objects
            filtered_objects = []
            for obj in objects:
                obj = obj.strip().lower()
                if obj in question.lower():
                    filtered_objects.append(obj)
                    
            # If filtered result is empty, use original question
            if not filtered_objects:
                filtered_objects = [question]
                    
            return filtered_objects
            
        else:
            # Use original extraction method
            prompt = f"Task: Extract all objects (including people) with their complete descriptions from the question. For example, from 'Is the person with white trousers on the left or right side of the person in blue?', extract 'person with white trousers' and 'person in blue'.\nQuestion: {self.row['question']}\nAction: Only list the objects separated by commas."
            response = await self.generate_local(prompt, self.blank_image, max_tokens=50)
            
            if "object" in response.lower() or "description" in response.lower():
                objects = response.split()[-1].lower()

            # Split response into list and strip whitespace
            objects = [obj.strip() for obj in response.split(',')]
            return objects
        
    async def get_expert_boxes(self, image, text):
        """Call visual expert to get boxes"""
        try:
            # Randomly select expert
            port = random.choice(self.expert_ports)
            
            expert_url = self.expert_base_url.format(port)
            timeout = aiohttp.ClientTimeout(total=10000)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    expert_url,
                    json={
                        "image": image,  # image is already base64 string
                        "text": text
                    }
                ) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        error_text = await response.text()
                        print(f"Visual expert API returned error status: {response.status}")
                        print(f"Error message: {error_text}")
                        print(f"Request URL: {expert_url}")
                        print(f"Request text: {text}")
                        return None
        except Exception as e:
            print(f"Error calling visual expert: {str(e)}")
            print(f"Request URL: {expert_url}")
            print(f"Request text: {text}")
            print(f"Exception stack: {traceback.format_exc()}")
            return None

    def selection(self, node):
        """Selection phase: Use UCB algorithm to select best child node"""
        if node.untried_actions:
            return node

        if not node.children:
            return node
            
        total_visits = sum(child.visits for child in node.children.values())
        
        def ucb_score(child):
            exploit = child.value / child.visits if child.visits > 0 else 0
            explore = math.sqrt(2 * math.log(total_visits) / (child.visits + 1e-8))
            return exploit + self.c_puct * explore
            
        best_child = max(node.children.values(), key=ucb_score)
        return self.selection(best_child)

    
    async def execute_repeat_question_action(self, node):
        """Execute repeat question action"""
        node_text = self.row['question']
        expert_result = await self.get_expert_boxes(node.state['image'], node_text)
        
        # If expert result contains boxes
        if expert_result and expert_result.get('boxes'):
            # Convert all boxes to numpy array
            boxes = np.array(expert_result['boxes'])
            
            # Calculate union of all boxes
            x1 = np.min(boxes[:, 0])
            y1 = np.min(boxes[:, 1]) 
            x2 = np.max(boxes[:, 2])
            y2 = np.max(boxes[:, 3])
            
            # Add some padding
            padding = 32
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(node.state['image_width'], x2 + padding)
            y2 = min(node.state['image_height'], y2 + padding)
            
            # Calculate new valid area ratio
            new_area = (x2 - x1) * (y2 - y1)
            total_area = node.state['image_width'] * node.state['image_height']
            valid_area_ratio = new_area / total_area

            # Update region_coords, considering parent node's coordinate offset
            parent_x1, parent_y1, _, _ = node.state['region_coords']
            new_region_coords = (
                parent_x1 + x1,
                parent_y1 + y1,
                parent_x1 + x2,
                parent_y1 + y2
            )

            # Crop image
            image_bytes = base64.b64decode(self.image)         
            img = Image.open(io.BytesIO(image_bytes))            
            cropped_img = img.crop(new_region_coords)
            
            # Convert cropped image back to base64
            buffered = io.BytesIO()
            cropped_img.save(buffered, format="PNG")
            cropped_image_base64 = base64.b64encode(buffered.getvalue()).decode()
                    
        else:
            # If no boxes obtained, use original image and region
            cropped_image_base64 = node.state['image']
            valid_area_ratio = node.valid_area_ratio
            new_region_coords = node.state['region_coords']
        
        # Create new state
        new_state = {
            'depth': node.state['depth'] + 1,
            'image': cropped_image_base64,
            'action_history': node.state['action_history'] + [self.action_prompts["repeat_question"]],
            'text': node_text,
            'image_width': node.state['image_width'],
            'image_height': node.state['image_height'],
            'region_coords': new_region_coords
        }

        # Create new node
        child = MCTSNode(new_state, parent=node, available_actions=self.actions)
        child.expert_info = expert_result
        child.valid_area_ratio = valid_area_ratio
        
        return child

    async def execute_zoom_out_action(self, node):
        """Execute zoom out action on region"""
        # Get current region coordinates
        x1, y1, x2, y2 = node.state['region_coords']
        
        # Calculate region center point
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        # Calculate current region width and height
        width = x2 - x1
        height = y2 - y1
        
        # Zoom out by 1.5x
        new_width = width * 1.5
        new_height = height * 1.5
        
        # Calculate new region coordinates
        new_x1 = max(0, center_x - new_width/2)
        new_y1 = max(0, center_y - new_height/2)
        new_x2 = min(node.state['image_width'], center_x + new_width/2)
        new_y2 = min(node.state['image_height'], center_y + new_height/2)
        
        # Crop original image
        image_bytes = base64.b64decode(self.image)
        img = Image.open(io.BytesIO(image_bytes))
        cropped_img = img.crop((new_x1, new_y1, new_x2, new_y2))
        
        # Convert cropped image to base64
        buffered = io.BytesIO()
        cropped_img.save(buffered, format="PNG")
        cropped_image_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        final_x1, final_y1, final_x2, final_y2 = new_x1, new_y1, new_x2, new_y2
        
        # If parent node has missing_objects, try to find them in zoomed out region
        if 'missing_objects' in node.state and node.state['missing_objects']:
            missing_objects_text = ', '.join(node.state['missing_objects'])
            expert_result = await self.get_expert_boxes(cropped_image_base64, missing_objects_text)
            
            # If boxes found, calculate union
            if expert_result and expert_result.get('boxes'):
                boxes = np.array(expert_result['boxes'])
                # Calculate union of expert boxes
                expert_x1 = np.min(boxes[:, 0]) + new_x1
                expert_y1 = np.min(boxes[:, 1]) + new_y1
                expert_x2 = np.max(boxes[:, 2]) + new_x1
                expert_y2 = np.max(boxes[:, 3]) + new_y1
                
                # Calculate union with parent node's region
                final_x1 = min(x1, expert_x1)
                final_y1 = min(y1, expert_y1)
                final_x2 = max(x2, expert_x2)
                final_y2 = max(y2, expert_y2)
                
                # Re-crop image
                cropped_img = img.crop((final_x1, final_y1, final_x2, final_y2))
                buffered = io.BytesIO()
                cropped_img.save(buffered, format="PNG")
                cropped_image_base64 = base64.b64encode(buffered.getvalue()).decode()
        else:
            expert_result = await self.get_expert_boxes(cropped_image_base64, ", ".join(self.key_objects))
        
        # Create new state
        new_state = {
            'depth': node.state['depth'] + 1,
            'image': cropped_image_base64,
            'action_history': node.state['action_history'] + [self.action_prompts["zoom_out"]],
            'text': node.state['text'],
            'image_width': node.state['image_width'],
            'image_height': node.state['image_height'],
            'region_coords': (final_x1, final_y1, final_x2, final_y2)
        }

        # Create new node
        child = MCTSNode(new_state, parent=node, available_actions=self.actions)
        child.expert_info = expert_result
            
        # Calculate new valid area ratio
        new_area = (final_x2 - final_x1) * (final_y2 - final_y1)
        total_area = node.state['image_width'] * node.state['image_height']
        child.valid_area_ratio = new_area / total_area
        
        return child

    async def expansion(self, node):
        """Expansion phase: add a new child node"""
        if node.state['depth'] >= self.max_depth or not node.untried_actions:
            return node

        action = random.choice(node.untried_actions)
        node.untried_actions.remove(action)
        child = await self.action_executors[action](node)
        node.children[action] = child
        
        return child

    async def simulation(self, node):
        # Get key objects
        key_objects = self.key_objects
        all_objects_present = True
        confirmed_objects = []
        missing_objects = []
        for obj in key_objects:
            prompt = f"Task: Only answer yes or no.\nQuestion: Is there a {obj} in this image?"
            response = await self.generate_local(prompt, node.state['image'], max_tokens=10)
            if 'yes' in response.lower():
                confirmed_objects.append(obj)
            else:
                missing_objects.append(obj)
                all_objects_present = False
                break

        node.state['caption'] = ', '.join(confirmed_objects)
        node.state['missing_objects'] = missing_objects
        if all_objects_present:
            reward = 1 - node.valid_area_ratio
        else:
            reward = 0
            
        return reward

    def backpropagation(self, node, reward):
        """Backpropagation phase: update node values"""
        while node:
            node.visits += 1
            node.value += reward
            node = node.parent
        
    async def single_run(self, root_state):
        if not self.root:
            # Create temporary root node
            temp_root = MCTSNode(root_state, available_actions=self.actions)
            # Execute repeat_question_action to get real root node
            self.root = await self.execute_repeat_question_action(temp_root)
            self.root.parent = None
     
        node = self.selection(self.root)
        if node.state['depth'] >= self.max_depth:
            return 0
    
        node = await self.expansion(node)
        reward = await self.simulation(node)
        node.leaf_reward = reward
        self.backpropagation(node, reward)
        
        return reward

    async def get_final_answer(self):
        """Run MCTS to search for best answer"""
        # Run multiple simulations
        for _ in range(self.n_simulations):
            # await self.single_run(initial_state)
            await self.single_run(self.initial_state)
            
        # Collect all nodes
        all_nodes = []
        nodes_to_visit = [self.root]
        while nodes_to_visit:
            node = nodes_to_visit.pop()
            all_nodes.append(node)
            nodes_to_visit.extend(node.children.values())
            
        # Generate final question
        final_qs = ''
        if not is_none(self.row['hint']):
            final_qs += self.row['hint'] + '\n'
        final_qs += self.row['question']
        
        for option_char, option in zip(self.cur_option_char, self.options):
            final_qs += '\n' + option_char + '. ' + option

        if self.args.single_pred_prompt:
            if self.args.lang == 'cn':
                final_qs += '\n' + "请直接回答选项字母。"
            else:
                final_qs += '\n' + "Answer with the option's letter from the given choices directly."
            
        # Generate answer for each node
        answers = []
        for node in all_nodes:
            answer = await self.generate_local(final_qs, node.state['image'])
            
            # Extract option letter from answer
            for letter in ['A', 'B', 'C', 'D']:
                if letter in answer:
                    answers.append((letter, node.leaf_reward))  # Use leaf reward as weight
                    break
            else:
                answers.append(('A', node.leaf_reward))  # If no valid option found, default to A with leaf reward
                
        # Find node with highest value/visits
        best_node = max(all_nodes, key=lambda x: (x.leaf_reward, all_nodes.index(x)))
        
        if self.use_ensemble:
            # Weighted voting for final answer
            from collections import defaultdict
            vote_result = defaultdict(float)
            for answer, weight in answers:
                vote_result[answer] += weight
                
            # Check if all weights are zero
            if all(weight == 0 for weight in vote_result.values()):
                # Regenerate answer using original image
                answer = await self.generate_local(final_qs, self.image)
                # Extract option letter from answer
                for letter in ['A', 'B', 'C', 'D']:
                    if letter in answer:
                        final_answer = letter
                        break
                else:
                    final_answer = 'A'  # If no valid option found, default to A
            else:
                final_answer = max(vote_result, key=vote_result.get)
        else:
            # Use best_node's answer
            final_answer = max(answers, key=lambda x: x[1])[0]
        
        return final_answer, final_qs, answers[-1][0], best_node.state['image'], best_node, self.root

    
    async def get_final_answer_vanilla(self):
        """
        Directly reasons using the original image and question without any MCTS.
        The return format is identical to get_final_answer for evaluation compatibility.
        """
        # 1. Prepare the final question prompt (same logic as in get_final_answer)
        final_qs = ''
        if not is_none(self.row['hint']):
            final_qs += self.row['hint'] + '\n'
        final_qs += self.row['question']

        for option_char, option in zip(self.cur_option_char, self.options):
            final_qs += '\n' + option_char + '. ' + option

        if self.args.single_pred_prompt:
            if self.args.lang == 'cn':
                final_qs += '\n' + "请直接回答选项字母。"
            else:
                final_qs += '\n' + "Answer with the option's letter from the given choices directly."

        # 2. Create a dummy MCTSNode to satisfy the return signature.
        #    This node represents the single "vanilla" reasoning step.
        vanilla_state = {
            'depth': 0,
            'image': self.image,
            'action_history': ['vanilla_reasoning'],
            'text': self.row['question'],
            'image_width': self.image_width,
            'image_height': self.image_height,
            'region_coords': (0, 0, self.image_width, self.image_height)
        }
        vanilla_node = MCTSNode(state=vanilla_state)
        # This single node acts as both the "best" node and the "root" node.
        # vanilla_node = MCTSNode(state=self.initial_state)
        
        # 3. Generate an answer using the original image and the prepared question
        full_answer_text = await self.generate_local(final_qs, vanilla_node.state['image'])

        # 4. Extract the option letter from the answer
        final_answer = 'A'  # Default to 'A' if no valid option is found
        for letter in ['A', 'B', 'C', 'D']:
            if letter in full_answer_text:
                final_answer = letter
                break

        # We can also populate some of its attributes for consistency.
        vanilla_node.leaf_reward = 1.0 # Assign a default reward
        vanilla_node.visits = 1
        vanilla_node.value = 1.0

        # 5. Return the results in the exact same format as get_final_answer
        return (
            final_answer,                     # The final predicted option letter
            final_qs,                         # The prompt sent to the model
            final_answer,                     # The "full" answer (same as final_answer in this case)
            vanilla_node.state['image'],      # The image used (the original image)
            vanilla_node,                     # The "best" node (our dummy node)
            vanilla_node                      # The "root" node (our dummy node)
        )
        
    def serialize_tree(self, node):
        """Serialize tree structure for saving to jsonl"""
        node_info = {
            "state": node.state,
            "visits": node.visits, 
            "value": node.value,
            "leaf_reward": node.leaf_reward,
            "expert_info": node.expert_info,
            "valid_area_ratio": node.valid_area_ratio,
            "region_coords": node.region_coords,
            "extra_info": node.extra_info,
            "children": {action: self.serialize_tree(child) for action, child in node.children.items()}
        }
        return node_info

        
    async def _process(self):
        # Extract key objects from question
        self.key_objects = await self.extract_key_objects()

        final_answer, prompt, full_answer, final_image, best_node, root_node = await self.get_final_answer()
    
        # Serialize tree structure for saving
        tree_info = self.serialize_tree(best_node)
         
        return {
            "question_id": self.row['index'],
            "round_id": self.round_idx,
            "prompt": prompt,
            "text": final_answer,
            "options": self.options,
            "option_char": self.cur_option_char,
            "answer_id": shortuuid.uuid(),
            "model_id": self.args.model_path,
            "answer": self.row['answer'],
        }, tree_info

import torch
import logging
from typing import List
from swift.plugin import ORM, orms
import re
import numpy as np
import ast


class FormatReward(ORM):
    
    def compute_format_reward(self, completion: str) -> float:
        think_pattern = r'<thinking>.*?</thinking>'
        answer_pattern = r'<answer>.*?</answer>'     
        object_pattern = r"Object Center:\s*(\[+.*?\]+)"
        camera_pattern = r"Camera Center:\s*(\[+.*?\]+)"
            
        has_think = bool(re.search(think_pattern, completion, re.DOTALL))
        has_answer = bool(re.search(answer_pattern, completion, re.DOTALL))
        has_object_data = bool(re.search(object_pattern, completion, re.DOTALL))
        has_camera_data = bool(re.search(camera_pattern, completion, re.DOTALL))
        think_match = re.search(think_pattern, completion, re.DOTALL)  
        think_length = len(think_match.group()) if think_match else 0

        format_score = 0.0
        if has_think and has_answer:
            format_score += 0.5
        if has_object_data and has_camera_data:
            format_score += 0.4
        if think_length > 20:
            format_score += 0.1

        return format_score
    

    def __call__(self, completions: List[str], solution: List[str], **kwargs) -> List[float]:
        batch_size = len(completions)
        rewards = [0 for _ in range(batch_size)]
        for i in range(batch_size):
            format_reward = self.compute_format_reward(completions[i])
            rewards[i] = 0.2 * format_reward

        return rewards
    
    
class ACCReward(ORM):
    
    def compute_accuracy_reward(self, completion: str, solution) -> float:
        accuracy_reward = 0.0
        answer_match = re.search(r'<answer>(.*?)</answer>', completion, re.DOTALL)
        generated_answer = answer_match.group(1).strip() if answer_match else completion.strip()
        ground_truth = solution[0]['answer']
        if generated_answer == ground_truth:
            accuracy_reward = 1.0
            
        return accuracy_reward
    

    def __call__(self, completions: List[str], solution: List[str], **kwargs) -> List[float]:
        batch_size = len(completions)
        rewards = [0 for _ in range(batch_size)]
        for i in range(batch_size):
            curr_sol = solution[i]        
            accuracy_reward = self.compute_accuracy_reward(completions[i], curr_sol)                  
            rewards[i] = 0.6 * accuracy_reward

        return rewards
    

class CameraReward(ORM):
        
    def extract_camera_centers(self, text):
        try:
            pattern = r"Camera Center:\s*(\[.*?\])"
            centers_str = re.findall(pattern, text)
            centers = [np.array(ast.literal_eval(c.strip())) for c in centers_str if 'null' not in c.lower()]
            return centers
        except Exception:
            return []


    def compute_camera_reward(self, completion: str, solution_data) -> float:
        if isinstance(solution_data, list) and len(solution_data) > 0:
            sol_dict = solution_data[0]
        else:
            sol_dict = solution_data
        gt_centers = [np.array(c) for c in sol_dict.get('camera_center', []) if c is not None]      
        pred_centers = self.extract_camera_centers(completion)
        if not gt_centers:
            return 1.0 if not pred_centers else 0.0
                       
        if len(pred_centers) != len(gt_centers):
            return 0.0

        errors = []
        for p, g in zip(pred_centers, gt_centers):
            p_np = np.array(p)
            g_np = np.array(g)
            if p_np.shape != g_np.shape:
                return 0.0           
            errors.append(np.linalg.norm(p_np - g_np))
            
        if not errors:
            return 0.0
        
        mean_error = np.mean(errors)

        return float(np.exp(-mean_error))
    

    def __call__(self, completions: List[str], solution: List[str], **kwargs) -> List[float]:
        batch_size = len(completions)
        rewards = [0 for _ in range(batch_size)]
        for i in range(batch_size):
            curr_sol = solution[i]        
            camera_reward = self.compute_camera_reward(completions[i], curr_sol)                   
            rewards[i] = 0.1 * camera_reward

        return rewards
    
    
class ObjectReward(ORM):

    def extract_object_centers(self, text):
        try:
            pattern = r"Object Center:\s*(\[.*?\])"
            centers_str = re.findall(pattern, text)
            centers = [np.array(ast.literal_eval(c.strip())) for c in centers_str if 'null' not in c.lower()]
            return centers
        except Exception:
            return []

    def compute_object_reward(self, completion: str, solution_data) -> float:
        if isinstance(solution_data, list) and len(solution_data) > 0:
            sol_dict = solution_data[0]
        else:
            sol_dict = solution_data

        gt_centers = [np.array(c) for c in sol_dict.get('object_center', []) if c is not None]   
        pred_centers = self.extract_object_centers(completion)
        if not gt_centers:
            return 1.0 if not pred_centers else 0.0

        if len(pred_centers) != len(gt_centers):
            return 0.0
        errors = []
        for p, g in zip(pred_centers, gt_centers):
            p_np = np.array(p)
            g_np = np.array(g)
            if p_np.shape != g_np.shape:
                return 0.0          
            errors.append(np.linalg.norm(p_np - g_np))
            
        if not errors:
            return 0.0
        
        mean_error = np.mean(errors)  
        return float(np.exp(-mean_error))
    
    def __call__(self, completions: List[str], solution: List[str], **kwargs) -> List[float]:
        batch_size = len(completions)
        rewards = [0 for _ in range(batch_size)]

        for i in range(batch_size):
            curr_sol = solution[i]        
            object_reward = self.compute_object_reward(completions[i], curr_sol)                 
            rewards[i] = 0.1 * object_reward

        return rewards


orms['format_reward_ours'] = FormatReward
orms['acc_reward_ours'] = ACCReward
orms['camera_reward_ours'] = CameraReward
orms['object_reward_ours'] = ObjectReward
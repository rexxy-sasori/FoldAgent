import torch
import numpy as np
from verl import DataProto


def compute_reward(data: DataProto) -> tuple[torch.Tensor, dict]:
    """Custom reward function that uses environment-computed rewards.
    
    This function extracts rewards from the batch's extra_data field,
    which were already computed by the environment (envs/local_search.py)
    using GPT-4.1 judge.
    
    Args:
        data: DataProto object containing batch data with environment-computed rewards
        
    Returns:
        Tuple of (reward_tensor, reward_extra_infos)
            - reward_tensor: Tensor of shape (batch_size, response_length) with rewards
            - reward_extra_infos: Dictionary with additional reward information
    """
    batch_size = len(data)
    response_length = data.batch['response_mask'].shape[1]
    
    reward_tensor = torch.zeros((batch_size, response_length), dtype=torch.float32)
    reward_extra_infos = {}
    
    for i in range(batch_size):
        reward_score = None
        
        if 'extra_data' in data.non_tensor_batch:
            extra_data = data.non_tensor_batch['extra_data']
            
            if isinstance(extra_data, np.ndarray):
                if extra_data.ndim == 0:
                    extra_data = extra_data.item()
                elif extra_data.ndim == 1:
                    extra_data = extra_data[i] if i < len(extra_data) else extra_data[0]
            
            if isinstance(extra_data, dict):
                reward_score = extra_data.get('reward', None)
        
        if reward_score is not None:
            reward_tensor[i, -1] = float(reward_score)
        else:
            reward_tensor[i, -1] = 0.0
    
    reward_extra_infos['reward_source'] = 'environment'
    reward_extra_infos['reward_computed_by'] = 'envs/local_search.py'
    
    return reward_tensor, reward_extra_infos

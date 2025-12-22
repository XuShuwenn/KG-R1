# Copyright 2025 Individual Contributor: Thibaut Barroyer
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import multiprocessing
import os
from functools import partial

import ray

from verl import DataProto
from verl.utils.reward_score import default_compute_score


def get_custom_reward_fn(config):
    import importlib.util
    import sys

    reward_fn_config = config.get("custom_reward_function") or {}
    file_path = reward_fn_config.get("path")
    if not file_path:
        return None

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Reward function file '{file_path}' not found.")

    spec = importlib.util.spec_from_file_location("custom_module", file_path)
    module = importlib.util.module_from_spec(spec)
    try:
        sys.modules["custom_module"] = module
        spec.loader.exec_module(module)
    except Exception as e:
        raise RuntimeError(f"Error loading module from '{file_path}': {e}") from e

    function_name = reward_fn_config.get("name")
    if not hasattr(module, function_name):
        raise AttributeError(f"Reward function '{function_name}' not found in '{file_path}'.")

    print(f"using customized reward function '{function_name}' from '{file_path}'")
    raw_fn = getattr(module, function_name)

    reward_kwargs = dict(reward_fn_config.get("reward_kwargs", {}))

    def wrapped_fn(*args, **kwargs):
        return raw_fn(*args, **kwargs, **reward_kwargs)

    return wrapped_fn


def load_reward_manager(config, tokenizer, num_examine, **reward_kwargs):
    reward_manager_name = config.reward_model.get("reward_manager", "naive")
    if reward_manager_name == "naive":
        from verl.workers.reward_manager import NaiveRewardManager

        reward_manager_cls = NaiveRewardManager
    elif reward_manager_name == "prime":
        from verl.workers.reward_manager import PrimeRewardManager

        reward_manager_cls = PrimeRewardManager
    elif reward_manager_name == "batch":
        from verl.workers.reward_manager import BatchRewardManager

        reward_manager_cls = BatchRewardManager
    elif reward_manager_name == "dapo":
        from verl.workers.reward_manager import DAPORewardManager

        reward_manager_cls = DAPORewardManager
    elif reward_manager_name == "format":
        from verl.workers.reward_manager import FormatRewardManager

        reward_manager_cls = FormatRewardManager
    elif reward_manager_name == "kg_format":
        from verl.workers.reward_manager import KGFormatRewardManager

        reward_manager_cls = KGFormatRewardManager
    elif reward_manager_name == "kg_format_multiturn":
        from verl.workers.reward_manager.kg_format_multiturn import KGFormatMultiTurnRewardManager

        reward_manager_cls = KGFormatMultiTurnRewardManager
    else:
        raise NotImplementedError

    compute_score = get_custom_reward_fn(config)
    final_compute_score = compute_score

    if compute_score is None:
        sandbox_config = config.reward_model.get("sandbox_fusion")
        sandbox_url = sandbox_config.get("url") if sandbox_config else None
        if sandbox_url:
            sandbox_manager = multiprocessing.Manager()
            _concurrent_semaphore = sandbox_manager.Semaphore(sandbox_config.get("max_concurrent", 64))
            final_compute_score = partial(default_compute_score, sandbox_fusion_url=sandbox_url, concurrent_semaphore=_concurrent_semaphore)
        else:
            final_compute_score = default_compute_score

    if config.trainer.mode in ['kg-search', 'search']:
        final_compute_score = None
    
    return reward_manager_cls(
        tokenizer=tokenizer,
        num_examine=num_examine,
        compute_score=final_compute_score,
        reward_fn_key=config.data.reward_fn_key,
        **reward_kwargs,
    )


def compute_reward(data: DataProto, reward_fn):
    """
    Compute reward for a batch of data.
    Args:
        data: DataProto object containing the input data.
        reward_fn: Reward function to compute the reward.
    Returns:
        Tuple of reward DataProto and extra info dictionary.
    """
    try:
        reward_result = reward_fn(data, return_dict=True)
        
        # Handle both old format (reward_tensor) and new format (structured_rewards)
        if isinstance(reward_result, dict):
            if "reward_tensor" in reward_result and "structured_rewards" not in reward_result:
                # Old format - token-level rewards only
                reward_tensor = reward_result["reward_tensor"]
                structured_rewards = None
            elif "structured_rewards" in reward_result:
                # New format - structured rewards (may also have reward_tensor)
                structured_rewards = reward_result["structured_rewards"]
                reward_tensor = reward_result.get("reward_tensor", None)
            else:
                raise ValueError("reward_result dict must contain either 'reward_tensor' or 'structured_rewards'")
            
            # Safely get reward_extra_info
            reward_extra_infos_dict = reward_result.get("reward_extra_info", {})
        elif isinstance(reward_result, list):
            # Fallback: if return_dict=True but got list, treat as structured_rewards
            structured_rewards = reward_result
            reward_tensor = None
            reward_extra_infos_dict = {}
        else:
            # Other format (e.g., tensor)
            reward_tensor = reward_result
            structured_rewards = None
            reward_extra_infos_dict = {}
    except Exception as e:
        print(f"Error in reward_fn: {e}")
        reward_result = reward_fn(data)
        reward_extra_infos_dict = {}
        
        # Handle both formats for fallback
        if isinstance(reward_result, list):
            # New format - structured rewards
            structured_rewards = reward_result
            reward_tensor = None
        else:
            # Old format - token-level rewards
            reward_tensor = reward_result
            structured_rewards = None

    # Create DataProto with appropriate format
    if structured_rewards is not None:
        # Store structured rewards in non_tensor_batch for multi-turn processing
        # If reward_tensor is available, use it as token_level_scores (already in correct format)
        batch_size = len(structured_rewards)
        import torch
        
        # Debug: Log what we're creating
        print(f"[DEBUG-REWARD] Creating DataProto with structured_rewards (batch_size={batch_size}), reward_tensor={'present' if reward_tensor is not None else 'None'}")
        
        # Check if reward_tensor was provided (from kg_format_multiturn)
        if reward_tensor is not None:
            # Use the reward_tensor directly as token_level_scores (already placed at last token)
            reward_data_proto = DataProto.from_dict(
                tensors={"token_level_scores": reward_tensor},
                non_tensors={"structured_rewards": structured_rewards}
            )
            print(f"[DEBUG-REWARD] Created DataProto with reward_tensor shape: {reward_tensor.shape}")
        else:
            # Fallback: create dummy tensor if reward_tensor not available
            # Need to match the shape of data.batch["responses"] for compatibility
            if hasattr(data, 'batch') and "responses" in data.batch:
                response_shape = data.batch["responses"].shape
                dummy_tensor = torch.zeros(response_shape, dtype=torch.float32)
                # Place reward values at the last valid token position for each sample
                for i in range(batch_size):
                    if i < len(data):
                        response_ids = data[i].batch["responses"]
                        attention_mask = data[i].batch["attention_mask"]
                        prompt_length = data[i].batch["prompts"].shape[-1]
                        valid_response_length = attention_mask[prompt_length:].sum()
                        if valid_response_length > 0:
                            reward_value = structured_rewards[i].get("total_score", 0.0) if isinstance(structured_rewards[i], dict) else 0.0
                            dummy_tensor[i, valid_response_length - 1] = reward_value
            else:
                # Last resort: create minimal dummy tensor
                dummy_tensor = torch.zeros(batch_size, 1, dtype=torch.float32)
            
            reward_data_proto = DataProto.from_dict(
                tensors={"token_level_scores": dummy_tensor},
                non_tensors={"structured_rewards": structured_rewards}
            )
            print(f"[DEBUG-REWARD] Created DataProto with dummy_tensor shape: {dummy_tensor.shape}")
        
        # Verify structured_rewards is in the DataProto
        if "structured_rewards" not in reward_data_proto.non_tensor_batch:
            print(f"[ERROR-REWARD] structured_rewards NOT in reward_data_proto.non_tensor_batch after creation!")
            print(f"[ERROR-REWARD] reward_data_proto.non_tensor_batch keys: {list(reward_data_proto.non_tensor_batch.keys())}")
            raise ValueError("Failed to create reward_data_proto with structured_rewards")
        else:
            print(f"[DEBUG-REWARD] Verified: structured_rewards is in reward_data_proto.non_tensor_batch")
    else:
        # Legacy format for single-turn
        reward_data_proto = DataProto.from_dict(
            tensors={"token_level_scores": reward_tensor}
        )
        print(f"[DEBUG-REWARD] Created DataProto in legacy format (no structured_rewards)")
    
    return reward_data_proto, reward_extra_infos_dict


@ray.remote(num_cpus=1)
def compute_reward_async(data: DataProto, config, tokenizer):
    """
    Load the reward manager and compute the reward for a batch of data.
    This is meant to be run in a separate Ray worker.
    """
    reward_fn = load_reward_manager(config, tokenizer, num_examine=0, **config.reward_model.get("reward_kwargs", {}))
    return compute_reward(data, reward_fn)

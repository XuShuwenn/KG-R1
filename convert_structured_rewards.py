"""
Utility helpers for converting structured multi-turn rewards into token-level tensors.

The training pipeline expects token-level rewards whenever we:
* apply KL penalty directly on rewards (so the KL term can be subtracted token-wise)
* compute PPO/GAE style advantages that operate over response tokens

`KGFormatMultiTurnRewardManager` produces *structured* rewards with two components:
    {
        "turn_rewards": {turn_id: scalar_reward, ...},
        "global_rewards": {"exact_match": ..., "retrieval_quality": ..., "_raw_*": ...}
    }

This module provides a single entry point, `convert_structured_to_token_level`, that maps
those structured rewards onto the per-token response span according to the requested
distribution strategy.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Union

import torch


def _normalize_turn_key(turn_key: Union[str, int]) -> Optional[int]:
    """Convert various turn key formats into integer turn IDs."""
    if isinstance(turn_key, int):
        return turn_key

    if isinstance(turn_key, str):
        # Accept formats like "turn_1", "Turn-2", or simple digit strings.
        digits = "".join(ch for ch in turn_key if ch.isdigit())
        if digits.isdigit():
            return int(digits)

    return None


def _sum_global_rewards(global_rewards: Dict[str, Any]) -> float:
    """Sum weighted global rewards, skipping any '_raw_' logging entries."""
    if not global_rewards:
        return 0.0

    return sum(
        float(value)
        for key, value in global_rewards.items()
        if not str(key).startswith("_raw_")
    )


def _get_valid_positions(mask_row: torch.Tensor) -> torch.Tensor:
    """Return indices of valid (non-zero) positions for a single sample."""
    if mask_row.ndim != 1:
        mask_row = mask_row.view(-1)
    return mask_row.nonzero(as_tuple=True)[0]


def convert_structured_to_token_level(
    structured_rewards: List[Dict[str, Any]],
    response_mask: torch.Tensor,
    turn_sequence_tensor: torch.Tensor,
    distribution_strategy: str = "turn_proportional",
) -> torch.Tensor:
    """
    Convert structured (turn + global) rewards into token-level tensors.

    Args:
        structured_rewards: List of dicts produced by KGFormatMultiTurnRewardManager.
        response_mask: Tensor of shape (batch, response_len) indicating valid response tokens.
        turn_sequence_tensor: Tensor of shape (batch, total_seq_len) mapping each token to a turn id.
        distribution_strategy: One of {"turn_proportional", "final_token_only"}.

    Returns:
        token_level_rewards: Tensor with the same shape as response_mask (float32).
    """

    if response_mask.ndim != 2:
        raise ValueError(f"response_mask must be 2D, got shape {response_mask.shape}")
    if turn_sequence_tensor.ndim != 2:
        raise ValueError(
            f"turn_sequence_tensor must be 2D, got shape {turn_sequence_tensor.shape}"
        )

    batch_size, response_len = response_mask.shape

    if turn_sequence_tensor.size(0) != batch_size:
        raise ValueError(
            "turn_sequence_tensor batch size mismatch: "
            f"{turn_sequence_tensor.size(0)} vs {batch_size}"
        )

    # Align turn ids with the response portion (right-aligned).
    if turn_sequence_tensor.size(1) < response_len:
        raise ValueError(
            "turn_sequence_tensor is shorter than response length. "
            f"turn_seq_len={turn_sequence_tensor.size(1)}, response_len={response_len}"
        )

    device = response_mask.device
    response_turn_tensor = turn_sequence_tensor[:, -response_len :].to(device)
    token_level = torch.zeros_like(response_mask, dtype=torch.float32, device=device)

    valid_mask = response_mask.bool()
    strategy = distribution_strategy.lower()

    for idx in range(batch_size):
        reward_entry = structured_rewards[idx] if idx < len(structured_rewards) else {}
        turn_rewards: Dict[Union[str, int], Any] = reward_entry.get("turn_rewards", {}) or {}
        global_rewards: Dict[str, Any] = reward_entry.get("global_rewards", {}) or {}

        sample_valid_positions = _get_valid_positions(valid_mask[idx])
        sample_turn_ids = response_turn_tensor[idx]

        if strategy == "turn_proportional":
            # Distribute each turn's reward evenly across the tokens belonging to that turn.
            if turn_rewards:
                for turn_key, turn_reward in turn_rewards.items():
                    turn_id = _normalize_turn_key(turn_key)
                    if turn_id is None:
                        continue

                    turn_mask = (sample_turn_ids == turn_id) & valid_mask[idx]
                    turn_positions = turn_mask.nonzero(as_tuple=True)[0]
                    if turn_positions.numel() == 0:
                        continue

                    reward_value = float(turn_reward)
                    if reward_value == 0.0:
                        continue

                    reward_per_token = reward_value / float(turn_positions.numel())
                    token_level[idx, turn_positions] += reward_per_token

            total_global_reward = _sum_global_rewards(global_rewards)
            if total_global_reward != 0.0 and sample_valid_positions.numel() > 0:
                per_token = total_global_reward / float(sample_valid_positions.numel())
                token_level[idx, sample_valid_positions] += per_token

        elif strategy == "final_token_only":
            # Place the combined reward on the last valid token (FormatRewardManager style).
            if sample_valid_positions.numel() == 0:
                continue

            if turn_rewards:
                turn_reward_mean = sum(float(v) for v in turn_rewards.values()) / len(turn_rewards)
            else:
                turn_reward_mean = 0.0

            total_reward = turn_reward_mean + _sum_global_rewards(global_rewards)
            if total_reward == 0.0:
                continue

            final_pos = sample_valid_positions[-1]
            token_level[idx, final_pos] += total_reward

        else:
            raise ValueError(
                f"Unsupported distribution_strategy '{distribution_strategy}'. "
                "Expected one of {'turn_proportional', 'final_token_only'}."
            )

    return token_level



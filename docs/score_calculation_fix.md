# Score 计算修复说明

## 问题描述

在计算 score 时，代码使用了 `batch.batch["response_mask"]`，这个值是通过 `compute_response_mask(batch)` 设置的，总是使用 `attention_mask`，包含了所有 tokens（包括 info tokens）。

这导致：
1. Score 包含了 info tokens（环境反馈）的 returns
2. 与 `apply_kl_penalty` 和 `compute_advantage` 的行为不一致（它们使用 `loss_mask` 排除 info tokens）
3. Score 不能准确反映模型策略的质量

## 修复方案

在计算 score 时，检查是否启用了 `state_masking`：
- **如果启用**：使用 `loss_mask` 排除 info tokens，只计算 state tokens 的 returns
- **如果未启用**：使用 `response_mask`（attention_mask）计算所有有效 tokens

## 修复后的逻辑

```python
if (self.use_search_generation and 
    self.config.actor_rollout_ref.actor.get('state_masking', False) and
    'loss_mask' in batch.batch):
    # 使用 loss_mask 排除 info tokens（只计算 state tokens）
    loss_mask = batch.batch['loss_mask']
    score_mask = loss_mask[:, -response_length:] if loss_mask.size(1) > response_length else loss_mask
    masked_returns = returns * score_mask
    scores = masked_returns.sum(-1).cpu().tolist()
else:
    # 使用 response_mask（attention_mask）计算所有有效 tokens
    response_mask = batch.batch.get("response_mask", None)
    # ...
```

## 一致性保证

修复后，所有相关计算都使用相同的 mask 逻辑：

| 阶段 | 使用的 Mask | 是否排除 Info Tokens |
|------|------------|---------------------|
| `apply_kl_penalty` (multi_turn=True) | `loss_mask` | ✅ 是 |
| `compute_advantage` (multi_turn=True) | `loss_mask` | ✅ 是 |
| 计算 score (multi_turn=True) | `loss_mask` | ✅ 是（修复后） |

## 影响

1. **Score 准确性**：Score 现在只反映模型策略的质量（state tokens），不包含环境反馈（info tokens）
2. **一致性**：与 KL penalty 和 advantage 计算保持一致
3. **训练目标**：符合训练目标（模型只对 state tokens 负责）

## 验证

修复后，score 计算应该：
- 在多轮对话中，如果启用了 `state_masking`，使用 `loss_mask` 排除 info tokens
- 在单轮对话中，或未启用 `state_masking` 时，使用 `response_mask`（attention_mask）
- 与 `apply_kl_penalty` 和 `compute_advantage` 的行为保持一致


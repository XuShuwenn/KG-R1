# Reward 计算方式总结

## 概述

当前系统使用**多轮对话结构化奖励**（Structured Multi-turn Rewards）机制，将奖励分为**轮次特定奖励**（turn-specific）和**全局奖励**（global），然后转换为 token 级别的奖励用于训练。

## 1. Reward 计算流程

### 1.1 整体流程

```
生成响应 (Generation)
    ↓
计算结构化奖励 (KGFormatMultiTurnRewardManager)
    ↓
转换为 Token-level 奖励 (convert_structured_to_token_level)
    ↓
应用 KL Penalty (apply_kl_penalty, 如果启用)
    ↓
计算 Advantage (compute_advantage: GAE 或 GRPO)
    ↓
计算 Score (用于轨迹保存)
```

### 1.2 详细步骤

#### 步骤 1: 生成响应
- **位置**: `ray_trainer_kg.py:1917-1934`
- **功能**: 通过 `LLMGenerationManager` 生成多轮对话响应
- **输出**: 
  - `responses`: 模型生成的 tokens（包含 state tokens 和 info tokens）
  - `turn_sequence_tensor`: 每个 token 所属的轮次编号
  - `info_mask`: 标记哪些 tokens 是 info tokens（环境反馈）

#### 步骤 2: 计算结构化奖励
- **位置**: `ray_trainer_kg.py:1965`
- **功能**: `KGFormatMultiTurnRewardManager` 计算结构化奖励
- **输出**: `structured_rewards` (List[Dict])
  ```python
  {
      "turn_rewards": {
          1: 0.15,  # 第1轮的奖励
          2: 0.20,  # 第2轮的奖励
          ...
      },
      "global_rewards": {
          "exact_match": 0.3,      # 精确匹配奖励
          "retrieval_quality": 0.4, # 检索质量奖励
          "_raw_exact_match": 0.6,   # 原始分数（仅用于日志）
          ...
      },
      "total_score": 0.65,  # 总分数
      ...
  }
  ```

**奖励组件**:
- **Turn-specific rewards** (每轮):
  - `kg_query_validity`: KG 查询有效性 (默认权重: 0.1)
  - `is_answer_score`: 答案分数 (默认权重: 0.1)
  - `format_score`: 格式分数 (默认权重: 0.15)
  
- **Global rewards** (全局):
  - `exact_match`: 精确匹配 (默认权重: 0.3)
  - `retrieval_quality`: 检索质量 (默认权重: 0.4)

**总分数计算**:
```python
total_turn_score = sum(turn_rewards.values()) / len(turn_rewards)  # 平均轮次奖励
total_global_score = sum(global_rewards.values())  # 全局奖励总和
total_score = total_turn_score + total_global_score
```

#### 步骤 3: 转换为 Token-level 奖励
- **位置**: `ray_trainer_kg.py:201` (KL penalty) 或 `ray_trainer_kg.py:350` (GAE)
- **功能**: 将结构化奖励转换为 token 级别的张量
- **模块**: `convert_structured_rewards.convert_structured_to_token_level`

**转换策略**:

1. **`turn_proportional`** (用于 KL penalty):
   - 将每轮的奖励均匀分配到该轮的所有 tokens
   - 将全局奖励均匀分配到所有有效 tokens
   ```python
   # 每轮奖励分配
   for turn_id, turn_reward in turn_rewards.items():
       turn_tokens = tokens where turn_sequence_tensor == turn_id
       reward_per_token = turn_reward / len(turn_tokens)
       token_level_rewards[turn_tokens] += reward_per_token
   
   # 全局奖励分配
   global_reward = sum(global_rewards.values())
   reward_per_token = global_reward / len(valid_tokens)
   token_level_rewards[valid_tokens] += reward_per_token
   ```

2. **`final_token_only`** (用于 GAE):
   - 将平均轮次奖励 + 全局奖励放在最后一个有效 token
   ```python
   turn_reward_mean = sum(turn_rewards.values()) / len(turn_rewards)
   total_reward = turn_reward_mean + sum(global_rewards.values())
   token_level_rewards[final_valid_token] = total_reward
   ```

#### 步骤 4: 应用 KL Penalty (可选)
- **位置**: `ray_trainer_kg.py:2050-2059`
- **条件**: `config.algorithm.use_kl_in_reward == True`
- **功能**: 从 token-level rewards 中减去 KL divergence 惩罚

**流程**:
```python
# 1. 转换结构化奖励为 token-level (使用 turn_proportional)
token_level_scores = convert_structured_to_token_level(
    structured_rewards=structured_rewards,
    response_mask=loss_mask,  # 排除 info tokens
    turn_sequence_tensor=turn_sequence_tensor,
    distribution_strategy="turn_proportional"
)

# 2. 计算 KL divergence
kld = kl_penalty(old_log_probs, ref_log_probs)
kld = kld * loss_mask  # 只计算 state tokens 的 KL

# 3. 应用 KL penalty
token_level_rewards = token_level_scores - beta * kld
```

**Mask 使用**:
- 如果 `multi_turn=True` 且 `state_masking=True`: 使用 `loss_mask`（排除 info tokens）
- 否则: 使用 `attention_mask`（包含所有有效 tokens）

#### 步骤 5: 计算 Advantage
- **位置**: `ray_trainer_kg.py:2074-2088`
- **模式**: GAE 或 GRPO

##### 5.1 GAE 模式

**流程**:
```python
# 1. 如果 token_level_rewards 不存在，转换结构化奖励
if "token_level_rewards" not in data.batch:
    token_level_rewards = convert_structured_to_token_level(
        structured_rewards=structured_rewards,
        response_mask=loss_mask,  # 排除 info tokens
        turn_sequence_tensor=turn_sequence_tensor,
        distribution_strategy="final_token_only"
    )

# 2. 计算 GAE advantage 和 returns
advantages, returns = compute_gae_advantage_return(
    token_level_rewards=token_level_rewards,
    values=values,  # Critic 网络的值估计
    response_mask=loss_mask,  # 排除 info tokens
    gamma=gamma,
    lam=lam,
)
```

**GAE 计算**:
```python
# 对于每个时间步 t (从后往前):
delta_t = r_t + gamma * V_{t+1} - V_t
A_t = delta_t + gamma * lam * A_{t+1}
returns_t = A_t + V_t
```

**Mask 使用**:
- 如果 `multi_turn=True` 且 `state_masking=True`: 使用 `loss_mask`（排除 info tokens）
- 否则: 使用 `attention_mask`（包含所有有效 tokens）

##### 5.2 GRPO 模式

**流程**:
```python
# 1. 选择计算模式
if enable_multiturn_advantage:
    # 多轮 GRPO: 分别计算轮次奖励和全局奖励的 advantage
    advantages, returns = compute_grpo_multiturn_advantage(
        structured_rewards=structured_rewards,
        response_mask=loss_mask,  # 排除 info tokens
        turn_sequence_tensor=turn_sequence_tensor,
        index=uid,  # 用于 GRPO 分组
    )
else:
    # 标准 GRPO: 统一计算
    advantages, returns = compute_grpo_outcome_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=loss_mask,  # 排除 info tokens
        index=uid,
    )
```

**GRPO 计算**:
```python
# 1. 计算每个样本的总奖励
scores = token_level_rewards.sum(dim=-1)  # 或从 structured_rewards 提取

# 2. 按 uid 分组，计算组内均值和标准差
for uid in unique_uids:
    group_scores = [scores[i] for i where uid[i] == uid]
    mean = mean(group_scores)
    std = std(group_scores)

# 3. 计算 advantage (归一化)
advantage = (score - mean) / (std + epsilon)

# 4. 应用到所有 action tokens
advantages[action_tokens] = advantage
```

**Mask 使用**:
- 如果 `multi_turn=True` 且 `state_masking=True`: 使用 `loss_mask`（排除 info tokens）
- 否则: 使用 `attention_mask`（包含所有有效 tokens）

#### 步骤 6: 计算 Score (用于轨迹保存)
- **位置**: `ray_trainer_kg.py:2262-2297`
- **功能**: 计算每个样本的总分数，用于保存到轨迹文件

**计算方式**:
```python
if "returns" in batch.batch:
    returns = batch.batch["returns"]
    if returns.dim() > 1:  # Token-level returns
        if multi_turn and state_masking:
            # 使用 loss_mask 排除 info tokens
            score_mask = loss_mask[:, -response_length:]
            masked_returns = returns * score_mask
            scores = masked_returns.sum(-1).cpu().tolist()
        else:
            # 使用 response_mask (attention_mask)
            masked_returns = returns * response_mask
            scores = masked_returns.sum(-1).cpu().tolist()
    else:
        # 已经是 per-sample
        scores = returns.cpu().tolist()
```

## 2. Mask 使用总结

### 2.1 Mask 类型

| Mask 类型 | 含义 | 包含 Info Tokens? |
|-----------|------|------------------|
| `attention_mask` | 所有有效 tokens | ✅ 是 |
| `info_mask` | State tokens = 1, Info tokens = 0 | ❌ 否 |
| `loss_mask` | 从 `info_mask` 提取 response 部分 | ❌ 否 |
| `response_mask` | 通常 = `attention_mask` | ✅ 是 |

### 2.2 各阶段 Mask 使用

| 阶段 | 条件 | 使用的 Mask | 是否排除 Info Tokens |
|------|------|------------|---------------------|
| `apply_kl_penalty` | `multi_turn=True` | `loss_mask` | ✅ 是 |
| `apply_kl_penalty` | `multi_turn=False` | `attention_mask` | ❌ 否 |
| `compute_advantage` (GAE) | `multi_turn=True` | `loss_mask` | ✅ 是 |
| `compute_advantage` (GAE) | `multi_turn=False` | `attention_mask` | ❌ 否 |
| `compute_advantage` (GRPO) | `multi_turn=True` | `loss_mask` | ✅ 是 |
| `compute_advantage` (GRPO) | `multi_turn=False` | `attention_mask` | ❌ 否 |
| 计算 score | `multi_turn=True` | `loss_mask` | ✅ 是 |
| 计算 score | `multi_turn=False` | `attention_mask` | ❌ 否 |

## 3. Reward 转换策略对比

### 3.1 turn_proportional (用于 KL penalty)

**优点**:
- 奖励均匀分配到每轮的所有 tokens
- 适合需要 token-level KL penalty 的场景
- 保持轮次内的奖励分布

**缺点**:
- 可能导致奖励信号稀释（如果轮次很长）

### 3.2 final_token_only (用于 GAE)

**优点**:
- 将所有奖励集中在最后一个 token
- 符合 PPO/GAE 的奖励信号设计
- 避免长度偏差（length bias）

**缺点**:
- 丢失了轮次内的奖励分布信息

## 4. 关键设计决策

### 4.1 为什么使用结构化奖励？

1. **灵活性**: 可以分别控制轮次奖励和全局奖励
2. **可解释性**: 可以单独查看每个组件的贡献
3. **可扩展性**: 容易添加新的奖励组件

### 4.2 为什么在 GAE 中使用 final_token_only？

- **历史原因**: 与 `FormatRewardManager` 的设计保持一致
- **PPO 特性**: PPO/GAE 通常将奖励放在序列末尾
- **避免长度偏差**: 防止长序列获得更多奖励

### 4.3 为什么在 KL penalty 中使用 turn_proportional？

- **Token-level KL**: KL penalty 需要 token-level 的奖励
- **轮次信息**: 保持轮次内的奖励分布
- **更细粒度**: 提供更细粒度的奖励信号

### 4.4 为什么使用 loss_mask 排除 info tokens？

- **训练目标**: 模型只应该对 state tokens（模型输出）负责
- **一致性**: 与 loss 计算保持一致
- **正确性**: Info tokens 是环境反馈，不应该影响模型训练

## 5. 配置参数

### 5.1 Reward Manager 配置

```python
# Turn-specific weights
turn_kg_query_validity = 0.1
turn_is_answer_score = 0.1
turn_format_score = 0.15

# Global weights
global_exact_match = 0.3
global_retrieval_quality = 0.4
```

### 5.2 Training 配置

```python
# KL penalty
use_kl_in_reward = True/False
kl_penalty = "kl"

# Advantage estimator
adv_estimator = "GAE" or "GRPO"
gamma = 1.0
lam = 1.0

# Multi-turn
state_masking = True/False
enable_multiturn_advantage = True/False
```

## 6. 总结

当前 reward 计算方式的核心特点：

1. **结构化设计**: 将奖励分为轮次特定和全局两部分
2. **灵活转换**: 根据使用场景选择不同的转换策略
3. **Mask 一致性**: 在多轮对话中统一使用 `loss_mask` 排除 info tokens
4. **模式支持**: 同时支持 GAE 和 GRPO 两种 advantage 计算模式
5. **可配置性**: 通过配置参数控制奖励权重和计算方式

这种设计既保持了灵活性，又确保了训练的正确性和一致性。


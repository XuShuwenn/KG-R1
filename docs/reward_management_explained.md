# 训练端 Reward 管理机制详解

## 概述

训练端使用 **`kg_format_multiturn`** Reward Manager，这是一个专门为多轮知识图谱推理设计的奖励系统。它继承自 `KGFormatRewardManager`，提供了 turn-wise（每轮）和 global（全局）两套奖励机制。

## Reward 组成部分

### 1. Turn-wise Rewards（每轮奖励）

每轮对话都会计算以下三个组件：

#### 1.1 `kg_query_validity`（KG 查询合法性）
- **计算方式**：
  - 仅对 `kg-query` 动作计算
  - 检查查询是否语法有效（`valid_action == True`）
  - 检查查询是否成功执行（`kg_metadata.success == True` 且 `error_type == KG_SUCCESS`）
  - **去重机制**：使用 `extract_query_identifier` 提取唯一查询 ID，重复查询不给予奖励（防止 reward hacking）
- **原始分数**：0.0（无效/重复）或 1.0（唯一且成功）
- **加权后分数**：`kg_query_validity * turn_kg_query_validity_weight`
- **默认权重**：`turn_kg_query_validity=0.5`（可在训练脚本中配置）

#### 1.2 `is_answer_score`（是否给出答案）
- **计算方式**：
  - 仅对 `answer` 动作计算
  - 二进制奖励：如果该轮是 `answer` 动作，返回 1.0；否则 0.0
- **原始分数**：0.0 或 1.0
- **加权后分数**：`is_answer_score * turn_is_answer_score_weight`
- **默认权重**：`turn_is_answer_score=0.5`

#### 1.3 `format_score`（格式合规性）
- **计算方式**：
  - 适用于 `kg-query` 和 `answer` 两种动作
  - 从 `interaction_history['responses_str']` 提取该轮的实际响应内容
  - **严格格式检查**：
    - **kg-query 格式**：必须完全匹配 `^<think>.*?</think>\s*<kg-query>.*?</kg-query>$`
      - 必须包含且仅包含一对 `<think>` 标签
      - 必须包含且仅包含一对 `<kg-query>` 标签
      - 两个标签之间只能有空白字符（空格、制表符、换行）
    - **answer 格式**：必须完全匹配 `^<think>.*?</think>\s*<answer>.*?</answer>$`
      - 必须包含且仅包含一对 `<think>` 标签
      - 必须包含且仅包含一对 `<answer>` 标签
      - 两个标签之间只能有空白字符
- **原始分数**：0.0（格式不符合）或 1.0（格式完全符合）
- **加权后分数**：`format_score * turn_format_score_weight`
- **默认权重**：`turn_format_score=0.5`

#### Turn-wise 总奖励计算

对于每一轮：
- **kg-query 动作**：`format_score * turn_format_score_weight + kg_query_validity * turn_kg_query_validity_weight`
- **answer 动作**：`format_score * turn_format_score_weight + is_answer_score * turn_is_answer_score_weight`

所有轮次的 turn-wise 奖励会被**平均**：
```
total_turn_score = sum(turn_rewards.values()) / len(turn_rewards)
```

### 2. Global Rewards（全局奖励）

在整个对话序列结束后计算，包含两个组件：

#### 2.1 `exact_match`（精确匹配）
- **计算方式**：
  - 从完整响应中提取 `<answer>` 标签内的答案
  - 解析 ground truth（支持 `target_text` 和 `target_kb_id` 两种形式）
  - 使用 `em_check_kg` 进行答案归一化和匹配
  - **支持两种评分模式**：
    - `binary` 模式（默认）：实体级精确匹配，返回 0.0 或 1.0
    - `f1` 模式：实体级 F1 分数（支持多实体答案）
  - **MultiTQ 数据集特殊处理**：自动处理年份 vs 年月的时间粒度问题
- **原始分数**：0.0-1.0（binary 模式）或 0.0-1.0（f1 模式）
- **加权后分数**：`exact_match_raw * global_exact_match_weight`
- **默认权重**：`global_exact_match=0.5`
- **可选 OTC 缩放**：如果启用 `otc_scaling`，会根据使用的 KG 查询轮次进行指数衰减缩放

#### 2.2 `retrieval_quality`（检索质量）
- **计算方式**：
  - 检查所有轮次的检索结果中是否包含正确答案
  - 使用 `is_retrieval_correct_kg` 函数检查
  - 如果任何一轮的检索结果包含正确答案，返回 1.0；否则 0.0
  - **二进制评分**：不提供部分奖励（要么找到答案，要么没找到）
- **原始分数**：0.0 或 1.0
- **加权后分数**：`retrieval_quality_raw * global_retrieval_quality_weight`
- **默认权重**：`global_retrieval_quality=0.5`

#### Global 总奖励计算

```
total_global_score = exact_match_weighted + retrieval_quality_weighted
```

### 3. 最终总奖励

```
total_score = total_turn_score + total_global_score
```

其中：
- `total_turn_score = avg(turn_rewards)`（所有轮次奖励的平均值）
- `total_global_score = exact_match_weighted + retrieval_quality_weighted`

## Reward 计算流程

### 1. 训练循环中的调用

在 `ray_trainer_kg.py` 的训练循环中：

```python
# 1. 生成响应（包含多轮 KG 交互）
batch = self.actor_wg.generate(batch, ...)

# 2. 计算 Reward
reward_data_proto, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)
batch = batch.union(reward_data_proto)

# 3. 应用 KL 惩罚（如果启用）
if self.config.algorithm.use_kl_in_reward:
    batch, kl_metrics = apply_kl_penalty(batch, ...)
else:
    batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

# 4. 计算 Advantage
advantages = compute_grpo_uniform_advantage(...)
```

### 2. Reward Manager 内部计算

`KGFormatMultiTurnRewardManager.__call__` 方法：

1. **解析交互历史**：从 `interaction_history` 中提取每轮的动作、搜索结果、服务器响应等
2. **计算 Turn-wise Rewards**：对每一轮调用 `_calculate_turn_reward_with_components`
3. **计算 Global Rewards**：调用 `_calculate_global_rewards`
4. **汇总总奖励**：`total_score = avg(turn_rewards) + sum(global_rewards)`
5. **转换为 Token-level**：将总奖励放置在响应序列的最后一个 token 位置

### 3. Token-level Reward 分配

虽然计算的是序列级奖励，但为了与 PPO 框架兼容，奖励会被放置在：
- **位置**：响应序列的最后一个有效 token
- **值**：`total_score`
- **其他 token**：0.0（在计算 advantage 时会通过 mask 处理）

## 配置参数

在训练脚本中（`train_debug_single_a100_7b.sh`）：

```bash
reward_model.reward_manager=kg_format_multiturn \
+reward_model.reward_kwargs.turn_kg_query_validity=0.5 \
+reward_model.reward_kwargs.turn_is_answer_score=0.5 \
+reward_model.reward_kwargs.turn_format_score=0.5 \
+reward_model.reward_kwargs.global_exact_match=0.5 \
+reward_model.reward_kwargs.global_retrieval_quality=0.5 \
```

### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `turn_kg_query_validity` | float | 0.5 | 每轮 KG 查询合法性的权重 |
| `turn_is_answer_score` | float | 0.5 | 每轮是否给出答案的权重 |
| `turn_format_score` | float | 0.5 | 每轮格式合规性的权重 |
| `global_exact_match` | float | 0.5 | 全局精确匹配的权重 |
| `global_retrieval_quality` | float | 0.5 | 全局检索质量的权重 |
| `answer_score_mode` | str | "binary" | 答案评分模式：`"binary"` 或 `"f1"` |
| `otc_scaling` | bool | False | 是否启用 OTC（Optimal Turn Count）缩放 |
| `max_turns` | int | 7 | OTC 缩放的最大轮次（仅当 `otc_scaling=True` 时有效） |
| `verbose` | bool | False | 是否打印详细的奖励计算日志 |

## 奖励计算示例

假设一个 3 轮对话：

### 轮次 1：kg-query
- `format_score = 1.0`（格式正确）
- `kg_query_validity = 1.0`（唯一且成功的查询）
- `turn_reward_1 = 1.0 * 0.5 + 1.0 * 0.5 = 1.0`

### 轮次 2：kg-query
- `format_score = 1.0`（格式正确）
- `kg_query_validity = 0.0`（重复查询）
- `turn_reward_2 = 1.0 * 0.5 + 0.0 * 0.5 = 0.5`

### 轮次 3：answer
- `format_score = 1.0`（格式正确）
- `is_answer_score = 1.0`（给出了答案）
- `turn_reward_3 = 1.0 * 0.5 + 1.0 * 0.5 = 1.0`

### Turn-wise 总奖励
```
total_turn_score = (1.0 + 0.5 + 1.0) / 3 = 0.833
```

### Global 奖励
- `exact_match_raw = 1.0`（答案正确）
- `retrieval_quality_raw = 1.0`（检索结果包含正确答案）
- `exact_match_weighted = 1.0 * 0.5 = 0.5`
- `retrieval_quality_weighted = 1.0 * 0.5 = 0.5`
- `total_global_score = 0.5 + 0.5 = 1.0`

### 最终总奖励
```
total_score = 0.833 + 1.0 = 1.833
```

## 与 PPO 的集成

### 1. Advantage 计算

在 GRPO（Group Relative Policy Optimization）模式下，advantage 计算会：
1. 将结构化奖励转换为统一的序列级奖励
2. 对同一 prompt 的多个响应进行分组
3. 计算组内均值和标准差
4. 使用组内标准化计算 advantage

### 2. KL 惩罚

如果启用 `algorithm.use_kl_in_reward`：
- 计算当前策略与参考策略之间的 KL 散度
- 从 token-level rewards 中减去 `beta * kld`
- 其中 `beta` 由 `AdaptiveKLController` 动态调整

### 3. State Masking

如果启用 `actor.state_masking`：
- 使用 `loss_mask` 来区分状态 token 和动作 token
- 只对动作 token 计算 loss 和 advantage
- 状态 token（如 `<information>` 内容）不参与梯度更新

## 调试和监控

### 1. 详细日志

设置 `verbose=True` 可以查看：
- 每轮的原始组件分数
- 权重应用后的分数
- 最终计算过程
- 格式验证结果

### 2. WandB 指标

Reward Manager 会自动记录以下指标到 WandB：
- `exact_match`：精确匹配分数
- `exact_match_binary`：二进制精确匹配
- `f1`、`precision`、`recall`：实体级 F1 指标
- `retrieval_quality`：检索质量分数
- `turn_kg_query_validity`：平均 KG 查询合法性
- `turn_format_score`：平均格式分数
- `turn_is_answer_score`：是否给出答案（二进制）
- `num_turns`：对话轮次数
- `total_score`：总奖励分数

### 3. 调试日志示例

```
[KG Multi-Turn Format Reward Manager - cwq_kg] - Sample 1
================================================================================
[turn_rewards]
  Turn 1 (kg-query):
    [raw_components] format=1.0, kg_validity=1.0, answer_validity=0.0
    [weights] format=0.5, kg_validity=0.5, answer_validity=0.5
    [weighted_scores] format=0.500, kg_validity=0.500, answer_validity=0.000
    [calculation] 0.500 + 0.500 = 1.000
    [final] expected=1.000, actual=1.000 ✅ CORRECT

[global_rewards]
  [raw_components] exact_match=1.000 (binary=1.000, f1=1.000), retrieval_quality=1.000
  [weights] exact_match=0.5, retrieval_quality=0.5, otc_scaling=1.000
  [weighted_scores] exact_match=0.500, retrieval_quality=0.500
  [calculation] 0.500 + 0.500 = 1.000
  [total_global] 1.000

[final_calculation]
  [components] turn_total=0.833, global_total=1.000
  [calculation] 0.833 + 1.000 = 1.833
  [final] expected=1.833, actual=1.833 ✅ CORRECT
```

## 关键代码位置

1. **Reward Manager 定义**：`verl/workers/reward_manager/kg_format_multiturn.py`
2. **奖励计算函数**：`verl/utils/reward_score/qa_em_format_kg.py`
3. **训练循环集成**：`verl/trainer/ppo/ray_trainer_kg.py` (line 1943-1953)
4. **Advantage 计算**：`verl/trainer/ppo/core_algos.py` (compute_grpo_uniform_advantage)
5. **KL 惩罚应用**：`verl/trainer/ppo/ray_trainer_kg.py` (apply_kl_penalty)

## 注意事项

1. **去重机制**：`kg_query_validity` 使用查询 ID 去重，防止模型通过重复查询刷奖励
2. **格式严格性**：`format_score` 要求严格的 MDP 格式，不允许额外文本
3. **二进制 vs F1**：`exact_match` 支持两种评分模式，F1 模式更适合多实体答案
4. **OTC 缩放**：如果启用，会根据使用的 KG 查询轮次对全局奖励进行缩放，鼓励更高效的推理
5. **Token-level 分配**：虽然计算的是序列级奖励，但为了兼容 PPO，奖励被放置在最后一个 token


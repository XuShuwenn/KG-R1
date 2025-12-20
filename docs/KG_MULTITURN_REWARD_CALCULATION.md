# KG-R1 多轮对话模式下的 Reward 计算详解

## 一、概述

KG-R1 的多轮对话模式使用 `KGFormatMultiTurnRewardManager` 来计算奖励。该管理器继承自 `KGFormatRewardManager`，专门为多轮KG交互设计。

**核心文件**: `verl/workers/reward_manager/kg_format_multiturn.py`

## 二、奖励结构

奖励分为两个层次：

### 2.1 轮次特定奖励（Turn-Specific Rewards）

每轮对话都会计算独立的奖励，包括：

1. **KG查询有效性** (`kg_query_validity`)
   - 权重: `turn_kg_query_validity` (默认: 0.1)
   - 计算方式: 
     - 检查查询是否语法有效 (`valid_action`)
     - 检查查询是否成功执行（通过 `kg_metadata.success`）
     - 检查查询是否唯一（防止重复查询获得奖励）
   - 奖励值: 
     - 1.0: 唯一且成功的KG查询
     - 0.0: 无效、失败或重复的查询

2. **答案分数** (`is_answer_score`)
   - 权重: `turn_is_answer_score` (默认: 0.1)
   - 计算方式: 二进制奖励，如果该轮产生了答案则为1.0，否则为0.0
   - 奖励值:
     - 1.0: 该轮包含 `<answer>...</answer>` 标签
     - 0.0: 该轮不包含答案

3. **格式分数** (`format_score`)
   - 权重: `turn_format_score` (默认: 0.15)
   - 计算方式: 检查该轮输出是否符合MDP格式要求
   - KG查询格式要求:
     ```
     <think>...</think>\s*<kg-query>...</kg-query>
     ```
   - 答案格式要求:
     ```
     <think>...</think>\s*<answer>...</answer>
     ```
   - 奖励值:
     - 1.0: 格式完全符合要求
     - 0.0: 格式不符合要求

### 2.2 全局奖励（Global Rewards）

对整个对话序列计算的奖励，包括：

1. **精确匹配** (`exact_match`)
   - 权重: `global_exact_match` (默认: 0.3)
   - 计算方式: 
     - 从最终响应中提取答案（`<answer>...</answer>` 标签内的内容）
     - 与ground truth进行实体级别的精确匹配
     - 支持两种模式:
       - `binary` 模式: 使用实体级别的精确匹配（0或1）
       - `f1` 模式: 使用实体级别的F1分数（0-1之间）
   - 奖励值: 根据 `answer_score_mode` 配置决定（默认 `binary`）

2. **检索质量** (`retrieval_quality`)
   - 权重: `global_retrieval_quality` (默认: 0.4)
   - 计算方式: 检查整个对话过程中，是否有任何一轮的检索结果包含正确答案
   - 奖励值:
     - 1.0: 任何一轮的检索结果包含正确答案
     - 0.0: 所有检索结果都不包含正确答案

## 三、奖励计算公式

### 3.1 轮次奖励计算

对于每一轮 `turn_num`，根据动作类型计算奖励：

**KG查询动作** (`kg-query`):
```python
turn_reward = format_score * 0.15 + kg_query_validity * 0.1
```

**答案动作** (`answer`):
```python
turn_reward = format_score * 0.15 + is_answer_score * 0.1
```

### 3.2 全局奖励计算

```python
# 应用OTC缩放（如果启用）
if otc_scaling:
    scaling_factor = e^(1 - kg_turns_used / max_turns)
    exact_match_scaled = exact_match_raw * scaling_factor
    retrieval_quality_scaled = retrieval_quality_raw * scaling_factor
else:
    exact_match_scaled = exact_match_raw
    retrieval_quality_scaled = retrieval_quality_raw

# 应用权重
exact_match_weighted = exact_match_scaled * 0.3
retrieval_quality_weighted = retrieval_quality_scaled * 0.4

total_global_score = exact_match_weighted + retrieval_quality_weighted
```

### 3.3 最终总奖励

```python
# 轮次奖励的平均值
total_turn_score = sum(turn_rewards.values()) / len(turn_rewards)

# 全局奖励的总和
total_global_score = exact_match_weighted + retrieval_quality_weighted

# 最终奖励
total_score = total_turn_score + total_global_score
```

## 四、详细计算流程

### 4.1 主计算函数：`_calculate_multiturn_rewards()`

**位置**: `kg_format_multiturn.py:141-225`

**流程**:

1. **解析交互历史**:
   ```python
   turn_data = self._parse_turns_from_interaction_history(
       sample_interaction_history, sample_idx
   )
   ```
   - 从 `interaction_history` 中提取每轮的动作、检索结果、有效性等信息
   - 返回字典: `{turn_num: {action, search_result, valid_action, is_search, ...}}`

2. **计算轮次特定奖励**:
   ```python
   for turn_num, turn_info in turn_data.items():
       turn_reward, components = self._calculate_turn_reward_with_components(
           turn_info, turn_num, assistant_response, seen_query_ids, ...
       )
       turn_rewards[turn_num] = turn_reward
   ```

3. **计算全局奖励**:
   ```python
   global_rewards = self._calculate_global_rewards(
       data_item, sample_interaction_history
   )
   ```

4. **汇总最终分数**:
   ```python
   total_turn_score = sum(turn_rewards.values()) / len(turn_rewards)
   total_global_score = sum(v for k, v in global_rewards.items() 
                           if not k.startswith('_'))
   total_score = total_turn_score + total_global_score
   ```

### 4.2 轮次奖励计算：`_calculate_turn_reward_with_components()`

**位置**: `kg_format_multiturn.py:263-310`

**流程**:

1. **计算格式分数**:
   ```python
   format_reward, extracted_content = self._calculate_turn_format_score_reward(
       turn_info, full_response, data_item, sample_idx, sample_interaction_history
   )
   format_score = format_reward * 0.15
   ```

2. **根据动作类型计算奖励**:
   
   **KG查询动作**:
   ```python
   kg_query_reward = self._calculate_kg_query_validity_reward(
       turn_info, seen_query_ids
   )
   kg_query_score = kg_query_reward * 0.1
   total_reward = format_score + kg_query_score
   ```
   
   **答案动作**:
   ```python
   is_answer_reward = self._calculate_is_answer_score_reward(turn_info, data_item)
   is_answer_score = is_answer_reward * 0.1
   total_reward = format_score + is_answer_score
   ```

### 4.3 KG查询有效性计算：`_calculate_kg_query_validity_reward()`

**位置**: `kg_format_multiturn.py:330-378`

**检查项**:

1. **动作类型检查**: 必须是 `kg-query` 动作
2. **语法有效性**: `valid_action` 必须为 True
3. **执行成功性**: 
   ```python
   kg_metadata = raw_response.get('kg_metadata', {})
   success = kg_metadata.get('success', False)
   error_type = kg_metadata.get('error_type')
   ```
   - 必须 `success=True` 且 `error_type='KG_SUCCESS'`
4. **唯一性检查**: 
   ```python
   query_id = self._extract_query_identifier(raw_response)
   if query_id in seen_query_ids:
       return 0.0  # 重复查询，无奖励
   else:
       seen_query_ids.add(query_id)
       return 1.0  # 唯一查询，给予奖励
   ```

### 4.4 格式分数计算：`_calculate_turn_format_score_reward()`

**位置**: `kg_format_multiturn.py:402-450`

**流程**:

1. **提取轮次内容**:
   ```python
   turn_content = self._extract_turn_content_server_interaction(
       turn_info, sample_interaction_history
   )
   ```
   - 优先使用 `responses_str` 从服务器交互历史中提取
   - 如果失败，回退到基于tensor的提取方法

2. **格式验证**:
   
   **KG查询格式** (`_has_proper_kg_query_format`):
   ```python
   pattern = r'^<think>.*?</think>\s*<kg-query>.*?</kg-query>$'
   ```
   - 必须严格匹配此模式
   - 标签之间只能有空白字符
   - 必须恰好有1个 `<think>` 和1个 `<kg-query>` 标签
   
   **答案格式** (`_has_proper_answer_format`):
   ```python
   pattern = r'^<think>.*?</think>\s*<answer>.*?</answer>$'
   ```
   - 必须严格匹配此模式
   - 标签之间只能有空白字符
   - 必须恰好有1个 `<think>` 和1个 `<answer>` 标签

### 4.5 全局奖励计算：`_calculate_global_rewards()`

**位置**: `kg_format_multiturn.py:847-908`

**流程**:

1. **计算精确匹配分数**:
   ```python
   exact_match_score_raw = self._calculate_exact_match_reward(
       data_item, interaction_history
   )
   ```
   - 提取最终答案: `extract_answer_kg(assistant_response)`
   - 解析ground truth: `_parse_ground_truth(ground_truth_raw)`
   - 实体级别匹配: `_calculate_entity_level_match(...)`
   - 根据 `answer_score_mode` 选择分数:
     - `binary`: 使用 `entity_level_exact_match` (0或1)
     - `f1`: 使用 `entity_level_f1` (0-1之间)

2. **计算检索质量分数**:
   ```python
   retrieval_quality_score_raw = self._calculate_global_retrieval_quality_reward(
       data_item, interaction_history
   )
   ```
   - 检查所有轮次的检索结果: `_check_global_retrieval_contains_answer(...)`
   - 使用 `is_retrieval_correct_kg()` 函数检查是否包含正确答案
   - 二进制评分: 1.0（包含）或 0.0（不包含）

3. **应用OTC缩放**（如果启用）:
   ```python
   if self.otc_scaling:
       kg_turns_used = self._count_kg_query_turns(interaction_history)
       ratio = kg_turns_used / self.max_turns
       scaling_factor = math.e ** (1 - ratio)
       exact_match_scaled = exact_match_raw * scaling_factor
       retrieval_quality_scaled = retrieval_quality_raw * scaling_factor
   ```
   - OTC (Optimal Turn Count) 缩放鼓励使用更少的轮次
   - 公式: `e^(1 - turns_used / max_turns)`
   - 0轮时缩放因子为 `e ≈ 2.718`
   - `max_turns` 轮时缩放因子为 `1.0`

4. **应用权重**:
   ```python
   global_rewards['exact_match'] = exact_match_scaled * 0.3
   global_rewards['retrieval_quality'] = retrieval_quality_scaled * 0.4
   ```

### 4.6 精确匹配计算：`_calculate_exact_match_reward()`

**位置**: `kg_format_multiturn.py:953-1085`

**流程**:

1. **提取完整解决方案文本**:
   ```python
   full_solution_text = self._extract_full_solution_text(data_item)
   assistant_response = extract_assistant_response(full_solution_text)
   ```

2. **提取预测答案**:
   ```python
   predicted_answer = extract_answer_kg(assistant_response)
   ```

3. **解析ground truth**:
   ```python
   ground_truth = self._parse_ground_truth(ground_truth_raw)
   ```
   - 支持多种格式: 字符串、列表、字典（包含 `target_text` 和 `target_kb_id`）

4. **实体级别匹配**:
   ```python
   entity_level_exact_match = self._calculate_entity_level_match(
       predicted_answer, ground_truth_answers, 
       interaction_history=interaction_history, 
       dataset_name=data_source
   )
   ```
   - 使用 `em_check_kg()` 函数进行匹配
   - 支持MultiTQ数据集的时间粒度处理（年 vs 年月）

5. **计算F1分数**（如果使用f1模式）:
   ```python
   entity_metrics = self._calculate_entity_level_f1_precision_recall(
       predicted_answer, ground_truth_raw
   )
   ```

### 4.7 检索质量计算：`_calculate_global_retrieval_quality_reward()`

**位置**: `kg_format_multiturn.py:1290-1338`

**流程**:

1. **获取ground truth答案**:
   ```python
   ground_truth = self._parse_ground_truth(ground_truth_raw)
   ```

2. **检查全局检索**:
   ```python
   retrieval_contains_answer = self._check_global_retrieval_contains_answer(
       interaction_history, ground_truth_answers
   )
   ```
   - 使用 `is_retrieval_correct_kg()` 函数
   - 检查所有轮次的 `search_results` 和 `raw_server_responses`

3. **二进制评分**:
   ```python
   if retrieval_contains_answer:
       retrieval_score = 1.0
   else:
       retrieval_score = 0.0
   ```

## 五、配置参数

### 5.1 轮次特定权重

```python
turn_specific_weights = {
    'kg_query_validity': 0.1,      # KG查询有效性权重
    'is_answer_score': 0.1,         # 答案分数权重
    'format_score': 0.15,           # 格式分数权重
}
```

### 5.2 全局权重

```python
global_weights = {
    'exact_match': 0.3,              # 精确匹配权重
    'retrieval_quality': 0.4,       # 检索质量权重
}
```

### 5.3 其他配置

- `answer_score_mode`: `'binary'` 或 `'f1'` (默认: `'binary'`)
- `otc_scaling`: 是否启用OTC缩放 (默认: `False`)
- `max_turns`: 最大轮次数，用于OTC缩放 (默认: 7)
- `verbose`: 是否输出详细日志 (默认: `False`)

## 六、奖励计算示例

### 示例1: 3轮对话，全部成功

**轮次1**: KG查询
- 格式分数: 1.0 → 加权: 0.15
- KG查询有效性: 1.0 → 加权: 0.1
- 轮次奖励: 0.15 + 0.1 = 0.25

**轮次2**: KG查询
- 格式分数: 1.0 → 加权: 0.15
- KG查询有效性: 1.0 → 加权: 0.1
- 轮次奖励: 0.15 + 0.1 = 0.25

**轮次3**: 答案
- 格式分数: 1.0 → 加权: 0.15
- 答案分数: 1.0 → 加权: 0.1
- 轮次奖励: 0.15 + 0.1 = 0.25

**轮次平均**: (0.25 + 0.25 + 0.25) / 3 = 0.25

**全局奖励**:
- 精确匹配: 1.0 → 加权: 0.3
- 检索质量: 1.0 → 加权: 0.4
- 全局总分: 0.3 + 0.4 = 0.7

**最终奖励**: 0.25 + 0.7 = **0.95**

### 示例2: 2轮对话，格式错误

**轮次1**: KG查询（格式错误）
- 格式分数: 0.0 → 加权: 0.0
- KG查询有效性: 1.0 → 加权: 0.1
- 轮次奖励: 0.0 + 0.1 = 0.1

**轮次2**: 答案（格式正确）
- 格式分数: 1.0 → 加权: 0.15
- 答案分数: 1.0 → 加权: 0.1
- 轮次奖励: 0.15 + 0.1 = 0.25

**轮次平均**: (0.1 + 0.25) / 2 = 0.175

**全局奖励**:
- 精确匹配: 1.0 → 加权: 0.3
- 检索质量: 0.0 → 加权: 0.0
- 全局总分: 0.3 + 0.0 = 0.3

**最终奖励**: 0.175 + 0.3 = **0.475**

## 七、关键设计特点

### 7.1 防止奖励黑客（Reward Hacking）

- **唯一性检查**: 重复的KG查询不会获得奖励
- **格式严格性**: 必须完全符合MDP格式要求
- **二进制评分**: 检索质量使用二进制评分，避免部分奖励

### 7.2 多轮支持

- **轮次独立性**: 每轮独立计算奖励
- **全局评估**: 最终答案和整体检索质量单独评估
- **交互历史**: 使用 `interaction_history` 追踪所有轮次的信息

### 7.3 实体级别匹配

- **精确匹配**: 使用实体级别的匹配，而非字符串匹配
- **MultiTQ支持**: 特殊处理时间粒度（年 vs 年月）
- **F1模式**: 支持F1分数用于多实体答案

### 7.4 OTC缩放（可选）

- **鼓励效率**: 使用更少轮次获得更高奖励
- **指数衰减**: 从 `e` 衰减到 `1.0`
- **可配置**: 通过 `otc_scaling` 参数启用/禁用

## 八、与训练流程的集成

### 8.1 调用位置

在 `RayPPOTrainer.fit()` 中:

```python
# 计算奖励
reward_data_proto, reward_extra_infos_dict = compute_reward(
    batch, reward_fn
)

# 合并到batch
batch = batch.union(reward_data_proto)
```

### 8.2 奖励格式

- **结构化奖励**: 返回 `structured_rewards` 列表，每个元素包含详细的奖励分解
- **WandB日志**: 通过 `_compute_wandb_metrics()` 生成指标用于日志记录
- **优势计算**: 在 `compute_grpo_multiturn_advantage()` 中使用结构化奖励计算优势

## 九、取值范围分析

### 9.1 多轮模式 - total_score (主要使用的score)

这是 `KGFormatMultiTurnRewardManager` 返回的最终奖励分数，用于训练：

**未启用OTC缩放**:
- 范围: **[0.00, 0.95]**
  - 最小值: 0.00（所有轮次0分 + 全局0分）
  - 最大值: 0.95（所有轮次满分 + 全局满分）
  - 轮次平均最大值: 0.25
  - 全局总分最大值: 0.70

**启用OTC缩放**:
- 范围: **[0.00, 2.1528]**
  - 最小值: 0.00
  - 最大值: 2.1528（理论值，0轮时）
  - OTC缩放因子范围: [1.0, 2.718] (e ≈ 2.718)
  - 实际最大值取决于使用的轮次数

### 9.2 多轮模式 - final_score (精确匹配原始分数)

在 `_calculate_exact_match_reward()` 中返回，用于计算全局奖励：

- **binary模式**: **[0.0, 1.0]**（只能是0或1）
- **f1模式**: **[0.0, 1.0]**（连续值）

这是精确匹配的原始分数，在应用权重（0.3）之前。

### 9.3 单轮模式 - score (单轮奖励管理器)

`compute_score_em_kg_refactored()` 返回的score，用于单轮模式：

- **范围**: **[-2.0, 1.5]**（理论范围）
  - `base_score`: [0.00, 1.00]
  - `kg_interaction_score`: [-2.00, 0.50]（可能为负，因为有错误惩罚）
  - `retrieval_score`: [0.0, 0.10]
  - 实际通常为正值，负值出现在大量错误查询时

### 9.4 各组件取值范围

#### 轮次特定奖励组件

- **格式分数** (`format_score`): [0.0, 1.0] → 加权后: [0.0, 0.15]
- **KG查询有效性** (`kg_query_validity`): [0.0, 1.0] → 加权后: [0.0, 0.1]
- **答案分数** (`is_answer_score`): [0.0, 1.0] → 加权后: [0.0, 0.1]
- **每轮奖励**: [0.0, 0.25]（KG查询动作或答案动作）

#### 全局奖励组件

- **精确匹配** (`exact_match`): 
  - 原始分数: [0.0, 1.0]
  - 加权后（未启用OTC）: [0.0, 0.3]
  - 加权后（启用OTC，0轮）: [0.0, 0.8155]
  
- **检索质量** (`retrieval_quality`):
  - 原始分数: [0.0, 1.0]
  - 加权后（未启用OTC）: [0.0, 0.4]
  - 加权后（启用OTC，0轮）: [0.0, 1.0873]

### 9.5 特殊情况示例

- **最差情况**: 0.00（所有轮次格式错误且查询无效）
- **中间情况**: 0.65（所有轮次格式正确且查询有效，但答案错误）
  - 轮次平均: 0.25
  - 检索质量: 0.40
  - 精确匹配: 0.00
- **最佳情况**: 0.95（所有轮次格式正确且查询有效，答案正确）
  - 轮次平均: 0.25
  - 全局总分: 0.70

### 9.6 取值范围总结表

| Score类型 | 模式 | 未启用OTC | 启用OTC | 说明 |
|-----------|------|-----------|---------|------|
| `total_score` | 多轮 | [0.00, 0.95] | [0.00, 2.1528] | 最终奖励分数 |
| `final_score` | 多轮 | [0.0, 1.0] | [0.0, 1.0] | 精确匹配原始分数 |
| `score` | 单轮 | [-2.0, 1.5] | N/A | 单轮模式分数 |

## 十、Score 和 Reward 的关系

### 10.1 核心关系

在 KG-R1 中，**Score 和 Reward 本质上是同一个值**，只是在不同阶段和不同上下文中的命名不同。

### 10.2 概念层面

- **Score**: 评估分数，表示模型输出的质量（计算层面的术语）
- **Reward**: 强化学习中的奖励信号，用于训练（RL层面的术语）
- **关系**: 在 KG-R1 中，它们表示同一个数值，只是在不同上下文中使用

### 10.3 代码转换流程

#### 多轮模式 (`KGFormatMultiTurnRewardManager`)

```
total_score (计算出的最终分数)
    ↓
reward_tensor (转换为tensor格式，放在最后一个token位置)
    ↓
token_level_scores (重命名，用于PPO训练)
```

**关键代码**:
```python
# kg_format_multiturn.py:127
reward_tensor[i, valid_response_length - 1] = reward_dict["total_score"]

# reward.py:172
tensors={"token_level_scores": reward_tensor}
```

#### 单轮模式 (`KGFormatRewardManager`)

```
score (compute_score_em_kg_refactored返回，dict或float)
    ↓
reward = score["score"] (如果是dict，提取score值)
    ↓
reward_tensor (转换为tensor)
```

**关键代码**:
```python
# kg_format.py:263-271
if isinstance(score, dict):
    reward = score["score"]
else:
    reward = score
reward_tensor[i, valid_response_length - 1] = reward
```

### 10.4 命名约定

| 名称 | 含义 | 使用场景 |
|------|------|----------|
| `score` | 计算函数返回的原始分数 | 评估计算阶段 |
| `reward` | score 的别名 | 赋值给 reward_tensor 时 |
| `total_score` | 多轮模式下的最终分数 | 多轮奖励计算 |
| `reward_tensor` | score 的 tensor 表示 | 奖励管理器输出 |
| `token_level_scores` | reward_tensor 的最终名称 | PPO 训练使用 |

### 10.5 关系总结

1. **Score = Reward**: 它们是同一个数值
2. **Score 是评估术语**: 表示评估分数
3. **Reward 是 RL 术语**: 表示训练信号
4. **最终用途**: 都转换为 `reward_tensor`/`token_level_scores` 用于 PPO 训练
5. **取值范围**: 相同，多轮模式 [0.00, 0.95]（未启用OTC）或 [0.00, 2.1528]（启用OTC）

在 KG-R1 中，score 和 reward 可以互换使用，它们表示同一个用于训练的信号值。

## 十一、总结

KG-R1的多轮对话reward计算采用**分层奖励结构**:

1. **轮次特定奖励**: 鼓励每轮的正确格式和有效操作
2. **全局奖励**: 评估最终答案正确性和整体检索质量
3. **最终奖励**: 轮次平均 + 全局总和

这种设计既鼓励中间步骤的正确性，又确保最终结果的准确性，同时通过唯一性检查和格式要求防止奖励黑客行为。

### 关键要点

- **取值范围**: 默认配置下，reward 在 [0.00, 0.95] 范围内
- **Score = Reward**: 它们是同一个值，只是命名不同
- **分层设计**: 轮次奖励 + 全局奖励，确保全面评估
- **防止奖励黑客**: 唯一性检查、格式要求、二进制评分


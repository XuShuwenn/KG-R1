# 奖励组件计算详解

本文档详细说明 `kg_format_multiturn.py` 中各个奖励组件的计算方式和输入。

## 1. kg_query_validity（KG查询有效性）

### 输入
- `turn_info: Dict` - 包含以下字段：
  - `action`: str - 动作类型（'kg-query' 或其他）
  - `valid_action`: bool - 查询是否语法有效
  - `raw_server_response`: Dict - KG服务器原始响应，包含：
    - `kg_metadata`: Dict - 包含 `success` 和 `error_type`
- `seen_query_ids: set` - 已见过的查询ID集合（用于去重）

### 计算逻辑
```python
def _calculate_kg_query_validity_reward(turn_info, seen_query_ids) -> float:
    # 1. 只处理 kg-query 动作
    if turn_info['action'] != 'kg-query':
        return 0.0
    
    # 2. 检查语法有效性
    if not turn_info['valid_action']:
        return 0.0
    
    # 3. 检查查询是否成功执行
    raw_response = turn_info.get('raw_server_response', {})
    if 'kg_metadata' in raw_response:
        success = kg_metadata.get('success', False)
        error_type = kg_metadata.get('error_type')
        if not success or error_type != 'KG_SUCCESS':
            return 0.0
    
    # 4. 提取查询唯一标识符（防止重复查询）
    query_id = extract_query_identifier(raw_response)
    # query_id 由以下字段组成（用 | 分隔）：
    # - action_type (如 'get_relations')
    # - entity_id
    # - relation
    # - sample_id
    # - dataset_name
    
    # 5. 检查是否重复
    if query_id in seen_query_ids:
        return 0.0  # 重复查询不给奖励
    else:
        seen_query_ids.add(query_id)
        return 1.0  # 唯一且成功的查询
```

### 返回值
- `0.0`: 非 kg-query 动作、语法无效、执行失败、或重复查询
- `1.0`: 唯一且成功的 kg-query

### 权重
- 默认权重：`turn_kg_query_validity = 0.1`
- 加权后：`kg_query_validity * 0.1`

> 说明：SPARQL bridge 的异常回包（例如超时、线程池失败）现在统一补齐 `kg_metadata`，因此即使是 fallback 也会被视作“失败查询”，不会误记奖励。

---

## 2. is_answer_score（答案动作奖励）

### 输入
- `turn_info: Dict` - 包含：
  - `action`: str - 动作类型（'answer' 或其他）
- `data_item` - 数据项（当前未使用）

### 计算逻辑
```python
def _calculate_is_answer_score_reward(turn_info, data_item=None) -> float:
    # 简单检查：是否为 answer 动作
    if turn_info['action'] != 'answer':
        return 0.0
    return 1.0  # 只要输出 answer 就给奖励
```

### 返回值
- `0.0`: 非 answer 动作
- `1.0`: answer 动作

### 权重
- 默认权重：`turn_is_answer_score = 0.1`
- 加权后：`is_answer_score * 0.1`

---

## 3. format_score（格式合规性）

### 输入
- `turn_info: Dict` - 包含：
  - `action`: str - 动作类型（'kg-query' 或 'answer'）
  - `turn_idx`: int - 轮次索引
  - `raw_server_response`: Dict - 服务器响应（未直接使用）
- `full_response: str` - 完整响应文本（未直接使用）
- `data_item` - 数据项（用于提取 token 信息，作为后备方案）
- `sample_interaction_history: Dict` - 单样本的交互历史，包含：
  - `responses_str: List[str]` - 每轮的模型响应字符串列表

### 计算逻辑
```python
def _calculate_turn_format_score_reward(turn_info, full_response, data_item, sample_interaction_history) -> tuple[float, str]:
    # 1. 只检查 kg-query 和 answer 动作
    if turn_info['action'] not in ['kg-query', 'answer']:
        return (0.0, "")
    
    # 2. 从 interaction_history 提取该轮的内容
    turn_idx = turn_info.get('turn_idx', 0)
    turn_content = sample_interaction_history['responses_str'][turn_idx]
    turn_content = _strip_chatml_wrappers(turn_content)  # 去除 ChatML token、<information> 等包裹
    
    # 3. 根据动作类型检查格式
    if action == 'kg-query':
        format_valid = _has_proper_kg_query_format(turn_content)
    elif action == 'answer':
        format_valid = _has_proper_answer_format(turn_content)
    
    return (1.0 if format_valid else 0.0, turn_content)
```

### 格式要求

#### kg-query 格式
```
^<think>.*?</think>\s*<kg-query>.*?</kg-query>$
```
- 必须包含且仅包含一对 `<think>` 标签
- 必须包含且仅包含一对 `<kg-query>` 标签
- 两个标签之间只能有空白字符（空格、制表符、换行）
- 整个内容必须完全匹配此模式（不能有其他文本）

#### answer 格式
```
^<think>.*?</think>\s*<answer>.*?</answer>$
```
- 必须包含且仅包含一对 `<think>` 标签
- 必须包含且仅包含一对 `<answer>` 标签
- 两个标签之间只能有空白字符
- 整个内容必须完全匹配此模式

### 返回值
- `0.0`: 格式不符合要求
- `1.0`: 格式完全符合要求

### 权重
- 默认权重：`turn_format_score = 0.15`
- 加权后：`format_score * 0.15`

---

## 4. retrieval_quality（检索质量）

### 输入
- `data_item` - 数据项，包含：
  - `non_tensor_batch["reward_model"]["ground_truth"]` - 标准答案（可以是字符串、列表或字典）
- `interaction_history: Dict` - 单样本的交互历史，包含：
  - `search_results: List[str]` - 格式化后的检索结果（展示给LLM的文本）
  - `raw_server_responses: List[Dict]` - KG服务器原始响应列表

### 计算逻辑
```python
def _calculate_global_retrieval_quality_reward(data_item, interaction_history) -> float:
    # 1. 解析标准答案
    ground_truth_raw = data_item.non_tensor_batch["reward_model"]["ground_truth"]
    ground_truth_answers = _parse_ground_truth(ground_truth_raw)  # 转换为列表
    
    # 2. 检查所有检索结果中是否包含正确答案
    retrieval_contains_answer = _check_global_retrieval_contains_answer(
        interaction_history, ground_truth_answers
    )
    
    # 3. 二进制评分
    return 1.0 if retrieval_contains_answer else 0.0
```

### 检索检查逻辑（`is_retrieval_correct_kg`）

- 预处理：统一移除 `<information>` 区块与 ChatML token（`<|im_start|>assistant`、`<|im_end|>`、`<s>` 等），并尝试解析 JSON。
- 生成候选文本：完整正文、冒号后的主体、逐行内容、以及 JSON 中的所有字符串值。
- 匹配策略：对每个候选文本执行 `normalize_answer`，与所有黄金答案做双向包含判断，命中即返回 True。
- `raw_server_responses` 仅在 `kg_metadata.success=True 且 error_type=KG_SUCCESS` 时参与，候选文本来源包括 `choices[].message.content`、`content`、`data` 等字段。

### 返回值
- `0.0`: 所有检索结果中都没有找到正确答案
- `1.0`: 至少一个检索结果包含正确答案

### 权重
- 默认权重：`global_retrieval_quality = 0.4`
- 加权后：`retrieval_quality * 0.4`

---

## 5. exact_match（精确匹配 - 全局奖励）

### 输入
- `data_item` - 数据项，包含：
  - `batch["prompts"]` - 提示词 token IDs
  - `batch["responses"]` - 响应 token IDs
  - `non_tensor_batch["reward_model"]["ground_truth"]` - 标准答案
- `interaction_history: Dict` - 单样本的交互历史，包含：
  - `data_source: str` - 数据源名称（用于判断是否使用 kgqa_agent 模式）

### 计算逻辑
```python
def _calculate_exact_match_reward(data_item, interaction_history) -> float:
    # 1. 提取完整序列（prompt + response）
    full_solution_text = _extract_full_solution_text(data_item)
    assistant_response = extract_assistant_response(full_solution_text)
    
    # 2. 提取预测答案（剥离 ChatML / <information>，允许不完整的 <answer> 标签）
    predicted_answer = extract_answer_kg(assistant_response)
    
    # 3. 解析标准答案
    ground_truth_raw = data_item.non_tensor_batch["reward_model"]["ground_truth"]
    ground_truth_answers = _parse_ground_truth(ground_truth_raw)
    
    # 4. 判断使用哪种计算模式
    use_kgqa_agent_metrics = _is_kgqa_agent_mode(interaction_history)
    # 判断条件：data_source 包含 'kgqa_agent' 或 'kgqa_agent_format'
    
    if use_kgqa_agent_metrics:
        # kgqa_agent 风格：双向子串匹配 + token F1
        entity_level_exact_match = _exact_match_kgqa_agent(predicted_answer, ground_truth_answers)
        entity_metrics = _token_f1_score_kgqa_agent(predicted_answer, ground_truth_answers)
    else:
        # KG-R1 风格：精确匹配 + 实体级 F1（支持 MultiTQ 特殊处理）
        entity_level_exact_match = _calculate_entity_level_match(
            predicted_answer, ground_truth_answers, 
            interaction_history=interaction_history, 
            dataset_name=data_source
        )
        entity_metrics = _calculate_entity_level_f1_precision_recall(
            predicted_answer, ground_truth_raw
        )
    
    # 5. 根据 answer_score_mode 选择最终分数
    if self.answer_score_mode == 'binary':
        final_score = entity_level_exact_match  # 0.0 或 1.0
    elif self.answer_score_mode == 'f1':
        final_score = entity_metrics['f1']  # 0.0-1.0
    else:
        final_score = entity_level_exact_match
    
    return final_score
```

### kgqa_agent 风格计算

#### Exact Match
```python
def _exact_match_kgqa_agent(pred, golds) -> float:
    # 1. 解析预测（支持 JSON 列表、管道分隔符、单字符串）
    preds = _parse_prediction_kgqa_agent(pred)
    
    # 2. 对每个候选答案尝试匹配
    for p in preds:
        npred = _qa_normalize_answer(p)  # 归一化（保留标点）
        for g in golds:
            ngold = _qa_normalize_answer(str(g))
            if npred == ngold:  # 精确匹配
                return 1.0
            if ngold in npred:  # 标准答案在预测中
                return 1.0
            if npred in ngold:  # 预测在标准答案中（反向匹配）
                return 1.0
    return 0.0
```

#### Token F1
```python
def _token_f1_score_kgqa_agent(pred, golds) -> Dict[str, float]:
    # 1. 合并所有预测候选答案的 token
    pred_tokens = set()
    for p in preds:
        normalized = _qa_normalize_answer(p)
        pred_tokens.update(normalized.split())  # 按空格分割成 token
    
    # 2. 合并所有标准答案的 token
    gold_tokens = set()
    for g in golds:
        normalized = _qa_normalize_answer(str(g))
        gold_tokens.update(normalized.split())
    
    # 3. 计算 token 集合的 F1
    common = len(pred_tokens & gold_tokens)
    precision = common / len(pred_tokens) if pred_tokens else 0.0
    recall = common / len(gold_tokens) if gold_tokens else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
```

### KG-R1 风格计算

#### Entity-level Exact Match
- 使用 `em_check_kg` 函数
- 支持 MultiTQ 时间粒度处理（年份 vs 年月）
- 支持列表匹配（逗号分隔的多个实体）
- 归一化：移除标点、冠词、规范化空白

#### Entity-level F1
- 将预测和标准答案按逗号分割成实体集合
- 对每个实体进行归一化
- 计算实体集合的交集
- 支持 `target_text` 和 `target_kb_id` 的替代表示

### 返回值
- `0.0-1.0`: 根据 `answer_score_mode` 返回 binary (0/1) 或 F1 分数

### 权重
- 默认权重：`global_exact_match = 0.3`
- 加权后：`exact_match * 0.3`
- 可选 OTC 缩放：如果启用 `otc_scaling`，会根据使用的 KG 查询轮次进行指数衰减

---

## 总奖励计算

### Turn-wise 奖励
```python
# 对于 kg-query 动作
turn_reward = format_score * 0.15 + kg_query_validity * 0.1

# 对于 answer 动作
turn_reward = format_score * 0.15 + is_answer_score * 0.1

# 所有轮次的平均
total_turn_score = sum(turn_rewards.values()) / len(turn_rewards)
```

### Global 奖励
```python
# 应用 OTC 缩放（如果启用）
if otc_scaling:
    scaling_factor = e^(1 - kg_turns_used / max_turns)
    exact_match_scaled = exact_match_raw * scaling_factor
    retrieval_quality_scaled = retrieval_quality_raw * scaling_factor
else:
    exact_match_scaled = exact_match_raw
    retrieval_quality_scaled = retrieval_quality_raw

# 加权
total_global_score = exact_match_scaled * 0.3 + retrieval_quality_scaled * 0.4
```

### 最终总奖励
```python
total_score = total_turn_score + total_global_score
```

---

## 默认权重总结

| 组件 | 权重 | 说明 |
|------|------|------|
| `turn_format_score` | 0.15 | 格式合规性 |
| `turn_kg_query_validity` | 0.1 | KG查询有效性 |
| `turn_is_answer_score` | 0.1 | 答案动作奖励 |
| `global_exact_match` | 0.3 | 精确匹配 |
| `global_retrieval_quality` | 0.4 | 检索质量 |

总权重：`0.15 + 0.1 + 0.1 + 0.3 + 0.4 = 1.05`（可能超过1.0，这是设计上的选择）

> 提示：在 reward_kwargs 中开启 `kgqa_reward_profile`（或 data_source 含 `kgqa_agent`）时，会自动将回合权重调整为 `format=0.1 / kg_query=0.05 / answer=0.05`，并将全局权重设置为 `exact=0.5 / retrieval=0.3`，使训练目标更贴合 kgqa_agent 的 EM/F1 评测。


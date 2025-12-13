# 多轮对话限制机制说明

## 当前限制机制

### ✅ 1. 对话轮次限制 (`max_turns`)

**配置位置**：`train_debug_single_a100.sh`
```bash
actor_rollout_ref.rollout.search.max_turns=6
```

**实现位置**：`kg_r1/llm_agent/generation.py:588`
```python
for step in range(self.config.max_turns):
    # 主循环：最多执行 max_turns 轮对话
    if not active_mask.sum():
        break
    # ... 执行预测、查询、更新状态
```

**限制方式**：
- ✅ **硬性限制**：代码层面强制执行
- 最多进行 **6 轮对话**（不包括最后的强制回答轮）
- 每轮对话包括：模型生成 → KG 查询（可选）→ 接收反馈 → 继续下一轮

**提前终止条件**：
- 模型输出 `<answer>` 标签（`dones.append(1)`）
- 所有样本都已完成（`active_mask.sum() == 0`）

### ⚠️ 2. 查询次数提示 (`max_calls`)

**配置位置**：`train_debug_single_a100.sh`
```bash
+kg_config.max_calls=10
```

**使用位置**：`verl/trainer/ppo/prompts.py:55`
```python
SEARCH_PROMPT_TEMPLATE = """You are a helpful assistant that answers questions based on knowledge graphs. You can query from knowledge base provided to you to answer the question up to {max_calls} times.
...
"""
```

**限制方式**：
- ⚠️ **软性提示**：仅在 prompt 中告知模型
- ❌ **没有代码层面的强制限制**
- 模型可以在每轮中查询多次，只要不超过 `max_turns` 轮次

**实际行为**：
- 模型在每轮对话中可以发出多个 `<kg-query>`
- 只要还在 `max_turns` 范围内，查询次数没有硬性限制
- `max_calls=10` 只是作为提示，告诉模型"建议最多查询 10 次"

## 限制机制对比

| 限制类型 | 参数 | 值 | 实现方式 | 是否强制执行 |
|---------|------|-----|---------|------------|
| 对话轮次 | `max_turns` | 6 | 代码循环 `for step in range(max_turns)` | ✅ 是 |
| 查询次数 | `max_calls` | 10 | Prompt 提示 | ❌ 否 |

## 实际运行流程

```
轮次 1: 模型生成 → [可能多个 <kg-query>] → 接收反馈 → 继续
轮次 2: 模型生成 → [可能多个 <kg-query>] → 接收反馈 → 继续
...
轮次 6: 模型生成 → [可能多个 <kg-query>] → 接收反馈 → 继续
最终轮: 模型生成 → [不允许查询] → 强制回答
```

**关键点**：
- 每轮可以包含**多个** `<kg-query>`
- 只要不超过 6 轮，查询总次数没有硬性限制
- 最后一轮（`do_search=False`）不允许查询，必须给出答案

## 代码实现细节

### 轮次限制实现

```python
# kg_r1/llm_agent/generation.py:588
for step in range(self.config.max_turns):  # 最多 6 轮
    # 执行预测
    responses_str = self._generate_with_gpu_padding(...)
    
    # 执行动作（包括 KG 查询）
    next_obs, dones, valid_action, is_search, raw_server_responses = \
        self.execute_predictions(responses_str, ..., do_search=True)
    
    # 更新活跃状态
    active_mask = active_mask * curr_active_mask
    
    # 如果所有样本都完成，提前退出
    if not active_mask.sum():
        break

# 最终轮：不允许查询
if active_mask.sum():
    # do_search=False 表示不允许查询
    _, dones, valid_action, is_search, raw_server_responses = \
        self.execute_predictions(responses_str, ..., do_search=False)
```

### 查询次数提示（无强制限制）

```python
# verl/trainer/ppo/prompts.py:87
def build_search_prompt(question_or_sample, max_calls=10, ...):
    return SEARCH_PROMPT_TEMPLATE.format(
        max_calls=max_calls,  # 仅在 prompt 中提示
        ...
    )
```

## 如果需要添加查询次数限制

如果需要在代码层面强制执行 `max_calls` 限制，需要：

1. **在会话状态中跟踪查询次数**
   ```python
   @dataclass
   class _SessionState:
       query_count: int = 0  # 添加查询计数
       ...
   ```

2. **在每次查询时检查并限制**
   ```python
   def run_query(self, sample_id: str, query_str: str, ...):
       session = self._get_session(sample_id, topic_entities)
       
       # 检查查询次数限制
       if session.query_count >= self._max_calls:
           return self._format_error(
               "Maximum query limit reached. Please provide your answer.",
               sample_id,
               KGErrorType.FORMAT_ERROR,
           )
       
       session.query_count += 1
       # ... 执行查询
   ```

3. **在会话重置时清零计数**
   ```python
   def reset(self, sample_id: Optional[str] = None):
       # ... 现有重置逻辑
       session.query_count = 0
   ```

## 当前配置总结

根据 `train_debug_single_a100.sh`：

- **对话轮次限制**：✅ 6 轮（硬性限制）
- **查询次数提示**：⚠️ 10 次（软性提示，无强制限制）

**实际行为**：
- 最多进行 6 轮对话
- 每轮可以包含多个查询
- 总查询次数理论上可以超过 10 次（如果模型在每轮都查询多次）
- 最后一轮强制回答，不允许查询

## 建议

1. **当前配置适合训练**：轮次限制已经足够，查询次数由模型自主控制
2. **如果需要严格限制查询次数**：需要实现上述的查询计数机制
3. **监控查询行为**：可以通过 `interaction_history` 中的 `is_search_actions` 统计实际查询次数


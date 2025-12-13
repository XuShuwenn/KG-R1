# 移除对话轮次限制，仅保留搜索次数限制

## ✅ 已完成的修改

### 1. 移除对话轮次限制

**文件**: `kg_r1/llm_agent/generation.py`

#### 修改前
```python
# Main generation loop
for step in range(self.config.max_turns):
    if not active_mask.sum():
        break
    # ... 生成和处理逻辑
```

#### 修改后
```python
# Main generation loop - continue until all samples are done (no turn limit, only query count limit)
step = 0
while active_mask.sum():
    step += 1
    # ... 生成和处理逻辑
```

**效果**：
- 移除了硬性的轮次限制（`max_turns`）
- 循环会一直继续，直到所有样本都完成（`active_mask.sum() == 0`）
- 样本完成的条件：模型输出 `<answer>` 标签，或达到搜索次数限制

### 2. 移除"最终轮"特殊处理

**修改前**：
- 在 `max_turns` 轮后，会执行一次特殊的"最终轮"生成
- 最终轮中 `do_search=False`，禁止搜索

**修改后**：
- 移除了所有"最终轮"的特殊处理代码
- 所有轮次都使用相同的逻辑，没有特殊限制

### 3. 搜索次数限制触发结束

**文件**: `kg_r1/llm_agent/generation.py` - `execute_predictions` 方法

#### 修改
```python
elif action == 'kg-query':
    if i in kg_response_map:
        kg_result, raw_kg_response = kg_response_map[i]
        # Check if max_calls limit was reached
        is_max_calls_reached = (
            isinstance(raw_kg_response, dict) and
            raw_kg_response.get("meta", {}).get("action") == "max_calls_reached"
        )
        
        if is_max_calls_reached:
            # Max calls reached: return FORCE_ANSWER_PROMPT and set done=True
            next_obs.append(f'\n\n<information>{kg_result.strip()}</information>\n\n')
            raw_server_responses.append(raw_kg_response)
            dones.append(1)  # Force done to end conversation after model answers
            valid_action.append(1)
            is_search.append(1)
        else:
            # Normal query: continue conversation
            next_obs.append(f'\n\n<information>{kg_result.strip()}</information>\n\n')
            raw_server_responses.append(raw_kg_response)
            dones.append(0)  # Continue conversation
            # ... 正常处理逻辑
```

**效果**：
- 当达到 `max_calls` 限制时，`KGQASparqlAdapter.run_query` 返回 `FORCE_ANSWER_PROMPT`
- `execute_predictions` 检测到 `max_calls_reached` 标记后，设置 `done=True`
- 模型会在下一轮看到 `FORCE_ANSWER_PROMPT` 并输出 `<answer>` 标签
- 输出 `<answer>` 后，`done=True` 会结束该样本的对话

## 🔄 新的对话流程

### 示例：max_calls=10

```
轮次 1: 查询 1 → query_count=1 → 继续 ✅
轮次 2: 查询 2 → query_count=2 → 继续 ✅
...
轮次 10: 查询 10 → query_count=10 → 继续 ✅
轮次 11: 查询 11 → query_count=11 >= max_calls(10) → 返回 FORCE_ANSWER_PROMPT, done=True ⚠️
轮次 12: 模型看到 FORCE_ANSWER_PROMPT → 输出 <answer> → done=True → 结束 ✅
```

### 提前结束的情况

```
轮次 3: 模型直接输出 <answer> → done=True → 结束 ✅
（不需要达到 max_calls）
```

## 📊 限制机制对比

| 特性 | 修改前 | 修改后 |
|------|--------|--------|
| 轮次限制 | `max_turns=6`（硬性限制） | 无限制（仅由搜索次数控制） |
| 搜索限制 | `max_calls=10`（提示，未强制执行） | `max_calls=10`（硬性限制） |
| 结束条件 | 达到 `max_turns` 或输出 `<answer>` | 达到 `max_calls` 或输出 `<answer>` |
| 最终轮 | 有（禁止搜索） | 无 |

## ⚙️ 配置说明

### 训练脚本配置

```bash
# 移除或注释掉 max_turns 相关配置（如果不再需要）
# actor_rollout_ref.rollout.search.max_turns=6  # 不再使用

# 保留搜索次数限制
+kg_config.max_calls=10  # 每个样本最多 10 次查询
```

### 参数传递

```
训练脚本 (kg_config.max_calls=10)
  ↓
ray_trainer_kg.py (_normalize_kg_bridge_config)
  ↓
GenerationConfig (kgqa_max_calls)
  ↓
LLMGenerationManager.__init__ (adapter_max_calls)
  ↓
KGQASparqlAdapter.__init__ (max_calls)
  ↓
run_query (检查 session.query_count >= self._max_calls)
  ↓
execute_predictions (检测 max_calls_reached，设置 done=True)
```

## 🎯 优势

1. **更灵活**：模型可以根据需要自由决定对话轮数
2. **更精确**：直接限制搜索次数，而不是间接通过轮次限制
3. **更符合 kgqa_agent 行为**：与 `kgqa_agent` 的实现逻辑一致
4. **更高效**：避免不必要的"最终轮"生成

## ⚠️ 注意事项

1. **无限循环风险**：理论上，如果模型既不输出 `<answer>` 也不查询，可能会无限循环
   - **缓解措施**：模型训练应该学会在适当时机输出 `<answer>`
   - **监控建议**：记录每个样本的实际轮数，设置合理的超时机制

2. **资源消耗**：没有轮次限制可能导致某些样本使用更多计算资源
   - **缓解措施**：通过 `max_calls` 限制搜索次数，间接控制对话长度

3. **训练稳定性**：移除轮次限制可能影响训练稳定性
   - **建议**：在训练初期监控平均轮数和资源使用情况

## 🔗 相关文档

- [max_calls 查询次数限制集成](./max_calls_integration_summary.md)
- [多轮对话限制说明](./multiturn_limits.md)
- [错误处理机制](./multiturn_conversation_error_handling.md)


# max_calls 查询次数限制集成总结

## ✅ 已完成的集成

### 1. 代码修改

#### `kg_r1/kgqa_bridge/sparql_adapter.py`

1. **`_SessionState` 添加查询计数**：
   ```python
   query_count: int = 0  # Track number of KG queries executed for this sample
   ```

2. **`KGQASparqlAdapter.__init__` 添加 `max_calls` 参数**：
   ```python
   def __init__(self, ..., max_calls: int = 10, ...):
       self._max_calls = max_calls
   ```

3. **`run_query` 方法添加限制检查**：
   ```python
   # 检查查询次数限制
   if session.query_count >= self._max_calls:
       return FORCE_ANSWER_PROMPT, payload
   
   # 增加计数
   session.query_count += 1
   # 执行查询
   ```

4. **`reset` 方法自动清零**：
   - 会话被删除时，计数自动清零（新会话从 0 开始）

#### `kg_r1/llm_agent/generation.py`

1. **初始化时传递 `max_calls`**：
   ```python
   adapter_max_calls = config.kgqa_max_calls or 10
   self.kgqa_adapter = KGQASparqlAdapter(
       sparql_endpoint=adapter_endpoint,
       kg_top_k=adapter_top_k,
       max_calls=adapter_max_calls,  # 传递 max_calls
   )
   ```

## 📋 kgqa_agent 实现逻辑分析

### 核心机制

在 `kgqa_agent/src/eval/kg_augmented_client.py` 的 `_interactive_generate` 方法中：

```python
calls = 0
while calls < self.max_calls:
    # 生成响应
    response = self.base_client.generate_from_messages(messages, **gen_kwargs)
    
    # 检测 <kg-query> 标签
    if kg_query_match:
        calls += 1  # 计数增加
        query_results = self._parse_and_execute_query(kg_query, question=question)
        # 继续循环
    
    # 检测 <answer> 标签
    elif answer_match:
        return response_truncated  # 提前结束

# 达到 max_calls 后
# 1. 检查最后响应中是否有答案
# 2. 如果没有，使用 FORCE_ANSWER_PROMPT 强制回答
```

### 关键特点

1. **计数时机**：在检测到 `<kg-query>` 标签时立即计数（`calls += 1`），而不是在执行查询后
2. **限制检查**：使用 `while calls < self.max_calls`，允许执行 `max_calls` 次查询
3. **强制回答**：达到限制后，先检查最后响应是否有答案，如果没有则使用 `FORCE_ANSWER_PROMPT`

## 🔄 集成后的行为

### 查询流程示例（max_calls=10）

```
查询 1: query_count=0 → 检查(0>=10?否) → 计数+1 → query_count=1 → 执行 ✅
查询 2: query_count=1 → 检查(1>=10?否) → 计数+1 → query_count=2 → 执行 ✅
...
查询 10: query_count=9 → 检查(9>=10?否) → 计数+1 → query_count=10 → 执行 ✅
查询 11: query_count=10 → 检查(10>=10?是) → 返回 FORCE_ANSWER_PROMPT ⚠️
```

### 与轮次限制的交互

- **轮次限制** (`max_turns=6`)：限制对话轮数
- **查询限制** (`max_calls=10`)：限制查询次数

**两者同时生效**：
- 如果模型在 6 轮内查询超过 10 次，会在达到 10 次时强制回答
- 如果模型在 10 次查询内完成 6 轮，会在第 6 轮后强制回答（最终轮不允许查询）

## 📝 配置使用

### 训练脚本配置

```bash
# 在 train_debug_single_a100.sh 中
+kg_config.max_calls=10  # 每个样本最多 10 次查询
actor_rollout_ref.rollout.search.max_turns=6  # 最多 6 轮对话
```

### 参数传递链

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
```

## 🎯 与 kgqa_agent 的差异

| 特性 | kgqa_agent | 训练框架（集成后） |
|------|-----------|------------------|
| 计数时机 | 检测到 `<kg-query>` 标签时 | `run_query` 调用时 |
| 限制检查 | `while calls < max_calls` | `if query_count >= max_calls` |
| 会话管理 | 每个问题独立（方法内局部变量） | 每个样本独立（会话状态） |
| 强制回答 | `FORCE_ANSWER_PROMPT` | `FORCE_ANSWER_PROMPT`（相同） |

**关键差异**：
- kgqa_agent 在检测到标签时计数（即使查询失败也会计数）
- 训练框架在 `run_query` 调用时计数（更符合实际执行）

## ✅ 验证清单

- [x] `_SessionState` 包含 `query_count` 字段
- [x] `KGQASparqlAdapter` 支持 `max_calls` 参数
- [x] `run_query` 检查查询次数限制
- [x] 达到限制时返回 `FORCE_ANSWER_PROMPT`
- [x] `reset` 时自动清零计数
- [x] `generation.py` 传递 `max_calls` 参数
- [x] 配置参数正确传递

## 🚀 使用效果

集成后，训练框架将：

1. **跟踪每个样本的查询次数**：通过 `session.query_count`
2. **强制执行查询限制**：达到 `max_calls` 后返回强制回答提示
3. **与 kgqa_agent 行为一致**：使用相同的 `FORCE_ANSWER_PROMPT`
4. **自动重置计数**：每个新样本从 0 开始计数

## 📊 监控建议

训练过程中可以监控：

1. **查询次数分布**：通过 `interaction_history` 中的查询记录
2. **达到限制的样本比例**：检查 `max_calls_reached` 标记
3. **平均查询次数**：统计每个样本的平均查询次数

## 🔗 相关文档

- [多轮对话限制说明](./multiturn_limits.md)
- [错误处理机制](./multiturn_conversation_error_handling.md)
- [训练检查清单](./training_checklist.md)


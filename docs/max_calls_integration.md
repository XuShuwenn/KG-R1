# max_calls 查询次数限制集成方案

## kgqa_agent 中的实现逻辑

### 核心机制

在 `kgqa_agent/src/eval/kg_augmented_client.py` 中：

1. **初始化**：`max_calls` 参数（默认 10）
   ```python
   def __init__(self, ..., max_calls: int = 10, ...):
       self.max_calls = max_calls
   ```

2. **查询计数**：在 `_interactive_generate` 方法中
   ```python
   calls = 0
   while calls < self.max_calls:
       # ... 生成响应
       if kg_query_match:
           calls += 1  # 每次执行查询时计数
           query_results = self._parse_and_execute_query(kg_query, question=question)
           # ... 继续循环
   ```

3. **达到限制后的处理**：
   ```python
   # 达到 max_calls 后：只再触发一次强制回答逻辑
   if messages and len(messages) >= 2 and messages[-1]["role"] == "assistant":
       last_response = messages[-1]["content"]
       answer_content = self._extract_answer_tag(last_response)
       if answer_content:
           return last_response_truncated
   
   # 否则，使用 FORCE_ANSWER_PROMPT 强制回答
   messages.append({"role": "user", "content": FORCE_ANSWER_PROMPT})
   final_response = self.base_client.generate_from_messages(messages, **gen_kwargs)
   ```

### 关键特点

- ✅ **硬性限制**：代码层面强制执行
- ✅ **每查询计数**：每次 `<kg-query>` 执行时 `calls += 1`
- ✅ **强制回答**：达到限制后使用 `FORCE_ANSWER_PROMPT` 提示模型回答
- ✅ **会话级别**：每个问题独立计数（在 `_interactive_generate` 开始时重置）

## 集成到训练框架

### 需要修改的地方

1. **`_SessionState`**：添加查询计数字段
2. **`KGQASparqlAdapter.__init__`**：添加 `max_calls` 参数
3. **`run_query`**：检查并限制查询次数
4. **`reset`**：清零查询计数
5. **`generation.py`**：传递 `max_calls` 参数

### 实现方案

#### 1. 修改 `_SessionState`

```python
@dataclass
class _SessionState:
    """Per-sample interaction state to mimic kgqa_agent's conversational memory."""
    entity_registry: Dict[str, str] = field(default_factory=dict)
    seen_relations: set[str] = field(default_factory=set)
    last_relations_text: str = ""
    last_entities_text: str = ""
    initial_entity_names: List[str] = field(default_factory=list)
    query_count: int = 0  # 新增：查询次数计数
```

#### 2. 修改 `KGQASparqlAdapter.__init__`

```python
def __init__(
    self,
    sparql_endpoint: str,
    *,
    timeout: int = 120,
    kg_top_k: int = 10,
    max_calls: int = 10,  # 新增参数
    logger: Optional[logging.Logger] = None,
) -> None:
    self._sparql_endpoint = self._normalize_endpoint(sparql_endpoint)
    self._timeout = timeout
    self._kg_top_k = kg_top_k
    self._max_calls = max_calls  # 新增
    self._logger = logger or logging.getLogger(__name__)
    # ...
```

#### 3. 修改 `run_query` 方法

```python
def run_query(self, sample_id: str, query_str: str, ...) -> Tuple[str, Dict[str, Any]]:
    session = self._get_session(sample_id, topic_entities)
    
    # 检查查询次数限制
    if session.query_count >= self._max_calls:
        from verl.trainer.ppo.prompts import FORCE_ANSWER_PROMPT
        return (
            FORCE_ANSWER_PROMPT,
            self._build_payload(
                success=False,
                content=FORCE_ANSWER_PROMPT,
                sample_id=sample_id,
                meta={"action": "max_calls_reached", "query_count": session.query_count},
            ),
        )
    
    # 执行查询前增加计数
    session.query_count += 1
    
    # ... 原有查询逻辑
```

#### 4. 修改 `reset` 方法

```python
def reset(self, sample_id: Optional[str] = None) -> None:
    """Clear adapter state. If sample_id is None, reset all sessions."""
    with self._sessions_lock:
        if sample_id is None:
            self._sessions.clear()
        else:
            session = self._sessions.pop(sample_id, None)
            if session:
                session.query_count = 0  # 重置计数
    # ...
```

#### 5. 修改 `generation.py` 中的初始化

```python
if self.use_sparql_bridge:
    adapter_endpoint = config.sparql_endpoint or config.search_url
    if not adapter_endpoint:
        raise ValueError("SPARQL bridge enabled but no sparql_endpoint/search_url provided.")
    adapter_top_k = config.kgqa_top_k or config.topk
    adapter_max_calls = config.kgqa_max_calls or 10  # 新增：从配置读取
    self.kgqa_adapter = KGQASparqlAdapter(
        sparql_endpoint=adapter_endpoint,
        kg_top_k=adapter_top_k,
        max_calls=adapter_max_calls,  # 新增：传递 max_calls
    )
```

## 集成后的行为

### 查询流程

```
查询 1: session.query_count = 0 → 执行 → query_count = 1 ✅
查询 2: session.query_count = 1 → 执行 → query_count = 2 ✅
...
查询 10: session.query_count = 9 → 执行 → query_count = 10 ✅
查询 11: session.query_count = 10 >= max_calls(10) → 返回 FORCE_ANSWER_PROMPT ⚠️
```

### 与轮次限制的关系

- **轮次限制** (`max_turns=6`)：限制对话轮数
- **查询限制** (`max_calls=10`)：限制查询次数

两者可以同时生效：
- 如果模型在 6 轮内查询超过 10 次，会在达到 10 次时强制回答
- 如果模型在 10 次查询内完成 6 轮，会在第 6 轮后强制回答

## 配置示例

```bash
# 训练脚本中
+kg_config.max_calls=10  # 每个样本最多 10 次查询
actor_rollout_ref.rollout.search.max_turns=6  # 最多 6 轮对话
```

## 注意事项

1. **计数时机**：在 `run_query` 开始时检查，执行前计数
2. **重置时机**：在 `reset` 时清零，确保每个新问题从 0 开始
3. **错误处理**：即使查询失败（如实体解析失败），也会计数
4. **强制回答格式**：使用 `FORCE_ANSWER_PROMPT` 提示模型，而不是直接终止


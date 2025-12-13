# 多轮对话管理与错误处理机制

本文档详细说明 KG-R1 框架中多轮对话的管理机制，以及 SPARQL 查询或解析失败时的处理方式。

## 目录

1. [多轮对话管理架构](#多轮对话管理架构)
2. [SPARQL 查询错误处理](#sparql-查询错误处理)
3. [解析错误处理](#解析错误处理)
4. [错误类型分类](#错误类型分类)
5. [会话状态管理](#会话状态管理)
6. [错误恢复机制](#错误恢复机制)

---

## 多轮对话管理架构

### 核心组件

#### 1. `LLMGenerationManager` (`kg_r1/llm_agent/generation.py`)

负责管理整个多轮对话流程：

```python
class LLMGenerationManager:
    def __init__(self, ...):
        self.use_sparql_bridge = bool(config.use_sparql_bridge)
        self.kgqa_adapter: KGQASparqlAdapter | None = None
        # 初始化 SPARQL 适配器（如果启用）
```

**关键方法：**
- `execute_predictions()`: 执行预测并处理多轮交互
- `batch_search()`: 批量处理 KG 查询
- `_batch_search_via_sparql_bridge()`: 通过 SPARQL 桥接执行查询

#### 2. `KGQASparqlAdapter` (`kg_r1/kgqa_bridge/sparql_adapter.py`)

SPARQL 查询适配器，管理每个样本的会话状态：

```python
class KGQASparqlAdapter:
    def __init__(self, ...):
        self._sessions: Dict[str, _SessionState] = {}  # 每个样本的会话状态
        self._sessions_lock = threading.Lock()  # 线程安全锁
```

**会话状态 (`_SessionState`)**：
- `entity_registry`: 实体名称到 ID 的映射
- `seen_relations`: 已见过的关系集合
- `last_relations_text`: 上次查询返回的关系文本
- `last_entities_text`: 上次查询返回的实体文本
- `initial_entity_names`: 初始实体名称列表

---

## SPARQL 查询错误处理

### 1. 查询执行错误

**位置**: `kg_r1/kgqa_bridge/sparql_adapter.py`

#### `get_relations` 错误处理

```python
def _handle_get_relations(self, ...):
    try:
        relations = self._client.get_relations(
            entity_resolved,
            question=question,
            top_k=self._kg_top_k * 3,
        )
    except Exception as exc:
        self._logger.warning("SPARQL get_relations failed: %s", exc)
        return self._format_error(
            f"SPARQL error: {exc}",
            sample_id,
            KGErrorType.SERVER_ERROR,
        )
```

**处理流程：**
1. 捕获所有异常（网络错误、超时等）
2. 记录警告日志
3. 返回格式化的错误消息
4. 错误类型标记为 `KGErrorType.SERVER_ERROR`

#### `get_triples` 错误处理

```python
def _handle_get_triples(self, ...):
    try:
        result = self._client.get_triples(
            entity_resolved,
            relations,
            limit_per_relation=5,
            question=question,
            return_with_cvt_info=True,
        )
        triples = result.get("triples", []) if isinstance(result, dict) else result
    except Exception as exc:
        self._logger.warning("SPARQL get_triples failed: %s", exc)
        return self._format_error(
            f"SPARQL error: {exc}",
            sample_id,
            KGErrorType.SERVER_ERROR,
        )
```

**处理流程：**
1. 捕获 SPARQL 执行异常
2. 记录警告日志
3. 返回错误消息，类型为 `SERVER_ERROR`

### 2. 实体解析错误

**位置**: `kg_r1/kgqa_bridge/sparql_adapter.py`

```python
def _entity_error(self, entity: str, session: _SessionState, sample_id: str):
    msg = "Invalid entity. Use an entity returned by the previous step and copy it exactly."
    if session.last_entities_text:
        msg += f"\n\nLast entities (names only):\n\n{session.last_entities_text}"
    return self._format_error(msg, sample_id, KGErrorType.FORMAT_ERROR)
```

**处理逻辑：**
- 实体无法解析时，返回友好的错误消息
- 如果会话中有上次查询的实体列表，会包含在错误消息中
- 错误类型：`KGErrorType.FORMAT_ERROR`

### 3. 关系选择错误

**位置**: `kg_r1/kgqa_bridge/sparql_adapter.py`

```python
def _relation_choice_error(self, relation: str, session: _SessionState, sample_id: str):
    msg = (
        f"The relation '{relation}' is not in the latest predicate list. "
        "Choose predicates from the list below:\n\n"
        f"{session.last_relations_text}"
    )
    return self._format_error(msg, sample_id, KGErrorType.FORMAT_ERROR)
```

**处理逻辑：**
- 当使用的关系不在已见过的关系列表中时触发
- 错误消息包含可用的关系列表
- 错误类型：`KGErrorType.FORMAT_ERROR`

### 4. HTTP 请求错误

**位置**: `kg_r1/llm_agent/generation.py` (FastAPI 模式)

#### 超时错误

```python
except requests.exceptions.Timeout as e:
    print(f"[KG_TIMEOUT_ERROR] Request timed out to {server_url}")
    error_responses = []
    for req in server_requests:
        error_msg = f"KG server timeout ({server_url})"
        error_responses.append({
            "error": error_msg,
            "query_time": 0,
            "total_results": 0,
            "request_payload": req,
            "kg_metadata": {"success": False, "error_type": KGErrorType.SERVER_ERROR},
        })
```

#### 连接错误

```python
except requests.exceptions.ConnectionError as e:
    print(f"[KG_CONNECTION_ERROR] Connection failed to {server_url}: {str(e)}")
    error_responses = []
    for req in server_requests:
        error_msg = f"KG server connection failed ({server_url})"
        error_responses.append({
            "error": error_msg,
            "query_time": 0,
            "total_results": 0,
            "request_payload": req,
            "kg_metadata": {"success": False, "error_type": KGErrorType.SERVER_ERROR},
        })
```

#### JSON 解析错误

```python
except json.JSONDecodeError as e:
    print(f"[KG_JSON_ERROR] Invalid JSON response from {server_url}: {str(e)}")
    error_msg = f"KG server response error ({server_url})"
    error_responses = [
        {
            "error": error_msg,
            "query_time": 0,
            "total_results": 0,
            "request_payload": req,
            "kg_metadata": {"success": False, "error_type": KGErrorType.SERVER_ERROR},
        }
        for req in server_requests
    ]
```

---

## 解析错误处理

### 1. 查询格式解析错误

**位置**: `kg_r1/llm_agent/generation.py`

```python
def _parse_kg_queries(self, query_contents: List[str], ...):
    parsed_requests = []
    for query_content_str in query_contents:
        try:
            # 解析查询内容
            # ...
        except Exception as e:
            # 创建 LLM 友好的错误消息
            error_msg = self._create_query_format_error(query_content_str, str(e))
            parsed_requests.append({
                "_is_client_error": True, 
                "error_message": error_msg,
            })
```

**处理流程：**
1. 捕获解析异常
2. 生成友好的错误消息
3. 标记为客户端错误（`_is_client_error`）
4. 错误类型：`KGErrorType.FORMAT_ERROR`

### 2. 无效动作处理

**位置**: `kg_r1/llm_agent/generation.py`

```python
def execute_predictions(self, ...):
    # ...
    else:
        # 检查响应是否过长
        if response_length_exceeded[i]:
            next_obs.append(
                f'\n\n<information>Your previous response was too long '
                f'({contents[i]}). Please provide shorter responses within the token limit.</information>\n\n'
            )
        else:
            next_obs.append(
                f'\n\n<information>Your previous action is invalid. '
                f'You should put the query between <kg-query> and </kg-query> '
                f'if you want to search, or put the answer between <answer> and </answer> '
                f'if you want to give the final answer.</information>\n\n'
            )
        dones.append(0)
        valid_action.append(0)
        is_search.append(0)
        raw_server_responses.append({
            "success": False, 
            "action": "invalid", 
            "kg_metadata": {
                "success": False, 
                "error_type": KGErrorType.FORMAT_ERROR
            }
        })
```

**处理逻辑：**
- 无效动作不会终止对话（`dones.append(0)`）
- 返回指导性错误消息
- 标记为无效动作（`valid_action.append(0)`）

---

## 错误类型分类

### `KGErrorType` 枚举 (`kg_r1/search/error_types.py`)

```python
class KGErrorType:
    # 系统/服务器级别错误
    SERVER_ERROR = "KG_SERVER_ERROR"           # 服务器不可用、HTTP 错误
    FORMAT_ERROR = "KG_FORMAT_ERROR"           # 无效请求格式、缺少必需字段
    
    # 数据未找到错误
    SAMPLE_NOT_FOUND = "KG_SAMPLE_NOT_FOUND"   # 样本/子图未找到
    ENTITY_NOT_FOUND = "KG_ENTITY_NOT_FOUND"   # 实体未找到
    RELATION_NOT_FOUND = "KG_RELATION_NOT_FOUND" # 关系未找到
    
    # 有效查询但无结果
    NO_RESULTS = "KG_NO_RESULTS"               # 空结果集
    
    # 成功
    SUCCESS = "KG_SUCCESS"                      # 操作成功完成
```

### 错误类型映射

| 错误场景 | 错误类型 | 位置 |
|---------|---------|------|
| SPARQL 查询执行失败 | `SERVER_ERROR` | `sparql_adapter.py` |
| 实体无法解析 | `FORMAT_ERROR` | `sparql_adapter.py` |
| 关系不在已见列表中 | `FORMAT_ERROR` | `sparql_adapter.py` |
| 查询格式解析失败 | `FORMAT_ERROR` | `generation.py` |
| HTTP 超时/连接失败 | `SERVER_ERROR` | `generation.py` |
| JSON 解析失败 | `SERVER_ERROR` | `generation.py` |
| 无效动作格式 | `FORMAT_ERROR` | `generation.py` |

---

## 会话状态管理

### 会话生命周期

#### 1. 会话创建

```python
def _get_session(self, sample_id: str, topic_entities: Optional[Dict[str, str]] = None):
    with self._sessions_lock:
        session = self._sessions.get(sample_id)
        if session is None:
            session = _SessionState()
            self._sessions[sample_id] = session
            # 从 topic_entities 初始化实体注册表
            self._seed_registry_from_topic_entities(session, topic_entities)
            session.initial_entity_names = self._extract_initial_entity_names(topic_entities)
        return session
```

#### 2. 会话重置

```python
def reset(self, sample_id: Optional[str] = None):
    """Clear adapter state. If sample_id is None, reset all sessions."""
    with self._sessions_lock:
        if sample_id is None:
            self._sessions.clear()
        else:
            self._sessions.pop(sample_id, None)
    if self._client:
        # 清除 flatten relations 缓存
        self._client.clear_pending_flatten_relations()
```

**调用时机：**
- 对话结束时（`dones[i] == True`）
- 位置：`generation.py:953-956`

```python
if self.use_sparql_bridge and session_keys:
    for idx, done in enumerate(dones):
        if done and idx < len(session_keys):
            self._reset_bridge_session(session_keys[idx])
```

### 实体注册表管理

#### 实体解析与注册

```python
def _resolve_and_register_entity(self, entity: str, session: _SessionState) -> Optional[str]:
    if not entity:
        return None
    # 如果已经是 ID 格式（m., g., en.），直接返回
    if entity.startswith(("m.", "g.", "en.")):
        session.entity_registry[entity] = entity
        return entity
    
    # 检查缓存
    key = entity.lower()
    if key in session.entity_registry:
        return session.entity_registry[key]
    
    # 解析新实体
    resolved = self._client._resolve_entity(entity)
    if resolved:
        session.entity_registry[key] = resolved
        session.entity_registry[resolved] = entity
    return resolved
```

#### 从三元组注册实体

```python
@staticmethod
def _register_entities(triples: List[Dict[str, Any]], session: _SessionState):
    for triple in triples:
        head_id = triple.get("head_id")
        head_name = triple.get("head")
        tail_id = triple.get("tail_id")
        tail_name = triple.get("tail")
        if head_id and head_name:
            session.entity_registry[head_name.lower()] = head_id
            session.entity_registry[head_id] = head_name
        if tail_id and tail_name:
            session.entity_registry[tail_name.lower()] = tail_id
            session.entity_registry[tail_id] = tail_name
```

---

## 错误恢复机制

### 1. 多轮对话中的错误恢复

**关键特性：**
- 错误不会立即终止对话（`dones.append(0)`）
- 错误消息会作为 `<information>` 反馈给模型
- 模型可以在下一轮修正错误

**示例流程：**

```
Turn 1: Model → <kg-query>get_relations("Invalid Entity")</kg-query>
         System → <information>Invalid entity. Use an entity returned by the previous step...</information>
         done = False  # 继续对话

Turn 2: Model → <kg-query>get_relations("Valid Entity")</kg-query>
         System → <information>Found relations: people.person.nationality, ...</information>
         done = False  # 继续对话

Turn 3: Model → <answer>["Answer"]</answer>
         System → (empty)
         done = True   # 对话结束
```

### 2. 错误消息格式化

**位置**: `kg_r1/kgqa_bridge/sparql_adapter.py`

```python
def _format_error(self, message: str, sample_id: str, error_type: KGErrorType):
    payload = {
        "object": "kg_retrieval",
        "success": False,
        "choices": [{"message": {"role": "tool", "content": message}}],
        "kg_metadata": {
            "success": False,
            "error_type": error_type,
        },
        "request_payload": {
            "sample_id": sample_id,
            "sparql_endpoint": self._sparql_endpoint,
        },
    }
    return message, payload
```

**特点：**
- 错误消息格式与成功响应一致
- 包含错误类型元数据
- 便于下游奖励计算和日志记录

### 3. 批量查询中的错误隔离

**位置**: `kg_r1/llm_agent/generation.py`

```python
def _batch_search_via_sparql_bridge(self, search_query_contents, meta_info_list):
    raw_payloads = []
    formatted_results = []
    
    for idx, query_content_str in enumerate(search_query_contents):
        try:
            formatted, payload = self.kgqa_adapter.run_query(...)
        except Exception as e:
            error_msg = f"SPARQL Bridge Error: {e}"
            self.logger.error(f"Error in SPARQL bridge for query '{query_content_str}': {e}")
            raw_payloads.append({
                "success": False,
                "error": error_msg,
                "kg_metadata": {"success": False, "error_type": KGErrorType.SERVER_ERROR},
            })
            formatted_results.append(error_msg)
    
    return raw_payloads, formatted_results
```

**特点：**
- 单个查询失败不影响其他查询
- 每个错误都被独立处理
- 保持批量处理的完整性

---

## 关键代码位置总结

| 功能 | 文件路径 | 关键函数/类 |
|------|---------|------------|
| 多轮对话管理 | `kg_r1/llm_agent/generation.py` | `LLMGenerationManager.execute_predictions()` |
| SPARQL 适配器 | `kg_r1/kgqa_bridge/sparql_adapter.py` | `KGQASparqlAdapter` |
| 错误类型定义 | `kg_r1/search/error_types.py` | `KGErrorType` |
| 会话状态 | `kg_r1/kgqa_bridge/sparql_adapter.py` | `_SessionState` |
| 错误格式化 | `kg_r1/kgqa_bridge/sparql_adapter.py` | `_format_error()`, `_entity_error()`, `_relation_choice_error()` |
| HTTP 错误处理 | `kg_r1/llm_agent/generation.py` | `_batch_search()` 中的异常处理 |

---

## 最佳实践

1. **错误消息设计**：错误消息应该对 LLM 友好，提供明确的修正指导
2. **会话隔离**：每个样本的会话状态独立管理，避免交叉污染
3. **错误恢复**：错误不应立即终止对话，给模型修正的机会
4. **日志记录**：所有错误都应记录日志，便于调试和监控
5. **错误分类**：使用明确的错误类型，便于下游处理和统计

---

## 相关文档

- [KG Retriever 文档](./kg_retriever.md)
- [SPARQL Bridge 集成](./retriever.md)
- [训练参数说明](../train_debug_single_a100_params.md)


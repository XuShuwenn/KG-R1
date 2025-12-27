# 评测端错误消息参考文档

本文档列出了 KG-Augmented 评测系统 (`kg_augmented_client.py`) 可能返回的所有错误消息类型。

## 错误消息分类

### 1. 系统级错误

#### 1.1 KG查询服务不可用
**错误消息**:
```
[KG query service not available]
```

**触发条件**: 
- KG客户端 (`kg_client`) 未初始化或为 None
- KG服务连接失败

**代码位置**: [kg_augmented_client.py](../kgqa_agent/src/eval/kg_augmented_client.py#L107)

---

### 2. 实体相关错误

#### 2.1 无效实体错误
**错误消息**:
```
Invalid entity. Use an entity returned by the previous step and copy it exactly.
```

**可能附加信息**:
```
Last entities (names only):

{上一步返回的实体列表}
```

**触发条件**:
- 模型使用了未在实体注册表 (`_entity_registry`) 中的实体
- 实体名称拼写错误或不准确
- 实体无法通过 `_resolve_and_register_entity()` 解析

**影响范围**:
- `get_relations(entity)` 调用
- `get_triples(entity, [relations])` 调用

**代码位置**: [kg_augmented_client.py](../kgqa_agent/src/eval/kg_augmented_client.py#L113-L119)

---

### 3. 关系/谓词相关错误

#### 3.1 关系选择错误
**错误消息**:
```
The relation '{relation}' is not in the latest predicate list. Choose predicates from the list below:

{最新的谓词列表}
```

**触发条件**:
- 在 `get_triples()` 中使用了不在最近一次 `get_relations()` 返回列表中的关系
- 关系名称标准化后 (通过 `normalize_relation()`) 不在 `_seen_relations_set` 中

**验证逻辑**:
```python
if self._seen_relations_set:
    for r in relations:
        norm_rel = normalize_relation(r)
        if norm_rel not in self._seen_relations_set:
            return relation_choice_error(r)
```

**代码位置**: [kg_augmented_client.py](../kgqa_agent/src/eval/kg_augmented_client.py#L121-L128)

---

### 4. 查询执行相关错误

#### 4.1 查询解析失败
**错误消息**:
```
[Could not parse query: {query_str}]
```

**触发条件**:
- KG查询字符串不符合以下任一格式:
  - `get_relations("entity_name")`
  - `get_triples("entity_name", ["rel1", "rel2", ...])`
- 正则表达式匹配失败

**有效查询格式示例**:
```python
# get_relations 格式
get_relations("Barack Obama")
get_relations('United States')

# get_triples 格式
get_triples("Barack Obama", ["place_of_birth", "date_of_birth"])
get_triples('United States', ["capital", "population"])
```

**代码位置**: [kg_augmented_client.py](../kgqa_agent/src/eval/kg_augmented_client.py#L301)

---

#### 4.2 无三元组结果
**错误消息**:
```
No triples found.
```

**触发条件**:
- `get_triples()` 查询成功执行但返回空结果
- SPARQL查询未找到匹配的三元组

**注意**: 这不是一个错误状态，而是正常的空结果提示

**代码位置**: [kg_augmented_client.py](../kgqa_agent/src/eval/kg_augmented_client.py#L260)

---

### 5. 格式验证错误

#### 5.1 响应格式错误
**错误消息**:
```
Your response did not contain a valid <kg-query> or <answer> tag. Please continue by outputting a valid <kg-query> or <answer>.
```

**触发条件**:
- 模型响应中既没有 `<kg-query>...</kg-query>` 标签
- 也没有 `<answer>...</answer>` 标签
- 响应格式不符合预期

**重试机制**:
- 最多重试 3 次 (`max_retries = 3`)
- 每次重试都会将错误消息作为新的用户消息添加到对话中
- 如果重试后仍无效，返回最后一次响应

**代码位置**: [kg_augmented_client.py](../kgqa_agent/src/eval/kg_augmented_client.py#L437)

---

### 6. 兜底错误

#### 6.1 无响应
**错误消息**:
```
[No response]
```

**触发条件**:
- 所有生成尝试都失败
- 极少出现的异常情况

**代码位置**: [kg_augmented_client.py](../kgqa_agent/src/eval/kg_augmented_client.py#L573)

---

## 错误处理流程

### 实体验证流程
```
模型使用实体 → _resolve_and_register_entity() 
              ↓
           是否已注册?
              ↓
         否 → 尝试通过kg_client解析
              ↓
         解析成功? 
              ↓
     否 → 返回 entity_error()
```

### 关系验证流程
```
模型选择关系 → normalize_relation()
              ↓
         在_seen_relations_set中?
              ↓
    否 → 返回 relation_choice_error()
```

### 查询解析流程
```
收到<kg-query> → 提取查询字符串
                ↓
            正则匹配get_relations?
                ↓
            是 → 执行get_relations
                ↓
            否 → 正则匹配get_triples?
                ↓
            是 → 执行get_triples
                ↓
            否 → 返回 "Could not parse query"
```

---

## Trace 记录

所有错误消息都会记录到 `_trace` 列表中，包含以下信息：

### 错误类型 Trace 条目
```python
{
    "type": "error",
    "content": "错误消息内容",
    "entity": "相关实体(如果适用)",
    "relation": "相关关系(如果适用)"
}
```

### 工具调用 Trace 条目
```python
{
    "type": "tool_call",
    "tool": "get_relations" | "get_triples",
    "args": {...},
    "result_count": 10,
    "tool_output": "格式化的结果"
}
```

---

## 调试建议

1. **实体错误**: 检查实体名称是否与之前返回的实体列表精确匹配
2. **关系错误**: 确保关系从最近的 `get_relations()` 结果中选择
3. **解析错误**: 验证查询字符串格式，注意引号匹配
4. **格式错误**: 确保响应包含正确的 XML 标签格式

---

## 相关配置

- **最大KG调用次数**: `max_calls` (默认 10)
- **Top-K关系数**: `kg_top_k` (默认 10)
- **格式错误重试次数**: 3次
- **SPARQL超时**: 120秒

---

*文档生成时间: 2025-12-27*  
*对应代码文件: `kgqa_agent/src/eval/kg_augmented_client.py`*

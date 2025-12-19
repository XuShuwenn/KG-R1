# kgqa_agent 与训练端交互流程对比分析

## 概述
本文档对比了 `kgqa_agent/src/eval/kg_augmented_client.py` 和训练端 `kg_r1/kgqa_bridge/sparql_adapter.py` 的交互流程，识别出不对齐之处。

## 关键差异总结

### 1. get_relations 的 LLM 过滤条件 ⚠️ **重要差异**

**kgqa_agent (kg_augmented_client.py:140-147)**:
```python
if len(relations) > 10:
    # Shuffle to avoid position bias before filtering
    random.shuffle(relations)
    # Filter with LLM
    relations = self._filter_relations_with_llm(relations, question, entity)

# Limit to top-k for returning to the model
relations = relations[:self.kg_top_k]
```

**训练端 (sparql_adapter.py:307-317)**:
```python
if (
    self._relation_filter_model
    and len(relations) > 10
):
    relations = self._filter_relations_with_llm(
        relations,
        question,
        entity,
        use_flatten_prompt=False,
    )
relations = relations[: self._kg_top_k]
```

**差异**:
- **kgqa_agent**: 总是进行 LLM 过滤（如果 `len(relations) > 10`），不检查是否有 filter model
- **训练端**: 只有在配置了 `_relation_filter_model` 时才过滤
- **影响**: 如果训练端没有配置 filter model，即使 relations > 10 也不会过滤，直接返回 BM25 排序的结果

**建议**: 
- 如果训练端应该完全对齐 kgqa_agent，应该移除 `self._relation_filter_model` 的检查，或者确保训练端总是配置了 filter model
- 但根据代码设计，训练端允许不配置 filter model（fallback 到 BM25），这可能是有意的设计差异

### 2. get_relations 的 shuffle 时机 ⚠️ **潜在差异**

**kgqa_agent (kg_augmented_client.py:141-142)**:
```python
# Shuffle to avoid position bias before filtering
random.shuffle(relations)
```

**训练端 (sparql_adapter.py:526-528)**:
```python
# Shuffle before building prompt to avoid positional bias
shuffled = relation_entries[:]
random.shuffle(shuffled)
```

**差异**:
- **kgqa_agent**: 在 `get_relations` 中直接 shuffle `relations` 列表
- **训练端**: 在 `_filter_relations_with_llm` 内部 shuffle，但 `get_relations` 中不 shuffle
- **影响**: 如果训练端没有配置 filter model，relations 不会被 shuffle，可能存在位置偏差

**建议**: 在训练端的 `_handle_get_relations` 中，即使不进行 LLM 过滤，也应该 shuffle relations（如果 `len(relations) > 10`）以避免位置偏差

### 3. CVT flatten relations 的 LLM 过滤 prompt ⚠️ **潜在问题**

**kgqa_agent (kg_augmented_client.py:226-228)**:
```python
combined_relations = self._filter_relations_with_llm(
    combined_relations, question, entity
)
# 没有传递 use_flatten_prompt，默认是 False
```

**训练端 (sparql_adapter.py:449-451)**:
```python
combined_relations = self._filter_relations_with_llm(
    combined_relations, question, entity, use_flatten_prompt=False
)
```

**差异**:
- 两者都使用 `use_flatten_prompt=False`，即使用普通的 `filter_prompt_template` 而不是 `flatten_rel_filter_prompt_template`
- 根据 `flatten_rel_filter_prompt_template` 的设计，它专门用于过滤 flatten relations（要求选择 8 个关系）
- **潜在问题**: 可能应该使用 `use_flatten_prompt=True` 来更好地处理 flatten relations

**建议**: 
- 检查 kgqa_agent 的原始设计意图：是否应该对 flatten relations 使用特殊的 prompt
- 如果应该使用，需要修复两处代码

### 4. BM25 排序的错误处理 ⚠️ **健壮性差异**

**kgqa_agent (kg_augmented_client.py:218-221)**:
```python
# Apply BM25 ranking if question provided
if augmented_question and combined_relations:
    combined_relations = self.kg_client.rank_by_similarity(
        combined_relations, augmented_question, "relation"
    )
```

**训练端 (sparql_adapter.py:436-446)**:
```python
if (
    question
    and combined_relations
    and hasattr(self._client, "rank_by_similarity")
):
    try:
        combined_relations = self._client.rank_by_similarity(
            combined_relations, question, "relation"
        )
    except Exception:
        pass
```

**差异**:
- **kgqa_agent**: 没有 try-except，如果 `rank_by_similarity` 失败会抛出异常
- **训练端**: 有 try-except 和 `hasattr` 检查，更健壮
- **影响**: 训练端在 BM25 排序失败时不会中断，会继续使用未排序的 relations

**建议**: 
- 训练端的错误处理更合理，但应该记录警告日志
- 可以考虑在 kgqa_agent 中也添加类似的错误处理

### 5. 关系解析和去重 ✅ **一致**

两者都使用相同的逻辑：
- 解析 relations 字符串，去除引号
- 去重（使用 set 或列表检查）
- 截断到前 4 个 relations

### 6. 实体注册 ✅ **一致**

两者都从 triples 中提取 `head`/`head_id` 和 `tail`/`tail_id`，然后注册到 entity_registry。

### 7. Triple 格式化 ✅ **一致**

两者都使用 `[{head}, {relation}, {tail}]` 格式，使用实体名称（不是 MID）。

### 8. 错误消息格式 ✅ **一致**

两者都返回简洁的错误消息（如 "No relations found." 或 "No triples found."），符合 kgqa_agent 的模式。

### 9. max_calls 检查 ✅ **一致**

两者都在执行查询前检查 `max_calls`，达到限制时返回 `FORCE_ANSWER_PROMPT`。

## 需要修复的问题

### 优先级 1: get_relations 的过滤条件
- **问题**: 训练端只有在配置了 filter model 时才过滤，而 kgqa_agent 总是过滤
- **影响**: 如果训练端没有配置 filter model，行为不一致
- **修复**: 
  1. 确保训练端总是配置了 filter model，或
  2. 修改训练端逻辑，即使没有 filter model 也进行过滤（使用默认的 filter client）

### 优先级 2: get_relations 的 shuffle ✅ **已修复**
- **问题**: 训练端在 `get_relations` 中不 shuffle（如果未配置 filter model）
- **影响**: 可能存在位置偏差
- **修复**: 已在 `_handle_get_relations` 中修复，现在如果 `len(relations) > 10`，即使不进行 LLM 过滤也会 shuffle

### 优先级 3: CVT flatten relations 的 prompt ✅ **已确认一致**
- **问题**: 两者都使用 `use_flatten_prompt=False` 来过滤 flatten relations
- **分析**: 
  - `flatten_rel_filter_prompt_template` 要求选择 **exactly 8** 个 flatten relations
  - `filter_prompt_template` 要求选择 **exactly 10** 个 relations
  - 在 `get_triples` 中，flatten relations 与原始 relations 合并后，使用普通的 `filter_prompt_template` 更合适
  - `direct_sparql_client.py` 内部处理 CVT 时使用 `use_flatten_prompt=True`，因为那里处理的是纯 flatten relations 列表
- **结论**: 当前行为是正确的，不需要修复

### 优先级 4: BM25 排序的错误处理
- **问题**: kgqa_agent 没有错误处理
- **影响**: 如果 BM25 排序失败，会中断执行
- **修复**: 在 kgqa_agent 中添加 try-except 和日志记录

## 验证建议

1. **检查训练配置**: 确认训练端是否总是配置了 `relation_filter_model`
2. **测试 shuffle**: 验证 relations 的顺序是否影响结果
3. **测试 flatten relations**: 对比使用 `use_flatten_prompt=True` 和 `False` 的效果
4. **测试 BM25 失败**: 验证在 BM25 排序失败时的行为

## 总结

主要差异集中在：
1. **get_relations 的过滤条件**: 训练端需要配置 filter model 才过滤，kgqa_agent 总是过滤
2. **shuffle 时机**: 训练端在 `get_relations` 中不 shuffle（如果未配置 filter model）
3. **CVT flatten relations 的 prompt**: 两者都使用普通 prompt，可能应该使用专门的 prompt

其他方面（关系解析、实体注册、triple 格式化、错误处理）基本一致。


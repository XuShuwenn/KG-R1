# 慢查询耗时分析

## 概述

为了诊断慢查询的根本原因，我们在查询的关键位置添加了详细的耗时日志。这些日志将帮助识别哪些步骤导致了查询延迟。

## 添加的耗时日志位置

### 1. `run_query` 函数（总体流程）

**位置**: `kg_r1/kgqa_bridge/sparql_adapter.py::run_query()`

**记录的步骤**:
- `session setup`: 获取或创建会话的时间
- `_ensure_client`: 初始化SPARQL客户端的时间
- `_augment_question`: 问题增强的时间
- `query parsing`: 查询字符串解析的时间
- `_parse_relations`: 关系列表解析的时间（仅get_triples）
- `TOTAL get_relations/get_triples`: 整个查询的总耗时

**日志格式**:
```
[TIMING] run_query: session setup took X.XXXs
[TIMING] run_query: _ensure_client took X.XXXs
[TIMING] run_query: _augment_question took X.XXXs
[TIMING] run_query: query parsing took X.XXXs
[TIMING] run_query: _parse_relations took X.XXXs
[TIMING] run_query: TOTAL get_relations took X.XXXs
[TIMING] run_query: TOTAL get_triples took X.XXXs
```

### 2. `_handle_get_relations` 函数

**位置**: `kg_r1/kgqa_bridge/sparql_adapter.py::_handle_get_relations()`

**记录的步骤**:
- `entity resolution`: 实体解析的时间
- `_client.get_relations`: SPARQL查询执行的时间
- `shuffle`: 关系列表打乱的时间（如果>10个关系）
- `LLM filtering`: LLM过滤的时间（如果启用）
- `format_relations_for_prompt`: 格式化关系列表的时间
- `post-processing`: 后处理的总时间
- `TOTAL`: 整个get_relations的总耗时

**日志格式**:
```
[TIMING] _handle_get_relations: entity resolution took X.XXXs: entity -> resolved
[TIMING] _handle_get_relations: _client.get_relations took X.XXXs, got N relations
[TIMING] _handle_get_relations: shuffle took X.XXXs
[TIMING] _handle_get_relations: LLM filtering took X.XXXs
[TIMING] _handle_get_relations: format_relations_for_prompt took X.XXXs
[TIMING] _handle_get_relations: post-processing took X.XXXs
[TIMING] _handle_get_relations: TOTAL took X.XXXs
```

### 3. `_handle_get_triples` 函数

**位置**: `kg_r1/kgqa_bridge/sparql_adapter.py::_handle_get_triples()`

**记录的步骤**:
- `entity resolution`: 实体解析的时间
- `relation validation`: 关系验证的时间
- `_client.get_triples`: SPARQL查询执行的时间（**最可能慢的地方**）
- `result parsing`: 结果解析的时间
- `formatting triples`: 格式化三元组的时间
- `_register_entities`: 注册实体的时间
- `get_pending_flatten_relations`: 获取待展开关系的时间
- `CVT info formatting`: CVT信息格式化的时间
- `rank_by_similarity`: 关系相似度排序的时间
- `shuffle`: 关系列表打乱的时间
- `LLM filtering`: LLM过滤的时间
- `flatten relations processing`: 展开关系处理的总时间
- `TOTAL`: 整个get_triples的总耗时

**日志格式**:
```
[TIMING] _handle_get_triples: entity resolution took X.XXXs: entity -> resolved
[TIMING] _handle_get_triples: relation validation took X.XXXs
[TIMING] _handle_get_triples: _client.get_triples took X.XXXs
[TIMING] _handle_get_triples: result parsing took X.XXXs
[TIMING] _handle_get_triples: formatting triples took X.XXXs
[TIMING] _handle_get_triples: _register_entities took X.XXXs
[TIMING] _handle_get_triples: get_pending_flatten_relations took X.XXXs
[TIMING] _handle_get_triples: CVT info formatting took X.XXXs
[TIMING] _handle_get_triples: rank_by_similarity took X.XXXs
[TIMING] _handle_get_triples: shuffle took X.XXXs
[TIMING] _handle_get_triples: LLM filtering took X.XXXs
[TIMING] _handle_get_triples: flatten relations processing took X.XXXs
[TIMING] _handle_get_triples: TOTAL took X.XXXs
```

### 4. `_resolve_and_register_entity` 函数

**位置**: `kg_r1/kgqa_bridge/sparql_adapter.py::_resolve_and_register_entity()`

**记录的步骤**:
- `validation`: 实体验证的时间
- `cache check`: 缓存查找的时间
- `_client._resolve_entity`: 实体解析的时间（**可能慢的地方**）
- `TOTAL`: 整个实体解析的总耗时

**日志格式**:
```
[TIMING] _resolve_and_register_entity: validation took X.XXXs
[TIMING] _resolve_and_register_entity: cache lookup took X.XXXs
[TIMING] _resolve_and_register_entity: _client._resolve_entity took X.XXXs for 'entity'
[TIMING] _resolve_and_register_entity: TOTAL took X.XXXs for 'entity' -> resolved
```

### 5. `_filter_relations_with_llm` 函数

**位置**: `kg_r1/kgqa_bridge/sparql_adapter.py::_filter_relations_with_llm()`

**记录的步骤**:
- `_ensure_filter_client`: 初始化过滤客户端的时间
- `prompt preparation`: 提示词准备的时间
- `LLM generate`: LLM生成的时间（**可能慢的地方**）
- `response parsing`: 响应解析的时间
- `TOTAL`: 整个LLM过滤的总耗时

**日志格式**:
```
[TIMING] _filter_relations_with_llm: _ensure_filter_client took X.XXXs
[TIMING] _filter_relations_with_llm: prompt preparation took X.XXXs
[TIMING] _filter_relations_with_llm: LLM generate (attempt N) took X.XXXs
[TIMING] _filter_relations_with_llm: response parsing took X.XXXs
[TIMING] _filter_relations_with_llm: TOTAL took X.XXXs, filtered N/M relations
```

## 慢查询的可能原因分析

基于代码分析，慢查询可能由以下原因导致：

### 1. SPARQL查询本身慢（最可能）

**位置**: `_client.get_relations()` 或 `_client.get_triples()`

**原因**:
- KG服务器响应慢
- 查询复杂度高（多个关系、大量数据）
- 网络延迟
- KG服务器负载高

**诊断**: 查看 `[TIMING] _handle_get_relations: _client.get_relations took X.XXXs` 或 `[TIMING] _handle_get_triples: _client.get_triples took X.XXXs` 的日志

### 2. 实体解析慢

**位置**: `_client._resolve_entity()`

**原因**:
- 实体名称模糊，需要多次查询
- 实体解析服务响应慢
- 网络延迟

**诊断**: 查看 `[TIMING] _resolve_and_register_entity: _client._resolve_entity took X.XXXs` 的日志

### 3. LLM过滤慢

**位置**: `_filter_client.generate()`

**原因**:
- LLM API响应慢
- 提示词过长
- LLM服务负载高

**诊断**: 查看 `[TIMING] _filter_relations_with_llm: LLM generate took X.XXXs` 的日志

### 4. CVT展开处理慢

**位置**: `_handle_get_triples()` 中的CVT信息处理

**原因**:
- CVT节点数量多
- 展开的三元组数量大

**诊断**: 查看 `[TIMING] _handle_get_triples: CVT info formatting took X.XXXs` 的日志

### 5. 关系相似度排序慢

**位置**: `_client.rank_by_similarity()`

**原因**:
- 关系数量多
- 相似度计算复杂

**诊断**: 查看 `[TIMING] _handle_get_triples: rank_by_similarity took X.XXXs` 的日志

## 日志输出条件

为了减少日志噪音，耗时日志只在以下情况下输出：

1. **关键步骤**: 总是输出（如 `_client.get_relations`, `_client.get_triples`）
2. **耗时阈值**: 
   - 超过 0.01秒（10ms）的操作
   - 超过 0.1秒（100ms）的操作（更详细的日志）
   - 超过 1.0秒（1s）的总耗时

## 使用建议

1. **运行训练**: 正常训练时，这些日志会自动输出
2. **查看日志**: 在训练日志中搜索 `[TIMING]` 来查看所有耗时信息
3. **分析慢查询**: 
   - 找到总耗时最长的查询
   - 查看该查询的各个步骤耗时
   - 识别瓶颈步骤
4. **优化建议**:
   - 如果 `_client.get_relations/get_triples` 慢 → 优化SPARQL查询或KG服务器
   - 如果 `_client._resolve_entity` 慢 → 优化实体解析或增加缓存
   - 如果 `LLM generate` 慢 → 优化LLM过滤或考虑禁用
   - 如果 `CVT info formatting` 慢 → 优化CVT处理逻辑

## 示例日志输出

```
[TIMING] run_query: session setup took 0.001s
[TIMING] run_query: _ensure_client took 0.002s
[TIMING] run_query: query parsing took 0.000s
[TIMING] _handle_get_triples: entity resolution took 0.150s: "The Bride with White Hair" -> m.0abc123
[TIMING] _handle_get_triples: relation validation took 0.001s
[TIMING] _handle_get_triples: _client.get_triples took 142.500s  <-- 瓶颈！
[TIMING] _handle_get_triples: result parsing took 0.002s
[TIMING] _handle_get_triples: formatting triples took 0.010s
[TIMING] _handle_get_triples: get_pending_flatten_relations took 0.050s
[TIMING] _handle_get_triples: CVT info formatting took 0.100s
[TIMING] _handle_get_triples: TOTAL took 142.813s
[TIMING] run_query: TOTAL get_triples took 142.815s
```

从这个示例可以看出，瓶颈在于 `_client.get_triples`，耗时142.5秒，占总时间的99.8%。

## 下一步

1. 运行训练并收集日志
2. 分析 `[TIMING]` 日志，找出最慢的步骤
3. 针对瓶颈步骤进行优化


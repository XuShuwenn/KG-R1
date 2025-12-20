# KG 查询执行流程详解

本文档详细描述评测端对 `get_relations` 和 `get_triples` 两种查询的完整处理流程，涵盖从解析到返回结果的每一个执行细节。

## 一、查询解析与分发

### 1.1 查询提取
- **位置**: `KGAugmentedModelClient._parse_and_execute_query()`
- **输入**: 从模型响应中提取的 `<kg-query>...</kg-query>` 标签内容
- **解析方式**: 使用正则表达式匹配函数调用格式

### 1.2 查询类型识别
- **get_relations**: 匹配模式 `get_relations("entity_name")`
- **get_triples**: 匹配模式 `get_triples("entity_name", ["rel1", "rel2", ...])`
- 如果无法匹配，返回错误信息 `[Could not parse query: ...]`

---

## 二、get_relations 查询流程

### 2.1 实体解析与注册
**步骤**:
1. 从查询字符串中提取实体名称（去除引号）
2. 调用 `_resolve_and_register_entity(entity)` 解析实体：
   - 如果实体名以 `m.`、`g.`、`en.` 开头，直接作为 ID 使用
   - 否则从 `_entity_registry` 中查找（key 为小写实体名）
   - 如果注册表中没有，调用 `kg_client._resolve_entity()` 通过 SPARQL 查询解析
   - 解析成功后，将实体名和 ID 的双向映射存入注册表
3. 如果解析失败，返回错误信息（包含上次查询返回的实体列表）

### 2.2 调用底层 get_relations
**参数**:
- `entity_resolved`: 解析后的实体 ID
- `question`: 增强后的问题（原问题 + 初始实体名称）
- `top_k`: `self.kg_top_k * 3`（例如，如果 `kg_top_k=10`，则请求 30 个候选）

### 2.3 DirectSPARQLKGClient.get_relations 内部流程

#### 2.3.1 实体 ID 解析
- 如果输入已经是 ID（以 `m.`、`g.`、`en.` 开头），直接使用
- 否则通过 `EntityResolver` 查询实体名称对应的 ID

#### 2.3.2 关系查询（SPARQL）
**查询出边关系**:
```sparql
SELECT DISTINCT ?relation WHERE {
    ns:{entity_id} ?relation ?tail .
    FILTER(isIRI(?relation))
    FILTER(STRSTARTS(STR(?relation), "http://rdf.freebase.com/ns/"))
} LIMIT {limit}
```

**查询入边关系**:
```sparql
SELECT DISTINCT ?relation WHERE {
    ?head ?relation ns:{entity_id} .
    FILTER(isIRI(?relation))
    FILTER(STRSTARTS(STR(?relation), "http://rdf.freebase.com/ns/"))
} LIMIT {limit}
```

**结果处理**:
- 合并出边和入边关系，去重
- 将 URI 格式转换为点分隔格式（`http://rdf.freebase.com/ns/people.person.children` → `people.person.children`）
- 结果缓存到 `_rel_cache[entity_id]` 中

#### 2.3.3 添加待处理的 Flatten 关系
- 如果该实体在之前的 `get_triples` 调用中发现了 CVT 节点并生成了 flatten 关系，将这些 flatten 关系添加到候选列表中
- 这些 flatten 关系存储在 `_pending_flatten_relations[entity_id]` 中

#### 2.3.4 白名单过滤
- 应用 CWQ 关系白名单（`filtered_cwq_white_list.json`）
- 如果过滤后结果为空，回退到原始结果

#### 2.3.5 过滤无意义模式关系
- 过滤掉元数据关系，如：
  - `type.object.type`
  - `type.type.instance`
  - `common.topic.article`
  - `freebase.type_hints.*`
  - 等等（详见 `_is_meaningless_pattern_relation()`）

#### 2.3.6 BM25 排序
- 如果提供了问题文本，使用 BM25 算法对关系进行排序
- 基于问题与关系名称的相似度计算分数
- 返回排序后的关系列表（最多 `top_k` 个）

### 2.4 LLM 过滤（可选）
**触发条件**: 候选关系数量 > 10

**流程**:
1. 随机打乱关系列表（避免位置偏差）
2. 调用 `_filter_relations_with_llm(relations, question, entity)`:
   - 构建过滤 prompt（使用 `filter_prompt_template`）
   - 调用 `filter_client.generate()`（默认使用 `gpt-4o-mini`）
   - 解析 LLM 返回的 JSON 列表格式的关系选择
   - 根据 LLM 选择重新排序关系列表
   - 如果解析失败或 API 调用失败，使用指数退避重试（最多 3 次）
   - 如果最终失败，回退到原始列表的前 `kg_top_k` 个

### 2.5 结果限制与格式化
- 限制到 `self.kg_top_k` 个关系
- 调用 `format_relations_for_prompt()` 格式化为文本：
  - 每行一个关系名称
  - 如果没有关系，返回 `"No relations found."`

### 2.6 状态更新
- 更新 `_seen_relations_set`：将返回的关系添加到已见关系集合（用于后续验证）
- 更新 `_last_relations_text`：保存格式化后的关系文本
- 清空 `_last_entities_text`
- 记录 trace 信息（工具调用、参数、结果数量、输出文本）

### 2.7 返回结果
- 返回格式化后的关系文本，供模型在下一轮中使用

---

## 三、get_triples 查询流程

### 3.1 查询解析
**步骤**:
1. 提取实体名称和关系列表
2. 解析关系列表（去除引号，去重）
3. **限制关系数量**: 最多取前 4 个关系（即使模型提供了更多）

### 3.2 实体解析与注册
- 与 `get_relations` 相同：调用 `_resolve_and_register_entity()`
- 如果解析失败，返回错误信息

### 3.3 关系验证
**验证逻辑**:
- 如果 `_seen_relations_set` 不为空（说明之前调用过 `get_relations`）
- 对每个关系调用 `normalize_relation()` 规范化
- 检查规范化后的关系是否在 `_seen_relations_set` 中
- 如果不在，返回错误信息（包含上次 `get_relations` 返回的关系列表）

**注意**: 这个验证确保模型只能使用之前 `get_relations` 返回的关系，防止使用未验证的关系。

### 3.4 调用底层 get_triples
**参数**:
- `entity_resolved`: 解析后的实体 ID
- `relations`: 验证后的关系列表（最多 4 个）
- `limit_per_relation`: 5（每个关系最多返回 5 个三元组）
- `question`: 增强后的问题
- `return_with_cvt_info`: `True`（返回 CVT 信息）

### 3.5 DirectSPARQLKGClient.get_triples 内部流程

#### 3.5.1 实体名称查询
- 查询实体的英文名称（`ns:type.object.name`，优先英文，回退到任意语言）
- 用于后续三元组格式化

#### 3.5.2 关系去重
- 对输入的关系列表去重（保留顺序）

#### 3.5.3 处理 Flatten 关系映射
- 如果某个关系在 `_cvt_mapping` 中（说明是之前生成的 flatten 关系），直接调用 `_query_flatten_relation()` 查询两跳路径
- 跳过常规查询流程

#### 3.5.4 常规关系查询（对每个关系）

**查询出边三元组**:
```sparql
SELECT ?tail (SAMPLE(?name_en) as ?preferred_name) (SAMPLE(?name_any) as ?fallback_name) WHERE {
    ns:{entity_id} ns:{relation_id} ?tail .
    OPTIONAL { ?tail ns:type.object.name ?name_en . FILTER(LANGMATCHES(LANG(?name_en), 'en')) }
    OPTIONAL { ?tail ns:type.object.name ?name_any . }
} GROUP BY ?tail LIMIT {limit_per_relation * 2}
```

**查询入边三元组**:
```sparql
SELECT ?head (SAMPLE(?name_en) as ?preferred_name) (SAMPLE(?name_any) as ?fallback_name) WHERE {
    ?head ns:{relation_id} ns:{entity_id} .
    OPTIONAL { ?head ns:type.object.name ?name_en . FILTER(LANGMATCHES(LANG(?name_en), 'en')) }
    OPTIONAL { ?head ns:type.object.name ?name_any . }
} GROUP BY ?head LIMIT {limit_per_relation * 4}
```

**结果处理**:
- 将 URI 转换为点分隔格式
- 优先使用英文名称，回退到任意名称，最后回退到 ID
- 检查是否为 CVT 节点（通过 `_is_cvt_node()`）：
  - ID 以 `m.` 开头
  - 名称为空或等于 ID
- 如果是普通实体，创建三元组数据：`{head, head_id, relation, tail, tail_id}`
- 如果是 CVT 节点，记录到 `cvt_nodes_found_out` 或 `cvt_nodes_found_in`

#### 3.5.5 CVT 节点处理（全局处理）

**收集阶段**（对每个关系中的 CVT 节点）:
1. 查询 CVT 节点的出边/入边关系
2. 过滤无意义模式关系
3. 应用白名单过滤
4. 为每个 CVT 关系创建 flatten 关系名称：
   - 出边: `_flatten_relation(original_rel, cvt_rel)`（去除公共前缀后拼接）
   - 入边: `_flatten_relation(cvt_rel, original_rel)`
   - 处理重名：添加数字后缀（`relation_1`, `relation_2`, ...）
5. 收集到 `all_flatten_candidates_out` 或 `all_flatten_candidates_in`

**全局排序与过滤**（所有关系处理完后）:
1. **BM25 排序**: 对所有 flatten 候选进行 BM25 排序（基于问题 + 实体名），取 top-50
2. **查询验证**: 对 top-50 候选查询实际三元组，过滤掉 `head_id == tail_id` 的无意义三元组
3. **LLM 过滤**: 对有意义候选调用 LLM 过滤（使用 `flatten_rel_filter_prompt_template`），取 top-8
4. **最终查询**: 对 top-8 候选再次查询三元组，确保有意义

**Flatten 关系映射存储**:
- 将最终选中的 flatten 关系存入 `_cvt_mapping[flatten_rel] = (original_rel, cvt_node_id, cvt_rel, direction)`
- 存入 `_pending_flatten_relations[entity_id]`，供后续 `get_relations` 使用

**CVT 信息收集**:
- 为每个选中的 flatten 关系创建 CVT 信息字典：
  ```python
  {
      "original_relation": original_rel,
      "cvt_node_id": cvt_node_id,
      "cvt_relation": cvt_rel_id,
      "flatten_relation": flatten_rel,
      "direction": "out" or "in",
      "flattened_triples": meaningful_triples
  }
  ```

#### 3.5.6 常规三元组采样
- 如果发现 CVT 节点，每个关系的常规三元组限制为 15 个；否则为 5 个
- 如果超过限制，随机采样

#### 3.5.7 结果合并
- 合并所有常规三元组和 flatten 三元组
- 返回格式：
  ```python
  {
      "triples": all_triples,  # 所有三元组列表
      "cvt_info": cvt_info     # CVT 信息列表（如果有）
  }
  ```

### 3.6 实体注册
- 从返回的三元组中提取所有实体（head 和 tail）
- 调用 `_register_entities()` 将实体名和 ID 的双向映射存入 `_entity_registry`

### 3.7 结果格式化
- 将三元组格式化为文本：`"[head_name, relation, tail_name]"`
- 每行一个三元组
- 如果没有三元组，返回 `"No triples found."`

### 3.8 状态更新
- 更新 `_last_entities_text`：保存格式化后的三元组文本
- 清空 `_last_relations_text`
- 记录 trace 信息（包括 CVT 信息，如果存在）

### 3.9 返回结果
- 返回格式化后的三元组文本，供模型在下一轮中使用

---

## 四、关键设计要点

### 4.1 实体注册表
- **目的**: 避免重复查询实体名称到 ID 的映射
- **存储**: `_entity_registry` 字典，支持双向查找（name ↔ ID）
- **初始化**: 从 `topic_entities` 参数中预填充
- **更新**: 每次 `get_triples` 返回后，自动注册新发现的实体

### 4.2 关系验证机制
- **目的**: 确保模型只能使用已验证的关系
- **实现**: `_seen_relations_set` 存储所有 `get_relations` 返回的关系
- **验证**: `get_triples` 调用前，检查所有关系是否在集合中
- **规范化**: 使用 `normalize_relation()` 统一关系格式

### 4.3 CVT 节点自动处理
- **检测**: 通过 `_is_cvt_node()` 识别 CVT 节点（无名称的中间节点）
- **Flatten**: 自动将两跳路径压缩为一跳关系
- **存储**: Flatten 关系存储在 `_pending_flatten_relations` 中，供后续 `get_relations` 使用
- **映射**: `_cvt_mapping` 存储 flatten 关系到原始路径的映射

### 4.4 多级过滤策略
1. **白名单过滤**: 基于 CWQ 数据集的关系白名单
2. **模式过滤**: 过滤元数据和无意义模式关系
3. **BM25 排序**: 基于问题相似度排序
4. **LLM 过滤**: 对大量候选使用 LLM 进行精确过滤
5. **有意义性验证**: 过滤 `head_id == tail_id` 的无意义三元组

### 4.5 错误处理
- **实体解析失败**: 返回错误信息，包含上次查询的实体列表
- **关系验证失败**: 返回错误信息，包含上次查询的关系列表
- **SPARQL 查询失败**: 记录错误日志，返回空结果或部分结果
- **LLM 过滤失败**: 使用指数退避重试，最终回退到 BM25 排序结果

---

## 五、执行流程图

```
get_relations 流程:
模型输出 <kg-query>get_relations("entity")</kg-query>
    ↓
解析查询字符串，提取实体名
    ↓
解析实体名 → 实体ID（从注册表或SPARQL查询）
    ↓
调用 DirectSPARQLKGClient.get_relations()
    ├─ 查询出边关系（SPARQL）
    ├─ 查询入边关系（SPARQL）
    ├─ 合并并去重
    ├─ 添加待处理 flatten 关系
    ├─ 白名单过滤
    ├─ 过滤无意义模式关系
    └─ BM25 排序（如果有问题）
    ↓
如果候选 > 10: LLM 过滤（可选）
    ↓
限制到 top_k 个关系
    ↓
格式化并返回

get_triples 流程:
模型输出 <kg-query>get_triples("entity", ["rel1", "rel2"])</kg-query>
    ↓
解析查询字符串，提取实体名和关系列表（最多4个）
    ↓
解析实体名 → 实体ID
    ↓
验证关系是否在 _seen_relations_set 中
    ↓
调用 DirectSPARQLKGClient.get_triples()
    ├─ 对每个关系：
    │   ├─ 查询出边三元组（SPARQL）
    │   ├─ 查询入边三元组（SPARQL）
    │   ├─ 检测 CVT 节点
    │   └─ 收集 CVT 节点候选
    ├─ 全局处理 CVT 节点：
    │   ├─ BM25 排序（top-50）
    │   ├─ 查询验证（过滤无意义）
    │   ├─ LLM 过滤（top-8）
    │   └─ 查询最终 flatten 三元组
    └─ 合并常规三元组和 flatten 三元组
    ↓
注册新发现的实体到注册表
    ↓
格式化三元组并返回
```

---

## 六、相关配置参数

- `kg_top_k`: 每次 `get_relations` 返回的关系数量（默认 10）
- `max_calls`: 每个问题最多允许的 KG 查询次数（默认 10）
- `limit_per_relation`: 每个关系最多返回的三元组数量（默认 5）
- `timeout`: SPARQL 查询超时时间（默认 120 秒）
- `filter_client`: LLM 过滤使用的模型客户端（默认 `gpt-4o-mini`）

---

## 七、Trace 记录

每次查询都会记录详细的 trace 信息，包括：
- `type: "tool_call"`: 工具调用信息（工具名、参数、结果数量）
- `type: "kg_query"`: 原始查询内容
- `type: "information"`: 查询返回的结果
- `type: "error"`: 错误信息
- `cvt_info`: CVT 节点详细信息（如果存在）

这些 trace 信息用于调试、分析和评估模型的行为。


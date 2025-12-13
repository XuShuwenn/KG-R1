# Conjunction-Type Path Generation

本目录包含用于生成 conjunction 类型路径的配置和说明。

## 什么是 Conjunction 类型路径？

Conjunction 类型的路径是指两个不同的实体通过不同的关系都指向同一个目标实体：

```
entity1 -> relationA -> entity2
entity3 -> relationB -> entity2
```

**重要说明**：
- `entity1` 和 `entity3` 是 **topic entities**（主题实体）
- `entity2` 是 **answers**（答案实体）
- `entity2` 可能有多个候选，但只接受候选个数 **小于等于 3** 的路径作为合法路径
- 生成的路径中会明确记录 `topic_entities` 和 `answers` 列表

这种路径模式在复杂问答中很常见，例如：
- "哪些人既是某个组织的成员，又是某个奖项的获得者？"
- "哪些城市既是某个国家的首都，又是某个河流的流经地？"

## 使用方法

### 1. 准备配置文件

使用示例配置文件 `conjunction_walk_example.yaml` 或创建自己的配置：

```yaml
seed: 42
walks:
  num_starts: 1000          # 要生成的路径数量
  max_attempts: 4000        # 最大尝试次数
  require_unique_centers: true  # 是否要求每个路径的中心实体唯一

output:
  dir: kgqa_agent/data/conjunction/paths
  file_name: conjunction_paths_1k

sparql:
  config: kgqa_agent/configs/sparql.yaml

# 可选：过滤谓词（与 random_walk 相同的格式）
allow-predicates-json: datasets/cwq/cwq_predicates/filtered_cwq_white_list.json
allow-predicates-top-k: 0
```

### 2. 运行生成脚本

```bash
python kgqa_agent/scripts/conjunction_walk.py --config kgqa_agent/configs/random_walk/conjunction/conjunction_walk_example.yaml
```

### 3. 输出格式

生成的文件包含以下字段：

- `path_id`: 路径ID
- `name_path`: 路径的字符串表示，格式为 `entity1 -> relation1 -> center AND entity3 -> relation2 -> center`
- `topic_entities`: **主题实体列表** `[entity1, entity3]`，每个包含 `uri` 和 `name`
- `answers`: **答案实体列表** `[entity2]`，每个包含 `uri` 和 `name`
- `answer_candidates_count`: 答案实体的候选数量（用于验证是否符合要求）
- `path_nodes`: 详细的节点和关系信息
- `center_entity`: 中心实体的URI和名称（向后兼容字段，等同于answers[0]）
- `path1`: 第一条路径的实体和关系信息（entity1 -> relation1 -> center）
- `path2`: 第二条路径的实体和关系信息（entity3 -> relation2 -> center）
- `hop_count`: 每条路径的跳数（conjunction路径中为1）
- `path_length`: 路径长度（与hop_count相同）

## 配置参数说明

### walks 部分

- `num_starts`: 要生成的路径数量
- `max_attempts`: 生成路径时的最大尝试次数（建议设为 `num_starts * 4`）
- `require_unique_centers`: 如果为 `true`，每个路径必须有不同的中心实体（答案实体）
- `max_answer_candidates`: 答案实体的最大候选数量（默认3）。只接受候选数量 <= 此值的路径
- `start_pool_size`: 起点实体候选池的大小（默认16）。从候选池中随机选择起点实体
- `pred_sample_m`: 起点采样时，从谓词池中随机抽取的谓词数量（默认4）
- `per_pred_pool_k`: 每个谓词采样的实体数量（默认4）

### start-predicates-json

- 可选：指定一个JSON文件，包含用于起点实体采样的谓词列表（in-domain谓词）
- 格式可以是列表 `["relation1", "relation2", ...]` 或字典 `{"relation1": freq1, ...}`
- `start-predicates-top-k`: 如果指定，只使用前k个最频繁的谓词
- 如果提供了 `start-predicates-json`，起点实体将从这些谓词对应的实体中采样
- 如果未提供，则使用完全随机采样

### sparql 部分

- `config`: SPARQL配置文件路径（相对于repo根目录）
- 或直接指定 `endpoint`, `timeout_s`, `graph_uri`

### allow-predicates-json

- 可选：指定一个JSON文件，包含允许使用的谓词列表
- 格式可以是列表 `["relation1", "relation2", ...]` 或字典 `{"relation1": freq1, ...}`
- `allow-predicates-top-k`: 如果指定，只使用前k个最频繁的谓词

## 与 Random Walk 的区别

1. **路径结构**：
   - Random Walk: 生成线性的多跳路径 `entity1 -> relation1 -> entity2 -> relation2 -> entity3`
   - Conjunction Walk: 生成两个指向同一实体的路径

2. **生成方式**：
   - Random Walk: 从起点开始，逐步扩展路径
   - Conjunction Walk: 先找到中心实体，然后找到两个不同的入关系及其对应的实体

3. **用途**：
   - Random Walk: 适合生成多跳推理路径
   - Conjunction Walk: 适合生成需要同时满足多个条件的查询路径

## 注意事项

1. 生成的路径中，所有实体都必须有可读的名称（不是纯ID如 `m.xxx`）
2. 如果指定了 `require_unique_centers=true`，每个路径的中心实体都会不同
3. 路径的唯一性基于 `name_path` 字符串，相同的路径不会重复生成
4. 支持断点续传：如果输出文件已存在，会自动加载已有路径并继续生成


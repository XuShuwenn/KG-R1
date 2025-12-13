# Random Walk Configuration Reference

## 配置文件结构说明

### 必需参数

#### `seed` (顶层)
- **类型**: 整数
- **说明**: 随机种子，用于复现结果
- **示例**: `seed: 42`

#### `walks` 部分
- **`num_starts`**: 需要生成的路径数量
- **`max_hops`**: 最大跳数（路径长度）
- **`min_hops`**: 最小跳数（路径长度）
- **`max_attempts`**: 最大尝试次数（建议设为 num_starts 的 10-100 倍）
- **`start_pool_size`**: 起始节点池大小
- **`pred_sample_m`**: 采样的谓词数量
- **`per_pred_pool_k`**: 每个谓词的候选节点数量

#### `output` 部分
- **`dir`**: 输出目录路径（相对于项目根目录）
- **`file_name`** (可选): 输出文件基础名称（不含扩展名）

#### `sparql` 部分
- **`config`**: SPARQL 配置文件路径（相对于项目根目录）
  - 示例: `kgqa_agent/configs/sparql.yaml`

#### 谓词过滤
- **`allow-predicates-json`**: 允许的谓词白名单 JSON 文件路径
- **`allow-predicates-top-k`**: 使用前 K 个高频谓词（0 表示全部使用）
- **`start-predicates-json`**: 起始节点谓词白名单
- **`start-predicates-top-k`**: 起始谓词的 top-K 设置

### 可选参数

#### `walks` 部分（高级）
- **`neigh_pool_size`**: 邻居节点池大小（默认: 16）
- **`batch_size`**: 批处理大小，>0 时启用并行生成（默认: 0，单线程）
- **`include_names`**: 是否包含实体名称（默认: true）

#### `debug` 部分
- **`start_progress`**: 是否显示起始节点采样进度（默认: false）


## 配置示例

### 基础配置（单线程）
```yaml
seed: 42
walks:
  num_starts: 1000
  max_hops: 10
  min_hops: 5
  max_attempts: 10000
  start_pool_size: 512
  pred_sample_m: 32
  per_pred_pool_k: 32

output:
  dir: kgqa_agent/data/5-10-hop/test_1k/paths
  
sparql:
  config: kgqa_agent/configs/sparql.yaml

allow-predicates-json: datasets/cwq/cwq_predicates/filtered_cwq_white_list.json
allow-predicates-top-k: 0

start-predicates-json: datasets/cwq/cwq_predicates/filtered_cwq_white_list.json
start-predicates-top-k: 0
```

### 高级配置（并行批处理）
```yaml
seed: 42
walks:
  num_starts: 1000
  max_hops: 5
  min_hops: 3
  max_attempts: 20000
  start_pool_size: 4096
  neigh_pool_size: 20
  pred_sample_m: 256
  per_pred_pool_k: 32
  batch_size: 16  # 启用 16 个并行 worker

output:
  dir: kgqa_agent/data/3-5-hop/test_1k/paths
  file_name: st4096_bs16
  
sparql:
  config: kgqa_agent/configs/sparql.yaml

allow-predicates-json: datasets/cwq/cwq_predicates/filtered_cwq_white_list.json
allow-predicates-top-k: 0

start-predicates-json: datasets/cwq/cwq_predicates/filtered_cwq_white_list.json
start-predicates-top-k: 0

debug:
  start_progress: false
```

## 参数调优建议

### 路径数量相关
- `num_starts` 越大，生成时间越长
- `max_attempts` 建议设为 `num_starts` 的 10-100 倍，确保能生成足够的唯一路径

### 采样相关
- `start_pool_size` 越大，起始节点多样性越好，但采样时间越长
- `pred_sample_m` 控制每次采样多少个谓词，影响起始节点分布
- `per_pred_pool_k` 控制每个谓词采样多少候选节点

### 并行处理
- `batch_size > 0` 时启用并行批处理，显著提升大规模生成速度
- 建议 `batch_size` 设为 8-32 之间，取决于 CPU 核心数和 SPARQL 端点负载能力

### 路径长度
- 短路径 (3-5 hops): 适合快速测试和简单问答
- 长路径 (5-10 hops): 适合复杂推理和多跳问答
- `max_attempts` 需要相应调整，长路径生成难度更大

## 输出文件

运行配置后会生成两个 JSON 文件：

1. **`<file_name>.json`** 或 **`random_walk_paths.json`**
  - 完整路径详情。自 2025-11 起，默认保存每一步的“查询结果”明细（query_steps），便于复现与约束校验：
    - steps: 若存在 `query_steps`，则优先保存为每步查询记录；否则回退为原始节点-关系-节点路径
    - hop_count: 路径跳数（节点数-1），便于快速统计

2. **`<file_name>_names.json`** 或 **`random_walk_paths_names.json`**
   - 简化版本，仅包含 path_id、name_path 和 path_length

示例输出结构（含 query_steps）：
```json
[
  {
    "path_id": 1,
    "name_path": "Entity A -> relation -> Entity B -> relation -> Entity C",
    "path_length": 2,
    "hop_count": 2,
    "steps": [
      {
        "step_index": "1",
        "node_uri": "http://rdf.freebase.com/ns/m.xxx",
        "node_name": "Entity A",
        "relation_uri": "http://rdf.freebase.com/ns/people.person.place_of_birth",
        "relation_name": "people.person.place_of_birth",
        "type": "tail",
        "query_results": {
          "http://rdf.freebase.com/ns/people.person.place_of_birth": "people.person.place_of_birth",
          "http://rdf.freebase.com/ns/people.person.nationality": "people.person.nationality"
        }
      },
      {
        "step_index": "2",
        "node_uri": "http://rdf.freebase.com/ns/m.yyy",
        "node_name": "Entity B",
        "relation_uri": "http://rdf.freebase.com/ns/location.location.part_of",
        "relation_name": "location.location.part_of",
        "type": "tail",
        "query_results": {
          "http://rdf.freebase.com/ns/m.zzz": "Entity C"
        }
      }
    ]
  }
]
```

说明：
- 每步会记录两条查询（关系列表 + 实体列表），均标注为 `type: "tail"`。若将来加入入边采样，将相应标注为 `head`。
- 约束：每步查询命中数量不超过 `walks.max_index`；最终一步实体候选必须唯一（否则丢弃该路径）。

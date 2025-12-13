# 训练脚本检查清单

## ✅ 已检查项目

### 1. 数据文件
- ✅ `data_kg/cwq_kgqa_agent_format/train.parquet` (27,590 samples, 13 MB)
- ✅ `data_kg/cwq_kgqa_agent_format/val.parquet` (3,512 samples, 1.6 MB)
- ✅ 数据格式正确：包含 `prompt`, `extra_info`, `reward_model` 等必需字段
- ✅ 所有样本都有有效的 `topic_entity`（已过滤 None 值）

### 2. 训练脚本配置 (`train_debug_single_a100.sh`)

#### 基础配置
- ✅ 模型路径：`/mnt/usercache/huggingface/Qwen2.5-3B-Instruct`
- ✅ 数据路径：`data_kg/cwq_kgqa_agent_format/`
- ✅ GPU 配置：单卡 A100 (`CUDA_VISIBLE_DEVICES=0`)
- ✅ W&B 项目：`KG-R1-debug`

#### KG 配置
- ✅ SPARQL bridge 已启用：`+kg_config.use_sparql_bridge=true`
- ✅ SPARQL endpoint：`http://210.75.240.141:18890/sparql`
- ✅ `kg_top_k=3`
- ✅ `max_calls=10`
- ✅ `relation_filter_model="gpt-4o-mini"`

#### Prompt 配置
- ✅ Prompt augmentation 已关闭：`+data.prompt_augmentation.enable=false`
- ✅ 使用自定义 prompt（`verl/trainer/ppo/prompts.py`）

#### 训练参数
- ✅ 训练步数：20（调试模式）
- ✅ Batch size：16
- ✅ Max prompt length：3072
- ✅ Max response length：3072

### 3. 代码检查

#### `kg_r1/llm_agent/generation.py`
- ✅ SPARQL bridge 初始化逻辑正确
- ✅ 条件判断：仅在非 SPARQL bridge 模式下初始化 FastAPI 路由
- ✅ `KGQASparqlAdapter` 初始化传递了 `sparql_endpoint` 和 `kg_top_k`

#### `kg_r1/kgqa_bridge/sparql_adapter.py`
- ✅ 适配器实现完整
- ✅ 错误处理机制完善
- ✅ 会话状态管理正确

### 4. 潜在问题

#### ⚠️ 注意：`max_calls` 和 `relation_filter_model` 参数

**当前状态：**
- 训练脚本中配置了 `max_calls=10` 和 `relation_filter_model="gpt-4o-mini"`
- 但这些参数目前**未传递**给 `KGQASparqlAdapter`

**影响：**
- `max_calls`：主要用于 prompt 构建（在 `prompts.py` 中），不影响适配器行为
- `relation_filter_model`：如果需要在适配器中使用 LLM 过滤关系，需要添加支持

**建议：**
- 如果当前不需要 LLM 关系过滤，可以忽略
- 如果需要，需要修改 `KGQASparqlAdapter.__init__` 和 `generation.py` 中的初始化代码

### 5. 环境检查

#### 必需依赖
- ✅ `kgqa_agent` 包可用（用于 SPARQL 客户端）
- ✅ `SPARQLWrapper` 已安装
- ✅ `rank_bm25` 已安装（如果使用关系过滤）

#### 网络连接
- ⚠️ 需要确保可以访问 SPARQL endpoint：`http://210.75.240.141:18890/sparql`
- ⚠️ 如果使用 `relation_filter_model="gpt-4o-mini"`，需要确保可以访问对应的 LLM API

### 6. 运行前检查

#### 快速验证命令

```bash
# 1. 检查数据文件
ls -lh data_kg/cwq_kgqa_agent_format/*.parquet

# 2. 检查模型路径
ls -d /mnt/usercache/huggingface/Qwen2.5-3B-Instruct

# 3. 测试 SPARQL endpoint 连接（如果可能）
curl -X POST http://210.75.240.141:18890/sparql \
  -H "Content-Type: application/sparql-query" \
  -d "SELECT * WHERE { ?s ?p ?o } LIMIT 1"

# 4. 检查 W&B 配置
cat .env | grep WANDB_KEY
```

## 🚀 启动训练

```bash
cd /netcache/yuanchenhao/KG-R1
bash train_debug_single_a100.sh
```

## 📊 监控要点

1. **初始化阶段**：
   - 检查是否看到 `[KG_BRIDGE] Initialized kgqa_agent SPARQL adapter` 消息
   - 确认没有 FastAPI 路由初始化消息（SPARQL bridge 模式）

2. **训练阶段**：
   - 监控 SPARQL 查询错误率
   - 检查会话重置是否正常
   - 观察 W&B 日志

3. **错误处理**：
   - 如果出现 SPARQL 连接错误，检查 endpoint 地址
   - 如果出现数据加载错误，检查 parquet 文件格式
   - 如果出现内存错误，考虑减小 batch size

## 🔧 常见问题排查

### 问题 1: SPARQL 连接失败
**症状**：`SPARQL error: Connection refused` 或超时
**解决**：
- 检查 endpoint 地址是否正确
- 确认网络连接
- 检查防火墙设置

### 问题 2: 数据加载错误
**症状**：`KeyError` 或格式错误
**解决**：
- 验证 parquet 文件格式
- 检查 `prompt` 和 `extra_info` 字段是否存在

### 问题 3: 内存不足
**症状**：`CUDA out of memory`
**解决**：
- 减小 `train_batch_size`
- 减小 `max_prompt_length` 或 `max_response_length`
- 启用 `enable_activation_offload`（已启用）

### 问题 4: W&B 登录失败
**症状**：`wandb login failed`
**解决**：
- 检查 `.env` 文件中的 `WANDB_KEY`
- 手动运行 `wandb login`


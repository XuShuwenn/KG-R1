# 训练启动指南

## 📋 训练前检查清单

### 1. 环境准备

#### 1.1 检查数据文件
```bash
# 检查训练数据是否存在
ls -lh data_kg/cwq_kgqa_agent_format/train.parquet
ls -lh data_kg/cwq_kgqa_agent_format/val.parquet

# 预期输出：文件大小约 13MB (train) 和 1.6MB (val)
```

#### 1.2 检查模型路径
```bash
# 检查模型是否存在
ls -d /mnt/usercache/huggingface/Qwen2.5-3B-Instruct

# 预期输出：显示模型目录路径
```

#### 1.3 检查 W&B 配置
```bash
# 检查 .env 文件
cat .env | grep WANDB_KEY

# 如果不存在，创建 .env 文件
if [ ! -f .env ]; then
    echo "WANDB_KEY=your_wandb_api_key_here" > .env
    echo "⚠ 请编辑 .env 文件，填入你的 WANDB_KEY"
fi
```

#### 1.4 检查 SPARQL 端点连接（可选）
```bash
# 测试 SPARQL 端点是否可访问
curl -X POST "http://210.75.240.141:18890/sparql" \
  -H "Content-Type: application/sparql-query" \
  -d "SELECT * WHERE { ?s ?p ?o } LIMIT 1" 2>/dev/null | head -20

# 如果连接失败，检查网络或联系管理员
```

### 2. 代码检查

#### 2.1 验证关键文件
```bash
# 检查关键代码文件
test -f kg_r1/llm_agent/generation.py && echo "✓ generation.py 存在"
test -f kg_r1/kgqa_bridge/sparql_adapter.py && echo "✓ sparql_adapter.py 存在"
test -f verl/trainer/ppo/prompts.py && echo "✓ prompts.py 存在"
```

#### 2.2 运行单元测试（可选但推荐）
```bash
# 运行 SPARQL 适配器测试
PYTHONPATH=. pytest tests/test_sparql_adapter.py -v
```

## 🚀 启动训练

### 步骤 1: 进入项目目录
```bash
cd /netcache/yuanchenhao/KG-R1
```

### 步骤 2: 检查 GPU 可用性
```bash
# 检查 GPU
nvidia-smi

# 确认 CUDA_VISIBLE_DEVICES 设置（脚本中已设置为 0）
echo $CUDA_VISIBLE_DEVICES  # 应该为空或 0
```

### 步骤 3: 运行训练脚本
```bash
# 使用默认配置
bash train_debug_single_a100.sh

# 或使用自定义配置
export DATA_DIR=data_kg
export BASE_MODEL=/mnt/usercache/huggingface/Qwen2.5-3B-Instruct
export WAND_PROJECT=KG-R1-debug
export EXPERIMENT_NAME=my-test-run
bash train_debug_single_a100.sh
```

## 📊 训练监控

### 1. 实时日志
训练开始后，你会看到：
- `[KG_BRIDGE] Initialized kgqa_agent SPARQL adapter @ ...` - SPARQL 适配器初始化
- `[INFO] Using BASE_MODEL=...` - 模型路径信息
- W&B 日志链接（如果配置正确）

### 2. W&B 监控
- 打开 W&B 项目：`KG-R1-debug`
- 查看实验：`cwq-single-a100-debug`（或你设置的 `EXPERIMENT_NAME`）
- 监控指标：
  - `train/loss` - 训练损失
  - `train/reward` - 奖励值
  - `train/valid_action_rate` - 有效动作率
  - `train/search_count` - 搜索次数统计

### 3. 检查点保存
- 位置：`verl_checkpoints/${EXPERIMENT_NAME}/`
- 频率：每 50 步保存一次（`trainer.save_freq=50`）

## ⚠️ 常见问题排查

### 问题 1: SPARQL 端点连接失败
**症状**：日志中出现 `ConnectionError` 或 `Timeout`

**解决方案**：
```bash
# 1. 检查网络连接
ping 210.75.240.141

# 2. 检查防火墙/代理设置
# 3. 联系管理员确认端点状态
# 4. 如果端点不可用，可以临时禁用 SPARQL bridge（不推荐）
```

### 问题 2: 模型加载失败
**症状**：`FileNotFoundError` 或 `OSError: Can't load tokenizer`

**解决方案**：
```bash
# 1. 检查模型路径
ls -la /mnt/usercache/huggingface/Qwen2.5-3B-Instruct

# 2. 检查模型文件完整性
# 3. 如果路径不同，修改脚本中的 BASE_MODEL
```

### 问题 3: 内存不足 (OOM)
**症状**：`CUDA out of memory`

**解决方案**：
```bash
# 1. 减小 batch size
# 在脚本中修改：
# data.train_batch_size=8  # 从 16 减小到 8
# data.val_batch_size=8

# 2. 减小序列长度
# data.max_prompt_length=2048  # 从 3072 减小
# data.max_response_length=2048

# 3. 减小 rollout 数量
# +actor_rollout_ref.rollout.grpo_rollout_n=4  # 从 8 减小到 4
```

### 问题 4: W&B 登录失败
**症状**：`wandb: ERROR Not logged in`

**解决方案**：
```bash
# 1. 检查 .env 文件
cat .env | grep WANDB_KEY

# 2. 手动登录
wandb login

# 3. 或设置环境变量
export WANDB_API_KEY=your_key_here
```

### 问题 5: 数据文件不存在
**症状**：`FileNotFoundError: data_kg/cwq_kgqa_agent_format/train.parquet`

**解决方案**：
```bash
# 1. 检查数据文件
ls -lh data_kg/cwq_kgqa_agent_format/*.parquet

# 2. 如果不存在，运行数据转换脚本
python scripts/data_process_kg/convert_cwq_splits.py

# 3. 或使用其他数据路径
export DATA_DIR=/path/to/your/data
```

## 📝 训练参数说明

### 关键参数（当前配置）

| 参数 | 值 | 说明 |
|------|-----|------|
| `trainer.total_training_steps` | 20 | 调试模式，仅训练 20 步 |
| `trainer.total_epochs` | 1 | 最多 1 个 epoch |
| `data.train_batch_size` | 16 | 训练批次大小 |
| `+kg_config.max_calls` | 10 | 每个样本最多 10 次查询 |
| `+kg_config.use_sparql_bridge` | true | 启用 SPARQL 直连模式 |
| `+kg_config.sparql_endpoint` | `http://210.75.240.141:18890/sparql` | SPARQL 端点地址 |

### 修改建议

**如果要进行完整训练**：
```bash
# 修改脚本中的以下参数
trainer.total_training_steps=2000  # 从 20 增加到 2000
trainer.total_epochs=5  # 从 1 增加到 5
data.train_batch_size=32  # 从 16 增加到 32（如果内存允许）
```

**如果遇到性能问题**：
```bash
# 减小批次大小
data.train_batch_size=8
data.val_batch_size=8

# 减小序列长度
data.max_prompt_length=2048
data.max_response_length=2048
```

## ✅ 训练成功标志

训练正常运行时，你应该看到：

1. **初始化阶段**：
   ```
   [KG_BRIDGE] Initialized kgqa_agent SPARQL adapter @ http://210.75.240.141:18890/sparql (max_calls=10)
   [INFO] Using BASE_MODEL=/mnt/usercache/huggingface/Qwen2.5-3B-Instruct
   ```

2. **训练阶段**：
   ```
   Step 1/20: loss=..., reward=..., valid_action_rate=...
   Step 2/20: loss=..., reward=..., valid_action_rate=...
   ```

3. **W&B 日志**：
   - 在 W&B 项目中看到实时更新的指标
   - 实验名称正确显示

4. **检查点保存**：
   ```
   Saving checkpoint to verl_checkpoints/cwq-single-a100-debug/step_50
   ```

## 🔗 相关文档

- [训练参数详解](./train_debug_single_a100_params.md)
- [训练检查清单](./training_checklist.md)
- [单卡 A100 训练指南](./training_debug_single_a100.md)
- [多轮对话限制说明](./multiturn_limits.md)


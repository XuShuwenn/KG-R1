# kgqa_agent 数据格式转换指南

本文档说明如何将 `kgqa_agent` 格式的数据（如 `cwq_train.json`）转换为 VERL 训练所需的 parquet 格式，并使用自定义的 prompt 构建逻辑。

## 概述

- **目标**: 将 `kgqa_agent/src/eval/datasets/cwq_train.json` 转换为 VERL 可用的 parquet 文件
- **Prompt 来源**: 使用 `verl/trainer/ppo/prompts.py` 中的 `build_search_prompt` 函数
- **Prompt Augmentation**: 已关闭，使用自定义 prompt 逻辑

## 数据格式说明

### 输入格式 (kgqa_agent)

```json
{
  "id": "cwq_0",
  "question": "What state is home to the university...",
  "answers": [
    "Washington, D.C.",
    "Washington D.C.",
    ...
  ],
  "topic_entity": {
    "m.03d0l76": "George Washington Colonials men's basketball"
  }
}
```

### 输出格式 (VERL parquet)

每条记录包含：
- `prompt`: `[{"role": "user", "content": "..."}]` - 使用 `build_search_prompt` 生成
- `reward_model`: 包含 `ground_truth` 信息
- `extra_info`: 包含 `sample_id`, `question`, `initial_entities` 等

## 使用步骤

### 1. 运行转换脚本

```bash
cd /netcache/yuanchenhao/KG-R1

# 基本用法：将整个文件作为 train split
python scripts/data_process_kg/cwq_kgqa_agent_format.py \
    --input kgqa_agent/src/eval/datasets/cwq_train.json \
    --output_dir data_kg/cwq_kgqa_agent_format \
    --split train \
    --max_calls 10

# 自动分割为 train/val/test (90%/5%/5%)
python scripts/data_process_kg/cwq_kgqa_agent_format.py \
    --input kgqa_agent/src/eval/datasets/cwq_train.json \
    --output_dir data_kg/cwq_kgqa_agent_format \
    --split auto \
    --train_ratio 0.9 \
    --val_ratio 0.05 \
    --test_ratio 0.05 \
    --max_calls 10
```

### 2. 更新训练脚本

修改 `train_debug_single_a100.sh` 中的数据路径：

```bash
data.train_files=$DATA_DIR/cwq_kgqa_agent_format/train.parquet \
data.val_files=$DATA_DIR/cwq_kgqa_agent_format/test.parquet \
```

### 3. 验证数据

转换完成后，检查输出：

```bash
# 查看生成的 parquet 文件
ls -lh data_kg/cwq_kgqa_agent_format/

# 使用 Python 快速检查
python -c "
import pandas as pd
df = pd.read_parquet('data_kg/cwq_kgqa_agent_format/train.parquet')
print(f'Total samples: {len(df)}')
print(f'Columns: {df.columns.tolist()}')
print(f'\nFirst sample:')
sample = df.iloc[0]
print(f'  - sample_id: {sample[\"extra_info\"][\"sample_id\"]}')
print(f'  - question: {sample[\"extra_info\"][\"question\"][:80]}...')
print(f'  - initial_entities: {sample[\"extra_info\"][\"initial_entities\"]}')
print(f'  - prompt length: {len(sample[\"prompt\"][0][\"content\"])} chars')
"
```

## 脚本参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--input` | 输入 JSON 文件路径 | `kgqa_agent/src/eval/datasets/cwq_train.json` |
| `--output_dir` | 输出 parquet 文件目录 | `data_kg/cwq_kgqa_agent_format` |
| `--split` | 数据分割方式 | `train` |
| | - `train`/`val`/`test`: 作为指定 split | |
| | - `auto`: 自动分割为 train/val/test | |
| `--max_calls` | Prompt 中的最大 KG 查询次数 | `10` |
| `--train_ratio` | Train split 比例（auto 模式） | `0.9` |
| `--val_ratio` | Val split 比例（auto 模式） | `0.05` |
| `--test_ratio` | Test split 比例（auto 模式） | `0.05` |

## Prompt 构建逻辑

转换脚本使用 `verl/trainer/ppo/prompts.py` 中的 `build_search_prompt` 函数：

```python
prompt_text = build_search_prompt(
    question_or_sample=question,
    max_calls=max_calls,
    topic_entities=topic_entities,
)
```

该函数会：
1. 提取问题文本和初始实体
2. 使用 `SEARCH_PROMPT_TEMPLATE` 构建完整 prompt
3. 包含 KG 查询指令和初始实体信息

## 训练配置

### 关闭 Prompt Augmentation

训练脚本中已设置：

```bash
+data.prompt_augmentation.enable=false
```

这确保使用 parquet 中预构建的 prompt，而不是动态添加 hint。

### 完整训练命令示例

```bash
bash train_debug_single_a100.sh
```

确保脚本中的数据路径指向转换后的 parquet 文件。

## 注意事项

1. **Prompt 一致性**: 转换时使用的 `max_calls` 应与训练配置中的 `kg_config.max_calls` 一致
2. **数据量**: `cwq_train.json` 包含约 48 万条样本，转换可能需要几分钟
3. **内存**: 确保有足够内存加载整个 JSON 文件
4. **验证**: 转换后建议检查几个样本的 prompt 格式是否正确

## 故障排除

### 问题: 导入错误

如果遇到 `ModuleNotFoundError: No module named 'ray'`，脚本已使用直接导入方式避免此问题。

### 问题: 数据格式不匹配

检查：
1. JSON 文件是否包含必需的字段（`id`, `question`, `answers`, `topic_entity`）
2. `topic_entity` 是否为字典格式 `{entity_id: entity_name}`

### 问题: Prompt 格式错误

验证：
1. `build_search_prompt` 函数是否正常工作
2. 生成的 prompt 是否包含 `Question:` 和初始实体信息

## 相关文件

- 转换脚本: `scripts/data_process_kg/cwq_kgqa_agent_format.py`
- Prompt 定义: `verl/trainer/ppo/prompts.py`
- 训练脚本: `train_debug_single_a100.sh`
- 原始数据: `kgqa_agent/src/eval/datasets/cwq_train.json`


# Prompt Augmentation 参数详解

本文档详细说明 `train_debug_single_a100.sh` 中 `data.prompt_augmentation.*` 相关参数的作用机制和代码调用位置。

## 参数概览

| 参数 | 默认值 | 作用 |
|------|--------|------|
| `enable` | `false` | 是否启用 prompt augmentation |
| `guideline_level` | `None` | 指引模板的详细程度级别 |
| `hint_steps` | `0` | hint 应用的训练步数上限（0=无限制） |

---

## 1. `prompt_augmentation.enable`

### 作用
控制是否在训练过程中向 prompt 中插入 KG 查询指引。当启用时，系统会在原始 prompt 的 "Question:" 之前插入一个 `[Hint]:` 块，包含如何正确使用 `<kg-query>` 标签和查询函数的详细说明。

### 代码调用链

#### 初始化阶段
```python
# 位置: verl/utils/dataset/rl_dataset.py:103-106
augmentation_config = config.get("prompt_augmentation", None)
if augmentation_config:
    from verl.trainer.ppo.prompt_augmentation_kg import create_prompt_augmentor_from_config
    self.prompt_augmentor = create_prompt_augmentor_from_config(config)
```

#### 应用阶段
```python
# 位置: verl/utils/dataset/rl_dataset.py:221, 254
# 在 __getitem__ 方法中，对每个样本的 prompt 应用 augmentation
raw_prompt = self.processor.apply_chat_template(messages, ...)
raw_prompt = self.build_augmented_prompt(raw_prompt)  # ← 这里应用 augmentation

# build_augmented_prompt 内部调用:
# 位置: verl/utils/dataset/rl_dataset.py:133-145
def build_augmented_prompt(self, base_prompt: str) -> str:
    if self.prompt_augmentor:
        return self.prompt_augmentor.augment_prompt(base_prompt)
    return base_prompt
```

#### 实际 augmentation 逻辑
```python
# 位置: verl/trainer/ppo/prompt_augmentation_kg.py:551-612
def augment_prompt(self, base_prompt: str) -> str:
    if not self.should_apply_hints():  # 检查是否应该应用（考虑 hint_steps）
        return base_prompt
    
    instruction_hint = self.get_instruction_hint()  # 根据 guideline_level 获取指引文本
    # ... 在 "Question:" 之前插入 [Hint]: {instruction_hint}
    return augmented_prompt
```

---

## 2. `prompt_augmentation.guideline_level`

### 作用
选择不同详细程度的指引模板。不同级别提供不同长度的 KG 查询函数说明、示例和注意事项。

### 可选值

| 值 | 说明 | 适用场景 |
|---|------|---------|
| `"concise"` | 简洁版指引（当前脚本中实际未使用此值） | 快速训练，减少 prompt 长度 |
| `"detailed"` | 默认详细指引，最多 7 次 kg-query | 标准训练 |
| `"detailed_flat"` | 扁平格式，最多 5 次 kg-query | 常用配置 |
| `"detailed_flat_turn7"` | 多轮推理，最多 7 次 kg-query，支持 7 轮 | 复杂多跳问题 |
| `"detailed_flat_multiTQ"` | MultiTQ 时序推理专用 | 时序知识图谱任务 |
| `"minimal"` | 最小化指引，仅函数说明 | 评估阶段，减少干扰 |
| `"vanilla"` | 无 KG 指引，直接回答问题 | 基线对比 |
| `"cot"` | Chain-of-Thought，鼓励推理步骤 | 推理能力评估 |

### 代码调用链

#### 模板选择
```python
# 位置: verl/trainer/ppo/prompt_augmentation_kg.py:509-538
def get_instruction_hint(self) -> str:
    if self.guideline_level == "extensive":
        return EXTENSIVE_GUIDELINE
    elif self.guideline_level == "detailed":
        return DETAILED_GUIDELINE
    elif self.guideline_level == "detailed_flat":
        return DETAILED_GUIDELINE_FLAT  # ← 脚本中使用 "concise" 但实际未定义，可能回退
    # ... 其他级别
```

#### 模板定义
```python
# 位置: verl/trainer/ppo/prompt_augmentation_kg.py:264-299
DETAILED_GUIDELINE_FLAT = """
You are allowed to make up to 5 kg-queries. 
If you encounter a KG-related error, read the error message carefully and correct your query.

Use exactly these query functions:
- get_relations_out(entity): Returns outgoing relations...
- get_relations_in(entity): Returns incoming relations...
- get_entities_out(entity, relation): Returns entities...
- get_entities_in(entity, relation): Returns entities...

IMPORTANT:
- Always begin with think after getting question or information.
- Always prefer information retrieved from the KG over your internal knowledge.
...
"""
```

**注意**: 脚本中设置 `guideline_level=concise`，但在 `prompt_augmentation_kg.py` 中未定义 `CONCISE_GUIDELINE` 常量，实际会回退到空字符串，导致 augmentation 不生效。建议改为 `"detailed_flat"` 或其他已定义的级别。

---

## 3. `prompt_augmentation.hint_steps`

### 作用
控制 hint 应用的训练步数上限。当 `current_step >= hint_steps` 时，停止向 prompt 中插入指引，让模型逐渐独立推理。这是一种**课程学习（Curriculum Learning）**策略：前期提供强指引，后期逐渐撤除。

### 代码调用链

#### 步数更新
```python
# 位置: verl/trainer/ppo/ray_trainer_kg.py:1777-1778
# 在每个训练 batch 开始前更新当前步数
if hasattr(self.train_dataset, 'set_current_step'):
    self.train_dataset.set_current_step(self.global_steps)

# 位置: verl/utils/dataset/rl_dataset.py:128-131
def set_current_step(self, step: int):
    """Update the current training step for hint scheduling."""
    if self.prompt_augmentor:
        self.prompt_augmentor.set_current_step(step)
```

#### 判断是否应用 hint
```python
# 位置: verl/trainer/ppo/prompt_augmentation_kg.py:540-549
def should_apply_hints(self) -> bool:
    if not self.enable:
        return False
    
    # 关键判断：如果当前步数 >= hint_steps，停止应用 hint
    if self.hint_steps > 0 and self.current_step >= self.hint_steps:
        return False  # ← 超过步数上限，不再插入 hint
        
    return True

# 位置: verl/trainer/ppo/prompt_augmentation_kg.py:551-565
def augment_prompt(self, base_prompt: str) -> str:
    if not self.should_apply_hints():  # ← 检查 hint_steps
        return base_prompt  # 直接返回原始 prompt，不插入 hint
    # ... 否则插入 hint
```

### 工作流程示例

假设 `hint_steps=200`：

1. **Step 0-199**: 每个 prompt 都会插入 `[Hint]: {指引文本}`
2. **Step 200+**: 不再插入 hint，模型使用原始 prompt 进行推理

这种设计帮助模型：
- **前期**：通过强指引学习正确的 KG 查询格式
- **后期**：逐渐独立，避免过度依赖 hint

---

## 完整调用流程图

```
训练脚本 (train_debug_single_a100.sh)
  ↓
  data.prompt_augmentation.enable=true
  data.prompt_augmentation.guideline_level=concise
  data.prompt_augmentation.hint_steps=200
  ↓
┌─────────────────────────────────────────────────┐
│ 1. 初始化阶段                                   │
│ verl/utils/dataset/rl_dataset.py:103-106       │
│   → create_prompt_augmentor_from_config()      │
│   → PromptAugmentor(enable=True,                │
│                    guideline_level="concise",   │
│                    hint_steps=200)              │
└─────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────┐
│ 2. 训练循环 (每个 batch)                         │
│ verl/trainer/ppo/ray_trainer_kg.py:1777-1778   │
│   → train_dataset.set_current_step(global_steps)│
│   → prompt_augmentor.set_current_step(step)     │
└─────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────┐
│ 3. 数据加载 (每个样本)                           │
│ verl/utils/dataset/rl_dataset.py:207-254       │
│   → __getitem__(item)                           │
│   → build_augmented_prompt(raw_prompt)          │
│   → prompt_augmentor.augment_prompt(base_prompt)│
└─────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────┐
│ 4. Augmentation 逻辑                            │
│ verl/trainer/ppo/prompt_augmentation_kg.py      │
│   → should_apply_hints()                        │
│     • 检查 enable                                │
│     • 检查 current_step < hint_steps            │
│   → get_instruction_hint()                      │
│     • 根据 guideline_level 返回模板             │
│   → 在 "Question:" 前插入 [Hint]: {hint}       │
└─────────────────────────────────────────────────┘
  ↓
返回增强后的 prompt 给模型
```

---

## 实际效果示例

### 原始 Prompt（未启用 augmentation）
```
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.

Question: What is the capital of France?
```

### 启用 augmentation 后（guideline_level="detailed_flat", hint_steps=200, step=50）
```
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.

[Hint]: You are allowed to make up to 5 kg-queries. 
If you encounter a KG-related error, read the error message carefully and correct your query.

Use exactly these query functions:
- get_relations_out(entity): Returns outgoing relations where the entity is the subject/head (entity → relation → ?).
- get_relations_in(entity): Returns incoming relations where the entity is the object/tail (? → relation → entity).
- get_entities_out(entity, relation): Returns entities connected from the given entity by the specified relation (entity → relation → ?).
- get_entities_in(entity, relation): Returns entities from which the given entity is connected by the specified relation (? → relation → entity).

IMPORTANT:
- Always begin with think after getting question or information.
- Always prefer information retrieved from the KG over your internal knowledge.
- Use KG data as your primary source if relevant information is available.

Examples of entities:
- Named entities: "Barack Obama", "Taylor Swift", "Albert Einstein", "New York City", "France", "Mount Everest", "Google", "United Nations", "Harvard University"
- Entity IDs: "m.02mjmr", "m.09c7w0"

Examples of relations:
- "people.person.nationality"
- "people.person.spouse_s"
- "location.location.contains"
- "location.country.capital"
...

KG Query Examples:
- get_relations_out("Bahamas")
- get_relations_in("Barack Obama")
- get_entities_out("Bahamas", "location.location.contains")
- get_entities_in("Barack Obama", "people.person.nationality")

Question: What is the capital of France?
```

### Step 200+ 后（hint_steps=200）
```
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.

Question: What is the capital of France?
```
（不再插入 hint，模型独立推理）

---

## 注意事项

1. **`guideline_level="concise"` 问题**: 脚本中使用了 `concise`，但代码中未定义对应的模板，实际会回退到空字符串，导致 augmentation 不生效。建议改为 `"detailed_flat"` 或其他已定义的级别。

2. **`hint_steps=0`**: 表示无限制，hint 会一直应用直到训练结束。

3. **性能影响**: 插入 hint 会增加 prompt 长度，可能影响：
   - Token 预算（需要调整 `max_prompt_length`）
   - 训练速度（更长的序列需要更多计算）

4. **课程学习效果**: `hint_steps` 的合理设置需要根据模型收敛情况调整。过早撤除（步数过小）可能导致模型未充分学习；过晚撤除（步数过大）可能导致过度依赖 hint。

---

## 相关文件清单

| 文件路径 | 关键函数/类 | 说明 |
|---------|------------|------|
| `verl/trainer/ppo/prompt_augmentation_kg.py` | `PromptAugmentor` | 核心 augmentation 逻辑 |
| `verl/utils/dataset/rl_dataset.py` | `RLHFDataset.__init__`<br>`RLHFDataset.build_augmented_prompt`<br>`RLHFDataset.set_current_step` | 数据集集成 |
| `verl/trainer/ppo/ray_trainer_kg.py` | `RayPPOTrainer.fit` | 训练循环中更新步数 |

---

## 参考

- 完整参数说明: `train_debug_single_a100_params.md`
- 训练调试文档: `docs/training_debug_single_a100.md`


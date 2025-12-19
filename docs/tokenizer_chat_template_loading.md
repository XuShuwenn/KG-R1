# Tokenizer 和 Chat Template 加载流程

本文档详细说明训练端如何加载 tokenizer 以及如何使用 chat template。

## 1. Tokenizer 加载流程

### 1.1 主入口 (`verl/trainer/main_ppo.py`)

在 `TaskRunner.run()` 方法中：

```python
# 第65行：从HDFS或本地路径下载模型
local_path = copy_to_local(config.actor_rollout_ref.model.path, use_shm=config.actor_rollout_ref.model.get("use_shm", False))

# 第68-72行：加载 tokenizer 和 processor
from verl.utils import hf_processor, hf_tokenizer

trust_remote_code = config.data.get("trust_remote_code", False)
tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)
```

### 1.2 Tokenizer 初始化函数 (`verl/utils/tokenizer.py`)

`hf_tokenizer()` 函数（第48-73行）：

```python
def hf_tokenizer(name_or_path, correct_pad_token=True, correct_gemma2=True, **kwargs):
    """创建 HuggingFace tokenizer，正确处理 eos 和 pad tokens"""
    from transformers import AutoTokenizer
    
    # 特殊处理 Gemma2 模型
    if correct_gemma2 and isinstance(name_or_path, str) and "gemma-2-2b-it" in name_or_path:
        kwargs["eos_token"] = "<end_of_turn>"
        kwargs["eos_token_id"] = 107
    
    # 使用 AutoTokenizer 从模型路径加载
    tokenizer = AutoTokenizer.from_pretrained(name_or_path, **kwargs)
    
    # 自动设置 pad_token（如果为 None）
    if correct_pad_token:
        set_pad_token_id(tokenizer)
    
    return tokenizer
```

**关键点：**
- 使用 `AutoTokenizer.from_pretrained()` 从模型路径加载
- 自动从模型配置中读取 chat template（如果模型支持）
- 自动处理 pad_token（LLaMA 使用 unk_token，其他模型使用 eos_token）
- 支持 `trust_remote_code` 参数以加载自定义 tokenizer

### 1.3 Tokenizer 传递给 Trainer

在 `main_ppo.py` 第170-184行，tokenizer 被传递给 trainer：

```python
trainer = trainer_cls(
    config=config,
    tokenizer=tokenizer,  # 传递给 trainer
    processor=processor,
    # ... 其他参数
)
```

Trainer 将 tokenizer 存储在 `self.tokenizer` 中，供后续使用。

## 2. Chat Template 使用流程

### 2.1 Dataset 中的 Chat Template 应用 (`verl/utils/dataset/rl_dataset.py`)

在 `RLHFDataset.__getitem__()` 方法中（第207-326行）：

#### 步骤1：构建 Messages 列表

```python
# 第212行：从数据行构建 messages（list of dicts）
messages = self._build_messages(row_dict)
```

`_build_messages()` 方法将数据行转换为标准的 messages 格式：
```python
[
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    # ...
]
```

#### 步骤2：应用 Chat Template

```python
# 第251行：使用 tokenizer.apply_chat_template 将 messages 转换为字符串
raw_prompt = self.tokenizer.apply_chat_template(
    messages, 
    add_generation_prompt=True,  # 添加生成提示（如 <|im_start|>assistant）
    tokenize=False  # 只返回字符串，不进行 tokenization
)
```

**关键点：**
- `apply_chat_template()` 使用模型自带的 chat template（如 Qwen 的 ChatML 格式）
- `add_generation_prompt=True` 会在末尾添加 assistant 角色的开始标记
- `tokenize=False` 返回字符串格式，而不是 token IDs

#### 步骤3：Tokenization

```python
# 第256行：将字符串 prompt 转换为 token IDs
model_inputs = self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
input_ids = model_inputs.pop("input_ids")
attention_mask = model_inputs.pop("attention_mask")
```

#### 步骤4：保存原始 Messages（可选）

```python
# 第305-306行：如果 return_raw_chat=True，保存原始 messages
if self.return_raw_chat:
    row_dict["raw_prompt"] = messages  # 保存为 list of dicts
```

**配置方式：**
在配置文件中设置 `data.return_raw_chat: True` 来启用此功能。

### 2.2 Trainer 中的 Chat Template 使用

#### 在 Rollout 过程中

在 `ray_trainer_kg.py` 的 `fit()` 方法中（第1825-1831行）：

```python
# 从 batch 中提取 raw_prompt（如果存在）
non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
if "raw_prompt" in batch.non_tensor_batch:
    non_tensor_batch_keys_to_pop.append("raw_prompt")
```

`raw_prompt` 会被传递给 rollout worker，用于生成。

#### 在保存 Rollout 轨迹时

在 `ray_trainer_kg.py` 的 `fit()` 方法中（第2151行）：

```python
# 当前实现：直接从 token IDs 解码（会丢失 ChatML 格式）
inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
```

**问题：**
- `skip_special_tokens=True` 会移除 ChatML 特殊 token（如 `<|im_start|>`, `<|im_end|>`）
- 即使 `skip_special_tokens=False`，如果原始 `raw_prompt` 不在 batch 中，也无法恢复正确的 ChatML 格式

**解决方案：**
应该从 `batch.non_tensor_batch["raw_prompt"]` 中获取原始 messages，然后使用 `tokenizer.apply_chat_template()` 重新格式化：

```python
# 尝试从 non_tensor_batch 获取原始 messages
original_messages = None
if hasattr(batch, 'non_tensor_batch') and 'raw_prompt' in batch.non_tensor_batch:
    original_messages = batch.non_tensor_batch['raw_prompt']

formatted_inputs = []
if original_messages is not None:
    for msg_list in original_messages:
        # 确保 messages 是 list of dicts 格式
        if isinstance(msg_list, np.ndarray):
            msg_list = msg_list.tolist()
        if isinstance(msg_list, list) and all(isinstance(m, dict) for m in msg_list):
            # 使用 apply_chat_template 格式化
            formatted_inputs.append(
                self.tokenizer.apply_chat_template(
                    msg_list, 
                    add_generation_prompt=True, 
                    tokenize=False
                )
            )
        else:
            # Fallback：直接解码 token IDs
            formatted_inputs.append(
                self.tokenizer.batch_decode(
                    batch.batch["prompts"], 
                    skip_special_tokens=False
                )[len(formatted_inputs)]
            )
else:
    # Fallback：直接解码 token IDs（保留特殊 token）
    formatted_inputs = self.tokenizer.batch_decode(
        batch.batch["prompts"], 
        skip_special_tokens=False
    )

inputs = [inp.strip() for inp in formatted_inputs]
```

## 3. Chat Template 格式说明

### 3.1 Qwen 模型的 ChatML 格式

Qwen 模型使用 ChatML 格式，示例：

```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
What is the capital of France?<|im_end|>
<|im_start|>assistant
```

### 3.2 如何查看模型的 Chat Template

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
print(tokenizer.chat_template)
```

或者查看模型的 `tokenizer_config.json` 文件。

## 4. 配置选项

### 4.1 Dataset 配置 (`ppo_trainer.yaml`)

```yaml
data:
  # 是否返回原始 messages（不应用 chat template）
  return_raw_chat: False
  
  # 是否返回完整 prompt（应用 chat template 后的字符串）
  return_full_prompt: False
  
  # 是否信任远程代码（用于加载自定义 tokenizer）
  trust_remote_code: False
```

### 4.2 启用 `return_raw_chat` 的好处

- 可以在保存 rollout 轨迹时重新应用 chat template，确保格式正确
- 可以灵活切换不同的 chat template（如果需要）
- 保留原始消息结构，便于调试和分析

## 5. 常见问题

### Q1: 为什么保存的 JSONL 文件中 `input` 字段不是 ChatML 格式？

**原因：**
- 当前代码使用 `batch_decode(prompts, skip_special_tokens=True)` 解码
- 这会移除所有特殊 token，包括 ChatML 标记

**解决：**
- 启用 `data.return_raw_chat: True`
- 在保存时从 `batch.non_tensor_batch["raw_prompt"]` 获取原始 messages
- 使用 `tokenizer.apply_chat_template()` 重新格式化

### Q2: 如何确保保存的格式与模型训练时使用的格式一致？

**方法：**
- 始终使用 `tokenizer.apply_chat_template()` 来格式化 messages
- 不要直接解码 token IDs（除非确实需要）
- 保存时使用与训练时相同的 `add_generation_prompt` 参数

### Q3: 不同模型的 chat template 是否兼容？

**答案：**
- 不同模型的 chat template 格式可能不同（如 Qwen 使用 ChatML，LLaMA 使用其他格式）
- 应该使用模型自带的 tokenizer 和 chat template
- 不要混用不同模型的 tokenizer 和 chat template

## 6. 总结

1. **Tokenizer 加载：** 通过 `hf_tokenizer()` 从模型路径加载，自动处理 pad_token
2. **Chat Template 应用：** 在 dataset 中使用 `tokenizer.apply_chat_template()` 将 messages 转换为字符串
3. **原始 Messages 保存：** 通过 `return_raw_chat=True` 配置保存原始 messages
4. **保存时格式化：** 应该从 `raw_prompt` 重新应用 chat template，而不是直接解码 token IDs

这样可以确保保存的 rollout 轨迹使用正确的 ChatML 格式，与模型训练时使用的格式一致。


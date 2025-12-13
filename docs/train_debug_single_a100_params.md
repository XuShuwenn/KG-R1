# `train_debug_single_a100.sh` 超参数说明（含来源）

> **来源分类说明**
>
> - **KG-R1**：指 VERL/KG-R1 框架原生配置项（含 Hydra/Lightning/VERL 相关字段）。
> - **kgqa_agent**：指为了复用 `kgqa_agent` 交互式查询逻辑而新增/沿用的字段（主要在 `kg_config.*` 下）。

---

## 1. 脚本级环境变量

| 参数 | 说明 | 来源 |
| --- | --- | --- |
| `CUDA_VISIBLE_DEVICES=0` | 锁定单张 GPU（A100）。 | KG-R1 |
| `DATA_DIR` | 数据根目录，缺省为 `data_kg`。 | KG-R1 |
| `BASE_MODEL` | HF 权重路径，默认 `/mnt/usercache/huggingface/Qwen2.5-3B-Instruct`。 | KG-R1 |
| `WAND_PROJECT` | wandb Project 名；默认 `KG-R1-debug`。 | KG-R1 |
| `EXPERIMENT_NAME` | 实验 ID，用于 wandb run 名以及 checkpoint 目录。 | KG-R1 |

---

## 2. `trainer.*`（训练控制）

| 参数 | 说明 | 来源 |
| --- | --- | --- |
| `+trainer.mode=kg-search` | 启用 KG 交互模式，替换标准搜索。 | KG-R1 |
| `trainer.project_name / experiment_name` | wandb 日志元数据。 | KG-R1 |
| `trainer.n_gpus_per_node=1` | 单机使用 1 GPU。 | KG-R1 |
| `trainer.nnodes=1` | 仅 1 台机器。 | KG-R1 |
| `trainer.total_training_steps=20` | 训练总步数（此脚本做 quick sanity check）。 | KG-R1 |
| `trainer.total_epochs=1` | epoch 上限为 1。 | KG-R1 |
| `trainer.default_local_dir=verl_checkpoints/$EXPERIMENT_NAME` | checkpoint 输出目录。 | KG-R1 |
| `trainer.save_freq=50`, `trainer.test_freq=50` | 每 50 步存/测一次（若尚未达到 50，不会触发）。 | KG-R1 |
| `trainer.logger='["wandb"]'` | 仅使用 wandb 日志后端。 | KG-R1 |
| `+trainer.use_ref_model=true` | 启用 reference model（用于 KL 约束）。 | KG-R1 |

---

## 3. `data.*`

| 参数 | 说明 | 来源 |
| --- | --- | --- |
| `data.train_files`, `data.val_files` | 训练/验证 parquet 路径（CWQ）。 | KG-R1 |
| `data.train_batch_size=16`, `data.val_batch_size=16` | 全局 batch size。 | KG-R1 |
| `data.max_prompt_length=3072` | prompt 最大长度。 | KG-R1 |
| `data.max_response_length=3072` | model response 最大长度（本脚本放宽以观察长回答）。 | KG-R1 |
| `data.max_obs_length=1024` | `<information>` 片段最大长度。 | KG-R1 |
| `data.shuffle=true` | 启用训练集 shuffle。 | KG-R1 |
| `data.trust_remote_code=true` | HF 模型允许远程代码。 | KG-R1 |
| `+data.prompt_augmentation.enable=true` | 开启题面 augmentation。在训练过程中向 prompt 中插入 KG 查询指引，帮助模型学习正确的 `<kg-query>` 格式。**调用位置**: `verl/utils/dataset/rl_dataset.py:103-106` 创建 `PromptAugmentor`，`rl_dataset.py:221,254` 在 `__getitem__` 中调用 `build_augmented_prompt()` 应用 augmentation。 | KG-R1 |
| `+data.prompt_augmentation.guideline_level=concise` | 使用简洁指引模板。可选值包括：`"concise"`（简洁版，当前脚本未使用，实际使用的是 `"detailed_flat"` 等）、`"detailed"`、`"detailed_flat"`、`"detailed_flat_turn7"`、`"detailed_flat_multiTQ"`、`"minimal"`、`"vanilla"`、`"cot"` 等。不同级别提供不同详细程度的 KG 查询函数说明和示例。**调用位置**: `verl/trainer/ppo/prompt_augmentation_kg.py:509-538` 的 `get_instruction_hint()` 方法根据 `guideline_level` 返回对应的指引模板字符串（如 `DETAILED_GUIDELINE_FLAT`）。 | KG-R1 |
| `+data.prompt_augmentation.hint_steps=200` | hint 在 200 步内逐渐减弱。当训练步数 `>= hint_steps` 时，停止向 prompt 中插入指引 hint，让模型逐渐独立推理。**调用位置**: `verl/trainer/ppo/prompt_augmentation_kg.py:540-549` 的 `should_apply_hints()` 检查 `current_step >= hint_steps`；`ray_trainer_kg.py:1777-1778` 在每个训练 batch 前调用 `train_dataset.set_current_step(global_steps)` 更新步数；`prompt_augmentation_kg.py:546` 判断是否应用 hint。 | KG-R1 |

---

## 4. `algorithm.*`

| 参数 | 说明 | 来源 |
| --- | --- | --- |
| `algorithm.adv_estimator=grpo` | 使用 GRPO 优势估计。 | KG-R1 |
| `algorithm.norm_adv_by_std_in_grpo=true` | 对优势按标准差归一。 | KG-R1 |
| `+algorithm.enable_multiturn_advantage=true` | 多轮对话累积优势。 | KG-R1 |
| `algorithm.use_kl_in_reward=false` | 奖励阶段不加 KL 惩罚。 | KG-R1 |
| `algorithm.kl_ctrl.kl_coef=0` | KL controller 系数设 0（仅 Actor 端 KL Loss 生效）。 | KG-R1 |
| `algorithm.kg_token_masking.enable=false` | 不对 KG token 施加特殊 KL mask。 | KG-R1 |

---

## 5. `actor_rollout_ref.model.*`

| 参数 | 说明 | 来源 |
| --- | --- | --- |
| `model.path=$BASE_MODEL` | 指定基础模型。 | KG-R1 |
| `enable_gradient_checkpointing=true` | 节省显存。 | KG-R1 |
| `enable_activation_offload=true` | 把部分激活迁移到 CPU。 | KG-R1 |
| `use_remove_padding=true` | 移除 padding 提升效率。 | KG-R1 |

---

## 6. `actor_rollout_ref.actor.*`

| 参数 | 说明 | 来源 |
| --- | --- | --- |
| `use_dynamic_bsz=true` | 动态微批大小，自适应序列长度。 | KG-R1 |
| `ppo_mini_batch_size=16` | PPO/GRPO 优化的 mini-batch 大小。 | KG-R1 |
| `ppo_max_token_len_per_gpu=16384` | 控制单 GPU token budget。 | KG-R1 |
| `entropy_coeff=0.0` | 关闭额外 entropy regularization。 | KG-R1 |
| `kl_loss_coef=0.01` | KL loss 系数。 | KG-R1 |
| `kl_loss_type=k3` | 使用 k3 范式计算 KL。 | KG-R1 |
| `state_masking=true` | 只对动作 token backprop。 | KG-R1 |
| `grad_clip=1.0` | 全局梯度裁剪。 | KG-R1 |

---

## 7. `actor_rollout_ref.rollout.*`（vLLM 配置）

| 参数 | 说明 | 来源 |
| --- | --- | --- |
| `name=vllm` | 采用 vLLM 作为推理后端。 | KG-R1 |
| `dtype=bfloat16` | 推理时使用 bfloat16。 | KG-R1 |
| `temperature=1.0` | 采样温度。 | KG-R1 |
| `n=1` | 每个输入生成 1 个样本。 | KG-R1 |
| `+grpo_rollout_n=8` | GRPO 展开倍数（扩充样本数）。 | KG-R1 |
| `max_num_batched_tokens=12288` | vLLM 批处理 token 上限。 | KG-R1 |
| `max_num_seqs=64` | vLLM 并行序列数限制。 | KG-R1 |
| `free_cache_engine=false` | 关闭空闲缓存引擎。 | KG-R1 |
| `enable_chunked_prefill=true` | 分块 prefill，缓解长序列显存压力。 | KG-R1 |
| `rollout.log_prob_max_token_len_per_gpu=80000` | logprob 计算 token 上限。 | KG-R1 |
| `ref.log_prob_max_token_len_per_gpu=80000` | reference model 同上。 | KG-R1 |

### Rollout 搜索段

| 参数 | 说明 | 来源 |
| --- | --- | --- |
| `rollout.search.enable=true` | 允许 `<kg-query>`。 | KG-R1 |
| `enable_during_training/validation=true` | 训练/验证均可查询。 | KG-R1 |
| `search_url="http://127.0.0.1:8001/retrieve"` | FastAPI RoG KG server 地址。 | KG-R1 |
| `max_turns=6` | 每条样本最多进行 6 次对话轮。 | KG-R1 |
| `topk=3` | KG server 单次返回 top3。 | KG-R1 |
| `timeout=3` | HTTP 请求超时秒数。 | KG-R1 |

---

## 8. `reward_model.*`

| 参数 | 说明 | 来源 |
| --- | --- | --- |
| `reward_model.enable=false` | 不启用独立 RM worker。 | KG-R1 |
| `reward_model.reward_manager=kg_format_multiturn` | 使用多轮 KG Reward manager。 | KG-R1 |
| `reward_kwargs.turn_kg_query_validity=0.5` | 每轮查询合法性权重。 | KG-R1 |
| `reward_kwargs.turn_is_answer_score=0.5` | 回答出现在该轮时的奖励。 | KG-R1 |
| `reward_kwargs.turn_format_score=0.5` | 标签格式奖励。 | KG-R1 |
| `reward_kwargs.global_exact_match=0.5` | 全局 EM 奖励。 | KG-R1 |
| `reward_kwargs.global_retrieval_quality=0.5` | 检索质量奖励。 | KG-R1 |

---

## 9. `kg_config.*`（KG/桥接配置）

| 参数 | 说明 | 来源 |
| --- | --- | --- |
| `enable_kg_during_training=true` | 训练过程中启用 KG 接口。 | KG-R1 |
| `server_url="http://127.0.0.1:8001/retrieve"` | FastAPI fallback server。 | KG-R1 |
| `use_sparql_bridge=true` | 打开 `kgqa_agent` 直连 Virtuoso 的适配器。 | kgqa_agent |
| `sparql_endpoint="http://210.75.240.141:18890/sparql"` | Virtuoso SPARQL 端点地址。 | kgqa_agent |
| `kg_top_k=10` | 每次 `get_relations` 返回 top3（遵循 kgqa_agent client 的接口）。 | kgqa_agent |
| `max_calls=10` | 每个样本最多 10 次 `<kg-query>`，与 kgqa_agent 交互逻辑一致。 | kgqa_agent |
| `relation_filter_model="gpt-4o-mini"` | relation rerank/filter 所用 LLM（kgqa_agent 中同款）。 | kgqa_agent |

---

## 10. 其他说明

- `set -euo pipefail`：Bash 严格模式，遇到错误立即退出。来源：KG-R1 脚本惯例。
- `echo "[INFO] ..."`：实用日志，帮助确认环境变量读取正确。来源：KG-R1。

如需扩展/修改其它参数，可在此文档基础上补充说明，便于追踪每个超参数的含义与来源。#


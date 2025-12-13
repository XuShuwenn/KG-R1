# 单卡 A100 训练与调试指南

本文档描述如何在一张 A100 GPU（24GB/40GB/80GB 均可）上使用 `Qwen2.5-3B-Instruct` 模型运行 KG-R1 训练调试流程，并详细解释 `train_debug_single_a100.sh` 中的每个参数，便于按需修改。

---

## 1. 环境准备

1. **安装依赖**
   ```bash
   git clone <repo_url> KG-R1
   cd KG-R1
   bash run.sh          # 自动安装 python 依赖、准备 wandb 登录
   ```
   > `run.sh` 会尝试读取 `.env` 中的 `WANDB_KEY` 自动登录 wandb。若还未配置，可复制模版：
   > ```bash
   > cp .env.example .env
   > echo "WANDB_KEY=你的apikey" >> .env
   > ```

2. **准备模型与数据**
   - 模型：`/mnt/usercache/huggingface/Qwen2.5-3B-Instruct`
   - 数据：`scripts/setup_data_kg.sh` 可一键下载 CWQ/WebQSP 处理好的 parquet 数据，默认放在 `data_kg`。

3. **启动（可选）KG 检索服务**
   - RoG 子图 FastAPI：
     ```bash
     python kg_r1/search/server.py --base_data_path data_kg
     ```
   - Virtuoso SPARQL（若要走 `kgqa_agent` 直连模式）：确保本地 8890 端口可访问。

---

## 2. 快速启动训练

脚本位于仓库根目录：`train_debug_single_a100.sh`

```bash
bash train_debug_single_a100.sh
```

默认参数：
- 使用单卡（`CUDA_VISIBLE_DEVICES=0`）
- 模型路径：`/mnt/usercache/huggingface/Qwen2.5-3B-Instruct`
- 数据：`data_kg/cwq_search_augmented_initial_entities`
- 日志：W&B 项目 `KG-R1-debug`；本地 checkpoint 写到 `verl_checkpoints/<experiment_name>`

如需修改，可在命令前导出环境变量：
```bash
export DATA_DIR=/path/to/data
export BASE_MODEL=/path/to/model
export EXPERIMENT_NAME=my-debug-run
bash train_debug_single_a100.sh
```

---

## 3. 参数详解（与脚本顺序相同）

| 分类 | 参数 | 说明 |
| --- | --- | --- |
| **全局** | `CUDA_VISIBLE_DEVICES=0` | 强制使用第 0 张 GPU。若多卡可改成 `0,1` 并调整 `trainer.n_gpus_per_node`。 |
| | `DATA_DIR` | 数据根目录，默认 `data_kg`。 |
| | `BASE_MODEL` | HuggingFace 权重路径，默认 `/mnt/usercache/huggingface/Qwen2.5-3B-Instruct`。 |
| | `WAND_PROJECT` | wandb 项目名，默认 `KG-R1-debug`。 |
| | `EXPERIMENT_NAME` | 实验标识，影响 wandb run 名称以及 checkpoint 目录。 |
| **Trainer** | `trainer.mode=kg-search` | 打开 KG 交互训练模式。 |
| | `trainer.project_name/experiment_name` | wandb 日志配置。 |
| | `trainer.n_gpus_per_node=1`、`trainer.nnodes=1` | 指定资源拓扑，单机单卡。 |
| | `trainer.total_training_steps=200`、`trainer.total_epochs=5` | 迭代上限（两者满足任一达到就结束）。 |
| | `trainer.default_local_dir` | checkpoint 输出目录。 |
| | `trainer.save_freq/test_freq=50` | 每 50 步存 ckpt / 做一次验证。 |
| | `trainer.logger=["wandb"]` | 仅推送到 wandb。 |
| **数据** | `data.train_files/val_files` | 训练/验证 parquet。可替换为 WebQSP 数据。 |
| | `data.train_batch_size=32`、`data.val_batch_size=64` | 全局 batch size（GRPO 展开后实际会更大）。 |
| | `data.max_prompt_length=3072`、`data.max_response_length=192`、`data.max_obs_length=384` | 控制 prompt/answer/信息块最大 token。 |
| | `data.shuffle=true` | 训练集随机打乱。 |
| | `data.trust_remote_code=true` | 允许加载远程 tokenizer/config（Qwen 需要）。 |
| | `data.prompt_augmentation.*` | 控制提示注入——此处启用简洁指引、200 步后逐渐减少。 |
| **算法** | `algorithm.adv_estimator=grpo` | 使用 GRPO 估计器。 |
| | `algorithm.norm_adv_by_std_in_grpo=true` | 对优势归一化，稳定训练。 |
| | `algorithm.enable_multiturn_advantage=true` | 多轮优势累积。 |
| | `algorithm.use_kl_in_reward=false` / `algorithm.kl_ctrl.kl_coef=0` | 禁止在奖励阶段做 KL 惩罚（只保留 Actor 侧 KL Loss）。 |
| **Actor / 模型** | `actor_rollout_ref.model.path` | 基础模型路径。 |
| | `enable_gradient_checkpointing/enable_activation_offload/use_remove_padding` | 控制显存优化策略。 |
| | `actor.use_dynamic_bsz=true` | 允许根据序列长度自动调节 micro-batch。 |
| | `actor.ppo_mini_batch_size=32` | 每次优化使用的样本数。 |
| | `actor.ppo_max_token_len_per_gpu=14000` | 限制单 GPU token 预算，防 OOM。 |
| | `actor.entropy_coeff/kl_loss_*` | 控制策略正则化。默认仅保留 KL loss（k3 范式）。 |
| | `actor.state_masking=true` | 仅针对动作 token 反传（忽略 observation token）。 |
| | `actor.grad_clip=1.0` | 全局梯度裁剪。 |
| **Rollout/VLLM** | `rollout.name=vllm`、`dtype=bfloat16` | 使用 vLLM 执行 rollout，数据类型 bfloat16。 |
| | `rollout.temperature=1.0`、`rollout.n=1` | 单样本生成的采样温度/次数。 |
| | `+rollout.grpo_rollout_n=8` | GRPO 面向策略梯度的展开次数（最终 batch ×8）。 |
| | `rollout.search.*` | 控制 `<kg-query>` 的 KG Server 调用：启用训练/验证流程、URL、最大回合数、topk、超时等。 |
| | `rollout.max_num_batched_tokens=12288`、`max_num_seqs=64` | vLLM 批处理限制，适配单卡。 |
| | `rollout.enable_chunked_prefill=true` | 减少长序列显存压力。 |
| | `rollout.log_prob_max_token_len_per_gpu=80000` 等 | 控制 logprob 计算时的 token 上限。 |
| **Reward** | `reward_model.enable=false` | 关闭独立 RM，改由 `kg_format_multiturn` 打分器。 |
| | `reward_kwargs.*` | 结构/答案/检索等 reward 组成部分的权重。 |
| **KG 设置** | `kg_config.enable_kg_during_training=true` | PPL 训练阶段启用 KG。 |
| | `kg_config.server_url` | FastAPI RoG 服务地址（用于 `<kg-query>` fallback）。 |
| | `kg_config.use_sparql_bridge=true` | 开启 `kgqa_agent` SPARQL 直连模式。 |
| | `kg_config.sparql_endpoint="http://127.0.0.1:8890/sparql"` | Virtuoso 端点。 |
| | `kg_config.kg_top_k=3`、`kg_config.max_calls=6` | 每轮最多返回 3 条关系，最多 6 次 `<kg-query>`。 |
| | `kg_config.relation_filter_model="gpt-4o-mini"` | relation 过滤时调用的 LLM（直接复用 `kgqa_agent` 策略）。 |

---

## 4. 调试建议

1. **检查 GPU 使用**
   ```bash
   watch -n 5 nvidia-smi
   ```
   若显存超过 24GB，可下调 `data.max_prompt_length` 或 `rollout.max_num_batched_tokens`。

2. **验证数据路径**
   - `ls $DATA_DIR/cwq_search_augmented_initial_entities` 应包含 `train.parquet/test.parquet`.

3. **Wandb 日志**
   - 首次需 `wandb login`，`run.sh` 已自动尝试。若登录失败可手动执行。
   - `WANDB_MODE=offline` 可离线调试。

4. **Virtuoso/Sparql 直连**
   - 若不需要 SPARQL Bridge，可将脚本中的 `+kg_config.use_sparql_bridge=true` 删除或设为 false。
   - 当端点不可达时 `<kg-query>` 会 fallback 到 RoG server，注意日志中的 `[KG_BRIDGE]` 提示。

5. **加速/稳定技巧**
   - 训练不稳定可将 `rollout.temperature` 降至 0.7。
   - 想快速 sanity check，可将 `trainer.total_training_steps` 改为 20 或更小。

---

## 5. 常见问题（FAQ）

| 问题 | 解决方案 |
| --- | --- |
| `wandb: ERROR Failed to set settings...` | 确认 `.env` 中的 `WANDB_KEY` 正确；或 `wandb login` 手动登录后再运行脚本。 |
| `FileNotFoundError: ... train.parquet` | 数据目录不完整，重新运行 `scripts/setup_data_kg.sh` 或手动指定 `DATA_DIR`。 |
| `CUDA out of memory` | 减小 `data.train_batch_size`、`actor_rollout_ref.rollout.max_num_batched_tokens` 或 `rollout.grpo_rollout_n`。 |
| 无法连接 SPARQL | 检查 Virtuoso 8890 端口是否监听；或者关闭 `use_sparql_bridge` 改走 RoG server。 |

---

## 6. 监控与调试指标

- **Wandb 面板**：关注 `actor/loss`, `critic/value_error`, `reward/avg`, `kg/query_success_rate`。
- **日志**：`verl_checkpoints/<exp>/actor` 会记录生成样本，可用 `tail -f` 观察。
- **Eval**：脚本默认 `trainer.test_freq=50`，可以在 wandb 中查看验证集指标。

祝训练顺利！如需进一步定制，可基于本脚本继续扩展（例如改为多卡、替换数据集、接入不同 reward manager 等）。#


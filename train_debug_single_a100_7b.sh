#!/bin/bash

set -euo pipefail

# -----------------------------
# Basic experiment setup
# -----------------------------
export CUDA_VISIBLE_DEVICES=0
export DATA_DIR="${DATA_DIR:-data_kg}"
export BASE_MODEL="${BASE_MODEL:-/mnt/usercache/yuanchenhao/LLaMA-Factory/saves/synth_7.6k_sft_traj_roll_10k_4o-mini_cutoff4096_filtered2_sample_3k/lr1e-5_ep2}"
export WAND_PROJECT="${WAND_PROJECT:-KG-R1-debug}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-cwq-single-a100-debug_7b}"

export RAY_TMPDIR=/mnt/usercache/yuanchenhao/KG-R1/ray_tmp

echo "[INFO] Using BASE_MODEL=${BASE_MODEL}"
echo "[INFO] Using DATA_DIR=${DATA_DIR}"
echo "[INFO] Writing checkpoints to verl_checkpoints/${EXPERIMENT_NAME}"

# -----------------------------
# Launch PPO training (single A100)
# -----------------------------
python -m verl.trainer.main_ppo \
    +trainer.mode=kg-search \
    trainer.project_name=$WAND_PROJECT \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.total_training_steps=100 \
    trainer.total_epochs=1 \
    trainer.default_local_dir=verl_checkpoints/$EXPERIMENT_NAME \
    trainer.save_freq=50 \
    trainer.test_freq=50 \
    trainer.logger='["wandb"]' \
    data.train_files=$DATA_DIR/cwq_kgqa_agent_format/train.parquet \
    data.val_files=$DATA_DIR/cwq_kgqa_agent_format/val.parquet \
    data.train_batch_size=8 \
    data.val_batch_size=2 \
    data.max_prompt_length=12288 \
    data.max_response_length=1024 \
    data.max_obs_length=1024 \
    data.shuffle=true \
    data.trust_remote_code=true \
    +data.prompt_augmentation.enable=false \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=true \
    +algorithm.enable_multiturn_advantage=true \
    algorithm.use_kl_in_reward=false \
    algorithm.kl_ctrl.kl_coef=0 \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.enable_activation_offload=true \
    actor_rollout_ref.model.use_remove_padding=true \
    actor_rollout_ref.actor.use_dynamic_bsz=true \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=14000 \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=k3 \
    actor_rollout_ref.actor.state_masking=true \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.dtype=bfloat16 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.n=1 \
    +actor_rollout_ref.rollout.grpo_rollout_n=4 \
    '+actor_rollout_ref.rollout.stop=["</kg-query>","</answer>"]' \
    actor_rollout_ref.rollout.search.enable=true \
    actor_rollout_ref.rollout.search.enable_during_training=true \
    actor_rollout_ref.rollout.search.enable_during_validation=true \
    actor_rollout_ref.rollout.search.timeout=15 \
    actor_rollout_ref.rollout.max_num_batched_tokens=16384 \
    actor_rollout_ref.rollout.max_num_seqs=12 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.70 \
    actor_rollout_ref.rollout.free_cache_engine=false \
    actor_rollout_ref.rollout.enable_chunked_prefill=true \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=80000 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=80000 \
    reward_model.enable=false \
    reward_model.reward_manager=kg_format_multiturn \
    +reward_model.reward_kwargs.turn_kg_query_validity=0.0 \
    +reward_model.reward_kwargs.turn_is_answer_score=0.0 \
    +reward_model.reward_kwargs.turn_format_score=0.0 \
    +reward_model.reward_kwargs.global_exact_match=1.0 \
    +reward_model.reward_kwargs.global_retrieval_quality=0.0 \
    algorithm.kg_token_masking.enable=false \
    +trainer.use_ref_model=true \
    trainer.val_before_train=false \
    +kg_config.enable_kg_during_training=true \
    +kg_config.use_sparql_bridge=true \
    +kg_config.sparql_endpoint="http://210.75.240.141:18890/sparql" \
    +kg_config.kg_top_k=10 \
    +kg_config.max_calls=10 \
    +kg_config.relation_filter_model="gpt-4o-mini"
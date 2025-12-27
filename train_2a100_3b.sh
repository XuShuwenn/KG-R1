#!/bin/bash

set -euo pipefail
# -----------------------------
# Basic experiment setup
# -----------------------------
export CUDA_VISIBLE_DEVICES=0,1
export DATA_DIR="${DATA_DIR:-data_kg}"
export BASE_MODEL="${BASE_MODEL:-/mnt/usercache/yuanchenhao/LLaMA-Factory/saves/qwen2.5_3b_it_sft_traj_roll_10k_4o-mini_cutoff4096_filter_max_calls_sample_3k/lr1e-5_ep2}"
export WAND_PROJECT="${WAND_PROJECT:-KG-R1-debug}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-cwq-2a100-test-3b-roll-3k}"

export RAY_TMPDIR=/mnt/usercache/yuanchenhao/KG-R1/ray_tmp

# Generate timestamp for unique rollout directory
export ROLLOUT_TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
export ROLLOUT_DIR="${EXPERIMENT_NAME}_${ROLLOUT_TIMESTAMP}"

# Log file in logs/ directory
export LOG_FILE="logs/${EXPERIMENT_NAME}_${ROLLOUT_TIMESTAMP}.log"

echo "[INFO] Using BASE_MODEL=${BASE_MODEL}"
echo "[INFO] Using DATA_DIR=${DATA_DIR}"
echo "[INFO] Writing checkpoints to verl_checkpoints/${EXPERIMENT_NAME}"
echo "[INFO] Writing rollout trajectories to rollout_trajectories/${ROLLOUT_DIR}"
echo "[INFO] Writing logs to ${LOG_FILE}"

# -----------------------------
# Launch PPO training (2 A100, 3B model)
# -----------------------------
python -m verl.trainer.main_ppo \
    +trainer.mode=kg-search \
    trainer.project_name=$WAND_PROJECT \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.total_training_steps=200 \
    trainer.total_epochs=1 \
    trainer.default_local_dir=verl_checkpoints/$EXPERIMENT_NAME \
    trainer.resume_mode=disable \
    trainer.rollout_data_dir=rollout_trajectories/$ROLLOUT_DIR \
    trainer.save_freq=40 \
    trainer.test_freq=200 \
    trainer.logger='["wandb"]' \
    data.train_files=$DATA_DIR/cwq_kgqa_agent_format/train.parquet \
    data.val_files=$DATA_DIR/cwq_kgqa_agent_format/val.parquet \
    data.train_batch_size=32 \
    data.val_batch_size=32 \
    data.max_prompt_length=12240 \
    data.max_response_length=1536 \
    data.max_obs_length=1536 \
    data.shuffle=true \
    data.trust_remote_code=true \
    data.return_raw_chat=true \
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
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16000 \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=k3 \
    actor_rollout_ref.actor.state_masking=true \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.dtype=bfloat16 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.0 \
    actor_rollout_ref.rollout.n=1 \
    +actor_rollout_ref.rollout.grpo_rollout_n=8 \
    '+actor_rollout_ref.rollout.stop=["</kg-query>","</answer>"]' \
    actor_rollout_ref.rollout.search.enable=true \
    actor_rollout_ref.rollout.search.enable_during_training=true \
    actor_rollout_ref.rollout.search.enable_during_validation=true \
    actor_rollout_ref.rollout.search.max_turns=10 \
    actor_rollout_ref.rollout.search.topk=10 \
    actor_rollout_ref.rollout.search.timeout=15 \
    actor_rollout_ref.rollout.max_num_batched_tokens=16384 \
    actor_rollout_ref.rollout.max_num_seqs=12 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.60 \
    actor_rollout_ref.rollout.free_cache_engine=true \
    actor_rollout_ref.rollout.enable_chunked_prefill=true \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=80000 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=80000 \
    reward_model.enable=false \
    reward_model.reward_manager=kg_format_multiturn \
    +reward_model.reward_kwargs.turn_kg_query_validity=0.0 \
    +reward_model.reward_kwargs.turn_is_answer_score=0.0 \
    +reward_model.reward_kwargs.turn_format_score=0.5 \
    +reward_model.reward_kwargs.global_exact_match=1.0 \
    +reward_model.reward_kwargs.global_retrieval_quality=0.5 \
    +reward_model.reward_kwargs.verbose=true \
    algorithm.kg_token_masking.enable=false \
    +trainer.use_ref_model=true \
    +kg_config.enable_kg_during_training=true \
    +kg_config.use_sparql_bridge=true \
    +kg_config.sparql_endpoint="http://210.75.240.141:18890/sparql" \
    +kg_config.kgqa_thread_pool_size=64 \
    +kg_config.kg_top_k=10 \
    +kg_config.max_calls=10 \
    +kg_config.relation_filter_model="deepseek-chat" \
    trainer.val_before_train=false \
    2>&1 | tee "${LOG_FILE}"

#!/bin/bash

# WebQuestionsSP Knowledge Graph Training with PPO
# This script trains a model on WebQSP dataset using KG-based retrieval with RoG subgraphs

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# Update DATA_DIR to point to the new RoG data structure
# Assuming the output of download_kg.py is in a directory like 'data_kg_rog'
# And that script saves webqsp data under 'data_kg_rog/webqsp/'
export DATA_DIR_ROOT='data_kg_rog' # Base directory for RoG data
export DATASET_NAME='webqsp' # Specify dataset being used (webqsp or cwq)
export DATA_DIR="$DATA_DIR_ROOT/$DATASET_NAME" # Path to specific dataset QA files

WAND_PROJECT='Search-R1-KG-RoG' # Updated project name for RoG

# Model configuration
export BASE_MODEL='Qwen/Qwen2.5-3B'
export EXPERIMENT_NAME=${DATASET_NAME}-kg-rog-r1-ppo-qwen2.5-3b-em
# export BASE_MODEL='Qwen/Qwen2.5-3B-Instruct'
# export EXPERIMENT_NAME=webqsp-kg-r1-ppo-qwen2.5-3b-it-em
# export BASE_MODEL='Qwen/Qwen2.5-7B'
# export EXPERIMENT_NAME=webqsp-kg-r1-ppo-qwen2.5-7b-em

# Set VLLM backend
export VLLM_ATTENTION_BACKEND=XFORMERS

# Calculate max_prompt_length based on configuration
# max_prompt_length = max_start_length + max_response_length * (max_turns - 1) + max_obs_length * max_turns

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \\\
    data.train_files=$DATA_DIR/train.json \\\
    data.val_files=$DATA_DIR/test.json \\\
    data.train_data_num=null \\\
    data.val_data_num=null \\\
    data.train_batch_size=512 \\\
    data.val_batch_size=256 \\\
    data.max_prompt_length=4096 \\\
    data.max_response_length=500 \\\
    data.max_start_length=2048 \\\
    data.max_obs_length=500 \\\
    data.shuffle_train_dataloader=True \\\
    data.input_format_type=rog_json \\ # Specify new input format type for RoG JSON
    algorithm.adv_estimator=gae \\\
    actor_rollout_ref.model.path=$BASE_MODEL \\\
    actor_rollout_ref.actor.optim.lr=1e-6 \\\
    actor_rollout_ref.model.enable_gradient_checkpointing=true \\\
    actor_rollout_ref.model.use_remove_padding=True \\\
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.285 \\\
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \\\
    actor_rollout_ref.actor.ppo_micro_batch_size=64 \\\
    actor_rollout_ref.actor.fsdp_config.param_offload=true \\\
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \\\
    actor_rollout_ref.rollout.log_prob_micro_batch_size=64 \\\
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \\\
    actor_rollout_ref.rollout.name=vllm \\\
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \\\
    actor_rollout_ref.ref.log_prob_micro_batch_size=64 \\\
    actor_rollout_ref.ref.fsdp_config.param_offload=True \\\
    actor_rollout_ref.rollout.n_agent=1 \\\
    actor_rollout_ref.rollout.temperature=1 \\\
    actor_rollout_ref.rollout.top_p=1.0 \\\
    actor_rollout_ref.actor.state_masking=true \\\
    critic.optim.lr=1e-5 \\\
    critic.model.use_remove_padding=True \\\
    critic.optim.lr_warmup_steps_ratio=0.015 \\\
    critic.model.path=$BASE_MODEL \\\
    critic.model.enable_gradient_checkpointing=true \\\
    critic.ppo_micro_batch_size=8 \\\
    critic.model.fsdp_config.param_offload=true \\\
    critic.model.fsdp_config.grad_offload=true \\\
    critic.model.fsdp_config.optimizer_offload=true \\\
    algorithm.kl_ctrl.kl_coef=0.001 \\\
    algorithm.no_think_rl=false \\\
    trainer.critic_warmup=0 \\\
    trainer.logger=['wandb'] \\\
    +trainer.val_only=false \\\
    +trainer.val_before_train=true \\\
    trainer.default_hdfs_dir=null \\\
    trainer.n_gpus_per_node=8 \\\
    trainer.nnodes=1 \\\
    trainer.save_freq=100 \\\
    trainer.test_freq=100 \\\
    trainer.project_name=$WAND_PROJECT \\\
    trainer.experiment_name=$EXPERIMENT_NAME \\\
    trainer.total_epochs=15 \\\
    trainer.total_training_steps=500 \\\
    trainer.default_hdfs_dir=null \\\
    trainer.default_local_dir=verl_checkpoints/$EXPERIMENT_NAME \\\
    reward_model.enable=false \\\
    reward_model.reward_manager=format \\\
    +reward_model.reward_kwargs.structure_format_score=0.1 \\\
    +reward_model.reward_kwargs.final_format_score=0.1 \\\
    +reward_model.reward_kwargs.retrieval_score=0.8 \\\
    +reward_model.reward_kwargs.format_score=0 \\\
    actor_rollout_ref.rollout.search.enable=true \\\
    actor_rollout_ref.rollout.search.enable_during_training=true \\\
    actor_rollout_ref.rollout.search.enable_during_validation=true \\\
    actor_rollout_ref.rollout.search.search_url="http://127.0.0.1:8000/retrieve" \\\
    actor_rollout_ref.rollout.search.max_turns=3 \\\
    actor_rollout_ref.rollout.search.topk=1 \\ # topk is 1 as we retrieve the whole subgraph as one doc
    actor_rollout_ref.rollout.search.search_request_extra_kwargs.dataset_name=$DATASET_NAME \\ # Pass dataset_name to server
    2>&1 | tee $EXPERIMENT_NAME.log

#!/bin/bash
set -e

echo "🎯 MultiTQ Integration Example"
echo "============================="

# Show how MultiTQ fits with the existing KG-R1 structure
echo "📁 File structure after setup:"
echo ""
echo "KG-R1/"
echo "├── scripts/data_multitq_kg/           # Processing scripts (this directory)"
echo "│   ├── download_multitq.py"
echo "│   ├── process_multitq.py"
echo "│   ├── multitq_search_augmented_initial_entities.py"
echo "│   └── setup_multitq.sh"
echo "├── data_multitq_kg/                   # Raw MultiTQ dataset"
echo "│   ├── MultiTQ/"
echo "│   └── MultiQA/"
echo "├── data_kg/multitq/                   # Processed KG-R1 format"
echo "│   ├── train_simple.json"
echo "│   ├── dev_simple.json"
echo "│   ├── test_simple.json"
echo "│   ├── entities.txt"
echo "│   ├── entities_text.txt"
echo "│   └── relations.txt"
echo "└── data_kg/multitq_search_augmented_initial_entities/  # Training data"
echo "    ├── train.parquet"
echo "    ├── dev.parquet"
echo "    ├── test.parquet"
echo "    └── DATASET_SUMMARY.md"
echo ""

echo "🔄 Usage comparison with other datasets:"
echo ""
echo "# Existing datasets:"
echo "data.train_files=\"data_kg/cwq_search_augmented_initial_entities/train.parquet\""
echo "data.val_files=\"data_kg/cwq_search_augmented_initial_entities/test.parquet\""
echo ""
echo "# MultiTQ dataset:"
echo "data.train_files=\"data_kg/multitq_search_augmented_initial_entities/train.parquet\""
echo "data.val_files=\"data_kg/multitq_search_augmented_initial_entities/test.parquet\""
echo ""

echo "🚀 Quick setup command:"
echo "cd scripts/data_multitq_kg && bash setup_multitq.sh"
echo ""

echo "🎯 Training script example:"
cat << 'EOF'

# Create a MultiTQ training script similar to existing ones:
# ./train_grpo_kg_qwen_3b_multitq_temporal.sh

#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export DATA_DIR="data_kg"

python -m verl.trainer.main_ppo \
    mode=kg-search \
    trainer.experiment_name=multitq-temporal-reasoning \
    trainer.rollout.n_sample_reuse=1 \
    trainer.rollout.tensor_parallel_size=2 \
    trainer.rollout.num_gpu_per_node=4 \
    data.train_files="data_kg/multitq_search_augmented_initial_entities/train.parquet" \
    data.val_files="data_kg/multitq_search_augmented_initial_entities/test.parquet" \
    data.prompt_augmentation.enable=true \
    data.prompt_augmentation.guideline_level=temporal_detailed \
    data.max_response_length=256 \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-3B-Instruct
EOF
#!/bin/bash

# WebQuestionsSP Knowledge Graph Data Processing Pipeline
# This script processes the WebQSP dataset and prepares it for KG-based training

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOCAL_DIR=$WORK_DIR/data/webqsp_kg

echo "Processing WebQuestionsSP dataset for Knowledge Graph training..."
echo "Output directory: $LOCAL_DIR"

# Create output directory
mkdir -p $LOCAL_DIR

# Process WebQSP dataset
echo "Processing WebQuestionsSP dataset..."
python $WORK_DIR/scripts/data_process/webqsp_kg.py \
    --local_dir $LOCAL_DIR \
    --template_type base

echo "WebQSP data processing completed!"
echo "Generated files:"
ls -la $LOCAL_DIR

echo ""
echo "Next steps:"
echo "1. Start the KG retrieval server: bash kg_retrieval_launch.sh"
echo "2. Run training: bash scripts/webqsp_kg/train_ppo.sh"
echo "3. Or run GRPO training: bash scripts/webqsp_kg/train_grpo.sh"

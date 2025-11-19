#!/bin/bash
set -e

echo "🚀 MultiTQ Dataset Setup for KG-R1 Training"
echo "============================================="

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data_multitq_kg"
PROCESSED_DIR="$PROJECT_ROOT/data_kg/multitq"
TRAINING_DIR="$PROJECT_ROOT/data_kg/multitq_search_augmented_initial_entities"

echo "📁 Script directory: $SCRIPT_DIR"
echo "📁 Raw data directory: $DATA_DIR"
echo "📁 Processed data directory: $PROCESSED_DIR"
echo "📁 Training data directory: $TRAINING_DIR"

# Step 1: Download MultiTQ dataset
echo ""
echo "📥 Step 1: Downloading MultiTQ dataset..."
echo "========================================="

if [ -d "$DATA_DIR/MultiTQ" ]; then
    echo "⚠️ MultiTQ data already exists, skipping download"
else
    python "$SCRIPT_DIR/download_multitq.py" --output_dir "$DATA_DIR"
    
    if [ $? -ne 0 ]; then
        echo "❌ Failed to download MultiTQ dataset"
        exit 1
    fi
fi

# Verify download
if [ ! -d "$DATA_DIR/MultiTQ" ]; then
    echo "❌ MultiTQ directory not found after download"
    exit 1
fi

echo "✅ MultiTQ dataset ready"

# Step 2: Process dataset for KG-R1 compatibility
echo ""
echo "🔄 Step 2: Processing dataset for KG-R1 compatibility..."
echo "======================================================="

python "$SCRIPT_DIR/process_multitq.py" \
    --input_dir "$DATA_DIR/MultiTQ" \
    --output_dir "$PROCESSED_DIR"

if [ $? -ne 0 ]; then
    echo "❌ Failed to process MultiTQ dataset"
    exit 1
fi

# Verify processing
required_files=("entities.txt" "entities_text.txt" "relations.txt")
for file in "${required_files[@]}"; do
    if [ ! -f "$PROCESSED_DIR/$file" ]; then
        echo "❌ Required file not found: $PROCESSED_DIR/$file"
        exit 1
    fi
done

echo "✅ Dataset processing completed"

# Step 3: Create search-augmented training data
echo ""
echo "🎯 Step 3: Creating search-augmented training data..."
echo "===================================================="

python "$SCRIPT_DIR/multitq_search_augmented_initial_entities.py" \
    --input_dir "$PROCESSED_DIR" \
    --output_dir "$TRAINING_DIR"

if [ $? -ne 0 ]; then
    echo "❌ Failed to create search-augmented training data"
    exit 1
fi

echo "✅ Search-augmented training data created"

# Step 4: Verify final output
echo ""
echo "🔍 Step 4: Verifying final output..."
echo "==================================="

echo "📊 Checking processed data files:"
for split in train dev test; do
    split_file="$PROCESSED_DIR/${split}_simple.json"
    if [ -f "$split_file" ]; then
        sample_count=$(wc -l < "$split_file")
        echo "   ✅ ${split}_simple.json: $sample_count samples"
    else
        echo "   ⚠️ ${split}_simple.json: not found"
    fi
done

echo ""
echo "📊 Checking training data files:"
for split in train dev test; do
    parquet_file="$TRAINING_DIR/${split}.parquet"
    if [ -f "$parquet_file" ]; then
        file_size=$(du -h "$parquet_file" | cut -f1)
        echo "   ✅ ${split}.parquet: $file_size"
    else
        echo "   ⚠️ ${split}.parquet: not found"
    fi
done

# Count entities and relations
if [ -f "$PROCESSED_DIR/entities.txt" ]; then
    entity_count=$(wc -l < "$PROCESSED_DIR/entities.txt")
    echo "   ✅ entities.txt: $entity_count entities"
fi

if [ -f "$PROCESSED_DIR/relations.txt" ]; then
    relation_count=$(wc -l < "$PROCESSED_DIR/relations.txt")
    echo "   ✅ relations.txt: $relation_count relations"
fi

# Step 5: Integration guidance
echo ""
echo "🎉 MultiTQ dataset setup completed!"
echo "==================================="
echo ""
echo "📋 Available files for KG-R1 training:"
echo "   📄 Raw data: $DATA_DIR"
echo "   📄 Processed data: $PROCESSED_DIR"
echo "   📄 Training data: $TRAINING_DIR"
echo ""
echo "🎯 Usage in training scripts:"
echo "   data.train_files=\"data_kg/multitq_search_augmented_initial_entities/train.parquet\""
echo "   data.val_files=\"data_kg/multitq_search_augmented_initial_entities/test.parquet\""
echo ""
echo "🔧 Key features:"
echo "   • Temporal reasoning with multi-granularity questions"
echo "   • Search-augmented prompts with initial entity hints"
echo "   • Compatible with existing KG-R1 training pipeline"
echo "   • Temporal fact integration for time-sensitive queries"
echo ""
echo "📖 For more details, see:"
echo "   • $DATA_DIR/SETUP_SUMMARY.md"
echo "   • $TRAINING_DIR/DATASET_SUMMARY.md"
#!/bin/bash

# Complete setup script for WebQuestionsSP Knowledge Graph Training
# This script downloads KG data, processes the dataset, and sets up the environment

set -e  # Exit on any error

echo "=== WebQuestionsSP Knowledge Graph Setup ==="

# Configuration
DATA_DIR="data"
KG_DATA_DIR="$DATA_DIR/kg_data"
WEBQSP_DATA_DIR="$DATA_DIR/webqsp_kg"

# Step 1: Create directories
echo "Step 1: Creating directories..."
mkdir -p $KG_DATA_DIR
mkdir -p $WEBQSP_DATA_DIR

# Step 2: Download and prepare KG data
echo "Step 2: Downloading and preparing knowledge graph data..."
python scripts/download_kg.py --save_path $KG_DATA_DIR

# Step 3: Process WebQuestionsSP dataset
echo "Step 3: Processing WebQuestionsSP dataset..."
python scripts/data_process/webqsp_kg.py --local_dir $WEBQSP_DATA_DIR

# Step 4: Verify setup
echo "Step 4: Verifying setup..."

# Check KG data files
if [ -f "$KG_DATA_DIR/freebase_subset.json" ] && [ -f "$KG_DATA_DIR/entity_index.json" ]; then
    echo "✓ Knowledge graph data ready"
else
    echo "✗ Knowledge graph data missing"
    exit 1
fi

# Check WebQSP data files  
if [ -f "$WEBQSP_DATA_DIR/train.parquet" ] && [ -f "$WEBQSP_DATA_DIR/test.parquet" ]; then
    echo "✓ WebQuestionsSP dataset processed"
else
    echo "✗ WebQuestionsSP dataset missing"
    exit 1
fi

echo ""
echo "=== Setup Complete! ==="
echo ""
echo "Next steps:"
echo "1. Start the KG retrieval server:"
echo "   bash kg_retrieval_launch.sh"
echo ""
echo "2. In another terminal, start training:"
echo "   bash scripts/webqsp_kg/train_ppo.sh"
echo ""
echo "3. Or run evaluation:"
echo "   bash scripts/webqsp_kg/evaluate.sh"
echo ""

# Optional: Test the KG retrieval server
echo "Would you like to test the KG retrieval server? (y/n)"
read -r response
if [[ "$response" =~ ^[Yy]$ ]]; then
    echo "Testing KG retrieval server..."
    echo "(This will start the server in background and run a quick test)"
    
    # Start server in background
    bash kg_retrieval_launch.sh &
    SERVER_PID=$!
    
    # Wait for server to start
    sleep 5
    
    # Test the server
    python -c "
import requests
import json

try:
    response = requests.post('http://127.0.0.1:8000/retrieve', 
                           json={'queries': ['Who is Barack Obama?'], 'topk': 3})
    if response.status_code == 200:
        print('✓ KG retrieval server test successful')
        result = response.json()
        print('Sample result:', json.dumps(result, indent=2)[:200] + '...')
    else:
        print('✗ Server test failed:', response.status_code)
except Exception as e:
    print('✗ Server test failed:', str(e))
"
    
    # Stop the server
    kill $SERVER_PID 2>/dev/null || true
    echo "Test complete. Server stopped."
fi

echo ""
echo "Setup completed successfully!"

# WebQuestionsSP Knowledge Graph Training

This directory contains scripts and configurations for training Search-R1 models on the WebQuestionsSP dataset using knowledge graph-based retrieval.

## Overview

WebQuestionsSP is a dataset of question-answer pairs where the questions require multi-hop reasoning over a knowledge graph (Freebase). This implementation adapts the Search-R1 framework to use knowledge graph retrieval instead of document retrieval.

## Quick Start

### 1. Process the Dataset

First, process the WebQuestionsSP dataset:

```bash
bash scripts/webqsp_kg/data_process.sh
```

This will:
- Download and process the WebQuestionsSP dataset
- Format it for KG-based training
- Save processed data to `data/webqsp_kg/`

### 2. Start Knowledge Graph Retrieval Server

Launch the KG retrieval server:

```bash
bash kg_retrieval_launch.sh
```

The server will start on `http://127.0.0.1:8000` and provide KG-based retrieval for entities and relations.

### 3. Train the Model

#### PPO Training
```bash
bash scripts/webqsp_kg/train_ppo.sh
```

#### GRPO Training
```bash
bash scripts/webqsp_kg/train_grpo.sh
```

### 4. Evaluate the Model

```bash
# Set your trained model path in the script first
bash scripts/webqsp_kg/evaluate.sh
```

## Configuration

### Model Options

You can modify the base model in the training scripts:

```bash
# 3B models
export BASE_MODEL='Qwen/Qwen2.5-3B'
export BASE_MODEL='Qwen/Qwen2.5-3B-Instruct'

# 7B models  
export BASE_MODEL='Qwen/Qwen2.5-7B'
export BASE_MODEL='Qwen/Qwen2.5-7B-Instruct'
```

### Training Parameters

Key parameters in the training scripts:

- `max_turns=3`: Maximum number of search iterations
- `topk=3`: Number of KG entities/relations to retrieve
- `retrieval_score=0.8`: Weight for retrieval-based rewards
- `structure_format_score=0.1`: Weight for output format compliance
- `final_format_score=0.1`: Weight for final answer format

### Knowledge Graph Configuration

The KG retrieval server can be configured with:

- `--kg_data_path`: Path to knowledge graph data file (optional)
- `--entity_index_path`: Path to entity linking index (optional)
- `--host`: Server host (default: 127.0.0.1)
- `--port`: Server port (default: 8000)

## Dataset Format

The processed WebQSP data follows this structure:

```json
{
    "data_source": "webqsp",
    "prompt": [{
        "role": "user", 
        "content": "Answer the given question using knowledge graph information..."
    }],
    "ability": "knowledge-reasoning",
    "reward_model": {
        "style": "rule",
        "ground_truth": {"target": ["answer1", "answer2", ...]}
    },
    "extra_info": {
        "split": "train/test",
        "index": 0,
        "qid": "question_id",
        "parse": {...},
        "sparql": "..."
    }
}
```

## Expected Output Format

The model learns to generate responses in this format:

```
<think>
The question asks about [analysis of the question]...
I need to search for information about [entities/concepts].
</think>

<search>[entity or relation query]</search>

<information>
Based on the retrieved information: [summary of KG results]
</information>

<answer>[final answer]</answer>
```

## Reward Function

The reward function evaluates:

1. **Answer Correctness**: Exact match with ground truth answers
2. **Format Compliance**: Proper use of `<think>`, `<search>`, `<information>`, `<answer>` tags  
3. **Retrieval Quality**: Effectiveness of knowledge graph queries

## Files

- `train_ppo.sh`: PPO training script
- `train_grpo.sh`: GRPO training script  
- `evaluate.sh`: Model evaluation script
- `data_process.sh`: Data processing pipeline
- `README.md`: This documentation

## Troubleshooting

### Common Issues

1. **KG Server Not Running**: Make sure to start the KG retrieval server before training
2. **CUDA Memory Issues**: Reduce batch sizes or enable gradient checkpointing
3. **Dataset Not Found**: Run the data processing script first

### Monitoring Training

Monitor training progress with Weights & Biases:
- Project: `Search-R1-KG`  
- Experiment names follow pattern: `webqsp-kg-r1-{algorithm}-{model}-{variant}`

## Knowledge Graph Integration

The KG retrieval server (`kg_r1/search/kg_retrieval_server.py`) provides:

- Entity linking from natural language queries
- Relation retrieval for linked entities  
- Structured knowledge graph responses
- Batch processing for multiple queries

For production use, you can extend this with:
- Real Freebase/Wikidata integration
- Better entity linking models
- Relation extraction improvements
- Caching for frequent queries

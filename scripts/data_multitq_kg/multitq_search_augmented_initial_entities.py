#!/usr/bin/env python3
"""
MultiTQ Search Augmented Training Data Creator

This script creates training-ready .parquet files for MultiTQ temporal KGQA 
directly from the raw MultiTQ data, designed to work with the MultiTQ KG server.
"""

import json
import os
import pandas as pd
from pathlib import Path
import argparse
from typing import List, Dict, Any

def load_multitq_questions(questions_file: Path) -> List[Dict]:
    """Load MultiTQ questions from JSON file (handles both JSON and JSONL formats)"""
    print(f"📊 Loading questions from {questions_file}")
    
    data = []
    with open(questions_file, 'r', encoding='utf-8') as f:
        try:
            # Try loading as single JSON array first
            data = json.load(f)
        except json.JSONDecodeError:
            # Fall back to JSONL format (one JSON object per line)
            f.seek(0)
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
    
    print(f"   ✅ Loaded {len(data)} questions")
    return data

def create_multitq_kg_prompt(question: str) -> str:
    """Create standard KG prompt for MultiTQ (same format as other datasets)"""
    
    prompt = (
        "Answer the given question. You must conduct reasoning inside <think> and </think> first "
        "every time you get new information. After reasoning, if you find you lack some knowledge, "
        "you can query the knowledge graph by using <kg-query> function_name(arguments) </kg-query>, "
        "and it will return the top query results between <information> and </information>. "
        "You can query as many times as you want. If you find no further external knowledge needed, "
        "you can directly provide the answer inside <answer> and </answer> without detailed "
        f"illustrations. For example, <answer> Beijing </answer>.\n\nQuestion: {question}"
    )
    
    return prompt

def process_multitq_split(questions_data: List[Dict], split_name: str) -> List[Dict]:
    """Convert MultiTQ questions to KG-R1 training format"""
    print(f"🔄 Converting {len(questions_data)} {split_name} samples to training format...")
    
    training_samples = []
    
    for sample in questions_data:
        # Extract basic information
        quid = sample.get("quid", "")
        question = sample.get("question", "")
        answers = sample.get("answers", [])
        answer_type = sample.get("answer_type", "unknown")
        time_level = sample.get("time_level", "unknown")
        
        # Create KG prompt for MultiTQ
        kg_prompt = create_multitq_kg_prompt(question)
        
        # Format answers to match KG-R1 expected format (as lists)
        if isinstance(answers, list) and answers:
            target_text = [str(ans) for ans in answers]
        else:
            target_text = [str(answers)] if answers else [""]
            
        # MultiTQ doesn't have kb_id, so use empty list
        target_kb_id = [""] * len(target_text)
        
        # Create training sample in KG-R1 format
        training_sample = {
            "data_source": "kgR1_multitq",
            "prompt": [{"role": "user", "content": kg_prompt}],
            "ability": "kg-reasoning",
            "reward_model": {
                "style": "rule",
                "ground_truth": {
                    "target_text": target_text,
                    "target_kb_id": target_kb_id
                }
            },
            "extra_info": {
                "dataset_name": "multitq",  # Critical: evaluation system looks for this field
                "sample_id": f"{split_name}_{quid}",
                "split": split_name,
                "initial_entities": [],  # Will be populated during training
                "initial_entity_ids": []  # Will be populated during training
            }
        }
        
        training_samples.append(training_sample)
    
    print(f"   ✅ Converted {len(training_samples)} samples")
    return training_samples

def save_training_data(training_data: List[Dict], output_file: Path):
    """Save training data as parquet file"""
    print(f"💾 Saving {len(training_data)} samples to {output_file}")
    
    # Convert to DataFrame and save as parquet
    df = pd.DataFrame(training_data)
    
    # Ensure output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Save as parquet
    df.to_parquet(output_file, index=False)
    
    print(f"   ✅ Saved to {output_file}")

def create_dataset_summary(output_dir: Path, stats: Dict[str, int]):
    """Create dataset summary file"""
    summary_file = output_dir / "DATASET_SUMMARY.md"
    
    total_samples = sum(stats.values())
    
    summary_content = f"""# MultiTQ KG-R1 Training Dataset Summary

## Dataset Overview
- **Source**: MultiTQ Temporal KGQA Dataset
- **Task**: Multi-granularity temporal question answering
- **KG Integration**: Direct MultiTQ KG server integration
- **Server**: Uses specialized MultiTQ KG server with temporal filtering

## Training Data Statistics
- **Total samples**: {total_samples:,}
- **Train**: {stats.get('train', 0):,} samples
- **Dev**: {stats.get('dev', 0):,} samples  
- **Test**: {stats.get('test', 0):,} samples

## Files Generated
- `train.parquet` - {stats.get('train', 0):,} training samples
- `dev.parquet` - {stats.get('dev', 0):,} development samples
- `test.parquet` - {stats.get('test', 0):,} testing samples

## MultiTQ KG Server Integration
- **Server**: Runs on http://127.0.0.1:8001 (same port as CWQ/WebQSP)
- **Actions**: get_head_relations, get_tail_relations, get_head_entities, get_tail_entities
- **Temporal Filtering**: Supports timestamp parameter for precise temporal queries
- **Data Source**: Direct access to MultiTQ kg files (train.txt, valid.txt, test.txt)

## Temporal Reasoning Features
- Multi-granularity temporal questions (day/month/year)
- Temporal filtering in entity queries
- Time-sensitive relationship reasoning
- No initial entity augmentation (MultiTQ doesn't provide them)

## Usage with KG-R1 Training
```bash
# Start MultiTQ KG server first
./kg_retrieval_launch_multitq.sh

# Then use in training scripts:
data.train_files="data_multitq_kg/multitq_search_augmented/train.parquet"
data.val_files="data_multitq_kg/multitq_search_augmented/dev.parquet"
```

## KG Server Commands Available
- `get_head_relations(entity_id)` - Relations where entity is subject
- `get_tail_relations(entity_id)` - Relations where entity is object  
- `get_head_entities(entity_id, relation, timestamp=optional)` - Connected entities with time filter
- `get_tail_entities(entity_id, relation, timestamp=optional)` - Source entities with time filter

## Temporal Query Examples
- Year filter: `timestamp='2015'` matches `2015-06-20`
- Month filter: `timestamp='2015-06'` matches `2015-06-20`
- Day filter: `timestamp='2015-06-20'` matches exact date
"""
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(summary_content)
    
    print(f"📝 Created dataset summary: {summary_file}")

def main():
    parser = argparse.ArgumentParser(description="Create MultiTQ search-augmented training data")
    parser.add_argument("--input_dir", type=str,
                       default="data_kg/multitq",
                       help="Input processed MultiTQ data directory (relative to project root)")
    parser.add_argument("--output_dir", type=str,
                       default="data_kg/multitq_search_augmented_initial_entities",
                       help="Output directory for parquet files (relative to project root)")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Maximum samples per split (for testing)")
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    print("🚀 Creating MultiTQ Search-Augmented Training Data")
    print("=" * 60)
    print(f"📁 Input directory: {input_dir}")
    print(f"📁 Output directory: {output_dir}")
    if args.max_samples:
        print(f"🔢 Max samples per split: {args.max_samples}")
    
    # Process each split
    splits = [
        ("train", "train.json"),
        ("dev", "dev.json"),
        ("test", "test.json")
    ]
    
    stats = {}
    
    for split_name, questions_file in splits:
        print(f"\n📊 Processing {split_name} split...")
        
        # Try questions/ subdirectory first, then direct files
        questions_path = input_dir / "questions" / questions_file
        if not questions_path.exists():
            # Try direct files (e.g., train_simple.json)
            questions_file_simple = questions_file.replace('.json', '_simple.json')
            questions_path = input_dir / questions_file_simple
            
        if not questions_path.exists():
            print(f"   ⚠️ Questions file not found: {questions_path}")
            continue
        
        # Load questions
        questions_data = load_multitq_questions(questions_path)
        
        # Limit samples if specified
        if args.max_samples and len(questions_data) > args.max_samples:
            questions_data = questions_data[:args.max_samples]
            print(f"   ⚠️ Limited to {args.max_samples} samples (from original {len(questions_data)})")
        
        # Convert to training format
        training_data = process_multitq_split(questions_data, split_name)
        
        # Save as parquet
        output_file = output_dir / f"{split_name}.parquet"
        save_training_data(training_data, output_file)
        
        stats[split_name] = len(training_data)
    
    # Create dataset summary
    create_dataset_summary(output_dir, stats)
    
    total_samples = sum(stats.values())
    print(f"\n🎉 MultiTQ search-augmented data creation completed!")
    print(f"📁 Output saved to: {output_dir}")
    print(f"\n📊 Final statistics:")
    print(f"   📄 Total samples processed: {total_samples:,}")
    for split_name, count in stats.items():
        print(f"   📄 {split_name}.parquet: {count:,} samples")
    
    print(f"\n📋 Ready for KG-R1 training:")
    print(f"   🎯 MultiTQ temporal reasoning capabilities enabled")
    print(f"   🎯 Direct KG server integration (no initial entities)")
    print(f"   🎯 Temporal filtering support in queries")
    print(f"   🎯 Multi-granularity temporal questions supported")

if __name__ == "__main__":
    main()
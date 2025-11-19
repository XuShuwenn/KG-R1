#!/usr/bin/env python3
"""
Process CWQ (ComplexWebQuestions) dataset to parquet format for KG-augmented training.

This script load        "extra_info": {
            'split': split,
            'index': idx,
            'sample_id': sample_id,
            'dataset_name': 'cwq'  # Add explicit dataset name for CWQ
        }JSON files and converts them to the exact format required 
by the nq_search.py data processing pipeline, but adapted for KG interactions.
"""

import json
import os
import pandas as pd
import numpy as np
from pathlib import Path
import argparse


def create_kg_prompt_template(question):
    """Create KG-augmented prompt template"""
    
    prefix = (
        "Answer the given question. You must conduct reasoning inside <think> and </think> "
        "first every time you get new information. After reasoning, if you find you lack some "
        "knowledge, you can query the knowledge graph by using <kg-query> function_name(arguments) </kg-query>, and it will "
        "return the top query results between <information> and </information>. You "
        "can query as many times as you want. If you find no further external knowledge "
        "needed, you can directly provide the answer inside <answer> and </answer> without "
        f"detailed illustrations. For example, <answer> Beijing </answer>.\n\nQuestion: {question}"
    )
    
    return prefix


def load_cwq_json(file_path):
    """Load CWQ JSONL file and return list of samples."""
    print(f"Loading CWQ data from {file_path}")
    
    samples = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        # JSONL format - each line is a separate JSON object
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    sample = json.loads(line)
                    samples.append(sample)
                except json.JSONDecodeError as e:
                    print(f"Error parsing line {line_num}: {e}")
                    continue
    
    print(f"Loaded {len(samples)} samples from {file_path}")
    return samples


def process_cwq_sample(sample, idx, split):
    """Process a single CWQ sample into the required format."""
    
    # Extract question - based on CWQ structure
    if 'question' in sample:
        question = sample['question']
    else:
        raise KeyError(f"No question field found in sample {idx}. Available fields: {list(sample.keys())}")
    
    # Extract sample_id from 'id' field - based on CWQ structure  
    if 'id' in sample:
        sample_id = sample['id']
    else:
        # If no ID field found, use the index as fallback
        sample_id = f"{split}_{idx}"
        print(f"Warning: No 'id' field found for sample {idx}, using fallback: {sample_id}")
    
    # Extract answers - based on CWQ structure
    answers = []
    if 'answers' in sample:
        if isinstance(sample['answers'], list):
            answers = sample['answers']
        else:
            answers = [sample['answers']]
    else:
        print(f"Warning: No answers found for sample {idx}. Using empty answer.")
        answers = [""]
    
    # Clean and format question
    question = str(question).strip()
    if question and question[-1] != '?':
        question += '?'
    
    # Process answers to separate text and kb_id
    target_text = []
    target_kb_id = []
    
    for ans in answers:
        if isinstance(ans, dict):
            # Handle structured answers with text and kb_id
            if 'text' in ans:
                target_text.append(str(ans['text']).strip())
            if 'kb_id' in ans:
                target_kb_id.append(str(ans['kb_id']).strip())
        elif isinstance(ans, str) and ans.strip():
            # Handle simple string answers
            target_text.append(ans.strip())
            # For string answers, we don't have kb_id
        elif ans:  # Non-empty non-string
            target_text.append(str(ans).strip())
    
    # Ensure we have at least one answer
    if not target_text:
        target_text = [""]
    
    # Ensure kb_id list matches text list length (pad with empty strings if needed)
    while len(target_kb_id) < len(target_text):
        target_kb_id.append("")
    
    # Create prompt using template
    prompt_content = create_kg_prompt_template(question)
    
    # Create the data structure matching nq_search.py exactly
    data = {
        "data_source": "kgR1_cwq",  # Use kgR1 naming for reward function
        "prompt": [{
            "role": "user",
            "content": prompt_content,
        }],
        "ability": "kg-reasoning",
        "reward_model": {
            "style": "rule",
            "ground_truth": {
                "target_text": target_text,
                "target_kb_id": target_kb_id
            }
        },
        "extra_info": {
            'split': split,
            'index': idx,
            'sample_id': sample_id,
            'dataset_name': 'CWQ',  # Keep original dataset name
        }
    }
    
    return data


def process_cwq_dataset(json_file_path, split):
    """Process CWQ JSON file into the required format."""
    
    # Load JSON data
    samples = load_cwq_json(json_file_path)
    
    processed_data = []
    
    print(f"Processing {len(samples)} samples for {split} split...")
    
    for idx, sample in enumerate(samples):
        try:
            data = process_cwq_sample(sample, idx, split)
            processed_data.append(data)
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            print(f"Sample keys: {list(sample.keys()) if isinstance(sample, dict) else type(sample)}")
            continue
    
    print(f"Successfully processed {len(processed_data)} samples for {split}")
    return processed_data


def main():
    parser = argparse.ArgumentParser(description='Process CWQ dataset to parquet format')
    parser.add_argument('--train_json', default='data_kg/CWQ/train_simple.json',
                       help='Path to train JSON file (relative to project root)')
    parser.add_argument('--dev_json', default='data_kg/CWQ/dev_simple.json',
                       help='Path to dev JSON file (relative to project root)')
    parser.add_argument('--test_json', default='data_kg/CWQ/test_simple.json',
                       help='Path to test JSON file (relative to project root)')
    parser.add_argument('--output_dir', default='data_kg/cwq_search',
                       help='Output directory for parquet files')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Processing CWQ dataset...")
    print(f"Train JSON: {args.train_json}")
    print(f"Dev JSON: {args.dev_json}")
    print(f"Test JSON: {args.test_json}")
    print(f"Output directory: {args.output_dir}")
    
    # Process train data
    if os.path.exists(args.train_json):
        train_data = process_cwq_dataset(args.train_json, 'train')
        train_df = pd.DataFrame(train_data)
        train_file = output_dir / 'train.parquet'
        train_df.to_parquet(train_file, index=False)
        print(f"Saved train data: {train_file} ({len(train_df)} samples)")
    else:
        print(f"Warning: Train file not found: {args.train_json}")
    
    # Process dev data (validation)
    if os.path.exists(args.dev_json):
        dev_data = process_cwq_dataset(args.dev_json, 'dev')
        dev_df = pd.DataFrame(dev_data)
        dev_file = output_dir / 'dev.parquet'
        dev_df.to_parquet(dev_file, index=False)
        print(f"Saved dev data: {dev_file} ({len(dev_df)} samples)")
    else:
        print(f"Warning: Dev file not found: {args.dev_json}")
    
    # Process test data
    if os.path.exists(args.test_json):
        test_data = process_cwq_dataset(args.test_json, 'test')
        test_df = pd.DataFrame(test_data)
        test_file = output_dir / 'test.parquet'
        test_df.to_parquet(test_file, index=False)
        print(f"Saved test data: {test_file} ({len(test_df)} samples)")
    else:
        print(f"Warning: Test file not found: {args.test_json}")
    
    print("\nCWQ dataset processing complete!")
    
    if 'train_df' in locals():
        print(f"\nData verification:")
        print(f"  Train shape: {train_df.shape}")
        print(f"  Train columns: {list(train_df.columns)}")
        print(f"  Sample prompt preview: {train_df.iloc[0]['prompt'][0]['content'][:200]}...")
    
    if 'dev_df' in locals():
        print(f"  Dev shape: {dev_df.shape}")
    
    if 'test_df' in locals():
        print(f"  Test shape: {test_df.shape}")


if __name__ == "__main__":
    main()

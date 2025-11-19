#!/usr/bin/env python3
"""
Process WebQSP dataset to parquet format for KG-augmented training.

This script loads WebQSP JSON files and converts them to the exact format required 
by the nq_search.py data processing pipeline, but adapted for KG interactions.
"""

import json
import os
import pandas as pd
import numpy as np
from pathlib import Path 
import argparse
import ast

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


def load_webqsp_json(file_path):
    """Load WebQSP JSONL file and return list of samples."""
    print(f"Loading WebQSP data from {file_path}")
    
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


def process_webqsp_sample(sample, idx, split):
    """Process a single WebQSP sample into the required format."""
    
    # Extract question - based on knowledge_graph.py structure
    if 'question' in sample:
        question = sample['question']
    else:
        raise KeyError(f"No question field found in sample {idx}. Available fields: {list(sample.keys())}")
    
    # Extract sample_id from 'id' field - based on knowledge_graph.py structure  
    if 'id' in sample:
        sample_id = sample['id']
    else:
        # If no ID field found, use the index as fallback
        sample_id = f"{split}_{idx}"
        print(f"Warning: No 'id' field found for sample {idx}, using fallback: {sample_id}")
     # Extract answers - based on knowledge_graph.py structure
    answers = []
    if 'answers' in sample:
        if isinstance(sample['answers'], list):
            answers = sample['answers']
        else:
            answers = [sample['answers']]
    else:
        print(f"Warning: No answers found for sample {idx}. Using empty answer.")
        answers = []

    # Clean and format question
    question = str(question).strip()
    if question and question[-1] != '?':
        question += '?'
    
    # Parse answers to extract text and kb_id
    target_text = []
    target_kb_id = []
    
    for answer in answers:
        if isinstance(answer, dict):
            # Answer is already a dictionary with 'text' and 'kb_id' fields
            if 'text' in answer and answer['text']:
                target_text.append(answer['text'])
            if 'kb_id' in answer and answer['kb_id']:
                target_kb_id.append(answer['kb_id'])
        elif isinstance(answer, str):
            # If answer is a string, treat as plain text
            if answer.strip():
                target_text.append(answer.strip())
        else:
            # Convert other types to string
            if answer:
                target_text.append(str(answer))
    
    # Ensure we have at least one answer
    if not target_text:
        target_text = [""]
    if not target_kb_id:
        target_kb_id = []
    
    # Create prompt using template
    prompt_content = create_kg_prompt_template(question)
    
    # Create the data structure matching nq_search.py exactly
    data = {
        "data_source": "kgR1_webqsp",  # Use kgR1 naming for reward function
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
            'dataset_name': 'webqsp',  # Keep original dataset name
        }
    }
    
    return data

def process_webqsp_dataset(json_file_path, split):
    """Process WebQSP JSON file into the required format."""
    
    # Load JSON data
    samples = load_webqsp_json(json_file_path)
    
    processed_data = []
    
    print(f"Processing {len(samples)} samples for {split} split...")
    
    for idx, sample in enumerate(samples):
        try:
            data = process_webqsp_sample(sample, idx, split)
            processed_data.append(data)
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            print(f"Sample keys: {list(sample.keys()) if isinstance(sample, dict) else type(sample)}")
            continue
    
    print(f"Successfully processed {len(processed_data)} samples for {split}")
    return processed_data


def main():
    parser = argparse.ArgumentParser(description='Process WebQSP dataset to parquet format')
    parser.add_argument('--train_json', default='data_kg/webqsp/train_simple.json',
                       help='Path to train JSON file (relative to project root)')
    parser.add_argument('--test_json', default='data_kg/webqsp/test_simple.json',
                       help='Path to test JSON file (relative to project root)')
    parser.add_argument('--output_dir', default='data_kg/webqsp_search',
                       help='Output directory for parquet files')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Processing WebQSP dataset...")
    print(f"Train JSON: {args.train_json}")
    print(f"Test JSON: {args.test_json}")
    print(f"Output directory: {args.output_dir}")
    
    # Process train data
    if os.path.exists(args.train_json):
        train_data = process_webqsp_dataset(args.train_json, 'train')
        train_df = pd.DataFrame(train_data)
        train_file = output_dir / 'train.parquet'
        train_df.to_parquet(train_file, index=False)
        print(f"Saved train data: {train_file} ({len(train_df)} samples)")
    else:
        print(f"Warning: Train file not found: {args.train_json}")
    
    # Process test data
    if os.path.exists(args.test_json):
        test_data = process_webqsp_dataset(args.test_json, 'test')
        test_df = pd.DataFrame(test_data)
        test_file = output_dir / 'test.parquet'
        test_df.to_parquet(test_file, index=False)
        print(f"Saved test data: {test_file} ({len(test_df)} samples)")
    else:
        print(f"Warning: Test file not found: {args.test_json}")
    
    print("\nWebQSP dataset processing complete!")
    
    if 'train_df' in locals():
        print(f"\nData verification:")
        print(f"  Train shape: {train_df.shape}")
        print(f"  Train columns: {list(train_df.columns)}")
        print(f"  Sample prompt preview: {train_df.iloc[0]['prompt'][0]['content'][:200]}...")
    
    if 'test_df' in locals():
        print(f"  Test shape: {test_df.shape}")


if __name__ == "__main__":
    main()

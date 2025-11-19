#!/usr/bin/env python3
"""
Filter the existing MultiTQ 2K dataset to exclude samples with time_level='day'
"""

import pandas as pd
import json
import numpy as np
from pathlib import Path

def has_day_level_time(prompt_content):
    """
    Extract time_level from the prompt content and check if it's 'day'.
    We need to parse the original question data to get the time_level.
    """
    try:
        # For now, we'll use the original raw data to match questions
        # This is a simplified approach - in practice you might want to 
        # store the time_level in the parquet files directly
        return False  # We'll implement this by matching with original data
    except:
        return False

def filter_multitq_2k_dataset():
    """Filter the 2K dataset to exclude day-level time granularity samples."""
    
    # Paths (relative to project root)
    input_dir = Path("data_multitq_kg/multitq_search_augmented_2k")
    output_dir = Path("data_multitq_kg/multitq_search_augmented_2k_filtered")
    original_data_dir = Path("data_multitq_kg/MultiTQ/questions")
    
    output_dir.mkdir(exist_ok=True)
    
    print("🚀 Filtering MultiTQ 2K dataset to exclude day-level samples")
    print(f"📁 Input: {input_dir}")
    print(f"📁 Output: {output_dir}")
    
    # Load original question data to get time_level information
    print("\n🔍 Loading original question data for time_level mapping...")
    original_data = {}
    
    for split in ['train', 'dev', 'test']:
        original_file = original_data_dir / f"{split}.json"
        if original_file.exists():
            with open(original_file, 'r') as f:
                data = json.load(f)
                for item in data:
                    question = item['question'].strip()
                    time_level = item.get('time_level', '').strip().lower()
                    original_data[question] = time_level
    
    print(f"   ✅ Loaded {len(original_data)} original questions with time_level data")
    
    # Process each split
    total_original = 0
    total_filtered = 0
    total_excluded = 0
    
    for split in ['train', 'dev', 'test']:
        print(f"\n📊 Processing {split} split...")
        
        input_file = input_dir / f"{split}.parquet"
        output_file = output_dir / f"{split}.parquet"
        
        if not input_file.exists():
            print(f"   ⚠️ File not found: {input_file}")
            continue
            
        # Load parquet data
        df = pd.read_parquet(input_file)
        original_count = len(df)
        total_original += original_count
        
        print(f"   📊 Original samples: {original_count}")
        
        # Filter out day-level samples
        filtered_rows = []
        excluded_count = 0
        
        for idx, row in df.iterrows():
            # Extract question from prompt - the prompt is in chat message format
            prompt = row['prompt']
            
            # Extract question from chat message format (handle numpy arrays)
            question = None
            try:
                # Handle numpy array format
                if isinstance(prompt, np.ndarray):
                    prompt_list = prompt.tolist()
                    if len(prompt_list) > 0 and isinstance(prompt_list[0], dict):
                        content = prompt_list[0]['content']
                        if 'Question:' in content:
                            question_part = content.split('Question:')[1].strip()
                            # Remove initial entities hint if present
                            if '(Initial entities:' in question_part:
                                question = question_part.split('(Initial entities:')[0].strip()
                            else:
                                question = question_part.strip()
                elif isinstance(prompt, list) and len(prompt) > 0:
                    content = prompt[0]['content']
                    if 'Question:' in content:
                        question_part = content.split('Question:')[1].strip()
                        # Remove initial entities hint if present
                        if '(Initial entities:' in question_part:
                            question = question_part.split('(Initial entities:')[0].strip()
                        else:
                            question = question_part.strip()
                elif isinstance(prompt, str):
                    if 'Question:' in prompt:
                        question_part = prompt.split('Question:')[1].strip()
                        if '(Initial entities:' in question_part:
                            question = question_part.split('(Initial entities:')[0].strip()
                        else:
                            question = question_part.strip()
            except Exception as e:
                print(f"     ⚠️ Error extracting question: {e}")
            
            if question and question in original_data:
                time_level = original_data[question]
                if time_level == 'day':
                    excluded_count += 1
                    print(f"     🚫 Excluding day-level question: {question[:60]}...")
                    continue
            
            filtered_rows.append(row)
        
        # Create filtered dataframe
        if filtered_rows:
            filtered_df = pd.DataFrame(filtered_rows)
            filtered_count = len(filtered_df)
        else:
            filtered_df = pd.DataFrame()
            filtered_count = 0
            
        total_filtered += filtered_count
        total_excluded += excluded_count
        
        # Save filtered data
        filtered_df.to_parquet(output_file, index=False)
        
        print(f"   📊 Filtering results:")
        print(f"     Original samples: {original_count}")
        print(f"     Filtered samples: {filtered_count}")
        print(f"     Excluded (day-level): {excluded_count}")
        print(f"   ✅ Saved to {output_file}")
    
    # Create summary
    summary_content = f"""# MultiTQ 2K Filtered Dataset Summary

## Filtering Configuration
- **Exclusion criteria**: time_level='day'
- **Original dataset**: multitq_search_augmented_2k
- **Filtered dataset**: multitq_search_augmented_2k_filtered

## Filtering Results
- **Original total samples**: {total_original:,}
- **Filtered total samples**: {total_filtered:,}
- **Excluded (day-level)**: {total_excluded:,}
- **Exclusion rate**: {(total_excluded/total_original)*100:.1f}%

## Files Generated
- `train.parquet` - Training samples (day-level excluded)
- `dev.parquet` - Development samples (day-level excluded)  
- `test.parquet` - Testing samples (day-level excluded)

## Usage
Update your training/evaluation scripts to use:
```
data.train_files="data_multitq_kg/multitq_search_augmented_2k_filtered/train.parquet"
data.val_files="data_multitq_kg/multitq_search_augmented_2k_filtered/test.parquet"
```

## Compatibility
This filtered dataset only contains questions with:
- time_level='year' (YYYY format answers)
- time_level='month' (YYYY-MM format answers)

All day-level questions (YYYY-MM-DD format) have been excluded for evaluation system compatibility.
"""
    
    with open(output_dir / "DATASET_SUMMARY.md", 'w') as f:
        f.write(summary_content)
    
    print(f"\n🎉 Filtering completed!")
    print(f"📊 Overall statistics:")
    print(f"   📄 Original total samples: {total_original:,}")
    print(f"   📄 Filtered total samples: {total_filtered:,}")
    print(f"   📄 Excluded (day-level): {total_excluded:,}")
    print(f"   📄 Exclusion rate: {(total_excluded/total_original)*100:.1f}%")
    print(f"\n📁 Filtered dataset saved to: {output_dir}")

if __name__ == "__main__":
    filter_multitq_2k_dataset()
"""
Convert kgqa_agent format cwq_train.json to VERL parquet format.

This script processes kgqa_agent/src/eval/datasets/cwq_train.json and converts it
to VERL-compatible parquet files using the custom prompt builder from
verl/trainer/ppo/prompts.py (build_search_prompt).
"""
import json
import os
import sys
import pandas as pd
from typing import List, Dict, Any

# Add project root to path to import verl modules
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# Direct import to avoid loading verl.__init__ which requires ray
prompts_path = os.path.join(project_root, "verl", "trainer", "ppo", "prompts.py")
if os.path.exists(prompts_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("prompts", prompts_path)
    prompts_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prompts_module)
    build_search_prompt = prompts_module.build_search_prompt
else:
    raise FileNotFoundError(f"Could not find prompts.py at {prompts_path}")


def load_kgqa_agent_cwq(json_path: str) -> List[Dict[str, Any]]:
    """Load kgqa_agent format CWQ JSON file."""
    print(f"Loading data from: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} samples")
    return data


def process_sample(
    item: Dict[str, Any],
    split_name: str,
    max_calls: int = 10,
) -> Dict[str, Any]:
    """
    Convert a single kgqa_agent sample to VERL format.
    
    Args:
        item: Sample from kgqa_agent JSON (with id, question, answers, topic_entity)
        split_name: Dataset split name (train/val/test)
        max_calls: Maximum KG query calls (used in prompt)
    
    Returns:
        VERL-compatible sample dict
    """
    sample_id = item.get('id', '')
    question = item.get('question', '').strip()
    answers = item.get('answers', [])
    topic_entity_dict = item.get('topic_entity', {})
    
    # Handle None or empty topic_entity
    if topic_entity_dict is None:
        topic_entity_dict = {}
    if not isinstance(topic_entity_dict, dict):
        topic_entity_dict = {}
    
    # Extract topic entities: topic_entity is a dict {entity_id: entity_name}
    topic_entities = list(topic_entity_dict.values()) if topic_entity_dict else []  # List of entity names
    topic_entity_ids = list(topic_entity_dict.keys()) if topic_entity_dict else []   # List of entity IDs
    
    # Use custom prompt builder from verl/trainer/ppo/prompts.py
    prompt_text = build_search_prompt(
        question_or_sample=question,
        max_calls=max_calls,
        topic_entities=topic_entities if topic_entities else None,
    )
    
    # Format as chat messages
    prompt_messages = [{"role": "user", "content": prompt_text}]
    
    # Normalize answers: ensure they're strings and remove duplicates
    target_text = []
    seen = set()
    for ans in answers:
        if isinstance(ans, str):
            ans_clean = ans.strip()
            if ans_clean and ans_clean.lower() not in seen:
                target_text.append(ans_clean)
                seen.add(ans_clean.lower())
        elif isinstance(ans, dict):
            # Handle dict format if needed
            ans_str = ans.get('text', str(ans))
            ans_clean = ans_str.strip()
            if ans_clean and ans_clean.lower() not in seen:
                target_text.append(ans_clean)
                seen.add(ans_clean.lower())
    
    # Build VERL-compatible sample
    processed_item = {
        "data_source": "kgqa_agent_cwq",
        "prompt": prompt_messages,
        "ability": "kg-reasoning",
        "reward_model": {
            "style": "rule",
            "ground_truth": {
                "target_text": target_text,
                "target_kb_id": topic_entity_ids,  # Use topic entity IDs as kb_ids
            }
        },
        "extra_info": {
            "split": split_name,
            "sample_id": sample_id,
            "dataset_name": "cwq",
            "question": question,  # Store original question for reference
            "initial_entities": topic_entities,
            "initial_entity_ids": topic_entity_ids,
        }
    }
    
    return processed_item


def convert_kgqa_agent_to_verl(
    input_json_path: str,
    output_dir: str,
    split_name: str = "train",
    max_calls: int = 10,
    train_ratio: float = 0.9,
    val_ratio: float = 0.05,
    test_ratio: float = 0.05,
):
    """
    Convert kgqa_agent format JSON to VERL parquet files.
    
    Args:
        input_json_path: Path to kgqa_agent cwq_train.json
        output_dir: Output directory for parquet files
        split_name: If input is already split, specify "train"/"val"/"test"
        max_calls: Maximum KG query calls for prompt
        train_ratio: Ratio for train split (if auto-splitting)
        val_ratio: Ratio for val split (if auto-splitting)
        test_ratio: Ratio for test split (if auto-splitting)
    """
    # Load data
    data = load_kgqa_agent_cwq(input_json_path)
    
    # Process samples
    print(f"Processing {len(data)} samples...")
    processed_data = []
    skipped_count = 0
    
    for idx, item in enumerate(data):
        if (idx + 1) % 1000 == 0:
            print(f"  Processed {idx + 1}/{len(data)} samples (skipped {skipped_count} with None topic_entity)...")
        
        # Skip samples with None or empty topic_entity
        topic_entity_dict = item.get('topic_entity', {})
        if topic_entity_dict is None or (isinstance(topic_entity_dict, dict) and len(topic_entity_dict) == 0):
            skipped_count += 1
            continue
        
        processed_item = process_sample(item, split_name, max_calls=max_calls)
        processed_data.append(processed_item)
    
    print(f"Processed {len(processed_data)} samples (skipped {skipped_count} samples with None/empty topic_entity)")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine splits
    if split_name in ["train", "val", "test"]:
        # Use provided split name
        output_file = os.path.join(output_dir, f"{split_name}.parquet")
        df = pd.DataFrame(processed_data)
        df.to_parquet(output_file, index=False)
        print(f"Saved {len(processed_data)} {split_name} samples to {output_file}")
        
        # Print sample info
        if processed_data:
            sample = processed_data[0]
            print(f"\nSample {split_name} data structure:")
            print(f"  - sample_id: {sample['extra_info']['sample_id']}")
            print(f"  - question: {sample['extra_info']['question'][:80]}...")
            print(f"  - initial_entities: {sample['extra_info']['initial_entities']}")
            print(f"  - prompt length: {len(sample['prompt'][0]['content'])} chars")
            print(f"  - target_text: {sample['reward_model']['ground_truth']['target_text'][:3]}...")
    else:
        # Auto-split into train/val/test
        total = len(processed_data)
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)
        
        train_data = processed_data[:train_end]
        val_data = processed_data[train_end:val_end]
        test_data = processed_data[val_end:]
        
        # Save splits
        for split_data, split_name in [(train_data, "train"), (val_data, "val"), (test_data, "test")]:
            if split_data:
                output_file = os.path.join(output_dir, f"{split_name}.parquet")
                df = pd.DataFrame(split_data)
                df.to_parquet(output_file, index=False)
                print(f"Saved {len(split_data)} {split_name} samples to {output_file}")
        
        # Print sample info
        if train_data:
            sample = train_data[0]
            print(f"\nSample train data structure:")
            print(f"  - sample_id: {sample['extra_info']['sample_id']}")
            print(f"  - question: {sample['extra_info']['question'][:80]}...")
            print(f"  - initial_entities: {sample['extra_info']['initial_entities']}")
            print(f"  - prompt length: {len(sample['prompt'][0]['content'])} chars")
            print(f"  - target_text: {sample['reward_model']['ground_truth']['target_text'][:3]}...")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Convert kgqa_agent cwq_train.json to VERL parquet format"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="kgqa_agent/src/eval/datasets/cwq_train.json",
        help="Path to input kgqa_agent JSON file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data_kg/cwq_kgqa_agent_format",
        help="Output directory for parquet files"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val", "test", "auto"],
        help="Split name (or 'auto' to split into train/val/test)"
    )
    parser.add_argument(
        "--max_calls",
        type=int,
        default=10,
        help="Maximum KG query calls (used in prompt)"
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.9,
        help="Train split ratio (if auto-splitting)"
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.05,
        help="Val split ratio (if auto-splitting)"
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.05,
        help="Test split ratio (if auto-splitting)"
    )
    
    args = parser.parse_args()
    
    # Validate ratios if auto-splitting
    if args.split == "auto":
        total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
        if abs(total_ratio - 1.0) > 0.01:
            print(f"Warning: Split ratios sum to {total_ratio}, not 1.0. Normalizing...")
            args.train_ratio /= total_ratio
            args.val_ratio /= total_ratio
            args.test_ratio /= total_ratio
    
    # Convert
    convert_kgqa_agent_to_verl(
        input_json_path=args.input,
        output_dir=args.output_dir,
        split_name=args.split if args.split != "auto" else None,
        max_calls=args.max_calls,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )
    
    print(f"\n✓ Conversion complete! Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()


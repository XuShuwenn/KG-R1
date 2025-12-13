#!/usr/bin/env python3
"""
Batch convert kgqa_agent CWQ splits (train/dev/test) to VERL parquet format.

This script processes:
- cwq_train.json -> train.parquet
- cwq_dev.json -> val.parquet
- cwq_test.json -> test.parquet
"""
import os
import sys
import argparse

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from scripts.data_process_kg.cwq_kgqa_agent_format import convert_kgqa_agent_to_verl


def main():
    parser = argparse.ArgumentParser(
        description="Batch convert kgqa_agent CWQ splits to VERL parquet format"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="kgqa_agent/src/eval/datasets",
        help="Directory containing cwq_*.json files"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data_kg/cwq_kgqa_agent_format",
        help="Output directory for parquet files"
    )
    parser.add_argument(
        "--max_calls",
        type=int,
        default=10,
        help="Maximum KG query calls (used in prompt)"
    )
    
    args = parser.parse_args()
    
    # Define file mappings: (input_file, split_name)
    file_mappings = [
        ("cwq_train.json", "train"),
        ("cwq_dev.json", "val"),  # dev -> val for VERL convention
        ("cwq_test.json", "test"),
    ]
    
    print("=" * 60)
    print("Batch Converting kgqa_agent CWQ splits to VERL format")
    print("=" * 60)
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Max calls: {args.max_calls}")
    print()
    
    # Process each file
    success_count = 0
    for input_file, split_name in file_mappings:
        input_path = os.path.join(args.input_dir, input_file)
        
        if not os.path.exists(input_path):
            print(f"⚠️  Skipping {input_file}: file not found at {input_path}")
            continue
        
        print(f"\n{'='*60}")
        print(f"Processing: {input_file} -> {split_name}.parquet")
        print(f"{'='*60}")
        
        try:
            convert_kgqa_agent_to_verl(
                input_json_path=input_path,
                output_dir=args.output_dir,
                split_name=split_name,
                max_calls=args.max_calls,
            )
            success_count += 1
            print(f"✓ Successfully converted {input_file} to {split_name}.parquet")
        except Exception as e:
            print(f"✗ Error processing {input_file}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*60}")
    print(f"Conversion Summary: {success_count}/{len(file_mappings)} files processed")
    print(f"{'='*60}")
    
    if success_count == len(file_mappings):
        print(f"\n✓ All splits converted successfully!")
        print(f"Output directory: {args.output_dir}")
        print(f"\nFiles created:")
        for _, split_name in file_mappings:
            output_file = os.path.join(args.output_dir, f"{split_name}.parquet")
            if os.path.exists(output_file):
                size_mb = os.path.getsize(output_file) / (1024 * 1024)
                print(f"  - {split_name}.parquet ({size_mb:.2f} MB)")
    else:
        print(f"\n⚠️  Some files failed to convert. Please check the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()


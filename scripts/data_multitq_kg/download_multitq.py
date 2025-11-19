#!/usr/bin/env python3
"""
Download and setup MultiTQ dataset for KG-R1 training pipeline

This script downloads the MultiTQ temporal KGQA dataset from GitHub and sets up
the data structure compatible with the KG-R1 training system.
"""

import os
import json
import subprocess
import requests
import zipfile
from pathlib import Path
from typing import Dict, List, Any, Optional
import argparse
import shutil
from tqdm import tqdm

# MultiTQ Repository URL
MULTITQ_REPO_URL = "https://github.com/czy1999/MultiTQ.git"

# Hugging Face alternative (if needed)
HUGGINGFACE_DATASET_URL = "https://huggingface.co/datasets/chenziyang/MultiTQ"

def clone_multitq_repo(target_dir: Path) -> bool:
    """Clone MultiTQ repository from GitHub"""
    print(f"📥 Cloning MultiTQ repository to {target_dir}")
    
    try:
        if target_dir.exists():
            print(f"   ⚠️ Directory {target_dir} already exists, removing...")
            shutil.rmtree(target_dir)
        
        # Clone the repository
        subprocess.run([
            "git", "clone", MULTITQ_REPO_URL, str(target_dir)
        ], check=True, capture_output=True, text=True)
        
        print(f"   ✅ Successfully cloned MultiTQ repository")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Git clone failed: {e}")
        print(f"   ❌ stderr: {e.stderr}")
        return False
    except Exception as e:
        print(f"   ❌ Unexpected error during cloning: {e}")
        return False

def extract_dataset_files(repo_dir: Path, output_dir: Path) -> bool:
    """Extract dataset files from the cloned repository"""
    print(f"📦 Extracting dataset files from repository")
    
    try:
        # Path to Dataset.zip in the cloned repo
        dataset_zip_path = repo_dir / "data" / "Dataset.zip"
        
        if not dataset_zip_path.exists():
            print(f"   ❌ Dataset.zip not found at {dataset_zip_path}")
            return False
        
        print(f"   📦 Extracting {dataset_zip_path}")
        
        # Extract Dataset.zip to temporary location
        temp_extract_dir = output_dir / "temp_extract"
        temp_extract_dir.mkdir(parents=True, exist_ok=True)
        
        with zipfile.ZipFile(dataset_zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_extract_dir)
        
        # Move extracted contents to proper location
        extracted_multitq_dir = temp_extract_dir / "MultiTQ"
        if extracted_multitq_dir.exists():
            final_data_dir = output_dir / "MultiTQ"
            if final_data_dir.exists():
                shutil.rmtree(final_data_dir)
            shutil.move(str(extracted_multitq_dir), str(final_data_dir))
        
        # Copy the MultiQA code directory as well
        multiqA_source = repo_dir / "MultiQA"
        multiqA_target = output_dir / "MultiQA"
        if multiqA_source.exists():
            if multiqA_target.exists():
                shutil.rmtree(multiqA_target)
            shutil.copytree(str(multiqA_source), str(multiqA_target))
            print(f"   ✅ Copied MultiQA code directory")
        
        # Copy README
        readme_source = repo_dir / "README.md"
        readme_target = output_dir / "README_ORIGINAL.md"
        if readme_source.exists():
            shutil.copy2(str(readme_source), str(readme_target))
        
        # Clean up temporary directory
        shutil.rmtree(temp_extract_dir)
        
        print(f"   ✅ Successfully extracted dataset files")
        return True
        
    except Exception as e:
        print(f"   ❌ Error extracting dataset files: {e}")
        return False

def verify_dataset_structure(output_dir: Path) -> bool:
    """Verify that the dataset has the expected structure"""
    print(f"🔍 Verifying dataset structure in {output_dir}")
    
    required_paths = [
        "MultiTQ/kg/full.txt",
        "MultiTQ/kg/train.txt", 
        "MultiTQ/kg/test.txt",
        "MultiTQ/questions",
        "MultiQA"
    ]
    
    missing_paths = []
    for path_str in required_paths:
        path = output_dir / path_str
        if not path.exists():
            missing_paths.append(path_str)
    
    if missing_paths:
        print(f"   ❌ Missing required paths:")
        for path in missing_paths:
            print(f"      - {path}")
        return False
    
    # Check for question files
    questions_dir = output_dir / "MultiTQ" / "questions"
    question_files = list(questions_dir.glob("*.json"))
    
    print(f"   ✅ Found {len(question_files)} question files:")
    for file in sorted(question_files):
        file_size = file.stat().st_size / 1024  # KB
        print(f"      - {file.name} ({file_size:.1f} KB)")
    
    # Check KG files
    kg_dir = output_dir / "MultiTQ" / "kg"
    kg_files = ["full.txt", "train.txt", "test.txt", "valid.txt"]
    
    print(f"   ✅ KG files status:")
    for kg_file in kg_files:
        kg_path = kg_dir / kg_file
        if kg_path.exists():
            file_size = kg_path.stat().st_size / (1024 * 1024)  # MB
            print(f"      - {kg_file} ({file_size:.1f} MB)")
        else:
            print(f"      - {kg_file} (missing)")
    
    print(f"   ✅ Dataset structure verification complete")
    return True

def analyze_dataset_content(output_dir: Path) -> Dict[str, Any]:
    """Analyze the content of the dataset to understand its structure"""
    print(f"📊 Analyzing dataset content")
    
    analysis = {}
    
    # Analyze questions
    questions_dir = output_dir / "MultiTQ" / "questions"
    if questions_dir.exists():
        question_files = list(questions_dir.glob("*.json"))
        analysis['question_files'] = []
        
        for qfile in question_files:
            try:
                with open(qfile, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                file_analysis = {
                    'filename': qfile.name,
                    'sample_count': len(data),
                    'sample_keys': list(data[0].keys()) if data else [],
                    'example_question': data[0].get('question', '') if data else ''
                }
                
                analysis['question_files'].append(file_analysis)
                print(f"   📄 {qfile.name}: {len(data)} samples")
                
            except Exception as e:
                print(f"   ❌ Error analyzing {qfile.name}: {e}")
    
    # Analyze KG structure
    kg_dir = output_dir / "MultiTQ" / "kg"
    if kg_dir.exists():
        analysis['kg_files'] = {}
        
        # Count lines in KG files
        for kg_file in ["full.txt", "train.txt", "test.txt", "valid.txt"]:
            kg_path = kg_dir / kg_file
            if kg_path.exists():
                try:
                    with open(kg_path, 'r', encoding='utf-8') as f:
                        line_count = sum(1 for _ in f)
                    analysis['kg_files'][kg_file] = line_count
                    print(f"   📊 {kg_file}: {line_count:,} temporal facts")
                except Exception as e:
                    print(f"   ❌ Error counting lines in {kg_file}: {e}")
    
    return analysis

def create_setup_summary(output_dir: Path, analysis: Dict[str, Any]):
    """Create a summary file with setup information"""
    summary_file = output_dir / "SETUP_SUMMARY.md"
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("# MultiTQ Dataset Setup Summary\n\n")
        f.write("## Dataset Overview\n")
        f.write("- **Source**: MultiTQ Temporal KGQA Dataset\n")
        f.write("- **Repository**: https://github.com/czy1999/MultiTQ\n")
        f.write("- **Paper**: ACL 2023\n")
        f.write("- **Task**: Multi-granularity temporal question answering\n\n")
        
        f.write("## Dataset Structure\n")
        f.write("```\n")
        f.write("MultiTQ/\n")
        f.write("├── kg/                    # Temporal Knowledge Graph\n")
        f.write("│   ├── full.txt          # Complete temporal facts\n")
        f.write("│   ├── train.txt         # Training temporal facts\n")
        f.write("│   ├── test.txt          # Test temporal facts\n")
        f.write("│   └── valid.txt         # Validation temporal facts\n")
        f.write("└── questions/            # Question-answer pairs\n")
        f.write("    ├── train.json        # Training questions\n")
        f.write("    ├── dev.json          # Development questions\n")
        f.write("    └── test.json         # Test questions\n")
        f.write("```\n\n")
        
        if 'question_files' in analysis:
            f.write("## Question Files Analysis\n")
            for qfile in analysis['question_files']:
                f.write(f"- **{qfile['filename']}**: {qfile['sample_count']:,} samples\n")
                if qfile['sample_keys']:
                    f.write(f"  - Keys: {', '.join(qfile['sample_keys'])}\n")
                if qfile['example_question']:
                    f.write(f"  - Example: \"{qfile['example_question'][:100]}...\"\n")
        
        if 'kg_files' in analysis:
            f.write("\n## Temporal KG Files Analysis\n")
            for kg_file, line_count in analysis['kg_files'].items():
                f.write(f"- **{kg_file}**: {line_count:,} temporal facts\n")
        
        f.write("\n## Next Steps\n")
        f.write("1. Run processing scripts from `/scripts/data_multitq_kg/`\n")
        f.write("2. Create search augmented training data\n")
        f.write("3. Test with KG-R1 training pipeline\n")
        f.write("4. Evaluate temporal reasoning capabilities\n")
    
    print(f"   ✅ Created setup summary: {summary_file}")

def main():
    parser = argparse.ArgumentParser(description="Download and setup MultiTQ dataset")
    parser.add_argument("--output_dir", type=str,
                      default="data_multitq_kg",
                      help="Output directory for MultiTQ dataset (relative to project root)")
    parser.add_argument("--temp_dir", type=str,
                      default="/tmp/multitq_download",
                      help="Temporary directory for cloning repository")
    parser.add_argument("--skip_download", action="store_true",
                      help="Skip download if data already exists")
    
    args = parser.parse_args()
    
    print("🚀 MultiTQ Dataset Download and Setup")
    print("=" * 50)
    
    output_dir = Path(args.output_dir)
    temp_dir = Path(args.temp_dir)
    
    print(f"📁 Output directory: {output_dir}")
    print(f"📁 Temporary directory: {temp_dir}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if data already exists
    multitq_data_dir = output_dir / "MultiTQ"
    if args.skip_download and multitq_data_dir.exists():
        print(f"⏭️ Skipping download, data already exists at {multitq_data_dir}")
    else:
        # Step 1: Clone repository
        print(f"\n📥 Step 1: Cloning MultiTQ repository...")
        if not clone_multitq_repo(temp_dir):
            print(f"❌ Failed to clone repository")
            return 1
        
        # Step 2: Extract dataset files
        print(f"\n📦 Step 2: Extracting dataset files...")
        if not extract_dataset_files(temp_dir, output_dir):
            print(f"❌ Failed to extract dataset files")
            return 1
    
    # Step 3: Verify structure
    print(f"\n🔍 Step 3: Verifying dataset structure...")
    if not verify_dataset_structure(output_dir):
        print(f"❌ Dataset structure verification failed")
        return 1
    
    # Step 4: Analyze content
    print(f"\n📊 Step 4: Analyzing dataset content...")
    analysis = analyze_dataset_content(output_dir)
    
    # Step 5: Create summary
    print(f"\n📝 Step 5: Creating setup summary...")
    create_setup_summary(output_dir, analysis)
    
    # Cleanup
    if temp_dir.exists() and not args.skip_download:
        print(f"\n🧹 Cleaning up temporary directory...")
        shutil.rmtree(temp_dir)
    
    print(f"\n🎉 MultiTQ dataset setup completed!")
    print(f"📁 Data saved to: {output_dir}")
    print(f"\n📋 Next steps:")
    print(f"1. cd scripts/data_multitq_kg")
    print(f"2. python process_multitq.py")
    print(f"3. python multitq_search_augmented_initial_entities.py")
    print(f"   Or simply run: bash setup_multitq.sh")
    
    return 0

if __name__ == "__main__":
    exit(main())
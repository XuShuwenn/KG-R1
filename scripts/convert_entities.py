#!/usr/bin/env python3
"""
Convert entities using existing JSON mapping file for both CWQ and WebQSP
"""

import json
import sys
import os

def convert_entities_with_json(input_file, output_file, json_mapping_file, dataset_name):
    """Convert entities using pre-built JSON mapping"""
    
    print(f"🔄 Converting {dataset_name} entities using JSON mapping")
    print(f"📁 Input: {input_file}")
    print(f"📁 Output: {output_file}")
    print(f"🗂️  Mapping: {json_mapping_file}")
    print()
    
    # Check if files exist
    if not os.path.exists(input_file):
        print(f"❌ Input file not found: {input_file}")
        return False
    
    if not os.path.exists(json_mapping_file):
        print(f"❌ JSON mapping file not found: {json_mapping_file}")
        return False
    
    # Load JSON mappings
    print("📂 Loading JSON mappings...")
    try:
        with open(json_mapping_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Handle different JSON structures
        if 'mappings' in data:
            mappings = data['mappings']
            metadata = data.get('metadata', {})
            print(f"✅ Loaded {len(mappings):,} mappings from structured JSON")
            if metadata:
                print(f"   📅 Created: {metadata.get('created_at', 'unknown')}")
        else:
            # Direct mapping format
            mappings = data
            print(f"✅ Loaded {len(mappings):,} mappings from direct JSON")
        
    except Exception as e:
        print(f"❌ Error loading JSON mappings: {e}")
        return False
    
    # Convert entities
    print(f"🔄 Converting {dataset_name} entities...")
    converted_count = 0
    total_count = 0
    
    try:
        with open(input_file, 'r', encoding='utf-8') as infile, \
             open(output_file, 'w', encoding='utf-8') as outfile:
            
            for line in infile:
                entity = line.strip()
                total_count += 1
                
                if entity in mappings:
                    name = mappings[entity]
                    outfile.write(f"{name}\n")
                    converted_count += 1
                    
                    # Show first few conversions
                    if converted_count <= 5:
                        print(f"   ✅ {entity} → {name}")
                    elif converted_count == 6:
                        print("   ... (showing first 5 conversions)")
                        
                else:
                    outfile.write(f"{entity}\n")  # Keep original MID
        
        print(f"\n✅ {dataset_name} conversion complete!")
        print(f"   📊 Total entities: {total_count:,}")
        print(f"   ✅ Converted to text: {converted_count:,} ({converted_count/total_count*100:.1f}%)")
        print(f"   🔤 Kept as MIDs: {total_count - converted_count:,}")
        print(f"   📁 Output saved to: {output_file}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during {dataset_name} conversion: {e}")
        return False

def main():
    # Get the directory of this script and construct relative paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)  # Go up one level from scripts/
    
    json_mapping_file = os.path.join(script_dir, "freebase_mid_gid_to_names.json")
    
    print("🚀 Entity Text Converter for CWQ and WebQSP")
    print("=" * 60)
    
    # Dataset configurations with relative paths
    datasets = [
        {
            "name": "CWQ",
            "input": os.path.join(project_root, "data_kg", "CWQ", "entities.txt"),
            "output": os.path.join(project_root, "data_kg", "CWQ", "entities_text.txt")
        },
        {
            "name": "WebQSP", 
            "input": os.path.join(project_root, "data_kg", "webqsp", "entities.txt"),
            "output": os.path.join(project_root, "data_kg", "webqsp", "entities_text.txt")
        }
    ]
    
    success_count = 0
    
    for dataset in datasets:
        print(f"\n{'='*20} {dataset['name']} {'='*20}")
        
        if convert_entities_with_json(
            dataset["input"], 
            dataset["output"], 
            json_mapping_file, 
            dataset["name"]
        ):
            success_count += 1
            
            # Show sample results
            print(f"\n📋 Sample {dataset['name']} converted entities:")
            try:
                with open(dataset["output"], 'r') as f:
                    for i, line in enumerate(f):
                        if i >= 3:
                            break
                        print(f"  {i+1}. {line.strip()}")
            except:
                pass
        else:
            print(f"\n❌ {dataset['name']} conversion failed!")
    
    print(f"\n{'='*60}")
    print(f"🎉 Conversion Summary:")
    print(f"   ✅ Successful: {success_count}/{len(datasets)} datasets")
    print(f"   📁 CWQ output: {os.path.join(project_root, 'data_kg', 'CWQ', 'entities_text.txt')}")
    print(f"   📁 WebQSP output: {os.path.join(project_root, 'data_kg', 'webqsp', 'entities_text.txt')}")
    
    if success_count == len(datasets):
        print("\n🎊 All conversions completed successfully!")
    else:
        print(f"\n⚠️  {len(datasets) - success_count} conversion(s) failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()

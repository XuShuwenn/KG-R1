#!/usr/bin/env python3
"""
MultiTQ dataset processor for KG-R1 training pipeline

This script processes the MultiTQ temporal KGQA dataset and converts it to the KG-R1 compatible format:
- Generates *_simple.json files (train/dev/test) following the KG-R1 standard
- Creates entities.txt, entities_text.txt, relations.txt from temporal KG
- Handles temporal quadruples: (entity, relation, entity, timestamp)
- Builds subgraphs for each question with relevant temporal facts
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Any, Set
import re
from collections import defaultdict

def load_temporal_kg_facts(kg_file: Path) -> List[Dict[str, str]]:
    """Load temporal facts from MultiTQ KG file (format: head\trelation\ttail\ttimestamp)"""
    print(f"📊 Loading temporal facts from {kg_file}")
    
    facts = []
    with open(kg_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if line:
                parts = line.split('\t')
                if len(parts) == 4:
                    head, relation, tail, timestamp = parts
                    facts.append({
                        'head': head.strip(),
                        'relation': relation.strip(), 
                        'tail': tail.strip(),
                        'timestamp': timestamp.strip()
                    })
                else:
                    print(f"   ⚠️ Line {line_num}: Expected 4 parts, got {len(parts)}")
    
    print(f"   ✅ Loaded {len(facts):,} temporal facts")
    return facts

def load_questions_data(questions_file: Path, max_samples: int = None) -> List[Dict]:
    """Load MultiTQ questions from JSON file"""
    print(f"📊 Loading questions from {questions_file}")
    
    with open(questions_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if max_samples and len(data) > max_samples:
        data = data[:max_samples]
        print(f"   ⚠️ Limited to {max_samples} samples (from original {len(data)})")
    
    print(f"   ✅ Loaded {len(data)} question samples")
    return data


def extract_entities_and_relations(temporal_facts: List[Dict]) -> Tuple[List[str], List[str]]:
    """Extract unique entities and relations from temporal facts"""
    print("🔄 Extracting entities and relations from temporal facts")
    
    entities = set()
    relations = set()
    
    for fact in temporal_facts:
        entities.add(fact['head'])
        entities.add(fact['tail'])
        relations.add(fact['relation'])
    
    sorted_entities = sorted(list(entities))
    sorted_relations = sorted(list(relations))
    
    print(f"   ✅ Found {len(sorted_entities):,} unique entities")
    print(f"   ✅ Found {len(sorted_relations):,} unique relations")
    
    return sorted_entities, sorted_relations

def find_initial_entities_in_question(question: str, all_entities: List[str], max_entities: int = 10) -> List[str]:
    """Find entities mentioned in question text using string matching"""
    question_lower = question.lower()
    initial_entities = []
    
    # Score entities by how well they match the question
    entity_scores = []
    
    for entity in all_entities:
        # Convert entity name for matching (e.g., "Jack_Straw" -> "jack straw")
        entity_text = entity.replace('_', ' ').lower()
        
        if len(entity_text) < 3:  # Skip very short entities
            continue
            
        # Exact match gets highest score
        if entity_text in question_lower:
            score = len(entity_text) * 2  # Longer matches are better
            entity_scores.append((entity, score))
        # Partial word match
        elif any(word in question_lower for word in entity_text.split()):
            score = len(entity_text)
            entity_scores.append((entity, score))
    
    # Sort by score and take top entities
    entity_scores.sort(key=lambda x: x[1], reverse=True)
    initial_entities = [entity for entity, _ in entity_scores[:max_entities]]
    
    return initial_entities

def build_question_subgraph(initial_entities: List[str], temporal_facts: List[Dict], 
                          all_entities: List[str], all_relations: List[str],
                          max_hops: int = 2, max_facts: int = 500) -> Dict:
    """Build a subgraph around initial entities using temporal facts"""
    
    # Create mappings
    entity_to_idx = {entity: idx for idx, entity in enumerate(all_entities)}
    relation_to_idx = {relation: idx for idx, relation in enumerate(all_relations)}
    
    # Start with initial entities
    relevant_entities = set(initial_entities)
    relevant_relations = set()
    relevant_facts = []
    
    # Expand by hops
    for hop in range(max_hops):
        new_entities = set()
        hop_facts = []
        
        for fact in temporal_facts:
            # Include fact if it involves any relevant entity
            if fact['head'] in relevant_entities or fact['tail'] in relevant_entities:
                hop_facts.append(fact)
                new_entities.add(fact['head'])
                new_entities.add(fact['tail'])
                relevant_relations.add(fact['relation'])
                
                if len(hop_facts) >= max_facts:
                    break
        
        relevant_facts.extend(hop_facts)
        relevant_entities.update(new_entities)
        
        print(f"     Hop {hop + 1}: Added {len(hop_facts)} facts, {len(new_entities)} entities")
        
        if len(hop_facts) >= max_facts:
            break
    
    # Convert to indices for KG-R1 format
    subgraph_entities = []
    subgraph_relations = []
    subgraph_tuples = []
    
    # Map entities to subgraph indices
    relevant_entities_list = sorted(list(relevant_entities))
    subgraph_entity_to_idx = {entity: idx for idx, entity in enumerate(relevant_entities_list)}
    
    for entity in relevant_entities_list:
        if entity in entity_to_idx:
            subgraph_entities.append(entity_to_idx[entity])
    
    # Map relations to subgraph indices  
    relevant_relations_list = sorted(list(relevant_relations))
    subgraph_relation_to_idx = {relation: idx for idx, relation in enumerate(relevant_relations_list)}
    
    for relation in relevant_relations_list:
        if relation in relation_to_idx:
            subgraph_relations.append(relation_to_idx[relation])
    
    # Create tuples with subgraph indices
    for fact in relevant_facts:
        if (fact['head'] in subgraph_entity_to_idx and 
            fact['tail'] in subgraph_entity_to_idx and
            fact['relation'] in subgraph_relation_to_idx):
            
            head_subgraph_idx = subgraph_entity_to_idx[fact['head']]
            tail_subgraph_idx = subgraph_entity_to_idx[fact['tail']]
            relation_subgraph_idx = subgraph_relation_to_idx[fact['relation']]
            
            subgraph_tuples.append([head_subgraph_idx, relation_subgraph_idx, tail_subgraph_idx])
    
    # Map initial entities to subgraph indices
    initial_entity_indices = []
    for entity in initial_entities:
        if entity in subgraph_entity_to_idx:
            initial_entity_indices.append(subgraph_entity_to_idx[entity])
    
    subgraph = {
        'entities': subgraph_entities,
        'relations': subgraph_relations,
        'tuples': subgraph_tuples,
        'temporal_facts': relevant_facts  # Keep for temporal reasoning
    }
    
    return subgraph, initial_entity_indices

def process_multitq_sample(sample: Dict, sample_id: str, temporal_facts: List[Dict],
                          all_entities: List[str], all_relations: List[str]) -> Dict:
    """Convert MultiTQ sample to KG-R1 compatible format"""
    try:
        # Extract question
        question = sample.get('question', '').strip()
        if not question:
            return None
        
        # Process answers
        answers = []
        if 'answers' in sample:
            for ans in sample['answers']:
                if isinstance(ans, str):
                    # Convert answer to proper format
                    kb_id = ans.replace(' ', '_')  # Convert spaces to underscores for entity format
                    answers.append({
                        'text': ans,
                        'kb_id': kb_id
                    })
                elif isinstance(ans, dict):
                    answers.append({
                        'text': ans.get('text', str(ans)),
                        'kb_id': ans.get('kb_id', str(ans))
                    })
        
        # Find initial entities mentioned in the question
        print(f"   🔍 Finding entities in question: \"{question[:60]}...\"")
        initial_entities = find_initial_entities_in_question(question, all_entities)
        print(f"     Found {len(initial_entities)} initial entities: {initial_entities[:3]}...")
        
        # Build subgraph around initial entities
        print(f"   🕸️ Building subgraph for {len(initial_entities)} initial entities")
        subgraph, initial_entity_indices = build_question_subgraph(
            initial_entities, temporal_facts, all_entities, all_relations
        )
        
        # Create KG-R1 compatible sample
        processed_sample = {
            'id': sample_id,
            'question': question,
            'answers': answers,
            'entities': initial_entity_indices,  # Initial entities from question
            'subgraph': {
                'entities': subgraph['entities'],  # Global entity indices
                'relations': subgraph['relations'],  # Global relation indices  
                'tuples': subgraph['tuples']  # [subject_idx, relation_idx, object_idx]
            }
        }
        
        # Add temporal metadata if available
        if 'time_level' in sample or 'qtype' in sample:
            processed_sample['temporal_metadata'] = {
                'time_level': sample.get('time_level', 'unknown'),
                'qtype': sample.get('qtype', 'unknown')
            }
        
        print(f"   ✅ Created sample with {len(subgraph['tuples'])} tuples, {len(initial_entity_indices)} initial entities")
        return processed_sample
        
    except Exception as e:
        print(f"   ❌ Error processing sample {sample_id}: {e}")
        return None

def save_entities_and_relations(all_entities: List[str], all_relations: List[str], output_dir: Path):
    """Save entities.txt, entities_text.txt, and relations.txt files"""
    print(f"💾 Saving entity and relation files")
    
    # Save entities.txt (entity IDs)
    entities_file = output_dir / 'entities.txt'
    with open(entities_file, 'w', encoding='utf-8') as f:
        for entity in all_entities:
            f.write(f"{entity}\n")
    
    # Save entities_text.txt (human-readable names)
    entities_text_file = output_dir / 'entities_text.txt'
    with open(entities_text_file, 'w', encoding='utf-8') as f:
        for entity in all_entities:
            # Convert "Jack_Straw" -> "Jack Straw" 
            human_readable = entity.replace('_', ' ')
            f.write(f"{human_readable}\n")
    
    # Save relations.txt
    relations_file = output_dir / 'relations.txt'
    with open(relations_file, 'w', encoding='utf-8') as f:
        for relation in all_relations:
            f.write(f"{relation}\n")
    
    print(f"   ✅ Saved entities.txt ({len(all_entities):,} entities)")
    print(f"   ✅ Saved entities_text.txt ({len(all_entities):,} entity names)")
    print(f"   ✅ Saved relations.txt ({len(all_relations):,} relations)")

def save_split_data(processed_data: List[Dict], split_name: str, output_dir: Path):
    """Save processed data as JSONL file"""
    output_file = output_dir / f"{split_name}_simple.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in processed_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"   ✅ Saved {output_file} ({len(processed_data)} samples)")

def main():
    parser = argparse.ArgumentParser(description="Process MultiTQ dataset for KG-R1")
    parser.add_argument("--input_dir", type=str,
                      default="data_multitq_kg/MultiTQ",
                      help="Directory containing MultiTQ data (relative to project root)")
    parser.add_argument("--output_dir", type=str,
                      default="data_kg/multitq",
                      help="Output directory for processed KG-R1 compatible data (relative to project root)")
    parser.add_argument("--max_samples", type=int, default=None,
                      help="Maximum samples per split (for testing)")
    parser.add_argument("--kg_file", type=str, default="full.txt",
                      help="KG file to use (full.txt, train.txt, etc.)")
    
    args = parser.parse_args()
    
    print("🚀 Processing MultiTQ dataset for KG-R1")
    print("=" * 60)
    
    # Setup paths
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    kg_file = input_dir / "kg" / args.kg_file
    questions_dir = input_dir / "questions"
    
    print(f"📁 Input directory: {input_dir}")
    print(f"📁 Questions directory: {questions_dir}")
    print(f"📁 KG file: {kg_file}")
    print(f"📁 Output directory: {output_dir}")
    if args.max_samples:
        print(f"🔢 Max samples per split: {args.max_samples}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Load temporal knowledge graph
    print(f"\n🔍 Step 1: Loading temporal knowledge graph...")
    if not kg_file.exists():
        print(f"❌ KG file not found: {kg_file}")
        return 1
    
    temporal_facts = load_temporal_kg_facts(kg_file)
    
    # Step 2: Extract entities and relations
    print(f"\n🔄 Step 2: Extracting entities and relations...")
    all_entities, all_relations = extract_entities_and_relations(temporal_facts)
    
    # Step 3: Process each question split
    splits = ['train', 'dev', 'test']
    
    for split in splits:
        print(f"\n📊 Step 3.{splits.index(split) + 1}: Processing {split} split...")
        
        questions_file = questions_dir / f"{split}.json"
        if not questions_file.exists():
            print(f"   ⚠️ Questions file not found: {questions_file}, skipping {split}")
            continue
        
        # Load questions
        questions_data = load_questions_data(questions_file, args.max_samples)
        
        
        # Process samples
        processed_data = []
        print(f"   🔄 Processing {len(questions_data)} {split} samples...")
        
        for i, sample in enumerate(questions_data):
            if i % 50 == 0:
                print(f"     Processing sample {i + 1}/{len(questions_data)}")
                
            sample_id = f"multitq_{split}_{i}"
            processed_sample = process_multitq_sample(
                sample, sample_id, temporal_facts, all_entities, all_relations
            )
            
            if processed_sample:
                processed_data.append(processed_sample)
        
        # Save processed split
        save_split_data(processed_data, split, output_dir)
        print(f"   ✅ Processed {len(processed_data)}/{len(questions_data)} {split} samples")
    
    # Step 4: Save entity and relation files
    print(f"\n💾 Step 4: Saving entity and relation mappings...")
    save_entities_and_relations(all_entities, all_relations, output_dir)
    
    # Step 5: Summary
    print(f"\n🎉 MultiTQ processing completed!")
    print(f"📁 Output saved to: {output_dir}")
    
    print(f"\n📊 Generated files:")
    for split in ['train', 'dev', 'test']:
        split_file = output_dir / f"{split}_simple.json"
        if split_file.exists():
            print(f"   📄 {split}_simple.json")
    
    print(f"   📄 entities.txt ({len(all_entities):,} entities)")
    print(f"   📄 entities_text.txt ({len(all_entities):,} entity names)")  
    print(f"   📄 relations.txt ({len(all_relations):,} relations)")
    
    print(f"\n📋 Dataset characteristics:")
    print(f"   🕰️ Temporal facts: {len(temporal_facts):,}")
    print(f"   🏷️ Unique entities: {len(all_entities):,}")
    print(f"   🔗 Unique relations: {len(all_relations):,}")
    
    print(f"\n📋 Next steps:")
    print(f"   1. Run: python multitq_search_augmented_initial_entities.py")
    print(f"   2. Test data loading with KG-R1 training pipeline")
    print(f"   3. Evaluate temporal reasoning performance")
    
    return 0

if __name__ == "__main__":
    exit(main())
# MultiTQ Dataset Integration Scripts

This directory contains scripts for downloading, processing, and integrating the MultiTQ temporal KGQA dataset with the KG-R1 training system.

## Overview

**MultiTQ** is a large-scale dataset for multi-granularity temporal question answering over knowledge graphs. It features temporal facts and questions across different time scales (day/month/year level) for complex temporal reasoning.

- **Source**: [MultiTQ GitHub Repository](https://github.com/czy1999/MultiTQ)
- **Paper**: ACL 2023
- **Task**: Temporal Knowledge Graph Question Answering
- **Scale**: Large-scale temporal facts with multi-hop reasoning

## Scripts Overview

### Core Scripts

1. **`download_multitq.py`** - Downloads MultiTQ dataset from GitHub
2. **`process_multitq.py`** - Converts raw data to KG-R1 compatible format
3. **`multitq_search_augmented_initial_entities.py`** - Creates training-ready data
4. **`setup_multitq.sh`** - Complete automated setup script

### File Structure

```
scripts/data_multitq_kg/
├── README.md                                      # This file
├── download_multitq.py                           # Download script
├── process_multitq.py                            # Data processing script  
├── multitq_search_augmented_initial_entities.py  # Training data generator
└── setup_multitq.sh                              # Complete setup script

# Generated data structure:
data_multitq_kg/                      # Raw downloaded data
├── MultiTQ/                          # Downloaded MultiTQ dataset
├── MultiQA/                          # MultiTQ code directory
└── SETUP_SUMMARY.md                  # Download summary

data_kg/multitq/                      # Processed KG-R1 compatible data
├── *_simple.json                     # Processed questions
├── entities.txt                      # Entity IDs
├── entities_text.txt                 # Human-readable entity names
└── relations.txt                     # Relations

data_kg/multitq_search_augmented_initial_entities/  # Training data
├── train.parquet                     # Training samples
├── dev.parquet                       # Development samples  
├── test.parquet                      # Test samples
└── DATASET_SUMMARY.md                # Training data summary
```

## Quick Setup

Run the complete setup with a single command:

```bash
cd scripts/data_multitq_kg
bash setup_multitq.sh
```

This will:
1. Download MultiTQ dataset from GitHub → `/data_multitq_kg/`
2. Process data into KG-R1 compatible format → `/data_kg/multitq/`
3. Create search-augmented training data → `/data_kg/multitq_search_augmented_initial_entities/`
4. Generate summary reports

## Manual Setup Steps

### Step 1: Download Dataset

```bash
python download_multitq.py --output_dir data_multitq_kg
```

Downloads and extracts MultiTQ dataset from GitHub repository to the data directory.

### Step 2: Process for KG-R1 Compatibility

```bash
python process_multitq.py \
    --input_dir data_multitq_kg/MultiTQ \
    --output_dir data_kg/multitq
```

Converts raw MultiTQ data to KG-R1 standardized format:
- Creates `*_simple.json` files with proper entity indexing
- Generates `entities.txt`, `entities_text.txt`, `relations.txt`
- Builds subgraphs for each question with temporal facts

### Step 3: Create Training Data

```bash
python multitq_search_augmented_initial_entities.py \
    --input_dir data_kg/multitq \
    --output_dir data_kg/multitq_search_augmented_initial_entities
```

Generates training-ready `.parquet` files with:
- KG-augmented prompts with temporal reasoning instructions
- Initial entity hints for better context
- Temporal metadata for multi-granularity questions

## Script Parameters

### download_multitq.py
- `--output_dir`: Output directory for downloaded data (default: `/data_multitq_kg`)
- `--temp_dir`: Temporary directory for cloning (default: `/tmp/multitq_download`)
- `--skip_download`: Skip download if data already exists

### process_multitq.py
- `--input_dir`: MultiTQ data directory (default: `/data_multitq_kg/MultiTQ`)
- `--output_dir`: Processed data output (default: `/data_kg/multitq`)
- `--max_samples`: Limit samples per split (for testing)
- `--kg_file`: KG file to use (default: `full.txt`)

### multitq_search_augmented_initial_entities.py
- `--input_dir`: Processed data directory (default: `/data_kg/multitq`)
- `--output_dir`: Training data output (default: `/data_kg/multitq_search_augmented_initial_entities`)

## Data Processing Pipeline

### Raw Data → KG-R1 Format
1. **Entity Extraction**: Extract entities and relations from temporal KG
2. **Question Processing**: Parse temporal questions and answers
3. **Subgraph Building**: Create relevant subgraphs for each question
4. **Index Mapping**: Convert entities to consistent global indices
5. **Format Standardization**: Follow KG-R1 schema with temporal extensions

### KG-R1 Format → Training Data
1. **Entity Mapping**: Map indices to human-readable text
2. **Prompt Generation**: Create temporal reasoning prompts
3. **Initial Entity Augmentation**: Add entity hints to questions
4. **Training Format**: Convert to VERL-compatible parquet files

## Temporal Reasoning Features

### Multi-granularity Questions
- **Day-level**: "Who condemned Abhisit Vejjajiva in May 2010?"
- **Month-level**: "When did the Aam Aadmi Party first negotiate with Harish Rawat?"
- **Year-level**: "Who was the first to visit the Middle East in 2008?"

### Temporal KG Structure
- **Temporal Facts**: Knowledge graph quadruples with timestamps
- **Time-sensitive Relations**: Predicates that change over time
- **Temporal Constraints**: Questions requiring specific time reasoning

## Integration with KG-R1 Training

### Training Configuration
Add to your training script configuration:

```bash
# Use MultiTQ for temporal reasoning training
data.train_files="data_kg/multitq_search_augmented_initial_entities/train.parquet"
data.val_files="data_kg/multitq_search_augmented_initial_entities/test.parquet"

# Temporal reasoning parameters
data.prompt_augmentation.enable=true
data.prompt_augmentation.guideline_level=temporal_detailed
data.max_response_length=256
```

### Expected Output Structure
After running all scripts:

```
KG-R1/
├── data_multitq_kg/                  # Raw downloaded data
├── data_kg/multitq/                  # Processed KG-R1 format
└── data_kg/multitq_search_augmented_initial_entities/  # Training data
```

## Troubleshooting

### Common Issues
1. **Download fails**: Check internet connection and GitHub access
2. **Processing errors**: Ensure sufficient memory for large temporal KG
3. **Missing files**: Verify MultiTQ dataset structure after download
4. **Entity mapping issues**: Check entities.txt and entities_text.txt alignment

### Debug Mode
Run scripts with limited samples for testing:
```bash
python process_multitq.py --max_samples 100
```

### Memory Requirements
- **Download**: ~1GB disk space
- **Processing**: ~4GB RAM for full dataset
- **Training data**: ~500MB additional disk space

## Contributing

To extend MultiTQ integration:
1. **Add temporal query types**: Extend KG server with temporal operations
2. **Improve entity linking**: Better temporal entity resolution
3. **Add time normalization**: Standardize temporal expressions
4. **Create eval metrics**: Temporal-specific evaluation measures

## References

- **MultiTQ Paper**: [ACL 2023 Paper](https://github.com/czy1999/MultiTQ)
- **KG-R1 Training**: See main RL_KG repository documentation
- **Temporal KGQA**: Research on temporal knowledge graph reasoning
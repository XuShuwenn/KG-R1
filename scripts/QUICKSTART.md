# Data KG Setup - Quick Start Guide

## 🚀 Getting Started in 3 Steps

### Step 1: Run the Setup Script
```bash
cd /home/yeopjin/orcd/pool/workspace/KG-R1
bash scripts/setup_data_kg.sh
```

### Step 2: Select Your Dataset
Choose from the interactive menu:
- **Option 1**: ComplexWebQuestions (CWQ) - Multi-hop reasoning
- **Option 2**: WebQuestionsSP (WebQSP) - Single/simple multi-hop
- **Option 3**: MultiTQ - Temporal reasoning (includes download!)
- **Option 4**: Download Freebase KG only
- **Option 5**: Setup all datasets

### Step 3: Start Training!
```bash
# Example: Train with CWQ
python -m verl.trainer.main_ppo \
    mode=kg-search \
    data.train_files="data_kg/cwq_search_augmented_initial_entities/train.parquet" \
    data.val_files="data_kg/cwq_search_augmented_initial_entities/test.parquet" \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-3B-Instruct
```

---

## 📋 Alternative: Manual Setup

### For MultiTQ (easiest - no pre-download needed)
```bash
cd /home/yeopjin/orcd/pool/workspace/KG-R1/scripts/data_multitq_kg
bash setup_multitq.sh
```

### For CWQ (requires raw data)
```bash
cd /home/yeopjin/orcd/pool/workspace/KG-R1
python scripts/data_process_kg/cwq.py
python scripts/data_process_kg/cwq_search_augmented_initial_entities.py
```

### For WebQSP (requires raw data)
```bash
cd /home/yeopjin/orcd/pool/workspace/KG-R1
python scripts/data_process_kg/webqsp.py
python scripts/data_process_kg/webqsp_search_augmented_initial_entities.py
```

---

## 🎯 Expected Output Locations

After setup, you'll have training data here:

```
KG-R1/
├── data_kg/
│   ├── cwq_search_augmented_initial_entities/
│   │   ├── train.parquet
│   │   └── test.parquet
│   │
│   ├── webqsp_search_augmented_initial_entities/
│   │   ├── train.parquet
│   │   └── test.parquet
│   │
│   └── multitq_search_augmented_initial_entities/
│       ├── train.parquet
│       ├── dev.parquet
│       └── test.parquet
```

---

## ❓ Common Questions

**Q: Which dataset should I start with?**
A: MultiTQ is easiest (auto-downloads). For more complex reasoning, try CWQ.

**Q: Where do I get CWQ/WebQSP raw data?**
A: You need to download these separately and place in `data_kg/CWQ/` or `data_kg/webqsp/`

**Q: Do I need Freebase KG data?**
A: Yes, but the setup scripts handle this automatically.

**Q: Can I test with a small dataset first?**
A: Yes! Use `--max_samples 100` parameter when running the processing scripts.

---

## 📚 More Information

- **Full Documentation**: [scripts/README.md](README.md)
- **Patch Summary**: [../SCRIPTS_PATCH_SUMMARY.md](../SCRIPTS_PATCH_SUMMARY.md)
- **MultiTQ Details**: [data_multitq_kg/README.md](data_multitq_kg/README.md)
- **WebQSP Details**: [webqsp_kg/README.md](webqsp_kg/README.md)

---

## 🐛 Troubleshooting

**Setup script fails?**
```bash
# Check dependencies
python3 --version  # Should be 3.7+
git --version
wget --version

# Install missing dependencies
pip install pandas pyarrow requests
```

**Path errors?**
- Always run scripts from the KG-R1 project root
- Or use the provided setup scripts which handle paths

**Can't find data files?**
```bash
# Check what was created
ls -la data_kg/
tree data_kg/ -L 2
```

---

## 🎉 Success!

Once setup is complete, you should see:
```
✅ Dataset processing completed
✅ Training data created
📋 Available files for KG-R1 training
```

Now you're ready to train! See the main README for training examples.

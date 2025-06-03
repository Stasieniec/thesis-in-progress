# Google Colab Setup Guide

This guide shows you how to clone and run the ABIDE Multimodal Transformer framework in Google Colab.

## 🚀 Quick Start

### 1. Clone Repository and Install Dependencies

```python
# Clone the repository
!git clone https://github.com/your-username/thesis-in-progress.git
%cd thesis-in-progress

# Install dependencies
!pip install -r requirements.txt

# Mount Google Drive (for data access)
from google.colab import drive
drive.mount('/content/drive')
```

### 2. Verify Installation

```python
# Test that everything works
!python scripts/test_imports.py

# Optional: View configuration templates
!python scripts/train_fmri.py get_config_template
!python scripts/train_smri.py get_config_template
!python scripts/train_cross_attention.py get_config_template
```

You should see all green checkmarks (✅) indicating successful imports.

**Note for Local Testing**: If you're testing locally (not in Colab), use `python3` instead of `python`:
```bash
python3 scripts/test_imports.py
python3 scripts/train_fmri.py get_config_template
```

## 📊 Running Experiments

### fMRI-only Experiments

```python
# Basic fMRI experiment
!python scripts/train_fmri.py run

# Custom parameters
!python scripts/train_fmri.py run --num_folds=5 --batch_size=128 --num_epochs=100

# Quick test for debugging
!python scripts/train_fmri.py quick_test
```

### sMRI-only Experiments

```python
# Basic sMRI experiment  
!python scripts/train_smri.py run

# With feature selection
!python scripts/train_smri.py run --feature_selection_k=500 --num_epochs=50

# Analyze features only
!python scripts/train_smri.py analyze_features_only --top_k=50
```

### Cross-Attention Multimodal Experiments

```python
# Basic multimodal experiment
!python scripts/train_cross_attention.py run

# Custom architecture
!python scripts/train_cross_attention.py run --d_model=512 --num_cross_layers=4

# Check data overlap first
!python scripts/train_cross_attention.py analyze_data_overlap
```

## 🔧 Advanced Usage

### Custom Configuration

```python
# You can override any configuration parameter
!python scripts/train_fmri.py run \
  --num_folds=10 \
  --batch_size=256 \
  --learning_rate=1e-4 \
  --num_epochs=750 \
  --d_model=256 \
  --num_layers=4 \
  --dropout=0.1 \
  --output_dir="/content/drive/MyDrive/my_results"
```

### Interactive Usage in Cells

If you prefer to run code interactively in Colab cells:

```python
# Add the src directory to Python path
import sys
sys.path.insert(0, '/content/thesis-in-progress/src')

# Import the experiment classes
from scripts.train_fmri import FMRIExperiment
from scripts.train_smri import SMRIExperiment
from scripts.train_cross_attention import CrossAttentionExperiment

# Create experiment instance
fmri_exp = FMRIExperiment()

# Run with custom parameters
results = fmri_exp.run(
    num_folds=5,
    batch_size=128,
    num_epochs=100,
    verbose=True
)
```

## 📁 Data Organization

Make sure your data is organized as follows in Google Drive:

```
/content/drive/MyDrive/
├── b_data/
│   └── ABIDE_pcp/
│       ├── cpac/
│       │   └── filt_noglobal/
│       │       └── rois_cc200/
│       │           ├── 0050002_rois_cc200.1D
│       │           ├── 0050003_rois_cc200.1D
│       │           └── ...
│       └── Phenotypic_V1_0b_preprocessed1.csv
└── processed_smri_data/
    ├── features.npy
    ├── labels.npy
    ├── subject_ids.npy
    ├── feature_names.txt
    └── metadata.json
```

## ⚡ Performance Tips for Colab

### GPU Optimization

```python
# Check GPU availability and memory
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Enable GPU in Colab: Runtime > Change runtime type > Hardware accelerator > GPU
```

### Memory Management

```python
# For large experiments, use smaller batch sizes
!python scripts/train_cross_attention.py run --batch_size=16

# Clear GPU memory between runs
import torch
torch.cuda.empty_cache()
```

### Efficient Debugging

```python
# Use quick_test for debugging with minimal resources
!python scripts/train_fmri.py quick_test

# Or run with very few epochs
!python scripts/train_smri.py run --num_epochs=5 --num_folds=2
```

## 📈 Monitoring and Results

### Real-time Monitoring

```python
# The scripts provide progress bars and regular updates
# Look for outputs like:
# 🔧 Using device: cuda
# 📊 Loading fMRI data...
# 🔄 Starting 5-fold cross-validation...
# FOLD 1/5
# Training: 100%|██████████| 123/123 [02:34<00:00, 1.26s/it]
```

### Accessing Results

```python
# Results are automatically saved to the output directory
# Check the output directory structure:
!ls -la /content/drive/MyDrive/fmri_outputs_*/

# View results JSON
import json
with open('/content/drive/MyDrive/fmri_outputs_*/fmri_sat_results.json', 'r') as f:
    results = json.load(f)
    print(f"Mean accuracy: {results['summary']['accuracy']['mean']:.4f}")
```

### Downloading Results

```python
# Create a zip file of results for download
!zip -r experiment_results.zip /content/drive/MyDrive/fmri_outputs_*
from google.colab import files
files.download('experiment_results.zip')
```

## 🔄 Example Complete Workflow

Here's a complete example workflow for running all three experiments:

```python
# Setup
!git clone https://github.com/your-username/thesis-in-progress.git
%cd thesis-in-progress
!pip install -r requirements.txt

from google.colab import drive
drive.mount('/content/drive')

# 1. Run fMRI experiment
print("="*50)
print("RUNNING fMRI EXPERIMENT")
print("="*50)
!python scripts/train_fmri.py run --num_folds=5 --num_epochs=100

# 2. Run sMRI experiment  
print("="*50)
print("RUNNING sMRI EXPERIMENT")
print("="*50)
!python scripts/train_smri.py run --num_folds=5 --num_epochs=100

# 3. Run cross-attention experiment
print("="*50)
print("RUNNING CROSS-ATTENTION EXPERIMENT")
print("="*50)
!python scripts/train_cross_attention.py run --num_folds=5 --num_epochs=100

print("="*50)
print("ALL EXPERIMENTS COMPLETED!")
print("="*50)
```

## 🐛 Troubleshooting

### Common Issues

**1. Import errors:**
```python
# Make sure you're in the right directory
%cd /content/thesis-in-progress

# Test imports
!python scripts/test_imports.py
```

**2. Data not found:**
```python
# Verify data paths
!ls /content/drive/MyDrive/b_data/ABIDE_pcp/
!ls /content/drive/MyDrive/processed_smri_data/
```

**3. CUDA out of memory:**
```python
# Reduce batch size
!python scripts/train_cross_attention.py run --batch_size=8

# Clear GPU memory
import torch
torch.cuda.empty_cache()
```

**4. Long runtime limits:**
```python
# For very long experiments, consider Colab Pro
# Or run with fewer folds/epochs
!python scripts/train_fmri.py run --num_folds=3 --num_epochs=50
```

### Getting Help

```python
# Get help for any script
!python scripts/train_fmri.py --help

# Test imports
!python scripts/test_imports.py
```

## 📝 Notes

- **Colab Pro**: Recommended for faster GPUs and longer runtimes
- **Runtime limits**: Free Colab has ~12 hour limits
- **GPU memory**: T4 has ~15GB, adjust batch sizes accordingly
- **Data persistence**: Results are saved to Google Drive automatically
- **Reproducibility**: All experiments use fixed random seeds

## 🎯 Quick Commands Reference

```bash
# Test setup
python scripts/test_imports.py

# View configuration templates
python scripts/train_fmri.py get_config_template
python scripts/train_smri.py get_config_template
python scripts/train_cross_attention.py get_config_template

# Basic runs
python scripts/train_fmri.py run
python scripts/train_smri.py run  
python scripts/train_cross_attention.py run

# Quick tests
python scripts/train_fmri.py quick_test
python scripts/train_smri.py quick_test
python scripts/train_cross_attention.py quick_test

# Data analysis
python scripts/train_smri.py analyze_features_only
python scripts/train_cross_attention.py analyze_data_overlap
```

Happy experimenting! 🚀
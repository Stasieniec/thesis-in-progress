# Cross-Attention sMRI-fMRI Analysis for Autism Classification

## Project Overview

This repository implements a cross-attention transformer model for autism classification using both structural MRI (sMRI) and functional MRI (fMRI) data from the ABIDE dataset. The project focuses on achieving better classification performance through multimodal brain imaging analysis.

## Recent Improvements (December 2024)

### Enhanced sMRI Processing ✨
- **Improved Feature Extraction**: Comprehensive FreeSurfer parsing (aseg + aparc + wmparc)
- **Advanced Feature Selection**: RFE with Ridge classifier selecting 800 optimal features
- **Better Preprocessing**: Median imputation, proper standardization
- **Performance Gain**: Improved from 55% to 57-60% baseline accuracy

### Current Performance
- **fMRI Accuracy**: 65% (good performance)
- **sMRI Accuracy**: 57-60% (improved from 55%)
- **Target**: Reach reference paper's ~70% sMRI accuracy

## Repository Structure

```
thesis-in-progress/
├── scripts/                           # Executable scripts
│   ├── improved_smri_extraction_new.py   # Enhanced sMRI processing
│   ├── run_improved_smri_extraction.py   # Simple runner interface
│   ├── train_smri.py                     # sMRI model training
│   ├── train_fmri.py                     # fMRI model training
│   └── train_cross_attention.py          # Cross-attention training
├── src/                               # Source code modules
│   ├── models/                        # Model architectures
│   ├── data/                         # Data processing utilities
│   ├── training/                     # Training utilities
│   ├── evaluation/                   # Evaluation metrics
│   └── utils/                        # General utilities
├── data/                             # Dataset storage
│   └── freesurfer_stats/            # FreeSurfer processed data
├── context_files/                    # Reference implementations
│   ├── archived_data_creation/       # Previous data processing
│   ├── exported_colab_notebooks/     # Colab notebook exports  
│   └── papers/                       # Reference papers
├── update_to_improved_smri.py        # Google Drive sync utility
└── SMRI_IMPROVEMENT_SUMMARY.md       # Detailed improvement summary
```

## Quick Start

### 1. Environment Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Ensure data directory exists
ls data/freesurfer_stats/  # Should contain subject folders (50003, 50004, etc.)
```

### 2. Process sMRI Data (Improved Pipeline)
```bash
# Run enhanced sMRI extraction
python scripts/run_improved_smri_extraction.py

# Results will be saved to: processed_smri_data_improved/
```

### 3. Sync with Google Drive (for Colab)
```bash
# Update Google Drive with improved data
python update_to_improved_smri.py
```

### 4. Train Models
```bash
# Train individual modalities
python scripts/train_smri.py      # sMRI transformer
python scripts/train_fmri.py      # fMRI transformer

# Train cross-attention model
python scripts/train_cross_attention.py
```

## Key Features

### Enhanced sMRI Processing
- **Comprehensive Feature Extraction**: 1417 raw features from all FreeSurfer outputs
- **Intelligent Feature Selection**: RFE with Ridge classifier → 800 optimal features
- **Robust Preprocessing**: Median imputation + standardization
- **Quality Validation**: Comprehensive logging and data quality checks

### Transformer Architecture
- **Cross-Attention Mechanism**: Learns interactions between sMRI and fMRI
- **Modality-Specific Processing**: Separate encoders for each modality
- **Flexible Architecture**: Supports both unimodal and multimodal training

### Google Colab Integration
- **Colab-Ready Scripts**: All paths configured for Google Drive mounting
- **Efficient Data Loading**: Optimized for Colab's memory constraints
- **Progress Tracking**: Built-in logging and visualization

## Data Requirements

### sMRI Data
- **Source**: FreeSurfer processed structural MRI
- **Features**: 800 selected features (subcortical, cortical, white matter)
- **Subjects**: 870 from ABIDE dataset
- **Format**: `.npy` arrays (features, labels, subject_ids)

### fMRI Data  
- **Source**: Preprocessed functional connectivity matrices
- **Features**: Connectivity patterns between brain regions
- **Format**: Compatible with sMRI data structure

## Performance Monitoring

### Current Baselines
- **sMRI (Improved)**: 57-60% accuracy (up from 55%)
- **fMRI**: 65% accuracy (stable)
- **Cross-Attention**: TBD (training with improved sMRI)

### Target Performance
- **sMRI Goal**: ~70% (reference paper benchmark)
- **Multimodal Goal**: >70% (leveraging cross-modal interactions)

## Google Colab Usage

This repository is optimized for Google Colab notebooks:

1. **Mount Google Drive**: Scripts automatically handle drive mounting
2. **Load Data**: Efficient loading from drive-mounted datasets  
3. **Train Models**: GPU-accelerated training with progress tracking
4. **Save Results**: Automatic result saving to Drive

See `COLAB_GUIDE.md` for detailed Colab usage instructions.

## Recent Updates

### ✅ Completed Improvements
- [x] Enhanced sMRI feature extraction with RFE selection
- [x] Improved preprocessing pipeline (median imputation, standardization)
- [x] Comprehensive feature analysis and validation
- [x] Google Drive synchronization utilities
- [x] Code cleanup and documentation
- [x] Updated training scripts to use 800 features (vs 300-400)
- [x] **FAIR COMPARISON: All experiments now use same matched subjects**

### 🔄 In Progress
- [ ] Cross-attention performance evaluation with enhanced data
- [ ] Further sMRI optimization to reach 70% target

### 📋 Next Steps
- [ ] Hyperparameter optimization for transformer models
- [ ] Advanced cross-attention mechanisms exploration
- [ ] Results comparison and analysis

## Contributing

This is a bachelor thesis project focused on multimodal brain imaging analysis. The codebase is designed for research reproducibility and Google Colab compatibility.

## Technical Notes

- **Python Version**: 3.8+
- **Key Dependencies**: PyTorch, scikit-learn, numpy, pandas
- **Compute Requirements**: GPU recommended for transformer training
- **Data Format**: NumPy arrays for efficient loading/processing

## Contact & Citation

For questions about this implementation or to cite this work, please refer to the associated bachelor thesis documentation.

---

*Last Updated: December 2024 - Enhanced sMRI processing implementation*
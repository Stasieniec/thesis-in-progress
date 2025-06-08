# 🧠 ABIDE Cross-Attention Experiments Guide

This guide covers running cross-attention experiments between sMRI and fMRI for autism classification using the ABIDE dataset.

## 📋 Overview

This repository implements cross-attention between structural MRI (sMRI) and functional MRI (fMRI) for autism spectrum disorder classification. All experiments use matched subjects to ensure fair comparison.

**Key Improvements Implemented:**
- ✅ **Improved sMRI Features**: 800 features using RFE + Ridge (vs 300-400 before)
- ✅ **Fair Subject Matching**: All experiments use identical subject sets (~800 matched subjects)
- ✅ **Reference Paper Methodology**: Following "A Framework for Comparison and Interpretation of Machine Learning Classifiers to Predict Autism on the ABIDE Dataset"

## 🚀 Quick Start (Google Colab)

### 1. Setup
```python
# Clone repository
!git clone [your-repo-url]
%cd thesis-in-progress

# Install requirements
!pip install -r requirements.txt

# Upload improved sMRI data (if not already done)
from scripts.update_to_improved_smri import sync_improved_smri_data
sync_improved_smri_data()
```

### 2. Run Experiments
```python
# Run all three experiments with matched subjects
from scripts.run_experiments import run_all_experiments

results = run_all_experiments(
    fmri_data_path="/content/drive/MyDrive/b_data/ABIDE_pcp/cpac/filt_noglobal/rois_cc200",
    smri_data_path="/content/drive/MyDrive/processed_smri_data_improved", 
    phenotypic_file="/content/drive/MyDrive/b_data/ABIDE_pcp/Phenotypic_V1_0b_preprocessed1.csv",
    use_matched_subjects=True
)
```

### 3. Verify Results
```python
# Verify all experiments used same subjects
from verify_matched_subjects import verify_subject_matching
verify_subject_matching()
```

## 📊 Expected Results

**With Improved sMRI (800 features) + Fair Comparison:**
- fMRI: ~65% accuracy
- sMRI: ~58-60% accuracy (improved from 55%)
- Cross-Attention: ~59-62% accuracy

## 🔧 Technical Details

### sMRI Improvements
- **Features**: 1417 raw → 800 selected using RFE + Ridge
- **Sources**: FreeSurfer aseg + aparc + wmparc 
- **Selection**: Reference paper methodology (RFE with Ridge classifier)
- **Processing**: Median imputation + standardization

### Fair Comparison
- **Subject Matching**: All experiments use identical ~800 matched subjects
- **Verification**: `verify_matched_subjects.py` ensures consistency
- **Scientific Validity**: Eliminates subject selection bias

### Feature Breakdown (800 total)
- 145 subcortical volumes
- 184 left hemisphere features  
- 201 right hemisphere features
- 270 white matter features

## 📁 Repository Structure

```
thesis-in-progress/
├── scripts/                     # Main experiment scripts
│   ├── run_experiments.py       # Single entry point
│   ├── train_fmri.py           # fMRI experiment
│   ├── train_smri.py           # sMRI experiment
│   └── train_cross_attention.py # Cross-attention experiment
├── src/                        # Core modules
│   ├── data/                   # Data processing
│   ├── models/                 # Model definitions
│   ├── training/               # Training utilities
│   ├── evaluation/             # Evaluation metrics
│   └── utils/                  # Utilities (subject matching)
├── docs/                       # Documentation
└── verify_matched_subjects.py  # Verification script
```

## 🧪 Experiment Details

### 1. fMRI Experiment (`train_fmri.py`)
- **Input**: 200 ROI time series (CC200 atlas)
- **Architecture**: Transformer with positional encoding
- **Subjects**: Uses matched subject set only

### 2. sMRI Experiment (`train_smri.py`) 
- **Input**: 800 selected FreeSurfer features
- **Architecture**: MLP with dropout and batch normalization
- **Features**: Improved feature selection (RFE + Ridge)

### 3. Cross-Attention Experiment (`train_cross_attention.py`)
- **Input**: Both fMRI and sMRI for same subjects
- **Architecture**: Cross-attention between modalities
- **Innovation**: Attention mechanism between brain modalities

## 🔍 Verification & Debugging

### Subject Matching Verification
```python
python verify_matched_subjects.py
```

### Check Logs
- All scripts include detailed logging
- Progress bars for data loading
- Performance metrics after each epoch

## 📚 Research Context

This work builds on:
- **ABIDE Dataset**: Multi-site autism neuroimaging
- **Reference Paper**: Craddock et al. framework for ABIDE classification
- **Cross-Attention**: Novel application to multimodal brain imaging

## 🛠️ Troubleshooting

### Common Issues
1. **Path Errors**: Ensure Google Drive paths match your setup
2. **Memory Issues**: Use smaller batch sizes if needed
3. **Subject Mismatch**: Run verification script to check consistency

### Data Requirements
- fMRI: CC200 preprocessed timeseries
- sMRI: FreeSurfer processed structural features
- Phenotypic: ABIDE demographic data

## 🔄 Updating sMRI Data

If you need to update the improved sMRI data:
```python
from scripts.update_to_improved_smri import sync_improved_smri_data
sync_improved_smri_data()
```

---

**Next Steps:**
1. Run experiments with matched subjects
2. Compare results across modalities
3. Analyze attention patterns in cross-attention model
4. Consider ensemble approaches 
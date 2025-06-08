# Improved sMRI Feature Extraction Guide

## 🎯 Problem
Your current sMRI approach achieves 55% accuracy while the paper you referenced achieves ~70%. This guide provides an improved extraction method that follows the paper's methodology more closely.

## 📖 Paper Reference
**"A Framework for Comparison and Interpretation of Machine Learning Classifiers to Predict Autism on the ABIDE Dataset"**

## 🔧 Key Improvements

### 1. **More Comprehensive Feature Extraction**
- **Your approach**: `aseg.stats` + `lh/rh.aparc.stats`
- **Improved approach**: `aseg.stats` + `lh/rh.aparc.stats` + `wmparc.stats`
- **Benefit**: White matter features provide additional brain structure information

### 2. **Better Feature Selection**
- **Your approach**: Combined F-score + Mutual Information (top-k selection)
- **Improved approach**: Recursive Feature Elimination with Ridge classifier
- **Benefit**: RFE considers feature interactions, not just individual feature importance

### 3. **Optimal Feature Count**
- **Your approach**: 300-400 features
- **Improved approach**: 800 features (as specified in paper)
- **Benefit**: Paper research found 800 to be optimal for this specific task

### 4. **Robust Preprocessing**
- **Your approach**: StandardScaler + feature elimination
- **Improved approach**: Median imputation + StandardScaler + RFE
- **Benefit**: More robust handling of missing values and outliers

## 🚀 Quick Start

### Step 1: Run Improved Extraction
```bash
python run_improved_smri_extraction.py
```

This will:
- Process FreeSurfer data from `data/freesurfer_stats/`
- Apply paper's methodology
- Save improved features to `processed_smri_data_improved/`
- Provide baseline SVM performance comparison

### Step 2: Update Your Training Code
Update your sMRI data loading to use the new improved features:

```python
# Old approach
processor = SMRIDataProcessor(
    data_path="processed_smri_data",
    feature_selection_k=300
)

# New improved approach  
processor = SMRIDataProcessor(
    data_path="processed_smri_data_improved",
    feature_selection_k=800  # Use all 800 selected features
)
```

### Step 3: Compare Performance
```bash
python compare_smri_approaches.py
```

This script will show you detailed comparisons and expected improvements.

## 📊 Expected Results

| Metric | Current Approach | Improved Approach | Improvement |
|--------|------------------|-------------------|-------------|
| Features | ~300-400 | 800 | +100% |
| Baseline SVM | ~55% | ~70%+ | +15% |
| Transformer | ~55% | ~70-75% | +15-20% |

## 🗂️ Output Files

After running the improved extraction, you'll have:

```
processed_smri_data_improved/
├── features.npy              # 800 selected features
├── labels.npy                # 0=Control, 1=ASD  
├── subject_ids.npy           # Matched subject IDs
├── feature_names.txt         # Names of selected features
├── metadata.json             # Extraction details & performance
└── processed_data.mat        # MATLAB compatibility
```

## 🔬 Technical Details

### Feature Extraction Pipeline
1. **Parse FreeSurfer Stats**: Extract all 9 cortical + 7 subcortical + 7 white matter features per region
2. **Handle Missing Values**: Use median imputation (robust to outliers)
3. **Standardization**: Zero mean, unit variance scaling
4. **RFE Selection**: Iteratively remove features using Ridge classifier
5. **Validation**: Cross-validated baseline performance

### Feature Sources
- **Cortical (Desikan-Killiany)**: 68 regions × 9 features = 612 features
- **Subcortical (aseg)**: ~45 regions × 7 features = ~315 features  
- **White Matter (wmparc)**: ~100+ regions × 7 features = ~700+ features
- **Total**: ~1400+ raw features → 800 selected features

## 🛠️ Integration with Existing Code

The improved extraction is designed to be compatible with your existing codebase. Simply:

1. Run the improved extraction once
2. Update your data path in config files
3. Your transformer training code should work without changes

For Google Colab, upload the `processed_smri_data_improved/` folder to your Google Drive.

## ❓ Troubleshooting

### Common Issues
1. **Import errors**: Make sure you're running from the thesis root directory
2. **Missing packages**: `pip install numpy pandas scikit-learn matplotlib`
3. **Memory issues**: The extraction uses ~2-4GB RAM for large datasets
4. **Path issues**: Ensure `data/freesurfer_stats/` contains subject directories

### Performance Validation
- Baseline SVM should achieve ~70% accuracy with improved features
- If much lower, check data quality and feature selection parameters
- Compare with original approach using `compare_smri_approaches.py`

## 📈 Next Steps

1. **Run extraction**: Generate improved features
2. **Validate baseline**: Confirm ~70% SVM accuracy  
3. **Update transformer**: Use new features in your model
4. **Compare results**: Should see 55% → 70%+ improvement
5. **Fine-tune**: Experiment with transformer hyperparameters for even better performance

## 🎯 Expected Impact

With the improved sMRI features:
- **Immediate**: Better baseline performance validates feature quality
- **Short-term**: Transformer should reach 70%+ accuracy (matching paper)
- **Long-term**: Stronger foundation for cross-attention experiments

The goal is to bring your sMRI performance in line with the paper's results, providing a solid foundation for your cross-attention research. 
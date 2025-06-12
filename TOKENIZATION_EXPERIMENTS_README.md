# Tokenization Strategy Experiments for Cross-Attention Models

This document describes experiments for exploring different tokenization strategies for multimodal brain imaging data (fMRI + sMRI) in cross-attention transformer models.

## 🎯 Objective

**Current Approach**: Each modality (fMRI, sMRI) is treated as a single token for cross-attention.  
**New Approach**: Create multiple tokens per modality to enable richer cross-attention interactions.

## 🧠 Problem Analysis

### Current Data Dimensions
- **fMRI**: 19,900 features (CC200 connectivity matrix: 200×199/2)
- **sMRI**: 800 features (FreeSurfer after RFE selection)

### Tokenization Recommendation
**Start with sMRI tokenization** because:
- ✅ Lower dimensionality (800 vs 19,900)
- ✅ Already feature-selected through RFE  
- ✅ More stable and interpretable
- ✅ Lower overfitting risk
- ✅ Cleaner anatomical structure

## 🚀 Quick Start (Google Colab)

Follow your standard workflow, then run tokenization experiments:

```bash
# Your standard setup
!git clone https://github.com/Stasieniec/thesis-in-progress.git
%cd thesis-in-progress
!pip install -r requirements.txt -q

from google.colab import drive
drive.mount('/content/drive')

# Run tokenization experiments
!python scripts/tokenization_experiments.py --quick_test
```

## 📋 Available Commands

### Quick Test (Recommended First)
```bash
!python scripts/tokenization_experiments.py --quick_test
```
- Compares 2 strategies: `single_token` (baseline) vs `smri_grouped`
- 15 epochs each
- ~10-15 minutes runtime
- Perfect for initial validation

### Full Experiment
```bash
!python scripts/tokenization_experiments.py --full_experiment
```
- Tests all 6 tokenization strategies
- 30 epochs each
- ~1-2 hours runtime
- Comprehensive comparison

### Compare Specific Strategies
```bash
!python scripts/tokenization_experiments.py --compare_strategies single_token,hemisphere_tokens,pca_tokens
```
- Test only specified strategies (comma-separated)
- Custom runtime based on number of strategies

### Optional Parameters
```bash
!python scripts/tokenization_experiments.py --quick_test --output_dir "/content/drive/MyDrive/my_results"
```

## 🔬 Tokenization Strategies

| Strategy | Description | fMRI Tokens | sMRI Tokens | Focus |
|----------|-------------|-------------|-------------|-------|
| `single_token` | **Baseline**: Current approach | 1×19900 | 1×800 | Reference |
| `smri_grouped` | Simple sMRI grouping | 1×19900 | 4×200 | Basic tokenization |
| `smri_detailed` | Detailed sMRI groups | 1×19900 | 8×100 | Finer granularity |
| `pca_tokens` | PCA-based sMRI tokens | 1×19900 | 5×20 | Dimensionality reduction |
| `cluster_tokens` | Feature clustering | 1×19900 | 4×variable | Data-driven grouping |
| `hemisphere_tokens` | Anatomical regions | 1×19900 | 3×266 | Left/Right/Subcortical |

## 📊 Expected Results

### What to Look For
1. **Baseline Performance**: `single_token` should match your current results
2. **Improvement**: sMRI tokenization strategies should show gains
3. **Best Strategy**: Likely `hemisphere_tokens` or `pca_tokens`
4. **Model Size**: Similar parameter counts across strategies

### Typical Performance Pattern
```
single_token:      ~0.65-0.70 accuracy (baseline)
smri_grouped:      ~0.67-0.72 accuracy (+2-3%)  
hemisphere_tokens: ~0.68-0.74 accuracy (+3-5%)
pca_tokens:        ~0.69-0.75 accuracy (+4-6%)
```

## 📁 Output Structure

Results are saved to `/content/drive/MyDrive/tokenization_experiments_YYYYMMDD_HHMMSS/`:

```
tokenization_experiments_20240315_143022/
├── tokenization_results.json      # Detailed results
├── tokenization_summary.csv       # Comparison table
├── single_token_best_model.pth    # Best models
├── smri_grouped_best_model.pth
└── ...
```

## 🔧 Technical Details

### Architecture: `TokenizedCrossAttentionTransformer`
- **Token Projections**: Map each token type to d_model=128
- **Self-Attention**: Within-modality token interactions
- **Cross-Attention**: fMRI tokens attend to sMRI tokens
- **Fusion**: Combine [CLS] tokens from both modalities
- **Classification**: Final autism detection

### Data Flow
```
fMRI: (batch, 19900) → tokenize → (batch, n_tokens, token_dim) → project → (batch, n_tokens, 128)
sMRI: (batch, 800)   → tokenize → (batch, m_tokens, token_dim) → project → (batch, m_tokens, 128)
                                                ↓
                              Cross-Attention + Fusion → Classification
```

## 🧪 Integration with Your Workflow

### Step 1: Run Experiments
```bash
!python scripts/tokenization_experiments.py --quick_test
```

### Step 2: Analyze Results
- Check `tokenization_summary.csv` for strategy comparison
- Look for best performing strategy

### Step 3: Integrate Best Strategy
If tokenization improves performance, integrate into your main models:
1. Copy best tokenization strategy code
2. Modify your cross-attention models to use multiple tokens
3. Update training scripts

### Step 4: Extended Research
- Apply successful sMRI tokenization to fMRI
- Explore hierarchical tokenization
- Test on other datasets

## 🚨 Important Notes

### Data Compatibility
- ✅ Uses your existing data loading (`get_matched_datasets`)
- ✅ Preserves train/validation/test splits
- ✅ Google Drive paths compatible
- ✅ Fallback to synthetic data if needed

### Memory Considerations
- **Focus on sMRI**: Lower memory usage than fMRI tokenization
- **Batch Size**: Default 32, reduce if OOM errors
- **Model Size**: ~500K-1M parameters per strategy

### Limitations
- Currently focuses on sMRI tokenization (easier starting point)
- fMRI tokenization requires more sophisticated approaches
- Synthetic anatomical groupings (could be improved with real atlas data)

## 🔍 Troubleshooting

### Common Issues

**1. Module Import Errors**
```
⚠️ Project modules not available
```
- **Solution**: Synthetic data will be used automatically
- Results still valid for methodology testing

**2. CUDA Memory Errors**
```
RuntimeError: CUDA out of memory
```
- **Solution**: Reduce batch size: `--batch_size 16`

**3. Drive Mount Issues**
- **Solution**: Ensure `drive.mount('/content/drive')` runs first
- Check data paths in output

### Performance Debugging
- Low accuracy across all strategies → Check data loading
- Single strategy fails → Strategy-specific bug (check logs)
- All strategies similar → Need more sophisticated tokenization

## 📚 Next Steps

### If Experiments Show Promise
1. **Extend to fMRI**: Implement ROI-based or network-based fMRI tokenization
2. **Hierarchical Tokens**: Multi-scale brain representations
3. **Cross-Modal Attention Maps**: Visualize which sMRI tokens attend to fMRI
4. **Dataset Validation**: Test on other brain imaging datasets

### If No Improvement
1. **Analyze Attention**: Examine what the model learns with multiple tokens
2. **Feature Engineering**: Better anatomical groupings for sMRI
3. **Alternative Architectures**: Different fusion mechanisms
4. **Baseline Investigation**: Ensure single-token baseline is optimal

## 🎓 Research Contribution

This work explores **multimodal tokenization for brain imaging**, a novel approach that could:
- Improve autism detection accuracy
- Provide insights into cross-modal brain interactions
- Enable more interpretable attention mechanisms
- Contribute to multimodal medical AI research

The framework provides a foundation for exploring tokenization strategies without modifying your existing work, allowing you to validate the approach before full integration.

---

**Quick Command Reference:**
```bash
# Quick test (start here)
!python scripts/tokenization_experiments.py --quick_test

# Full evaluation
!python scripts/tokenization_experiments.py --full_experiment

# Custom comparison
!python scripts/tokenization_experiments.py --compare_strategies single_token,hemisphere_tokens
``` 
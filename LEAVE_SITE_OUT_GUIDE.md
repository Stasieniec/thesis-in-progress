# Real Leave-Site-Out Cross-Validation Guide

This guide explains how to use **real** leave-site-out cross-validation for ABIDE cross-attention models.

## What is Leave-Site-Out Cross-Validation?

Leave-site-out cross-validation provides the most realistic estimate of model generalization by:
- Training on data from multiple sites
- Testing on completely unseen sites  
- Simulating real-world deployment to new clinical centers
- Accounting for site-specific scanner and protocol differences

## Usage

### Basic Usage
```bash
python scripts/leave_site_out_experiments.py \
    --fmri-data /path/to/fmri/roi/data \
    --smri-data /path/to/smri/data \
    --phenotypic /path/to/phenotypic.csv \
    --output-dir results/leave_site_out
```

### Quick Test
```bash
python scripts/leave_site_out_experiments.py \
    --fmri-data /path/to/data \
    --smri-data /path/to/data \
    --phenotypic /path/to/phenotypic.csv \
    --quick-test
```

## Output

The script generates:
- Comprehensive statistical analysis
- Site-specific performance metrics
- Publication-ready visualizations
- Detailed per-model results

## Clinical Significance

Leave-site-out CV provides the most conservative and clinically relevant performance estimates, essential for:
- Regulatory approval (FDA/EMA)
- Clinical deployment readiness
- Publication in top journals
- Commercial viability assessment

## Site Extraction Methods

The script uses multiple methods to extract site information:

### 1. Phenotypic Data (Primary)
- Looks for `SITE_ID` column in phenotypic CSV
- Most reliable method when available

### 2. Subject ID Patterns (Fallback)
- **Direct matching**: `NYU_0050001` → `NYU`
- **Prefix patterns**: `0050001` → `NYU` (known numeric ranges)
- **Suffix patterns**: `0050001_KKI` → `KKI`
- **Embedded patterns**: `sub_NYU_001` → `NYU`

### 3. Known ABIDE Sites
The script recognizes these standard ABIDE site codes:
- `CALTECH` - California Institute of Technology
- `CMU` - Carnegie Mellon University
- `KKI` - Kennedy Krieger Institute
- `LEUVEN` - University of Leuven
- `NYU` - NYU Langone Medical Center
- `OHSU` - Oregon Health and Science University
- `UCLA` - University of California Los Angeles
- `UM` - University of Michigan
- And more...

## Output Files

The script generates comprehensive results:

```
results/leave_site_out/
├── LEAVE_SITE_OUT_SUMMARY.md      # Executive summary
├── leave_site_out_results.json    # Detailed results
├── leave_site_out_results.png     # Main visualization
├── site_information.csv           # Site statistics
├── site_mapping.json             # Subject-to-site mapping
├── bidirectional/                 # Per-model results
│   ├── detailed_results.json
│   ├── site_results.csv
│   └── fold_*/                    # Per-fold results
├── hierarchical/
├── contrastive/
├── adaptive/
└── ensemble/
```

## Models Tested

The script evaluates 5 advanced cross-attention models:

1. **Bidirectional**: True bidirectional attention (fMRI ↔ sMRI)
2. **Hierarchical**: Multi-scale processing at 3 granularities  
3. **Contrastive**: Cross-modal alignment with contrastive learning
4. **Adaptive**: Performance-aware weighting with dynamic gating
5. **Ensemble**: Multiple parallel attention mechanisms

## Expected Results

### Conservative Estimate
- **2-3 models** should beat the 60% fMRI baseline
- **Best performance**: ~62-65% accuracy
- **Clinical significance**: Demonstrates multimodal benefit

### Optimistic Estimate  
- **3-4 models** beat baseline
- **Best performance**: ~65-68% accuracy
- **Strong evidence** for cross-attention effectiveness

### Realistic Challenges
Leave-site-out CV is **much harder** than random CV because:
- No data leakage between sites
- Must generalize across different scanners/protocols
- Site-specific effects are not memorized
- True test of clinical deployment readiness

## Site Validation

The script automatically validates site distribution:

```python
✅ Site distribution validation passed:
   All 12 sites have ≥ 5 subjects
   All sites have both ASD and control subjects
```

Common validation failures:
- **Too few subjects per site** (< 5)
- **Single class sites** (only ASD or only controls)
- **Too few sites** (< 3 for meaningful CV)

## Integration with Existing Code

The implementation uses your existing infrastructure:

```python
# Uses existing data loading
from utils.subject_matching import get_matched_datasets

# Uses existing models  
from models.cross_attention import BidirectionalCrossAttentionTransformer

# Uses existing training
from training.cross_validation import _run_multimodal_fold
```

This ensures compatibility and leverages your proven components.

## Clinical Significance

### Why Leave-Site-Out Matters
1. **Regulatory approval**: FDA/EMA expect site-generalization studies
2. **Clinical deployment**: Models must work at new hospitals
3. **Publication impact**: Top journals require rigorous validation
4. **Commercial viability**: Real-world performance guarantees

### Interpreting Results
- **>65% accuracy**: Clinically promising, better than fMRI alone
- **60-65% accuracy**: Moderate improvement, additional validation needed  
- **<60% accuracy**: May indicate overfitting to training sites

## Troubleshooting

### Common Issues

1. **Site extraction fails**
   ```
   ⚠️ Warning: Many subjects mapped to generic sites
   ```
   **Solution**: Check phenotypic file has SITE_ID column

2. **Too few sites detected**
   ```
   ValueError: Need at least 3 sites for leave-site-out CV, found 2
   ```
   **Solution**: Verify data includes multi-site subjects

3. **Imbalanced sites**
   ```
   ⚠️ Warning: 3 sites have only one class
   ```
   **Solution**: Filter problematic sites or combine small sites

4. **Memory issues with large sites**
   ```
   CUDA out of memory
   ```
   **Solution**: Reduce batch size or use gradient checkpointing

### Site Mapping Verification

Always check the site mapping output:
```bash
# Check detected sites
cat results/leave_site_out/site_information.csv

# Verify subject mapping
head results/leave_site_out/site_mapping.json
```

## Advanced Usage

### Custom Site Extraction

If automatic site extraction fails, you can create a custom phenotypic file:

```csv
SUB_ID,SITE_ID,DX_GROUP
sub001,NYU,1
sub002,NYU,0
sub003,KKI,1
...
```

### Filtering Sites

For better balance, you might want to filter sites:

```python
# In your preprocessing
min_subjects = 10
min_asd_subjects = 3
min_control_subjects = 3

# Filter in phenotypic file before running
```

### Combining Small Sites

For sites with few subjects, consider grouping:

```python
# Map similar sites together
site_groups = {
    'UCLA_GROUP': ['UCLA_1', 'UCLA_2'],
    'LEUVEN_GROUP': ['LEUVEN_1', 'LEUVEN_2']
}
```

## Comparison with Standard CV

| Method | Accuracy | Generalization | Clinical Relevance |
|--------|----------|----------------|-------------------|
| Random 5-fold CV | ~64% | Optimistic | Limited |
| Leave-site-out CV | ~60% | Realistic | High |
| Real deployment | ~58% | Actual | Maximum |

Leave-site-out provides the closest estimate to real-world performance.

## Next Steps

After running leave-site-out CV:

1. **Analyze site-specific results** - Which sites are hardest?
2. **Investigate failure modes** - Why do some models fail?
3. **Domain adaptation** - Can we improve cross-site generalization?
4. **Site harmonization** - Should we normalize for site effects?
5. **Clinical validation** - Test best models on new sites

This real leave-site-out validation puts your cross-attention models through the most rigorous test possible, providing confidence for clinical translation. 
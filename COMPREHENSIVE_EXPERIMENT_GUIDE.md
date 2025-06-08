# 🎓 Comprehensive Experimental Framework for Thesis Results

**A robust framework for evaluating multiple model configurations on both regular cross-validation and leave-site-out cross-validation, designed specifically for generating publication-quality thesis results.**

## 🚀 Quick Start

### For Google Colab (Recommended)

```python
# 1. Mount Google Drive and navigate to your project
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/your-thesis-project

# 2. Install dependencies
!pip install -r requirements.txt

# 3. Run comprehensive evaluation (main thesis results)
!python scripts/comprehensive_experiments.py run_all

# 4. Quick test (for validation)
!python scripts/comprehensive_experiments.py quick_test
```

### Command Line Usage

```bash
# Run all experiments (comprehensive thesis evaluation)
python scripts/comprehensive_experiments.py run_all

# Quick test with reduced parameters
python scripts/comprehensive_experiments.py quick_test

# Run specific experiments only
python scripts/comprehensive_experiments.py run_selected --experiments smri_basic fmri_basic

# Validate setup before running
python scripts/comprehensive_experiments.py validate_setup

# List all available experiments
python scripts/comprehensive_experiments.py list_experiments

# Generate plots from existing results
python scripts/comprehensive_experiments.py generate_plots --results_dir comprehensive_results_20241201_120000
```

## 📊 What This Framework Does

### **Comprehensive Model Evaluation**
- **sMRI Models**: Optimized transformer architecture for structural MRI features
- **fMRI Models**: Enhanced SAT (Spatial Attention Transformer) for functional connectivity
- **Cross-Attention Models**: Basic and advanced cross-modal attention mechanisms
- **Advanced Models**: Bidirectional, hierarchical, contrastive, adaptive, and ensemble attention

### **Dual Validation Strategy**
1. **Regular Cross-Validation**: Traditional 5-fold CV for optimistic performance
2. **Leave-Site-Out Cross-Validation**: Site-based CV for realistic generalization assessment

### **Publication-Ready Outputs**
- **CSV Summary Table**: Ready for thesis tables
- **Statistical Report**: Comprehensive analysis with significance tests
- **High-Quality Plots**: Publication-ready figures (PNG + PDF)
- **Raw Results**: JSON format for further analysis

## 🧪 Available Experiments

### Basic Models
- `smri_basic`: sMRI Transformer (optimized architecture)
- `fmri_basic`: fMRI Transformer (enhanced SAT)
- `cross_attention_basic`: Basic cross-attention between modalities

### Advanced Cross-Attention Models
- `cross_attention_bidirectional`: Bidirectional attention
- `cross_attention_hierarchical`: Multi-scale hierarchical attention
- `cross_attention_contrastive`: Contrastive learning approach
- `cross_attention_adaptive`: Adaptive attention weights
- `cross_attention_ensemble`: Ensemble of attention mechanisms

## 📈 Output Structure

After running experiments, you'll get this directory structure:

```
comprehensive_results_YYYYMMDD_HHMMSS/
├── 📄 thesis_summary_table.csv          # Main results table for thesis
├── 📄 comprehensive_results.json        # Raw results (all data)
├── 📄 statistical_report.txt            # Statistical analysis
├── 📄 experiment_log.txt                # Detailed execution log
├── 📁 plots/                            # Publication-ready figures
│   ├── 📊 accuracy_comparison.png/.pdf  # Main comparison plot
│   ├── 📊 generalization_gap.png/.pdf   # Generalization analysis
│   ├── 📊 by_model_type.png/.pdf        # Results by model type
│   ├── 📊 statistical_significance.png/.pdf # Significance heatmap
│   └── 📊 performance_complexity.png/.pdf   # Performance vs complexity
└── 📁 individual_results/               # Detailed per-experiment results
```

## 🎯 Main Results Table (thesis_summary_table.csv)

| Experiment | Type | CV_Accuracy | CV_Std | LSO_Accuracy | LSO_Std | Generalization_Gap | P_Value | Significant |
|------------|------|-------------|--------|--------------|---------|-------------------|---------|-------------|
| sMRI Transformer | smri | 0.620 | 0.045 | 0.587 | 0.052 | 0.033 | 0.0124 | Yes |
| fMRI Transformer | fmri | 0.652 | 0.038 | 0.601 | 0.041 | 0.051 | 0.0089 | Yes |
| Cross-Attention | cross_attention | 0.634 | 0.042 | 0.598 | 0.048 | 0.036 | 0.0156 | Yes |
| Bidirectional CA | cross_attention_advanced | N/A | N/A | 0.612 | 0.044 | N/A | N/A | N/A |

## 🔬 Understanding the Results

### **Key Metrics**
- **CV_Accuracy**: Regular 5-fold cross-validation accuracy (optimistic)
- **LSO_Accuracy**: Leave-site-out cross-validation accuracy (realistic)
- **Generalization_Gap**: CV_Accuracy - LSO_Accuracy (lower = better generalization)
- **P_Value**: Statistical significance of the difference between CV types

### **Interpretation Guidelines**
- **LSO results are more clinically relevant** - use these as primary metrics
- **Generalization gap < 0.05** indicates good generalization
- **LSO accuracy > 0.60** beats fMRI baseline and shows clinical promise
- **Statistical significance** confirms the difference is not due to chance

### **Expected Results**
Based on ABIDE dataset characteristics:
- **sMRI**: ~58-62% (improved with better feature selection)
- **fMRI**: ~60-65% (baseline performance)
- **Cross-Attention**: ~59-64% (potential for multimodal benefits)
- **Leave-Site-Out**: Typically 2-5% lower than regular CV

## ⚙️ Configuration Options

### **Standard Run (Recommended for Thesis)**
```bash
python scripts/comprehensive_experiments.py run_all \
  --cv_folds 5 \
  --include_advanced True \
  --seed 42 \
  --verbose True
```

### **Quick Test (Validation)**
```bash
python scripts/comprehensive_experiments.py quick_test \
  --cv_folds 2 \
  --num_epochs 5
```

### **Custom Data Paths**
```python
# If using custom data locations
data_paths = {
    'fmri_data_path': "/path/to/fmri/data",
    'smri_data_path': "/path/to/smri/data", 
    'phenotypic_file': "/path/to/phenotypic.csv"
}

!python scripts/comprehensive_experiments.py run_all --data_paths '{data_paths}'
```

## 🛠️ Troubleshooting

### **Common Issues**

1. **"No module named" errors**
   ```bash
   # Ensure you're in the right directory
   %cd /content/drive/MyDrive/thesis-in-progress
   
   # Check imports
   python scripts/comprehensive_experiments.py validate_setup
   ```

2. **Data not found**
   ```python
   # Mount Google Drive first
   from google.colab import drive
   drive.mount('/content/drive')
   
   # Verify data paths exist
   import os
   print(os.path.exists('/content/drive/MyDrive/b_data/ABIDE_pcp/Phenotypic_V1_0b_preprocessed1.csv'))
   ```

3. **Out of memory**
   ```python
   # Use reduced batch sizes for Google Colab
   # The framework automatically handles this
   
   # Or use quick_test for validation
   !python scripts/comprehensive_experiments.py quick_test
   ```

4. **Advanced models not available**
   ```bash
   # This is normal - framework will use basic models
   # Advanced models are optional enhancements
   python scripts/comprehensive_experiments.py run_all --include_advanced False
   ```

## 📊 Using Results in Your Thesis

### **Main Results Table**
Use `thesis_summary_table.csv` directly in your thesis:
- Import into LaTeX with `\csvautotabular{thesis_summary_table.csv}`
- Import into Word/Google Docs as a table
- Copy-paste into any document format

### **Publication-Quality Figures**
All plots are generated in both PNG (for documents) and PDF (for LaTeX):
- `accuracy_comparison.pdf` - Main results figure
- `generalization_gap.pdf` - Shows clinical generalization
- `statistical_significance.pdf` - Supports claims of significance

### **Statistical Reporting**
Use `statistical_report.txt` for:
- Significance values for claims
- Effect size reporting
- Methodology validation

### **Example Results Section**
```latex
\section{Results}
Our comprehensive evaluation included both regular 5-fold cross-validation 
and leave-site-out cross-validation across X models. Table~\ref{tab:results} 
shows the complete results.

The leave-site-out cross-validation, which better represents real-world 
clinical deployment, showed that the [best model] achieved 61.2% ± 4.4% 
accuracy (p < 0.01), beating the fMRI baseline of 60.0% and demonstrating 
clinical promise for autism classification.

\begin{table}
\csvautotabular{thesis_summary_table.csv}
\caption{Comprehensive model evaluation results}
\label{tab:results}
\end{table}
```

## 🎯 Best Practices for Thesis Use

### **1. Run Complete Evaluation**
```bash
# Use this for final thesis results
python scripts/comprehensive_experiments.py run_all --seed 42
```

### **2. Document Your Setup**
```python
# Always validate before running
!python scripts/comprehensive_experiments.py validate_setup

# Save the validation output in your thesis appendix
```

### **3. Use Consistent Reporting**
- **Primary metric**: Leave-site-out cross-validation accuracy
- **Supporting metrics**: Regular CV for comparison, generalization gap
- **Statistical support**: P-values from significance tests

### **4. Reproducibility**
- Use fixed seed (`--seed 42`)
- Document data preprocessing steps
- Save complete results directory
- Include experiment log in appendix

## 🔄 Advanced Usage

### **Comparing Different Seeds**
```python
# Run multiple seeds for robustness
for seed in [42, 123, 456]:
    !python scripts/comprehensive_experiments.py run_all --seed {seed} --output_dir results_seed_{seed}
```

### **Generating Additional Plots**
```python
# From existing results
!python scripts/comprehensive_experiments.py generate_plots --results_dir comprehensive_results_20241201_120000
```

### **Custom Experiment Selection**
```bash
# Run only specific models
python scripts/comprehensive_experiments.py run_selected --experiments smri_basic fmri_basic cross_attention_basic
```

## 📝 Integration with Thesis Writing

### **Results Reporting Template**
1. **Load results**: Import `thesis_summary_table.csv`
2. **Main findings**: Report leave-site-out results as primary
3. **Statistical support**: Include p-values and significance
4. **Clinical relevance**: Emphasize generalization to new sites
5. **Comparison**: Show improvement over baselines

### **Figure Captions Template**
```latex
\caption{Model performance comparison on ABIDE dataset. 
Regular CV shows optimistic performance while leave-site-out CV 
represents realistic clinical deployment to new scanning sites. 
Error bars show standard deviation across folds. 
Dashed lines indicate baseline performance.}
```

---

## 🎉 Ready for Thesis-Quality Results!

This framework is designed to be robust, comprehensive, and publication-ready. The generated results can be directly used in your thesis with confidence in their statistical validity and clinical relevance.

**Happy experimenting! 🧠📊🎓** 
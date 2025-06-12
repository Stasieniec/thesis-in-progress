# 🧠 ABIDE Cross-Attention Experiments

**Cross-attention between sMRI and fMRI for autism classification using the ABIDE dataset.**

## 🚀 Quick Start

### For Google Colab (Recommended)
```python
# 1. Clone and setup
!git clone [your-repo-url]
%cd thesis-in-progress
!pip install -r requirements.txt

# 2. Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 3. Run comprehensive thesis evaluation (MAIN RESULTS)
!python scripts/comprehensive_experiments.py run_all

# 4. Quick test (for validation)
!python scripts/comprehensive_experiments.py quick_test
```

### Verify Setup
```python
# Check that everything is ready
!python scripts/comprehensive_experiments.py validate_setup

# List all available experiments
!python scripts/comprehensive_experiments.py list_experiments
```

## 📊 Key Improvements

✅ **Improved sMRI Features**: 800 features using RFE + Ridge (vs 300-400 before)  
✅ **Fair Subject Matching**: All experiments use identical subject sets (~800 matched subjects)  
✅ **Reference Paper Methodology**: Following established ABIDE classification framework  
✅ **Single Entry Point**: Clean `run_experiments.py` for all experiments  
✅ **Comprehensive Verification**: Subject matching validation  

## 📈 Expected Results

### Regular Cross-Validation (Optimistic)
- **fMRI**: ~62-65% accuracy
- **sMRI**: ~58-62% accuracy (improved from 55%)
- **Cross-Attention**: ~59-64% accuracy

### Leave-Site-Out CV (Realistic/Clinical)
- **fMRI**: ~58-62% accuracy
- **sMRI**: ~55-60% accuracy
- **Cross-Attention**: ~56-62% accuracy

**Note**: Leave-site-out results are more clinically relevant for thesis reporting.

## 📚 Documentation

🎓 **[Comprehensive Experiment Guide](context_files/COMPREHENSIVE_EXPERIMENT_GUIDE.md)** - **MAIN GUIDE** for thesis results  
📄 **[Leave-Site-Out Guide](context_files/COLAB_LEAVE_SITE_OUT.md)** - Site-based cross-validation  
📖 **[Complete Experiment Guide](docs/EXPERIMENT_GUIDE.md)** - Individual experiment setup  
📄 **[Colab Guide](context_files/COLAB_GUIDE.md)** - Google Colab specific instructions  

## 🏗️ Repository Structure

```
thesis-in-progress/
├── scripts/                     # Main experiment scripts
│   ├── run_experiments.py       # 🎯 Single entry point for all experiments
│   ├── improved_smri_extraction_new.py # sMRI data processing
│   └── update_to_improved_smri.py      # Google Drive sync
├── src/                        # Core modules
│   ├── data/                   # Data processing (fMRI, sMRI)
│   ├── models/                 # Model definitions
│   ├── training/               # Training utilities & individual trainers
│   ├── evaluation/             # Evaluation metrics
│   └── utils/                  # Subject matching utilities
├── docs/                       # Documentation
└── verify_matched_subjects.py  # 🔍 Verification script
```

## 🔬 Research Context

This bachelor thesis implements cross-attention between structural MRI (sMRI) and functional MRI (fMRI) for autism spectrum disorder classification using the ABIDE dataset. The work focuses on fair comparison methodologies and improved feature extraction techniques.

**Key Contributions:**
- Cross-modal attention mechanism for brain imaging
- Improved sMRI feature selection methodology
- Fair comparison framework ensuring identical subject sets
- Comprehensive evaluation and verification system

---

**For detailed setup, usage, and technical information, see [docs/EXPERIMENT_GUIDE.md](docs/EXPERIMENT_GUIDE.md)**
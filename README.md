# 🧠 ABIDE Cross-Attention Experiments

**Cross-attention between sMRI and fMRI for autism classification using the ABIDE dataset.**

## 🚀 Quick Start

### For Google Colab (Recommended)
```python
# 1. Clone and setup
!git clone [your-repo-url]
%cd thesis-in-progress
!pip install -r requirements.txt

# 2. Run all experiments with matched subjects
from scripts.run_experiments import run_all_experiments

results = run_all_experiments(
    fmri_data_path="/content/drive/MyDrive/b_data/ABIDE_pcp/cpac/filt_noglobal/rois_cc200",
    smri_data_path="/content/drive/MyDrive/processed_smri_data_improved", 
    phenotypic_file="/content/drive/MyDrive/b_data/ABIDE_pcp/Phenotypic_V1_0b_preprocessed1.csv"
)
```

### Verify Results
```python
from verify_matched_subjects import verify_subject_matching
verify_subject_matching()
```

## 📊 Key Improvements

✅ **Improved sMRI Features**: 800 features using RFE + Ridge (vs 300-400 before)  
✅ **Fair Subject Matching**: All experiments use identical subject sets (~800 matched subjects)  
✅ **Reference Paper Methodology**: Following established ABIDE classification framework  
✅ **Single Entry Point**: Clean `run_experiments.py` for all experiments  
✅ **Comprehensive Verification**: Subject matching validation  

## 📈 Expected Results

- **fMRI**: ~65% accuracy
- **sMRI**: ~58-60% accuracy (improved from 55%)
- **Cross-Attention**: ~59-62% accuracy

## 📚 Documentation

📖 **[Complete Experiment Guide](docs/EXPERIMENT_GUIDE.md)** - Comprehensive setup and usage guide  
📄 **[Colab Guide](COLAB_GUIDE.md)** - Google Colab specific instructions  

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
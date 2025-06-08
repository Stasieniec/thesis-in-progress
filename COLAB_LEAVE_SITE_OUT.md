# Google Colab: Real Leave-Site-Out Cross-Validation 🧠

**Three simple ways to run REAL leave-site-out cross-validation in Google Colab using your existing data paths.**

## 🚀 Option 1: Super Quick Test (One Command)

```bash
# Just run this one line!
!python quick_leave_site_out.py
```

**What it does:**
- Uses your existing Google Colab paths
- Tests the best model (contrastive) with 5 epochs
- Extracts real site information from ABIDE data
- Gives you leave-site-out performance in ~5 minutes

## 🧪 Option 2: Step-by-Step Functions

```python
# Cell 1: Setup
from colab_leave_site_out import setup_leave_site_out
setup_leave_site_out()

# Cell 2: Quick Test
from colab_leave_site_out import run_quick_test
results = run_quick_test()

# Cell 3: Full Experiment (Optional)
from colab_leave_site_out import run_full_experiment
results = run_full_experiment()
```

**What it does:**
- More control over each step
- Can run just quick test or full experiment
- Better for debugging and understanding

## 🔬 Option 3: Original Script (Most Features)

```python
# Full experiment
from scripts.leave_site_out_experiments import LeaveSiteOutExperiments
from utils.subject_matching import get_matched_datasets

# Your data paths (automatic)
matched_data = get_matched_datasets(
    fmri_roi_dir="/content/drive/MyDrive/b_data/ABIDE_pcp/cpac/filt_noglobal/rois_cc200",
    smri_data_path="/content/drive/MyDrive/processed_smri_data_improved",
    phenotypic_file="/content/drive/MyDrive/b_data/ABIDE_pcp/Phenotypic_V1_0b_preprocessed1.csv"
)

experiments = LeaveSiteOutExperiments()
results = experiments.run_leave_site_out_cv(
    matched_data=matched_data,
    num_epochs=50,
    output_dir=Path("/content/drive/MyDrive/leave_site_out_results")
)
```

## 📊 Expected Results

**Leave-site-out CV is harder than regular CV** because it tests true generalization across different clinical sites:

- **Regular CV**: ~64% (optimistic, may overfit to sites)
- **Leave-site-out CV**: ~60-62% (realistic clinical performance)

**Interpretation:**
- `> 60%`: Beats fMRI baseline - **clinically promising** 🎉
- `58-60%`: Beats sMRI baseline - **moderate improvement** ✅
- `< 58%`: Below baselines - **needs improvement** 📉

## 🏥 What is Leave-Site-Out CV?

**Traditional CV**: Randomly splits subjects → **optimistic results**
```
Train: [NYU_001, KKI_002, UCLA_003, NYU_004, ...]
Test:  [NYU_005, KKI_006, UCLA_007, ...]
```

**Leave-site-out CV**: Splits by site → **realistic results**
```
Train: [NYU_*, KKI_*, PITT_*]    (3 sites)
Test:  [UCLA_*]                  (1 site - completely unseen)
```

This simulates **real clinical deployment** where your model must work at new hospitals with different:
- MRI scanners
- Acquisition protocols  
- Patient populations
- Data processing pipelines

## 🔧 Troubleshooting

**Error: "Google Drive not mounted"**
```python
from google.colab import drive
drive.mount('/content/drive')
```

**Error: "Data not found"**
- Check your Google Drive has the ABIDE data
- Verify paths match your actual file locations
- Try running `setup_leave_site_out()` first

**Error: "Site extraction failed"**
- Check that phenotypic file contains SITE_ID column
- Verify phenotypic file path is correct
- Some ABIDE datasets have different column names

**Error: "Not enough sites"**
- Need at least 3 sites for meaningful leave-site-out CV
- Check site extraction is working correctly
- May indicate data quality issues

## 📈 Why This Matters for Your Thesis

**Academic Impact:**
- Most rigorous validation possible
- Addresses site effects (major issue in neuroimaging)
- Required for top-tier journal publication
- Shows clinical deployment readiness

**Clinical Relevance:**
- Simulates real-world performance
- FDA/EMA require multi-site validation
- Demonstrates generalization beyond training data
- Builds confidence for clinical translation

**Technical Excellence:**
- Uses actual ABIDE site information
- No data leakage between sites
- Proper stratification by diagnosis
- Multiple cross-attention architectures tested

This gives you the **gold standard** validation for your bachelor thesis! 🎓 
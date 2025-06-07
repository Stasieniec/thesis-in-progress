# 🧪 Google Colab Scripts for Cross-Attention Debugging

## 📊 **Problem Summary**
Cross-attention performance dropped from **63.6%** to **57.7%** (-5.9 points) despite individual sMRI improvements (49% → 54%). Need to find why and fix it to beat **65% fMRI baseline**.

## 🎯 **Hypothesis: sMRI Preprocessing Mismatch**
Enhanced sMRI preprocessing (RobustScaler + advanced feature selection) creates **distribution shift** that breaks cross-attention model compatibility.

---

## 🚀 **Script Execution Order**

### **STEP 1: Test Preprocessing Mismatch Hypothesis**
**File:** `colab_test_original_preprocessing.py`

**Purpose:** Test if original sMRI preprocessing recovers the 63.6% performance

**What it does:**
- Uses **original preprocessing**: StandardScaler + simple f_classif
- Tests original CrossAttentionTransformer 
- 3-fold cross-validation
- **Expected:** Recovery to ~63.6% if preprocessing was the issue

**Run this script first!**

**Expected Outcomes:**
```
✅ ~63.6% → Preprocessing mismatch CONFIRMED
⚠️  60-62% → Partial preprocessing impact  
❌ <60%   → NOT preprocessing issue
```

---

### **STEP 2A: If Preprocessing NOT the Issue**
**File:** `colab_hyperparameter_grid_search.py`

**Purpose:** Systematic hyperparameter search to find config beating 65%

**What it tests:**
- Learning rates: [1e-5, 5e-5, 1e-4, 2e-4]
- Batch sizes: [16, 32, 64]
- Architecture variations: hidden_dim, num_heads, num_layers
- Both original and enhanced preprocessing
- 12 strategic configurations

**Run if:** Script 1 shows NO RECOVERY or PARTIAL RECOVERY

---

### **STEP 2B: If Preprocessing Mismatch Confirmed**
**File:** `colab_gradual_enhancement.py`

**Purpose:** Gradually introduce enhancements from working baseline

**Enhancement Levels:**
```
Level 0: Original baseline (StandardScaler + f_classif)
Level 1: RobustScaler only  
Level 2: RobustScaler + improved F-score
Level 3: RobustScaler + F-score + MI
Level 4: Full enhanced preprocessing
```

**Models tested:**
- Original CrossAttentionTransformer
- WeightedFusionCrossAttention (55% fMRI, 45% sMRI)

**Run if:** Script 1 shows RECOVERY to ~63.6%

---

## 📋 **How to Use These Scripts**

### **1. Setup in Google Colab**
```python
# Each script includes these setup steps:
from google.colab import drive
drive.mount('/content/drive')

!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 -q
!pip install scikit-learn matplotlib seaborn pandas numpy scipy -q
```

### **2. Update Data Paths**
**IMPORTANT:** Update these paths in each script to match your Google Drive structure:
```python
FMRI_DATA_PATH = '/content/drive/MyDrive/processed_fmri_data/features.npy'
SMRI_DATA_PATH = '/content/drive/MyDrive/processed_smri_data/features.npy'  
LABELS_PATH = '/content/drive/MyDrive/processed_smri_data/labels.npy'
OUTPUT_PATH = '/content/drive/MyDrive/cross_attention_tests'
```

### **3. Run Scripts in Order**
1. **Always start with:** `colab_test_original_preprocessing.py`
2. **Based on results, run:**
   - Script 2A (hyperparameter search) OR
   - Script 2B (gradual enhancement)

### **4. Check Results**
Results are saved to your Google Drive in JSON format:
- `original_preprocessing_test_results.json`
- `hyperparameter_search_results.json` 
- `gradual_enhancement_results.json`

---

## 🎯 **Success Criteria**

### **Primary Goal**
Beat **65% fMRI baseline** consistently

### **Minimum Goal**  
Understand why performance dropped from 63.6% to 57.7%

### **Ideal Outcome**
**66-68%** accuracy proving multimodal benefit

---

## 🔍 **Expected Results & Next Steps**

### **Scenario 1: Preprocessing Mismatch Confirmed** ✅
```
Script 1 → ~63.6% recovery
Script 2B → Find optimal enhancement level
Next: Use best config for production
```

### **Scenario 2: Hyperparameter Issue** ⚠️
```
Script 1 → No recovery (<60%)
Script 2A → Test systematic hyperparameters  
Next: Use best hyperparameters found
```

### **Scenario 3: Mixed Factors** 🔄
```
Script 1 → Partial recovery (60-62%)
Both 2A and 2B → Compare approaches
Next: Hybrid approach
```

### **Scenario 4: No Solution Found** ❌
```
All scripts → <65% performance
Next: Architectural changes needed
- More sophisticated cross-attention
- Different fusion strategies  
- Ensemble methods
```

---

## 📊 **Performance Tracking**

| Metric | Original | Current | Target |
|--------|----------|---------|--------|
| **Cross-Attention** | 63.6% | 57.7% | >65% |
| **Pure fMRI** | - | 65.0% | Baseline |
| **Enhanced sMRI** | 49% | 54% | Working |

---

## 🛠️ **Script Features**

### **All Scripts Include:**
- ✅ Reproducible random seeds
- ✅ 3-fold cross-validation  
- ✅ Comprehensive metrics (Accuracy, Balanced Accuracy, AUC)
- ✅ Detailed diagnostic output
- ✅ JSON result saving
- ✅ Error handling
- ✅ Progress indicators

### **Diagnostic Output:**
- 📊 Performance summaries
- 🔍 Detailed analysis  
- 📋 Next step recommendations
- 🎯 Success/failure indicators
- ⚠️ Warning messages

---

## 💡 **Tips for Success**

### **1. Data Preparation**
Ensure your data files are properly preprocessed and accessible in Google Drive

### **2. Runtime Selection** 
Use GPU runtime for faster training:
`Runtime → Change runtime type → GPU`

### **3. Monitor Progress**
Scripts include detailed progress output - watch for:
- 🎯 "BEATS fMRI BASELINE!" messages
- ⚠️ Warning about performance drops
- 📊 Fold-by-fold results

### **4. Save Results**
Results auto-save to Google Drive, but also note key findings in your notebook

### **5. Interpret Diagnostics**
Each script provides clear diagnosis and next step recommendations

---

## 🔧 **Troubleshooting**

### **Common Issues:**
```python
# File not found
→ Check data paths in each script

# Out of memory  
→ Reduce batch_size in configs

# Import errors
→ Ensure all pip installs completed

# Poor performance across all tests
→ Check data quality and preprocessing
```

### **Performance Issues:**
- If all results <60%: Check data quality
- If inconsistent results: Increase CV folds
- If no improvement: Consider architectural changes

---

## 📈 **What Success Looks Like**

### **Ideal Outcome:**
```
🎯 Script 1: Recovers to 63.6% (preprocessing mismatch confirmed)
🎯 Script 2B: Finds enhancement level achieving 66-68%  
🏆 RESULT: Beat fMRI baseline with multimodal approach!
```

### **Alternative Success:**
```
⚠️  Script 1: Partial recovery to 61%
🎯 Script 2A: Finds hyperparameters achieving 66-67%
🏆 RESULT: Beat fMRI baseline through optimization!
```

---

## 🎉 **Final Notes**

These scripts provide a **systematic approach** to debugging the cross-attention performance drop. The key insight is that sometimes **working systems shouldn't be dramatically changed** - small, incremental improvements are often more successful than architectural overhauls.

**Remember:** Even achieving **66-67%** would be a clear success, proving that multimodal fusion provides meaningful benefit over single-modality approaches!

Good luck! 🚀 
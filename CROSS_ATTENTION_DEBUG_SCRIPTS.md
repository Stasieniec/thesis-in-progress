# 🔧 Cross-Attention Debugging Scripts

## 📊 **Problem**
Cross-attention performance dropped from **63.6%** to **57.7%** (-5.9 points) despite sMRI improvements (49% → 54%). Need to find why and beat **65% fMRI baseline**.

## 🎯 **Hypothesis**
Enhanced sMRI preprocessing (RobustScaler + advanced features) creates **distribution shift** that breaks cross-attention compatibility.

---

## 🚀 **Scripts Created**

### **1. Test Preprocessing Hypothesis** 🔍
**File:** `scripts/test_preprocessing_hypothesis.py`

**Purpose:** Test if original sMRI preprocessing recovers 63.6% performance

**Usage:**
```bash
# Full test (3-fold CV, 30 epochs)
!python scripts/test_preprocessing_hypothesis.py run

# Quick test (2-fold CV, 5 epochs)  
!python scripts/test_preprocessing_hypothesis.py quick_test
```

**Expected Results:**
- ✅ **~63.6%** → Preprocessing mismatch CONFIRMED
- ⚠️ **60-62%** → Partial preprocessing impact
- ❌ **<60%** → NOT preprocessing issue

---

### **2. Hyperparameter Search** ⚙️
**File:** `scripts/hyperparameter_search.py`

**Purpose:** Find hyperparameters that beat 65% fMRI baseline

**Usage:**
```bash
# Full search (12 strategic configs)
!python scripts/hyperparameter_search.py run

# Quick search (reduced epochs)
!python scripts/hyperparameter_search.py quick_test
```

**What it tests:**
- Learning rates: [1e-5, 5e-5, 1e-4, 2e-4]
- Batch sizes: [16, 32, 64]  
- Architecture variations
- Both original and enhanced preprocessing

**Run if:** Script 1 shows NO RECOVERY or PARTIAL RECOVERY

---

### **3. Gradual Enhancement** 🔄
**File:** `scripts/gradual_enhancement.py`

**Purpose:** Gradually introduce enhancements from working baseline

**Usage:**
```bash
# Full gradual test (5 enhancement levels)
!python scripts/gradual_enhancement.py run

# Quick test
!python scripts/gradual_enhancement.py quick_test
```

**Enhancement Levels:**
- **Level 0:** Original baseline (StandardScaler + f_classif)
- **Level 1:** RobustScaler only
- **Level 2:** RobustScaler + improved F-score  
- **Level 3:** RobustScaler + F-score + MI
- **Level 4:** Full enhanced preprocessing

**Run if:** Script 1 shows RECOVERY to ~63.6%

---

## 📋 **Execution Strategy**

### **Step 1: Always Start Here** 🔍
```bash
!python scripts/test_preprocessing_hypothesis.py run
```

### **Step 2: Based on Results**

**If recovers to ~63.6%** ✅
```bash
!python scripts/gradual_enhancement.py run
```

**If no recovery (<60%)** ❌  
```bash
!python scripts/hyperparameter_search.py run
```

**If partial recovery (60-62%)** ⚠️
```bash
# Run both
!python scripts/gradual_enhancement.py run
!python scripts/hyperparameter_search.py run
```

---

## 📁 **Output Files**

Each script saves results to JSON files:
- `preprocessing_hypothesis_analysis.json`
- `hyperparameter_search_results.json`
- `gradual_enhancement_results.json`

---

## 🎯 **Success Criteria**

**Primary Goal:** Beat **65% fMRI baseline** consistently

**Secondary Goals:**
- Understand why 63.6% → 57.7% drop occurred
- Find reproducible improvement strategy
- Achieve **66-68%** proving multimodal benefit

---

## 💡 **Key Features**

✅ **Uses your existing data loading system**  
✅ **Compatible with your Google Drive setup**  
✅ **Fire-based CLI like your other scripts**  
✅ **3-fold cross-validation for robust results**  
✅ **Comprehensive diagnostics and recommendations**  
✅ **JSON result saving for analysis**  

---

## 🔧 **Example Workflow**

```bash
# 1. Test the hypothesis first
!python scripts/test_preprocessing_hypothesis.py quick_test

# 2a. If preprocessing mismatch confirmed:
!python scripts/gradual_enhancement.py quick_test

# 2b. If no preprocessing recovery:  
!python scripts/hyperparameter_search.py quick_test

# 3. Run full tests on most promising approach
!python scripts/[best_approach].py run
```

**Remember:** Even **66-67%** would be a clear success proving multimodal benefit! 🚀 
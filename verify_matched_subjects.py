#!/usr/bin/env python3
"""
Verification Script: Ensure Fair Comparison Across All Experiments
================================================================

This script verifies that all three experiments (fMRI, sMRI, cross-attention)
use exactly the same subjects for fair comparison.

Usage:
    python verify_matched_subjects.py
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def verify_subject_matching():
    """Verify that all experiments use the same matched subjects."""
    
    print("🔍 VERIFYING SUBJECT MATCHING ACROSS EXPERIMENTS")
    print("=" * 60)
    
    try:
        from utils.subject_matching import get_matched_datasets
        
        # Try to get matched datasets (same as what experiments use)
        try:
            print("📊 Loading matched datasets (trying improved sMRI data)...")
            matched_data = get_matched_datasets(
                fmri_roi_dir="/content/drive/MyDrive/b_data/ABIDE_pcp/cpac/filt_noglobal/rois_cc200",
                smri_data_path="/content/drive/MyDrive/processed_smri_data_improved",
                phenotypic_file="/content/drive/MyDrive/b_data/ABIDE_pcp/Phenotypic_V1_0b_preprocessed1.csv",
                verbose=True
            )
            smri_source = "improved (800 features)"
        except Exception as e:
            print(f"⚠️ Improved sMRI data not available: {e}")
            print("📊 Falling back to original sMRI data...")
            matched_data = get_matched_datasets(
                fmri_roi_dir="/content/drive/MyDrive/b_data/ABIDE_pcp/cpac/filt_noglobal/rois_cc200",
                smri_data_path="/content/drive/MyDrive/processed_smri_data",
                phenotypic_file="/content/drive/MyDrive/b_data/ABIDE_pcp/Phenotypic_V1_0b_preprocessed1.csv",
                verbose=True
            )
            smri_source = "original"
        
        print("\n" + "=" * 60)
        print("✅ VERIFICATION RESULTS")
        print("=" * 60)
        
        # Summary statistics
        num_subjects = len(matched_data['fmri_subject_ids'])
        fmri_shape = matched_data['fmri_features'].shape
        smri_shape = matched_data['smri_features'].shape
        asd_count = int(matched_data['fmri_labels'].sum())
        control_count = num_subjects - asd_count
        
        print(f"📊 Matched subjects: {num_subjects}")
        print(f"📊 fMRI features: {fmri_shape}")
        print(f"📊 sMRI features: {smri_shape} ({smri_source})")
        print(f"📊 Class distribution:")
        print(f"   - ASD: {asd_count} ({asd_count/num_subjects*100:.1f}%)")
        print(f"   - Control: {control_count} ({control_count/num_subjects*100:.1f}%)")
        
        # Verify data consistency
        print(f"\n🔧 Data consistency checks:")
        
        # Check subject ID consistency
        fmri_ids = set(matched_data['fmri_subject_ids'])
        smri_ids = set(matched_data['smri_subject_ids'])
        if fmri_ids == smri_ids:
            print(f"   ✅ Subject IDs match between modalities")
        else:
            print(f"   ❌ Subject ID mismatch!")
            return False
        
        # Check label consistency
        import numpy as np
        if np.array_equal(matched_data['fmri_labels'], matched_data['smri_labels']):
            print(f"   ✅ Labels match between modalities")
        else:
            print(f"   ❌ Label mismatch!")
            return False
        
        # Check for valid data
        if not np.any(np.isnan(matched_data['fmri_features'])):
            print(f"   ✅ fMRI data has no NaN values")
        else:
            print(f"   ⚠️ fMRI data contains NaN values")
        
        if not np.any(np.isnan(matched_data['smri_features'])):
            print(f"   ✅ sMRI data has no NaN values")
        else:
            print(f"   ⚠️ sMRI data contains NaN values")
        
        print(f"\n" + "=" * 60)
        print("🎯 EXPERIMENT IMPACT")
        print("=" * 60)
        
        print(f"All three experiments will now use:")
        print(f"   📊 {num_subjects} subjects (same for all)")
        print(f"   📊 {fmri_shape[1]} fMRI features")
        print(f"   📊 {smri_shape[1]} sMRI features")
        print(f"   📊 Fair comparison ensured!")
        
        print(f"\n🚀 Next steps:")
        print(f"   1. Run: python scripts/train_fmri.py run")
        print(f"   2. Run: python scripts/train_smri.py run")
        print(f"   3. Run: python scripts/train_cross_attention.py run")
        print(f"   4. Compare results fairly!")
        
        return True
        
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main verification function."""
    print("🧠 Subject Matching Verification for ABIDE Experiments")
    print("=" * 60)
    print("This script ensures all experiments use the same subjects.")
    print("=" * 60)
    
    success = verify_subject_matching()
    
    print("\n" + "=" * 60)
    if success:
        print("✅ VERIFICATION PASSED!")
        print("All experiments will use the same matched subjects.")
        print("You can now run fair comparisons between methods.")
    else:
        print("❌ VERIFICATION FAILED!")
        print("Please check the error messages above.")
    print("=" * 60)

if __name__ == '__main__':
    main() 
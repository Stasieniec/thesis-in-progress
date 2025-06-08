#!/usr/bin/env python3
"""
One-Liner: Quick Leave-Site-Out Cross-Validation
================================================

Super simple script for Google Colab - just run and get results!

Usage:
    !python quick_leave_site_out.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def main():
    """Run quick leave-site-out test with one command."""
    
    print("🧠 QUICK LEAVE-SITE-OUT TEST")
    print("Using your existing Google Colab paths...")
    print("=" * 50)
    
    try:
        # Import modules
        from scripts.leave_site_out_experiments import LeaveSiteOutExperiments
        from utils.subject_matching import get_matched_datasets
        
        # Load data with your exact paths
        print("📊 Loading data...")
        try:
            # Try improved sMRI first
            matched_data = get_matched_datasets(
                fmri_roi_dir="/content/drive/MyDrive/b_data/ABIDE_pcp/cpac/filt_noglobal/rois_cc200",
                smri_data_path="/content/drive/MyDrive/processed_smri_data_improved",
                phenotypic_file="/content/drive/MyDrive/b_data/ABIDE_pcp/Phenotypic_V1_0b_preprocessed1.csv",
                verbose=False
            )
            print("✅ Using improved sMRI data (800 features)")
        except:
            # Fallback to original
            matched_data = get_matched_datasets(
                fmri_roi_dir="/content/drive/MyDrive/b_data/ABIDE_pcp/cpac/filt_noglobal/rois_cc200",
                smri_data_path="/content/drive/MyDrive/processed_smri_data",
                phenotypic_file="/content/drive/MyDrive/b_data/ABIDE_pcp/Phenotypic_V1_0b_preprocessed1.csv",
                verbose=False
            )
            print("✅ Using original sMRI data")
        
        print(f"📊 Loaded {len(matched_data['fmri_subject_ids'])} matched subjects")
        
        # Quick test with contrastive model (best performer)
        print("🧠 Testing contrastive model (5 epochs)...")
        
        experiments = LeaveSiteOutExperiments()
        
        # Extract site info
        site_labels, site_mapping, _ = experiments.extract_site_info(
            matched_data['fmri_subject_ids'],
            "/content/drive/MyDrive/b_data/ABIDE_pcp/Phenotypic_V1_0b_preprocessed1.csv"
        )
        
        print(f"🏥 Found {len(site_mapping)} sites: {list(site_mapping.keys())}")
        
        if len(site_mapping) < 3:
            print("❌ Need at least 3 sites for leave-site-out CV")
            print("   Site extraction may have failed - check phenotypic file")
            return
        
        # Run quick test
        result = experiments.test_strategy(
            strategy='contrastive',
            matched_data=matched_data,
            num_epochs=5,
            batch_size=16,
            d_model=128,
            output_dir=None,
            verbose=False
        )
        
        # Results
        acc = result['cv_results']['test_accuracies']
        print(f"\n📊 LEAVE-SITE-OUT RESULT:")
        print(f"   Contrastive Model: {acc.mean():.3f} ± {acc.std():.3f}")
        
        # Compare to baselines
        if acc.mean() > 0.60:
            print("   🎉 BEATS fMRI baseline (60%)!")
        elif acc.mean() > 0.58:
            print("   ✅ Beats sMRI baseline (58%)")
        else:
            print("   📉 Below current baselines")
        
        print(f"\n✅ Leave-site-out test completed!")
        print(f"   This is the most realistic clinical performance estimate.")
        
        return result
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure Google Drive is mounted")
        print("2. Check that data paths exist in your Drive")
        print("3. Run: from google.colab import drive; drive.mount('/content/drive')")
        return None


if __name__ == "__main__":
    main() 
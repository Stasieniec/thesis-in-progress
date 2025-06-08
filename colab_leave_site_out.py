"""
Google Colab: Leave-Site-Out Cross-Validation
=============================================

Simple script to run real leave-site-out cross-validation in Google Colab.
Copy and paste functions into Colab cells or run as script.

Functions:
- setup_leave_site_out() - Initial setup
- run_quick_test() - Quick 5-epoch test
- run_full_experiment() - Full experiment
"""

import sys
from pathlib import Path

# Setup for Google Colab
def setup_leave_site_out():
    """Setup for Google Colab."""
    print("🚀 Setting up Leave-Site-Out Cross-Validation...")
    
    # Mount Google Drive
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        print("✅ Google Drive mounted")
    except:
        print("⚠️ Not in Google Colab or Drive already mounted")
    
    # Add src to path
    sys.path.insert(0, str(Path.cwd() / 'src'))
    print("✅ Python path configured")
    
    # Check data paths
    paths = {
        'fMRI': '/content/drive/MyDrive/b_data/ABIDE_pcp/cpac/filt_noglobal/rois_cc200',
        'sMRI': '/content/drive/MyDrive/processed_smri_data_improved',
        'Phenotypic': '/content/drive/MyDrive/b_data/ABIDE_pcp/Phenotypic_V1_0b_preprocessed1.csv'
    }
    
    for name, path in paths.items():
        if Path(path).exists():
            print(f"✅ {name} data found")
        else:
            print(f"❌ {name} data not found: {path}")
    
    print("✅ Setup complete!")


def run_quick_test():
    """Run quick test (5 epochs, 2 models)."""
    print("🧪 Quick Leave-Site-Out Test")
    print("=" * 40)
    
    try:
        from scripts.leave_site_out_experiments import LeaveSiteOutExperiments
        from utils.subject_matching import get_matched_datasets
        
        # Load data
        print("📊 Loading data...")
        try:
            matched_data = get_matched_datasets(
                fmri_roi_dir="/content/drive/MyDrive/b_data/ABIDE_pcp/cpac/filt_noglobal/rois_cc200",
                smri_data_path="/content/drive/MyDrive/processed_smri_data_improved",
                phenotypic_file="/content/drive/MyDrive/b_data/ABIDE_pcp/Phenotypic_V1_0b_preprocessed1.csv",
                verbose=True
            )
        except:
            print("⚠️ Using original sMRI data")
            matched_data = get_matched_datasets(
                fmri_roi_dir="/content/drive/MyDrive/b_data/ABIDE_pcp/cpac/filt_noglobal/rois_cc200",
                smri_data_path="/content/drive/MyDrive/processed_smri_data",
                phenotypic_file="/content/drive/MyDrive/b_data/ABIDE_pcp/Phenotypic_V1_0b_preprocessed1.csv",
                verbose=True
            )
        
        # Quick test with best models
        experiments = LeaveSiteOutExperiments()
        
        # Test only contrastive model (best performer)
        print("\n🧠 Testing contrastive model...")
        results = experiments.test_strategy(
            strategy='contrastive',
            matched_data=matched_data,
            num_epochs=5,
            batch_size=16,
            d_model=128,
            output_dir=None,
            verbose=True
        )
        
        print(f"\n📊 Results:")
        acc = results['cv_results']['test_accuracies']
        print(f"Contrastive: {acc.mean():.3f} ± {acc.std():.3f}")
        
        if acc.mean() > 0.60:
            print("🎉 Beats fMRI baseline!")
        elif acc.mean() > 0.58:
            print("✅ Beats sMRI baseline")
        
        return results
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_full_experiment():
    """Run full leave-site-out experiment."""
    print("🚀 Full Leave-Site-Out Experiment")
    print("=" * 40)
    
    try:
        from scripts.leave_site_out_experiments import LeaveSiteOutExperiments
        from utils.subject_matching import get_matched_datasets
        
        # Load data
        print("📊 Loading data...")
        try:
            matched_data = get_matched_datasets(
                fmri_roi_dir="/content/drive/MyDrive/b_data/ABIDE_pcp/cpac/filt_noglobal/rois_cc200",
                smri_data_path="/content/drive/MyDrive/processed_smri_data_improved",
                phenotypic_file="/content/drive/MyDrive/b_data/ABIDE_pcp/Phenotypic_V1_0b_preprocessed1.csv",
                verbose=True
            )
        except:
            print("⚠️ Using original sMRI data")
            matched_data = get_matched_datasets(
                fmri_roi_dir="/content/drive/MyDrive/b_data/ABIDE_pcp/cpac/filt_noglobal/rois_cc200",
                smri_data_path="/content/drive/MyDrive/processed_smri_data",
                phenotypic_file="/content/drive/MyDrive/b_data/ABIDE_pcp/Phenotypic_V1_0b_preprocessed1.csv",
                verbose=True
            )
        
        # Run full experiment
        experiments = LeaveSiteOutExperiments()
        
        results = experiments.run_leave_site_out_cv(
            matched_data=matched_data,
            num_epochs=50,
            batch_size=32,
            learning_rate=0.0005,
            d_model=256,
            output_dir=Path("/content/drive/MyDrive/leave_site_out_results"),
            phenotypic_file="/content/drive/MyDrive/b_data/ABIDE_pcp/Phenotypic_V1_0b_preprocessed1.csv",
            seed=42,
            verbose=True
        )
        
        print(f"\n📊 FINAL RESULTS:")
        print("=" * 40)
        for model, result in results.items():
            acc = result['cv_results']['test_accuracies']
            baseline = ""
            if acc.mean() > 0.60:
                baseline = "🎉 BEATS fMRI!"
            elif acc.mean() > 0.58:
                baseline = "✅ Beats sMRI"
            print(f"{model:12} | {acc.mean():.3f} ± {acc.std():.3f} | {baseline}")
        
        print(f"\n💾 Results saved to: /content/drive/MyDrive/leave_site_out_results")
        return results
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


# Command line interface
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--setup', action='store_true', help='Setup only')
    parser.add_argument('--quick', action='store_true', help='Quick test')
    parser.add_argument('--full', action='store_true', help='Full experiment')
    args = parser.parse_args()
    
    if args.setup:
        setup_leave_site_out()
    elif args.quick:
        setup_leave_site_out()
        return run_quick_test()
    elif args.full:
        setup_leave_site_out()
        return run_full_experiment()
    else:
        print("🧠 Google Colab Leave-Site-Out Cross-Validation")
        print("Usage: python colab_leave_site_out.py [--setup|--quick|--full]")
        print("\nOr use functions directly in Colab:")
        print("  setup_leave_site_out()")
        print("  run_quick_test()")
        print("  run_full_experiment()")


if __name__ == "__main__":
    main()


# ================================================================
# GOOGLE COLAB NOTEBOOK USAGE
# ================================================================
"""
COPY AND PASTE INTO GOOGLE COLAB CELLS:

# Cell 1: Setup
from colab_leave_site_out import setup_leave_site_out
setup_leave_site_out()

# Cell 2: Quick Test
from colab_leave_site_out import run_quick_test
results = run_quick_test()

# Cell 3: Full Experiment (Optional)
from colab_leave_site_out import run_full_experiment
results = run_full_experiment()

OR RUN AS SCRIPT:

# Quick test
!python colab_leave_site_out.py --quick

# Full experiment
!python colab_leave_site_out.py --full
""" 
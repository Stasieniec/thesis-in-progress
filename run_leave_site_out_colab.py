#!/usr/bin/env python3
"""
🧠 Google Colab Script: Real Leave-Site-Out Cross-Validation
===========================================================

This script runs REAL leave-site-out cross-validation for ABIDE cross-attention models
directly in Google Colab using your existing data paths.

Usage in Google Colab:
    !python run_leave_site_out_colab.py
    !python run_leave_site_out_colab.py --quick-test
"""

import sys
import os
from pathlib import Path

# Add src to path (for Google Colab)
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def run_leave_site_out_experiment(quick_test=False):
    """
    Run leave-site-out cross-validation with Google Colab paths.
    
    Args:
        quick_test: If True, run with reduced epochs for testing
    """
    
    print("🧠 REAL LEAVE-SITE-OUT CROSS-VALIDATION")
    print("🌐 Optimized for Google Colab")
    print("=" * 60)
    
    # Google Colab paths (same as your existing experiments)
    FMRI_DATA_PATH = "/content/drive/MyDrive/b_data/ABIDE_pcp/cpac/filt_noglobal/rois_cc200"
    SMRI_DATA_PATH = "/content/drive/MyDrive/processed_smri_data_improved"  # Try improved first
    PHENOTYPIC_FILE = "/content/drive/MyDrive/b_data/ABIDE_pcp/Phenotypic_V1_0b_preprocessed1.csv"
    OUTPUT_DIR = "/content/drive/MyDrive/leave_site_out_results"
    
    print(f"📁 fMRI Data: {FMRI_DATA_PATH}")
    print(f"📁 sMRI Data: {SMRI_DATA_PATH}")
    print(f"📄 Phenotypic: {PHENOTYPIC_FILE}")
    print(f"💾 Output: {OUTPUT_DIR}")
    
    if quick_test:
        print("🚀 Quick test mode: 5 epochs, 2 models")
    
    print("=" * 60)
    
    try:
        # Import the leave-site-out experiments
        from scripts.leave_site_out_experiments import LeaveSiteOutExperiments
        
        # Create experiments instance
        experiments = LeaveSiteOutExperiments()
        
        # Load data with site information
        print("\n📊 Loading multimodal data with site extraction...")
        
        from utils.subject_matching import get_matched_datasets
        
        # Try improved sMRI first, fallback to original
        try:
            matched_data = get_matched_datasets(
                fmri_roi_dir=FMRI_DATA_PATH,
                smri_data_path=SMRI_DATA_PATH,
                phenotypic_file=PHENOTYPIC_FILE,
                verbose=True
            )
            print("✅ Using improved sMRI data (800 features)")
        except Exception as e:
            print(f"⚠️ Improved sMRI data not available: {e}")
            print("📊 Falling back to original sMRI data...")
            SMRI_DATA_PATH = "/content/drive/MyDrive/processed_smri_data"
            matched_data = get_matched_datasets(
                fmri_roi_dir=FMRI_DATA_PATH,
                smri_data_path=SMRI_DATA_PATH,
                phenotypic_file=PHENOTYPIC_FILE,
                verbose=True
            )
            print("✅ Using original sMRI data")
        
        # Configure experiment parameters
        if quick_test:
            config = {
                'num_epochs': 5,
                'batch_size': 16,
                'learning_rate': 0.001,
                'd_model': 128,
                'output_dir': Path(OUTPUT_DIR) / 'quick_test',
                'seed': 42,
                'verbose': True
            }
            print("\n🧪 Running quick test with 2 best models...")
        else:
            config = {
                'num_epochs': 50,
                'batch_size': 32,
                'learning_rate': 0.0005,
                'd_model': 256,
                'output_dir': Path(OUTPUT_DIR),
                'seed': 42,
                'verbose': True
            }
            print("\n🚀 Running full experiment with all 5 models...")
        
        # Extract site information
        subject_ids = matched_data['fmri_subject_ids']
        site_labels, site_mapping, site_stats = experiments.extract_site_info(
            subject_ids, PHENOTYPIC_FILE
        )
        
        # Validate site distribution
        print(f"\n🏥 Site Information:")
        print(f"   Total sites detected: {len(site_mapping)}")
        print(f"   Total subjects: {len(subject_ids)}")
        print(f"   Sites: {list(site_mapping.keys())}")
        
        # Check if we have enough sites for leave-site-out CV
        if len(site_mapping) < 3:
            print(f"\n❌ ERROR: Need at least 3 sites for leave-site-out CV, found {len(site_mapping)}")
            print("   This might indicate site extraction failed.")
            print("   Check if phenotypic file contains SITE_ID column.")
            return None
        
        # Check minimum subjects per site
        min_subjects = min(len(subjects) for subjects in site_mapping.values())
        if min_subjects < 2:
            print(f"\n❌ ERROR: Some sites have < 2 subjects (minimum: {min_subjects})")
            print("   This makes leave-site-out CV unreliable.")
            return None
        
        print(f"✅ Site validation passed: {len(site_mapping)} sites, min {min_subjects} subjects/site")
        
        # Run leave-site-out cross-validation
        if quick_test:
            # Test only the best performing models
            test_models = ['contrastive', 'hierarchical']  # Best from previous results
            results = {}
            
            for model_name in test_models:
                print(f"\n🧠 Testing {model_name} model...")
                model_results = experiments._run_model_leave_site_out(
                    model_name=model_name,
                    model_class=experiments.models[model_name],
                    matched_data=matched_data,
                    site_labels=site_labels,
                    site_mapping=site_mapping,
                    **config
                )
                results[model_name] = model_results
                
                # Quick summary
                avg_acc = model_results['cv_results']['test_accuracies'].mean()
                std_acc = model_results['cv_results']['test_accuracies'].std()
                print(f"   {model_name}: {avg_acc:.3f} ± {std_acc:.3f}")
        else:
            # Run full experiment
            results = experiments.run_leave_site_out_cv(
                matched_data=matched_data,
                phenotypic_file=PHENOTYPIC_FILE,
                **config
            )
        
        # Print summary
        print("\n" + "=" * 60)
        print("📊 LEAVE-SITE-OUT CROSS-VALIDATION RESULTS")
        print("=" * 60)
        
        for model_name, model_results in results.items():
            cv_results = model_results['cv_results']
            mean_acc = cv_results['test_accuracies'].mean()
            std_acc = cv_results['test_accuracies'].std()
            
            baseline_comparison = ""
            if mean_acc > 0.60:  # fMRI baseline
                baseline_comparison = "🎉 BEATS fMRI baseline!"
            elif mean_acc > 0.58:  # sMRI baseline
                baseline_comparison = "✅ Beats sMRI baseline"
            else:
                baseline_comparison = "📉 Below baselines"
            
            print(f"{model_name:12} | {mean_acc:.3f} ± {std_acc:.3f} | {baseline_comparison}")
        
        # Save results
        if config['output_dir']:
            config['output_dir'].mkdir(parents=True, exist_ok=True)
            experiments._save_results(results, config['output_dir'] / 'results.json')
            experiments.generate_summary(results, config['output_dir'])
            print(f"\n💾 Results saved to: {config['output_dir']}")
            print(f"📄 Executive summary: {config['output_dir']}/LEAVE_SITE_OUT_SUMMARY.md")
        
        print("\n🎉 Leave-site-out cross-validation completed!")
        print("   This provides the most realistic estimate of clinical deployment performance.")
        
        return results
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        
        print("\n🔧 Troubleshooting Tips:")
        print("1. Make sure Google Drive is mounted: drive.mount('/content/drive')")
        print("2. Check that data paths exist in your Google Drive")
        print("3. Verify you have the improved sMRI data or fallback to original")
        print("4. Try running with --quick-test first")
        return None


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Google Colab Leave-Site-Out CV')
    parser.add_argument('--quick-test', action='store_true',
                       help='Run quick test (5 epochs, 2 models)')
    
    args = parser.parse_args()
    
    # Check if in Google Colab
    try:
        import google.colab
        print("✅ Running in Google Colab")
    except ImportError:
        print("⚠️ Not in Google Colab - paths may need adjustment")
    
    # Check if Google Drive is mounted
    if not Path("/content/drive/MyDrive").exists():
        print("❌ Google Drive not mounted!")
        print("   Run: drive.mount('/content/drive')")
        return
    
    results = run_leave_site_out_experiment(quick_test=args.quick_test)
    return results


if __name__ == "__main__":
    main()


# ================================================================
# GOOGLE COLAB HELPER FUNCTIONS
# ================================================================

def quick_setup():
    """Quick setup function for Google Colab notebooks."""
    print("🚀 Setting up leave-site-out cross-validation...")
    
    # Mount Google Drive
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        print("✅ Google Drive mounted")
    except ImportError:
        print("⚠️ Not in Google Colab")
    
    # Install requirements if needed
    import subprocess
    import sys
    
    try:
        import torch
        import sklearn
        import seaborn
        print("✅ Required packages available")
    except ImportError:
        print("📦 Installing required packages...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", 
                             "torch", "torchvision", "scikit-learn", 
                             "seaborn", "matplotlib", "pandas", "numpy"])
    
    print("✅ Setup complete! Ready to run leave-site-out CV.")


def run_quick_test():
    """Run a quick test directly from a Colab cell."""
    return run_leave_site_out_experiment(quick_test=True)


def run_full_experiment():
    """Run the full experiment directly from a Colab cell."""
    return run_leave_site_out_experiment(quick_test=False)


# ================================================================
# COLAB NOTEBOOK USAGE EXAMPLES
# ================================================================
"""
GOOGLE COLAB USAGE EXAMPLES:

# Method 1: Command line style
!python run_leave_site_out_colab.py --quick-test
!python run_leave_site_out_colab.py

# Method 2: Direct function calls
from run_leave_site_out_colab import quick_setup, run_quick_test, run_full_experiment

# Setup
quick_setup()

# Quick test
results = run_quick_test()

# Full experiment  
results = run_full_experiment()

# Method 3: Step by step
from run_leave_site_out_colab import run_leave_site_out_experiment

# Quick test
results = run_leave_site_out_experiment(quick_test=True)

# Full experiment
results = run_leave_site_out_experiment(quick_test=False)
""" 
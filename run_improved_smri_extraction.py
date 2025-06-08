#!/usr/bin/env python3
"""
Simple runner script for the improved sMRI feature extraction.

This script runs the improved extraction method that follows the paper's methodology.
Can be easily executed locally to generate improved sMRI features.

Usage: python3 run_improved_smri_extraction.py
"""

import sys
import os
from pathlib import Path

# Add scripts directory to path
sys.path.append('scripts')

def main():
    """Run the improved sMRI extraction."""
    
    print("🚀 Improved sMRI Feature Extraction - Local Runner")
    print("="*60)
    
    # Check Python version
    if sys.version_info < (3, 6):
        print("❌ Python 3.6+ required")
        print("💡 Run with: python3 run_improved_smri_extraction.py")
        return
    
    # Import after path setup
    try:
        from improved_smri_extraction_new import ImprovedSMRIExtractor
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you're running from the thesis-in-progress directory")
        return
    
    # Configuration
    FREESURFER_PATH = "data/freesurfer_stats"
    OUTPUT_PATH = "processed_smri_data_improved"
    
    # Check if data exists
    if not os.path.exists(FREESURFER_PATH):
        print(f"❌ FreeSurfer data not found at: {FREESURFER_PATH}")
        print("Expected directory structure:")
        print("  data/freesurfer_stats/")
        print("    51149/")
        print("      aseg.stats")
        print("      lh.aparc.stats")
        print("      rh.aparc.stats")
        print("      wmparc.stats")
        print("    51197/")
        print("      ...")
        return
    
    # Check if we have some sample data
    sample_dirs = [d for d in os.listdir(FREESURFER_PATH) 
                   if os.path.isdir(os.path.join(FREESURFER_PATH, d))]
    
    if len(sample_dirs) == 0:
        print(f"❌ No subject directories found in {FREESURFER_PATH}")
        return
    
    print(f"✅ Found {len(sample_dirs)} subject directories")
    print(f"📁 FreeSurfer data: {FREESURFER_PATH}")
    print(f"💾 Output directory: {OUTPUT_PATH}")
    
    # Create and run extractor
    try:
        print("\n🔧 Initializing improved sMRI extractor...")
        extractor = ImprovedSMRIExtractor(
            freesurfer_path=FREESURFER_PATH,
            output_path=OUTPUT_PATH
        )
        
        print("\n🚀 Starting extraction pipeline...")
        extractor.run_complete_extraction()
        
        print("\n" + "="*60)
        print("🎉 SUCCESS! Improved sMRI extraction completed!")
        print("\n📋 Next steps:")
        print("  1. Check the results in the output directory")
        print("  2. Compare performance with your existing approach")
        print("  3. Use the new data for transformer training")
        print("\n📊 Expected improvements:")
        print("  - Better feature selection (RFE with Ridge)")
        print("  - 800 optimal features (as in paper)")
        print("  - Higher baseline accuracy (should be >70%)")
        print("  - Better transformer performance")
        
    except Exception as e:
        print(f"\n❌ Error during extraction: {e}")
        import traceback
        traceback.print_exc()
        
        print("\n🔧 Troubleshooting:")
        print("  1. Check that all required packages are installed:")
        print("     pip install numpy pandas sklearn matplotlib seaborn")
        print("  2. Ensure FreeSurfer data is in the correct format")
        print("  3. Check disk space for output files")


if __name__ == "__main__":
    main() 
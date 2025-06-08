#!/usr/bin/env python3
"""
Runner Script for Improved sMRI Data Extraction
===============================================

Simple interface to run the improved sMRI feature extraction pipeline.
This script orchestrates the complete processing workflow.

Usage:
    python scripts/run_improved_smri_extraction.py
    
The script will:
1. Process FreeSurfer statistics from data/freesurfer_stats/
2. Extract comprehensive features (aseg + aparc + wmparc)
3. Apply RFE feature selection to get 800 optimal features
4. Save results to processed_smri_data_improved/

Results are compatible with existing transformer training scripts.
"""

import os
import sys
import logging
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import the main extraction function
from scripts.improved_smri_extraction_new import main as run_extraction

def setup_logging():
    """Configure logging for the extraction process."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('smri_extraction.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )

def main():
    """
    Main function to run the improved sMRI extraction pipeline.
    """
    print("=" * 60)
    print("Improved sMRI Data Extraction Pipeline")
    print("=" * 60)
    
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # Check if FreeSurfer data directory exists
        freesurfer_dir = project_root / "data" / "freesurfer_stats"
        if not freesurfer_dir.exists():
            raise FileNotFoundError(f"FreeSurfer data directory not found: {freesurfer_dir}")
        
        logger.info(f"Starting sMRI extraction from: {freesurfer_dir}")
        logger.info("This process may take several minutes...")
        
        # Run the extraction
        run_extraction()
        
        logger.info("sMRI extraction completed successfully!")
        logger.info("Results saved to: processed_smri_data_improved/")
        
        # Provide next steps
        print("\n" + "=" * 60)
        print("EXTRACTION COMPLETE!")
        print("=" * 60)
        print("Next steps:")
        print("1. Review results in processed_smri_data_improved/")
        print("2. Use update_to_improved_smri.py to sync with Google Drive")
        print("3. Update your training scripts to use 800 features")
        print("4. Train your transformer model with improved sMRI data")
        
    except Exception as e:
        logger.error(f"Extraction failed: {str(e)}")
        print(f"\nERROR: {str(e)}")
        print("Check smri_extraction.log for detailed error information.")
        sys.exit(1)

if __name__ == "__main__":
    main() 
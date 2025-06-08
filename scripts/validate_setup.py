#!/usr/bin/env python3
"""
Setup validation script for ABIDE experiments.
Checks all imports and dependencies to ensure everything is working correctly.
"""

import sys
import traceback
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

def test_imports():
    """Test all critical imports."""
    import_tests = []
    
    # Core dependencies
    try:
        import torch
        import numpy as np
        import pandas as pd
        import sklearn
        import_tests.append(("✅ Core dependencies", True))
    except ImportError as e:
        import_tests.append(("❌ Core dependencies", str(e)))
    
    # Data processors
    try:
        from data.fmri_processor import FMRIDataProcessor
        from data.smri_processor import SMRIDataProcessor
        import_tests.append(("✅ Data processors", True))
    except ImportError as e:
        import_tests.append(("❌ Data processors", str(e)))
    
    # Models
    try:
        from models.fmri_transformer import SingleAtlasTransformer
        from models.smri_transformer import SMRITransformer
        from models.cross_attention import CrossAttentionTransformer
        import_tests.append(("✅ Model architectures", True))
    except ImportError as e:
        import_tests.append(("❌ Model architectures", str(e)))
    
    # Training modules
    try:
        from training.trainer import Trainer
        from training.utils import set_seed, create_data_loaders, create_multimodal_data_loaders
        import_tests.append(("✅ Training modules", True))
    except ImportError as e:
        import_tests.append(("❌ Training modules", str(e)))
    
    # Training functions
    try:
        from training.train_fmri import run_fmri_training
        from training.train_smri import run_smri_training  
        from training.train_cross_attention import run_cross_attention_training
        import_tests.append(("✅ Training functions", True))
    except ImportError as e:
        import_tests.append(("❌ Training functions", str(e)))
    
    # Evaluation
    try:
        from evaluation.metrics import calculate_metrics
        import_tests.append(("✅ Evaluation metrics", True))
    except ImportError as e:
        import_tests.append(("❌ Evaluation metrics", str(e)))
    
    # Subject matching
    try:
        from utils.subject_matching import get_matched_datasets, get_matched_subject_ids
        import_tests.append(("✅ Subject matching", True))
    except ImportError as e:
        import_tests.append(("❌ Subject matching", str(e)))
    
    return import_tests


def test_paths():
    """Test that key directories exist."""
    path_tests = []
    
    key_dirs = [
        "src/data",
        "src/models", 
        "src/training",
        "src/evaluation",
        "src/utils",
        "scripts",
        "docs",
        "configs"
    ]
    
    for dir_path in key_dirs:
        if Path(dir_path).exists():
            path_tests.append((f"✅ {dir_path}", True))
        else:
            path_tests.append((f"❌ {dir_path}", False))
    
    return path_tests


def test_main_scripts():
    """Test that main scripts can be imported."""
    script_tests = []
    
    try:
        # Test main experiment runner
        from run_experiments import run_all_experiments
        script_tests.append(("✅ Main experiment runner", True))
    except ImportError as e:
        script_tests.append(("❌ Main experiment runner", str(e)))
    
    try:
        # Test verification script
        sys.path.append(str(Path(__file__).parent.parent))
        from verify_matched_subjects import verify_subject_matching
        script_tests.append(("✅ Verification script", True))
    except ImportError as e:
        script_tests.append(("❌ Verification script", str(e)))
    
    return script_tests


def main():
    """Run all validation tests."""
    print("🧠 ABIDE Experiments Setup Validation")
    print("=" * 50)
    
    # Test imports
    print("\n📦 Testing Imports...")
    import_results = test_imports()
    for test_name, result in import_results:
        if isinstance(result, bool) and result:
            print(f"  {test_name}")
        else:
            print(f"  {test_name}: {result}")
    
    # Test paths
    print("\n📁 Testing Directory Structure...")
    path_results = test_paths()
    for test_name, result in path_results:
        print(f"  {test_name}")
    
    # Test main scripts
    print("\n🎯 Testing Main Scripts...")
    script_results = test_main_scripts()
    for test_name, result in script_results:
        if isinstance(result, bool) and result:
            print(f"  {test_name}")
        else:
            print(f"  {test_name}: {result}")
    
    # Summary
    print("\n" + "=" * 50)
    import_passed = sum(1 for _, result in import_results if isinstance(result, bool) and result)
    path_passed = sum(1 for _, result in path_results if result)
    script_passed = sum(1 for _, result in script_results if isinstance(result, bool) and result)
    
    total_tests = len(import_results) + len(path_results) + len(script_results)
    total_passed = import_passed + path_passed + script_passed
    
    if total_passed == total_tests:
        print(f"🎉 All {total_passed}/{total_tests} tests passed!")
        print("✅ Setup is ready for experiments")
    else:
        print(f"⚠️ {total_passed}/{total_tests} tests passed")
        print("❌ Some issues need to be resolved")
    
    print("\n🚀 Next Steps:")
    print("1. Run: python scripts/run_experiments.py --help")
    print("2. Or use the main runner function in Google Colab")
    print("3. Validate with: python verify_matched_subjects.py")


if __name__ == "__main__":
    main() 
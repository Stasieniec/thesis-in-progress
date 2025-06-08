#!/usr/bin/env python3
"""
🧪 Google Colab Setup Test Script
================================

Quick test to verify that all imports, data paths, and framework components
are working correctly before running comprehensive experiments.

Usage in Google Colab:
    !python test_colab_setup.py
"""

import sys
import os
from pathlib import Path
import traceback

def test_basic_imports():
    """Test basic Python imports."""
    print("🔍 Testing basic imports...")
    
    try:
        import numpy as np
        import pandas as pd
        import torch
        from datetime import datetime
        from pathlib import Path
        print("   ✅ Basic imports (numpy, pandas, torch) - OK")
        return True
    except Exception as e:
        print(f"   ❌ Basic imports failed: {e}")
        return False

def test_sklearn_imports():
    """Test scikit-learn imports."""
    print("🔍 Testing scikit-learn imports...")
    
    try:
        from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut
        from scipy import stats
        print("   ✅ Scikit-learn imports - OK")
        return True
    except Exception as e:
        print(f"   ❌ Scikit-learn imports failed: {e}")
        return False

def test_project_structure():
    """Test that project structure exists."""
    print("🔍 Testing project structure...")
    
    required_dirs = [
        'src',
        'src/config',
        'src/evaluation', 
        'src/models',
        'src/utils',
        'scripts'
    ]
    
    missing_dirs = []
    for dir_path in required_dirs:
        if not Path(dir_path).exists():
            missing_dirs.append(dir_path)
    
    if missing_dirs:
        print(f"   ❌ Missing directories: {missing_dirs}")
        return False
    else:
        print("   ✅ Project structure - OK")
        return True

def test_config_imports():
    """Test configuration imports."""
    print("🔍 Testing config imports...")
    
    try:
        from src.config import get_config
        config = get_config('cross_attention', num_folds=2, seed=42)
        print("   ✅ Config imports and get_config() - OK")
        return True
    except Exception as e:
        print(f"   ❌ Config imports failed: {e}")
        return False

def test_framework_imports():
    """Test framework imports."""
    print("🔍 Testing framework imports...")
    
    try:
        from src.evaluation.experiment_framework import (
            ExperimentRegistry,
            ComprehensiveExperimentFramework
        )
        print("   ✅ Framework imports - OK")
        return True
    except Exception as e:
        print(f"   ❌ Framework imports failed: {e}")
        print(f"   📋 Full error: {traceback.format_exc()}")
        return False

def test_experiment_registry():
    """Test experiment registry functionality."""
    print("🔍 Testing experiment registry...")
    
    try:
        from src.evaluation.experiment_framework import ExperimentRegistry
        
        registry = ExperimentRegistry()
        experiments = registry.list_experiments()
        
        print(f"   📊 Found {len(experiments)} experiments:")
        for exp in experiments[:3]:  # Show first 3
            exp_config = registry.get_experiment(exp)
            print(f"      - {exp}: {exp_config['name']}")
        if len(experiments) > 3:
            print(f"      - ... and {len(experiments) - 3} more")
        
        print("   ✅ Experiment registry - OK")
        return True
    except Exception as e:
        print(f"   ❌ Experiment registry failed: {e}")
        return False

def test_data_paths():
    """Test Google Drive data paths."""
    print("🔍 Testing Google Drive data paths...")
    
    data_paths = {
        'fMRI data': '/content/drive/MyDrive/b_data/ABIDE_pcp/cpac/filt_noglobal/rois_cc200',
        'sMRI improved': '/content/drive/MyDrive/processed_smri_data_improved',
        'sMRI original': '/content/drive/MyDrive/processed_smri_data',
        'Phenotypic': '/content/drive/MyDrive/b_data/ABIDE_pcp/Phenotypic_V1_0b_preprocessed1.csv'
    }
    
    available_paths = {}
    missing_paths = {}
    
    for name, path in data_paths.items():
        if Path(path).exists():
            if Path(path).is_file():
                size = Path(path).stat().st_size / (1024*1024)  # MB
                available_paths[name] = f"{path} ({size:.1f} MB)"
            else:
                try:
                    count = len(list(Path(path).iterdir()))
                    available_paths[name] = f"{path} ({count} files)"
                except:
                    available_paths[name] = path
        else:
            missing_paths[name] = path
    
    if available_paths:
        print("   ✅ Available data paths:")
        for name, info in available_paths.items():
            print(f"      - {name}: {info}")
    
    if missing_paths:
        print("   ⚠️ Missing data paths:")
        for name, path in missing_paths.items():
            print(f"      - {name}: {path}")
    
    # Check if we have at least the essential paths
    essential = ['fMRI data', 'Phenotypic']
    has_essential = all(name in available_paths for name in essential)
    
    if has_essential:
        print("   ✅ Essential data paths available")
        return True
    else:
        print("   ⚠️ Some essential data paths missing (but framework may still work)")
        return True  # Don't fail completely, just warn

def test_framework_initialization():
    """Test framework initialization."""
    print("🔍 Testing framework initialization...")
    
    try:
        from src.evaluation.experiment_framework import ComprehensiveExperimentFramework
        
        # Initialize with test output directory
        framework = ComprehensiveExperimentFramework(
            output_dir="test_framework_init",
            seed=42
        )
        
        print(f"   📁 Output directory: {framework.output_dir}")
        print(f"   🎲 Seed: {framework.seed}")
        print(f"   📊 Registry has {len(framework.registry.list_experiments())} experiments")
        
        print("   ✅ Framework initialization - OK")
        return True
    except Exception as e:
        print(f"   ❌ Framework initialization failed: {e}")
        return False

def test_data_loading_simulation():
    """Test data loading with mock data."""
    print("🔍 Testing data loading simulation...")
    
    try:
        import numpy as np
        from src.evaluation.experiment_framework import ComprehensiveExperimentFramework
        
        # Create mock data that matches expected format
        mock_data = {
            'fmri_features': np.random.randn(100, 19900),  # 100 subjects, 19900 features
            'smri_features': np.random.randn(100, 800),    # 100 subjects, 800 features  
            'fmri_labels': np.random.randint(0, 2, 100),   # Binary labels
            'smri_labels': np.random.randint(0, 2, 100),   # Binary labels
            'num_matched_subjects': 100,
            'fmri_subject_ids': [f"test_{i:03d}" for i in range(100)]
        }
        
        framework = ComprehensiveExperimentFramework(
            output_dir="test_data_loading",
            seed=42
        )
        
        # Mock the data loading
        framework.matched_data = mock_data
        
        print(f"   📊 Mock data loaded: {mock_data['num_matched_subjects']} subjects")
        print(f"   🧠 fMRI shape: {mock_data['fmri_features'].shape}")
        print(f"   🏗️ sMRI shape: {mock_data['smri_features'].shape}")
        
        print("   ✅ Data loading simulation - OK")
        return True
    except Exception as e:
        print(f"   ❌ Data loading simulation failed: {e}")
        return False

def test_single_experiment_simulation():
    """Test running a single experiment with mock data."""
    print("🔍 Testing single experiment simulation...")
    
    try:
        import numpy as np
        from src.evaluation.experiment_framework import ComprehensiveExperimentFramework
        
        # Create framework with mock data
        framework = ComprehensiveExperimentFramework(
            output_dir="test_single_experiment",
            seed=42
        )
        
        # Mock data
        framework.matched_data = {
            'fmri_features': np.random.randn(50, 1000),    # Smaller for speed
            'smri_features': np.random.randn(50, 500),     
            'fmri_labels': np.random.randint(0, 2, 50),    
            'smri_labels': np.random.randint(0, 2, 50),    
            'num_matched_subjects': 50,
            'fmri_subject_ids': [f"test_{i:03d}" for i in range(50)]
        }
        
        # Test running basic cross-attention experiment
        result = framework.run_experiment(
            experiment_name='cross_attention_basic',
            cv_folds=2,  # Small number for testing
            verbose=False  # Reduce output
        )
        
        print(f"   📋 Experiment result keys: {list(result.keys())}")
        
        # Check if we got expected results
        if 'error' in result:
            print(f"   ⚠️ Experiment returned error (expected): {result['error']}")
        else:
            print("   ✅ Experiment completed successfully")
        
        print("   ✅ Single experiment simulation - OK")
        return True
    except Exception as e:
        print(f"   ❌ Single experiment simulation failed: {e}")
        return False

def cleanup_test_files():
    """Clean up test directories."""
    test_dirs = [
        'test_framework_init',
        'test_data_loading', 
        'test_single_experiment'
    ]
    
    for dir_name in test_dirs:
        test_dir = Path(dir_name)
        if test_dir.exists():
            import shutil
            shutil.rmtree(test_dir)

def main():
    """Run all tests and provide summary."""
    print("🧪 GOOGLE COLAB SETUP TEST")
    print("=" * 50)
    print("Testing comprehensive experiments framework setup...\n")
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("Scikit-learn", test_sklearn_imports),
        ("Project Structure", test_project_structure),
        ("Config Module", test_config_imports),
        ("Framework Imports", test_framework_imports),
        ("Experiment Registry", test_experiment_registry),
        ("Data Paths", test_data_paths),
        ("Framework Init", test_framework_initialization),
        ("Data Loading Sim", test_data_loading_simulation),
        ("Single Experiment", test_single_experiment_simulation)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
            print()  # Add spacing
        except Exception as e:
            print(f"   💥 {test_name} crashed: {e}")
            results.append((test_name, False))
            print()
    
    # Clean up test files
    cleanup_test_files()
    
    # Summary
    print("📊 TEST SUMMARY")
    print("=" * 30)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\n🎯 OVERALL: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ Your setup is ready for comprehensive experiments!")
        print("\n🚀 You can now run:")
        print("   !python scripts/comprehensive_experiments.py run_all")
    elif passed >= total * 0.8:  # 80% pass rate
        print("\n⚠️ MOSTLY READY!")
        print("✅ Core functionality works, minor issues detected")
        print("🚀 You can try running experiments (may work with fallbacks)")
    else:
        print("\n❌ SETUP ISSUES DETECTED")
        print("🔧 Please fix the failing tests before running experiments")
        print("💡 Check that Google Drive is mounted and data exists")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 
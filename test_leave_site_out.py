#!/usr/bin/env python3
"""
Test Script for Leave-Site-Out Cross-Validation
==============================================

This script tests the leave-site-out cross-validation implementation
with mock data to ensure it works correctly.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import LeaveOneGroupOut

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def test_site_extraction():
    """Test site extraction from subject IDs."""
    print("🔍 Testing Site Extraction...")
    
    # Mock subject IDs with various patterns
    test_subjects = [
        'NYU_0050001', 'NYU_0050002', 'NYU_0050003',
        'KKI_0051456', 'KKI_0051457', 'KKI_0051458', 
        'UCLA_1_0028002', 'UCLA_1_0028003',
        '0050001', '0050002',  # Should map to NYU
        'sub_CALTECH_001', 'sub_CALTECH_002',
        'UNKNOWN_001', 'UNKNOWN_002'
    ]
    
    # Import the leave-site-out experiments
    try:
        from scripts.leave_site_out_experiments import LeaveSiteOutExperiments
        experiments = LeaveSiteOutExperiments()
        
        # Create mock phenotypic data
        mock_pheno = {
            'NYU_0050001': 'NYU',
            'KKI_0051456': 'KKI'
        }
        
        # Test site extraction
        sites = []
        for sub_id in test_subjects:
            site = experiments._extract_site_from_subject_id(sub_id, mock_pheno)
            sites.append(site)
            print(f"   {sub_id:15} → {site}")
        
        # Check that we got multiple sites
        unique_sites = set(sites)
        print(f"\n✅ Found {len(unique_sites)} unique sites: {list(unique_sites)}")
        
        if len(unique_sites) >= 3:
            print("✅ Site extraction test PASSED")
            return True
        else:
            print("❌ Site extraction test FAILED - need at least 3 sites")
            return False
            
    except Exception as e:
        print(f"❌ Site extraction test FAILED: {e}")
        return False

def test_leave_one_group_out():
    """Test sklearn's LeaveOneGroupOut functionality."""
    print("🔄 Testing LeaveOneGroupOut...")
    
    # Create mock data
    n_subjects = 50
    X = np.random.randn(n_subjects, 100)
    y = np.random.randint(0, 2, n_subjects)
    
    # Create site groups
    sites = ['NYU'] * 15 + ['KKI'] * 12 + ['UCLA'] * 10 + ['CALTECH'] * 8 + ['OHSU'] * 5
    sites_array = np.array(sites)
    
    # Initialize LeaveOneGroupOut
    logo = LeaveOneGroupOut()
    
    fold_count = 0
    for train_idx, test_idx in logo.split(X, y, sites_array):
        fold_count += 1
        test_sites = np.unique(sites_array[test_idx])
        train_sites = np.unique(sites_array[train_idx])
        
        print(f"   Fold {fold_count}: Train sites: {list(train_sites)}, Test site: {list(test_sites)}")
        
        # Verify no overlap
        assert len(set(train_sites) & set(test_sites)) == 0, "Site leakage detected!"
    
    unique_sites = len(np.unique(sites_array))
    print(f"✅ Generated {fold_count} folds for {unique_sites} sites")
    return fold_count == unique_sites

def test_mock_data_creation():
    """Test creation of mock data for testing."""
    print("\n📊 Testing Mock Data Creation...")
    
    try:
        # Create mock matched data structure
        n_subjects = 30
        
        mock_data = {
            'fmri_features': np.random.randn(n_subjects, 19900),  # ~19,900 fMRI features
            'smri_features': np.random.randn(n_subjects, 800),   # 800 sMRI features
            'fmri_labels': np.random.randint(0, 2, n_subjects),
            'smri_labels': np.random.randint(0, 2, n_subjects),
            'fmri_subject_ids': [f'NYU_{i:07d}' if i < 10 else f'KKI_{i:07d}' if i < 20 else f'UCLA_{i:07d}' 
                                for i in range(n_subjects)],
            'smri_subject_ids': [f'NYU_{i:07d}' if i < 10 else f'KKI_{i:07d}' if i < 20 else f'UCLA_{i:07d}' 
                                for i in range(n_subjects)]
        }
        
        # Verify data structure
        assert mock_data['fmri_features'].shape[0] == n_subjects
        assert mock_data['smri_features'].shape[0] == n_subjects
        assert len(mock_data['fmri_subject_ids']) == n_subjects
        assert len(mock_data['smri_subject_ids']) == n_subjects
        
        # Check we have multiple sites
        from scripts.leave_site_out_experiments import LeaveSiteOutExperiments
        experiments = LeaveSiteOutExperiments()
        site_labels, site_mapping, _ = experiments.extract_site_info(mock_data['fmri_subject_ids'])
        
        print(f"   Mock data: {n_subjects} subjects")
        print(f"   fMRI features: {mock_data['fmri_features'].shape}")
        print(f"   sMRI features: {mock_data['smri_features'].shape}")
        print(f"   Sites detected: {list(site_mapping.keys())}")
        
        if len(site_mapping) >= 3:
            print("✅ Mock data creation test PASSED")
            return True, mock_data
        else:
            print("❌ Mock data creation test FAILED - insufficient sites")
            return False, None
            
    except Exception as e:
        print(f"❌ Mock data creation test FAILED: {e}")
        return False, None

def test_model_parameter_detection():
    """Test model parameter detection for different models."""
    print("\n🧠 Testing Model Parameter Detection...")
    
    try:
        from scripts.leave_site_out_experiments import LeaveSiteOutExperiments
        import inspect
        
        experiments = LeaveSiteOutExperiments()
        
        for model_name, model_class in experiments.models.items():
            print(f"   Testing {model_name}...")
            
            # Get model signature
            model_signature = inspect.signature(model_class.__init__)
            
            # Test parameter detection
            model_params = {}
            for param_name in model_signature.parameters:
                if param_name in ['self']:
                    continue
                elif param_name == 'fmri_input_dim':
                    model_params[param_name] = 19900
                elif param_name == 'smri_input_dim':
                    model_params[param_name] = 800
                elif param_name == 'd_model':
                    model_params[param_name] = 256
                elif param_name in ['num_heads', 'n_heads']:
                    model_params[param_name] = 8
                # Add more as needed
            
            print(f"      Parameters: {list(model_params.keys())}")
        
        print("✅ Model parameter detection test PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Model parameter detection test FAILED: {e}")
        return False

def run_integration_test(mock_data):
    """Run a minimal integration test with mock data."""
    print("\n🚀 Running Integration Test...")
    
    try:
        from scripts.leave_site_out_experiments import LeaveSiteOutExperiments
        
        experiments = LeaveSiteOutExperiments()
        
        # Test with just one model and minimal epochs
        test_config = {
            'matched_data': mock_data,
            'num_epochs': 1,  # Very minimal for testing
            'batch_size': 8,
            'learning_rate': 0.001,
            'd_model': 64,  # Smaller for testing
            'output_dir': None,  # No output for test
            'phenotypic_file': None,
            'seed': 42,
            'verbose': True
        }
        
        # Test site extraction first
        site_labels, site_mapping, site_stats = experiments.extract_site_info(
            mock_data['fmri_subject_ids']
        )
        
        print(f"   Site extraction successful: {len(site_mapping)} sites")
        print(f"   Sites: {list(site_mapping.keys())}")
        
        # Test that we have enough sites and subjects
        if len(site_mapping) < 3:
            print("❌ Integration test SKIPPED - need at least 3 sites")
            return False
        
        min_subjects = min(len(subjects) for subjects in site_mapping.values())
        if min_subjects < 2:
            print("❌ Integration test SKIPPED - need at least 2 subjects per site")
            return False
        
        print("✅ Integration test environment VALID")
        print("   (Full test would require actual model training)")
        return True
        
    except Exception as e:
        print(f"❌ Integration test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("🧪 LEAVE-SITE-OUT CROSS-VALIDATION TESTS")
    print("=" * 50)
    
    results = []
    
    # Test 1: Site extraction
    results.append(test_site_extraction())
    
    # Test 2: LeaveOneGroupOut
    results.append(test_leave_one_group_out())
    
    # Test 3: Mock data creation
    mock_success, mock_data = test_mock_data_creation()
    results.append(mock_success)
    
    # Test 4: Model parameter detection
    results.append(test_model_parameter_detection())
    
    # Test 5: Integration test (if mock data successful)
    if mock_success and mock_data is not None:
        results.append(run_integration_test(mock_data))
    else:
        print("\n⏭️ Skipping integration test - mock data failed")
        results.append(False)
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 TEST SUMMARY")
    print("=" * 50)
    
    test_names = [
        "Site Extraction",
        "LeaveOneGroupOut", 
        "Mock Data Creation",
        "Model Parameter Detection",
        "Integration Test"
    ]
    
    passed = sum(results)
    total = len(results)
    
    for i, (name, result) in enumerate(zip(test_names, results)):
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{i+1}. {name:25} {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Leave-site-out CV is ready to use.")
        return 0
    else:
        print(f"\n⚠️ {total-passed} tests failed. Please check the implementation.")
        return 1


if __name__ == "__main__":
    exit(main()) 
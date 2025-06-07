#!/usr/bin/env python3
"""
Diagnose why cross-attention performance dropped from 63.6% to 57.7%.
Test if it's due to sMRI preprocessing mismatch.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

import os
import json

def analyze_cross_attention_issue():
    """Analyze potential causes of cross-attention performance drop."""
    print("🔍 DIAGNOSING Cross-Attention Performance Drop")
    print("=" * 60)
    
    print("📊 Performance Summary:")
    print("   Original Cross-Attention:    63.6%")
    print("   Complex Improved:            54.1% (-9.5 points)")  
    print("   Minimal Improved:            57.7% (-5.9 points)")
    print("   Pure fMRI Baseline:          65.0% (target to beat)")
    print()
    
    print("🎯 HYPOTHESIS: sMRI Preprocessing Mismatch")
    print("-" * 45)
    print("Theory: Enhanced sMRI preprocessing creates distribution shift")
    print("that breaks cross-attention model expectations.")
    print()
    
    print("📋 Evidence:")
    print("   ✓ Enhanced sMRI alone: 49% → 54% (works)")
    print("   ✓ Original cross-attention: 63.6% (worked)")  
    print("   ✗ Enhanced sMRI + cross-attention: 57.7% (broken)")
    print("   → Suggests: preprocessing incompatibility")
    print()
    
    print("🧪 RECOMMENDED TESTS:")
    print()
    
    print("1. 🔄 TEST ORIGINAL PREPROCESSING:")
    print("   - Use original sMRI preprocessing in cross-attention")
    print("   - Should recover ~63.6% performance")
    print("   - Command: Test with StandardScaler instead of RobustScaler")
    print()
    
    print("2. 📊 FEATURE DISTRIBUTION ANALYSIS:")
    print("   - Compare sMRI feature distributions:")
    print("     * Original preprocessing (StandardScaler)")
    print("     * Enhanced preprocessing (RobustScaler + F-score + MI)")
    print("   - Look for significant distribution shifts")
    print()
    
    print("3. 🎯 HYPERPARAMETER GRID SEARCH:")
    print("   - Keep original preprocessing")
    print("   - Search learning rate: [1e-5, 5e-5, 1e-4]")
    print("   - Search batch size: [16, 32, 64]")
    print("   - Target: Beat 65% fMRI baseline")
    print()
    
    print("🚀 IMMEDIATE ACTION PLAN:")
    print("-" * 30)
    print("Step 1: Test with original sMRI preprocessing")
    print("Step 2: If that works, gradually introduce improvements")
    print("Step 3: If that fails, focus on hyperparameter tuning")
    print()
    
    print("💡 EXPECTED OUTCOMES:")
    print("   IF preprocessing mismatch:")
    print("     → Original preprocessing: ~63.6% ✅")
    print("     → Enhanced preprocessing: ~57.7% ❌")
    print("   ")
    print("   IF hyperparameter issue:")
    print("     → Both preprocessings: ~57-60% ⚠️") 
    print("     → Better hyperparams: >65% ✅")
    print()
    
    print("🎯 ULTIMATE GOAL:")
    print("   Find combination that beats 65% fMRI baseline")
    print("   Even 66-67% would be a clear multimodal success!")

def create_test_script():
    """Create a script to test original preprocessing."""
    test_script = '''#!/usr/bin/env python3
"""
Test cross-attention with ORIGINAL sMRI preprocessing.
This should help diagnose if the issue is preprocessing mismatch.
"""

# Modify train_cross_attention.py to use:
# - StandardScaler instead of RobustScaler  
# - Basic feature selection instead of F-score + MI
# - Original cross-attention architecture

print("🧪 Testing original preprocessing hypothesis...")
print("Expected: Recovery to ~63.6% if preprocessing was the issue")
'''
    
    with open('test_original_preprocessing.py', 'w') as f:
        f.write(test_script)
    
    print("📝 Created test_original_preprocessing.py")

if __name__ == "__main__":
    analyze_cross_attention_issue()
    create_test_script()
    
    print("\n" + "="*60)
    print("🔍 DIAGNOSIS COMPLETE")
    print("📋 Next: Test preprocessing mismatch hypothesis")
    print("🎯 Goal: Recover 63.6% performance, then beat 65% fMRI") 
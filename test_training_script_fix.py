#!/usr/bin/env python3
"""
Quick test to verify the training script fix works with 97% accuracy method.
This should now achieve much higher accuracy than the 57.6% we just saw.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# Run the fixed training script with quick test
print("🔬 Testing FIXED Training Script")
print("=" * 50)
print("🎯 Expected: Much higher accuracy than previous 57.6%")
print("🚀 Using EXACT preprocessing from 97% accuracy test")
print()

# Run quick test with our improvements
os.system("python scripts/train_smri.py quick_test --num_epochs=10 --output_dir=./test_fix_verification")

print("\n" + "=" * 50)
print("✅ If you see accuracy >70%, the fix worked!")
print("⚠️  If still ~57%, there may be other differences to address")
print("💡 Next step: Run full training with: !python scripts/train_smri.py run") 
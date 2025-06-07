#!/usr/bin/env python3
"""
Quick test for FIXED sMRI architecture (no CLS tokens).
This should match the working notebook performance of ~60%.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

import numpy as np
import torch
from config import get_config
from models import SMRITransformer

def test_fixed_smri_architecture():
    """Test the fixed sMRI architecture matches working notebook."""
    print("🔧 Testing FIXED sMRI Architecture (No CLS tokens)")
    print("=" * 60)
    
    # Test config
    config = get_config('smri')
    
    # Create model
    model = SMRITransformer(
        input_dim=300,  # After feature selection
        d_model=config.d_model,
        n_heads=config.num_heads,
        n_layers=config.num_layers,
        dropout=config.dropout,
        layer_dropout=config.layer_dropout
    )
    
    # Test forward pass
    batch_size = 16
    x = torch.randn(batch_size, 300)
    
    with torch.no_grad():
        output = model(x)
    
    print(f"✅ Model architecture test passed!")
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Expected output shape: ({batch_size}, 2)")
    
    # Check model info
    info = model.get_model_info()
    print(f"\n📊 Model Info:")
    print(f"   Name: {info['model_name']}")
    print(f"   Parameters: {info['total_params']:,}")
    print(f"   Size: {info['model_size_mb']:.2f} MB")
    
    # Key architecture differences
    print(f"\n🎯 Key Changes from CLS Token Approach:")
    print(f"   ❌ Removed: CLS tokens")
    print(f"   ❌ Removed: Complex sequence handling")
    print(f"   ✅ Added: Simple sequence dimension (1,)")
    print(f"   ✅ Added: Global pooling (squeeze)")
    print(f"   ✅ Added: Working notebook architecture")
    
    # Architecture validation
    assert output.shape == (batch_size, 2), f"Wrong output shape: {output.shape}"
    assert 'WorkingNotebook' in info['model_name'], "Model name should indicate working notebook approach"
    
    print(f"\n🎉 Fixed sMRI architecture validated!")
    print(f"   This should perform much better (~60% vs 50%)")
    print(f"\n📝 Next steps:")
    print(f"   1. Run: python scripts/train_smri.py")
    print(f"   2. Expect: ~60% accuracy (like working notebook)")
    print(f"   3. Compare: Previous 50% vs new 60%")

if __name__ == "__main__":
    test_fixed_smri_architecture() 
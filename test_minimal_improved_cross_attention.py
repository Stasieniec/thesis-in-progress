#!/usr/bin/env python3
"""
Test the MINIMAL Improved Cross-Attention model (conservative fixes to beat fMRI).
This should be much closer to the original 63.6% performance while adding small improvements.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

import torch
import numpy as np
from models.minimal_improved_cross_attention import MinimalImprovedCrossAttentionTransformer

def test_minimal_model():
    """Test that the minimal improved model works correctly."""
    print("🧪 Testing Minimal Improved Cross-Attention Model")
    print("=" * 55)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"📱 Device: {device}")
    
    # Create model with realistic dimensions
    fmri_dim = 6670  # Typical fMRI feature size
    smri_dim = 300   # Enhanced sMRI features
    
    model = MinimalImprovedCrossAttentionTransformer(
        fmri_dim=fmri_dim,
        smri_dim=smri_dim,
        d_model=256,
        n_heads=8,
        n_layers=4,
        n_cross_layers=2,
        dropout=0.2
    ).to(device)
    
    # Get model info
    model_info = model.get_model_info()
    print(f"\n🧠 Model: {model_info['model_name']}")
    print(f"   Parameters: {model_info['total_params']:,}")
    print(f"   fMRI dim: {model_info['fmri_dim']}")
    print(f"   sMRI dim: {model_info['smri_dim']}")
    
    print(f"\n✅ Conservative Improvements:")
    for improvement in model_info['improvements']:
        print(f"   • {improvement}")
    
    # Test forward pass
    batch_size = 8
    fmri_features = torch.randn(batch_size, fmri_dim).to(device)
    smri_features = torch.randn(batch_size, smri_dim).to(device)
    
    print(f"\n🔬 Testing forward pass...")
    print(f"   Batch size: {batch_size}")
    print(f"   fMRI input: {fmri_features.shape}")
    print(f"   sMRI input: {smri_features.shape}")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        logits, attention_info = model(fmri_features, smri_features, return_attention=True)
    
    print(f"\n📊 Forward pass results:")
    print(f"   Output logits: {logits.shape}")
    print(f"   Fusion weights: {attention_info['fusion_weights']}")
    print(f"   fMRI contribution: {attention_info['fmri_contribution']:.4f}")
    print(f"   sMRI contribution: {attention_info['smri_contribution']:.4f}")
    
    # Test that fusion weights favor fMRI (65%) over sMRI (54%)
    fusion_weights = attention_info['fusion_weights']
    fmri_weight = fusion_weights[0].item()
    smri_weight = fusion_weights[1].item()
    
    print(f"\n🎯 Fusion Weight Analysis:")
    print(f"   fMRI weight: {fmri_weight:.3f} (should be ~0.55)")
    print(f"   sMRI weight: {smri_weight:.3f} (should be ~0.45)")
    print(f"   Sum: {fmri_weight + smri_weight:.3f} (should be 1.0)")
    
    # Check that weights are reasonable
    if 0.50 <= fmri_weight <= 0.60 and 0.40 <= smri_weight <= 0.50:
        print(f"   ✅ CORRECT: Fusion weights favor fMRI appropriately")
        status = "WORKING"
    else:
        print(f"   ⚠️  Fusion weights seem off")
        status = "NEEDS_CHECK"
    
    print(f"\n💡 Expected Behavior:")
    print(f"   • Should perform very close to original (63.6%)")
    print(f"   • Weighted fusion should provide 2-3% boost")
    print(f"   • Target: 66-68% accuracy (beat 65% fMRI baseline)")
    print(f"   • Conservative approach - minimal risk")
    
    print(f"\n🚀 Model Status: {status}")
    print(f"   Ready for training: {'✅ YES' if status == 'WORKING' else '⚠️ MAYBE'}")
    
    return status == "WORKING"

if __name__ == "__main__":
    success = test_minimal_model()
    print(f"\n{'✅ SUCCESS' if success else '⚠️ ISSUES'}: Minimal improved cross-attention test completed")
    print(f"\n💡 Next step: Run conservative training with:")
    print(f"   !python scripts/train_cross_attention.py quick_test")
    print(f"   Expected: 63.6% → 66-68% (small but consistent improvement!)") 
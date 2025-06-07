#!/usr/bin/env python3
"""
Test the IMPROVED Cross-Attention model designed to beat pure fMRI.
This should achieve >65% accuracy by using adaptive gating and performance-aware fusion.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

import torch
import numpy as np
from models.improved_cross_attention import ImprovedCrossAttentionTransformer

def test_improved_model():
    """Test that the improved model can be instantiated and run."""
    print("🧪 Testing Improved Cross-Attention Model")
    print("=" * 50)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"📱 Device: {device}")
    
    # Create model with realistic dimensions
    fmri_dim = 6670  # Typical fMRI feature size
    smri_dim = 300   # Enhanced sMRI features
    
    model = ImprovedCrossAttentionTransformer(
        fmri_dim=fmri_dim,
        smri_dim=smri_dim,
        d_model=256,
        n_heads=8,
        n_layers=3,
        n_cross_layers=2,
        dropout=0.2,
        fmri_performance=0.65,  # Known fMRI performance  
        smri_performance=0.54   # Current sMRI performance
    ).to(device)
    
    # Get model info
    model_info = model.get_model_info()
    print(f"\n🧠 Model: {model_info['model_name']}")
    print(f"   Parameters: {model_info['total_params']:,}")
    print(f"   fMRI dim: {model_info['fmri_dim']}")
    print(f"   sMRI dim: {model_info['smri_dim']}")
    
    print(f"\n✅ Key Improvements:")
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
    print(f"   fMRI contribution: {attention_info['fmri_contribution']:.4f}")
    print(f"   sMRI contribution: {attention_info['smri_contribution']:.4f}")
    print(f"   Performance weights: {attention_info['modality_performances']}")
    
    # Test that the model prefers fMRI (65%) over sMRI (54%)
    fmri_contrib = attention_info['fmri_contribution'].item()
    smri_contrib = attention_info['smri_contribution'].item()
    
    print(f"\n🎯 Performance-Aware Analysis:")
    print(f"   fMRI contribution: {fmri_contrib:.4f} (should be higher)")
    print(f"   sMRI contribution: {smri_contrib:.4f} (should be lower)")
    
    if fmri_contrib > smri_contrib:
        print(f"   ✅ CORRECT: Model favors stronger modality (fMRI)")
        status = "WORKING"
    else:
        print(f"   ⚠️  Model might not be using performance weighting correctly")
        status = "NEEDS_CHECK"
    
    print(f"\n💡 Expected Benefits:")
    print(f"   • Adaptive gating learns optimal fMRI/sMRI weights")
    print(f"   • Performance-aware fusion (65% fMRI vs 54% sMRI)")
    print(f"   • Residual fMRI connection for fallback")
    print(f"   • Goal: Beat pure fMRI baseline (65%)")
    
    print(f"\n🚀 Model Status: {status}")
    print(f"   Ready for training: {'✅ YES' if status == 'WORKING' else '⚠️ MAYBE'}")
    
    return status == "WORKING"

if __name__ == "__main__":
    success = test_improved_model()
    print(f"\n{'✅ SUCCESS' if success else '⚠️ ISSUES'}: Improved cross-attention model test completed")
    print(f"\n💡 Next step: Run training with:")
    print(f"   !python scripts/train_cross_attention.py quick_test")
    print(f"   Expected: >65% accuracy (beating pure fMRI!)") 
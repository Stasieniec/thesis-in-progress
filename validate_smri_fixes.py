#!/usr/bin/env python3
"""Validate sMRI fixes are working correctly."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def validate_fixes():
    print("🔍 Validating sMRI fixes...")
    
    # Test imports
    try:
        from config import get_config
        from data import SMRIDataProcessor
        from models import SMRITransformer
        from training import Trainer
        print("✅ All imports successful")
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False
    
    # Test config improvements
    try:
        config = get_config('smri')
        assert hasattr(config, 'use_class_weights')
        assert config.use_class_weights == True
        assert hasattr(config, 'warmup_epochs')
        assert config.weight_decay == 1e-4
        print("✅ Config improvements applied")
    except Exception as e:
        print(f"❌ Config error: {e}")
        return False
    
    # Test model with scaling
    try:
        import torch
        model = SMRITransformer(input_dim=300)
        assert hasattr(model, 'scale')
        print("✅ Model improvements applied")
    except Exception as e:
        print(f"❌ Model error: {e}")
        return False
    
    print("🎉 All sMRI fixes validated successfully!")
    return True

if __name__ == "__main__":
    validate_fixes() 
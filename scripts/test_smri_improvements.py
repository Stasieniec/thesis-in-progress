#!/usr/bin/env python3
"""
Quick test script for sMRI improvements.
Tests the fixes applied to improve sMRI performance from ~52% to hopefully ~60%+

Usage in Google Colab:
  !python scripts/test_smri_improvements.py
"""

import sys
from pathlib import Path

# Add src to path (for Google Colab)
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# Import after path setup
import numpy as np
import torch
from config import get_config
from data import SMRIDataProcessor
from models import SMRITransformer
from training import Trainer
from training.utils import create_data_loaders, set_seed
from sklearn.model_selection import train_test_split

def test_smri_improvements():
    """Quick test to verify sMRI improvements."""
    print("🧠 Testing sMRI Improvements...")
    print("=" * 60)
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Get config with improvements
    config = get_config('smri')
    print(f"📁 Output directory: {config.output_dir}")
    print(f"🔧 Config improvements:")
    print(f"   - Class weights: {config.use_class_weights}")
    print(f"   - Weight decay: {config.weight_decay}")
    print(f"   - Warmup epochs: {config.warmup_epochs}")
    print(f"   - Feature selection: {config.feature_selection_k}")
    
    # Load sMRI data
    print("\n📊 Loading sMRI data...")
    processor = SMRIDataProcessor(
        data_path=config.smri_data_path,
        feature_selection_k=config.feature_selection_k,
        scaler_type=config.scaler_type
    )
    
    features, labels, subject_ids = processor.process_all_subjects(
        phenotypic_file=config.phenotypic_file,
        verbose=True
    )
    
    print(f"✅ Loaded {len(features)} subjects")
    print(f"📊 Original feature dimension: {features.shape[1]}")
    print(f"📊 Class distribution: ASD={np.sum(labels)}, Control={len(labels)-np.sum(labels)}")
    print(f"📊 Class balance: {np.sum(labels)/len(labels)*100:.1f}% ASD")
    
    # Quick data split for testing
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    # Apply preprocessing
    print("\n🔄 Applying preprocessing with improvements...")
    X_train_processed = processor.fit(X_train, y_train)
    X_val_processed = processor.transform(X_val)
    X_test_processed = processor.transform(X_test)
    
    print(f"📊 Processed feature dimension: {X_train_processed.shape[1]}")
    print(f"📊 Train/Val/Test sizes: {len(X_train_processed)}/{len(X_val_processed)}/{len(X_test_processed)}")
    
    # Create data loaders
    print("📦 Creating data loaders with improved settings...")
    train_loader, val_loader = create_data_loaders(
        X_train_processed, y_train, X_val_processed, y_val,
        batch_size=config.batch_size,
        augment_train=True,
        dataset_type='smri'
    )
    
    # Initialize improved model
    print("🤖 Initializing improved sMRI transformer...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🎯 Using device: {device}")
    
    model = SMRITransformer(
        input_dim=X_train_processed.shape[1],
        d_model=config.d_model,
        n_heads=config.num_heads,
        n_layers=config.num_layers,
        dropout=config.dropout,
        layer_dropout=config.layer_dropout
    )
    
    model_info = model.get_model_info()
    print(f"📊 Model info:")
    print(f"   - Parameters: {model_info['total_params']:,}")
    print(f"   - Model size: {model_info['model_size_mb']:.2f} MB")
    
    # Initialize improved trainer
    print("🏋️ Initializing trainer with class weights and label smoothing...")
    trainer = Trainer(model, device, config, model_type='single')
    
    # Quick training test (just a few epochs)
    print("\n🚀 Running quick training test (5 epochs)...")
    checkpoint_path = config.output_dir / 'test_model.pt'
    
    try:
        history = trainer.fit(
            train_loader, val_loader,
            num_epochs=5,  # Quick test
            checkpoint_path=checkpoint_path,
            y_train=y_train  # This triggers class weights
        )
        
        print("\n✅ Training test completed successfully!")
        print(f"📊 Final validation metrics:")
        if 'val_accuracy' in history:
            print(f"   - Accuracy: {history['val_accuracy'][-1]:.4f}")
        if 'val_balanced_accuracy' in history:
            print(f"   - Balanced Accuracy: {history['val_balanced_accuracy'][-1]:.4f}")
        if 'val_auc' in history:
            print(f"   - AUC: {history['val_auc'][-1]:.4f}")
            
        # Test evaluation
        test_loader, _ = create_data_loaders(
            X_test_processed, y_test, X_test_processed, y_test,
            batch_size=config.batch_size,
            augment_train=False,
            dataset_type='smri'
        )
        
        test_metrics = trainer.evaluate_final(test_loader)
        print(f"\n🎯 Quick test results:")
        print(f"   - Test Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"   - Test Balanced Accuracy: {test_metrics['balanced_accuracy']:.4f}")
        print(f"   - Test AUC: {test_metrics['auc']:.4f}")
        
        if test_metrics['accuracy'] > 0.55:
            print("\n🎉 Good! Test accuracy > 55% - improvements look promising!")
        elif test_metrics['accuracy'] > 0.52:
            print("\n✅ Okay - slight improvement over baseline 52%")
        else:
            print("\n⚠️ Test accuracy still low - may need more training or adjustments")
            
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        print("Check your data paths and configurations.")
        return False
    
    print("\n📝 Summary of improvements applied:")
    print("   ✅ Class weights with label smoothing")
    print("   ✅ Improved input scaling")
    print("   ✅ Better feature selection")
    print("   ✅ Reduced data augmentation noise")
    print("   ✅ Weight decay regularization")
    print("   ✅ Warmup learning rate scheduling")
    print("   ✅ Data quality checks")
    
    print(f"\n🎯 Ready to run full experiment:")
    print("   python scripts/train_smri.py run")
    
    return True

if __name__ == "__main__":
    success = test_smri_improvements()
    if success:
        print("\n✅ All improvements applied successfully!")
    else:
        print("\n❌ Some issues found - check logs above") 
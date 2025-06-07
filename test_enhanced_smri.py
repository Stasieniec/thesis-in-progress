#!/usr/bin/env python3
"""
Enhanced sMRI test based on data creation script insights.
This should achieve the 60% performance from the working notebook.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

import numpy as np
import torch
from config import get_config
from data import SMRIDataProcessor
from models import SMRITransformer
from training import Trainer
from training.utils import create_data_loaders, set_seed
from sklearn.model_selection import train_test_split

def test_enhanced_smri():
    """Test enhanced sMRI with all data creation script improvements."""
    print("🚀 Enhanced sMRI Test (Data Creation Script Optimizations)")
    print("=" * 70)
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Get enhanced config
    config = get_config('smri')
    
    print(f"📊 Enhanced Configuration:")
    print(f"   Feature selection: {config.feature_selection_k} features")
    print(f"   Scaler type: {config.scaler_type}")
    print(f"   Label smoothing: {config.label_smoothing}")
    print(f"   Learning rate: {config.learning_rate}")
    print(f"   Batch size: {config.batch_size}")
    
    # Load and process data with enhanced preprocessing
    print(f"\n📁 Loading sMRI data with enhanced preprocessing...")
    processor = SMRIDataProcessor(
        data_path=config.smri_data_path,
        feature_selection_k=config.feature_selection_k,
        scaler_type=config.scaler_type
    )
    
    # Load data
    features, labels, subject_ids = processor.process_all_subjects(
        phenotypic_file=config.phenotypic_file
    )
    
    print(f"✅ Loaded {len(labels)} subjects")
    print(f"📊 Original features: {features.shape[1]}")
    print(f"🎯 Class distribution: ASD={np.sum(labels)}, Control={len(labels) - np.sum(labels)}")
    
    # Enhanced data split (stratified)
    X_temp, X_test, y_temp, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.2, random_state=42, stratify=y_temp
    )
    
    print(f"\n📊 Enhanced Data Split:")
    print(f"   Train: {X_train.shape[0]} ({np.sum(y_train==1)} ASD, {np.sum(y_train==0)} Control)")
    print(f"   Val: {X_val.shape[0]} ({np.sum(y_val==1)} ASD, {np.sum(y_val==0)} Control)")
    print(f"   Test: {X_test.shape[0]} ({np.sum(y_test==1)} ASD, {np.sum(y_test==0)} Control)")
    
    # Apply enhanced preprocessing
    print(f"\n🔧 Applying enhanced preprocessing...")
    X_train_processed = processor.fit(X_train, y_train)
    X_val_processed = processor.transform(X_val)
    X_test_processed = processor.transform(X_test)
    
    print(f"✅ Processed features: {X_train_processed.shape[1]}")
    
    # Create enhanced model
    model = SMRITransformer(
        input_dim=X_train_processed.shape[1],
        d_model=config.d_model,
        n_heads=config.num_heads,
        n_layers=config.num_layers,
        dropout=config.dropout,
        layer_dropout=config.layer_dropout
    )
    
    print(f"\n🧠 Enhanced Model Architecture:")
    model_info = model.get_model_info()
    print(f"   Name: {model_info['model_name']}")
    print(f"   Parameters: {model_info['total_params']:,}")
    print(f"   Input dim: {model_info['input_dim']}")
    
    # Create enhanced data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        X_train_processed, y_train,
        X_val_processed, y_val, 
        X_test_processed, y_test,
        config, 'smri'
    )
    
    # Enhanced training
    print(f"\n🎯 Starting enhanced training...")
    trainer = Trainer(
        model=model,
        config=config,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    )
    
    # Train with enhanced settings
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        y_train=y_train
    )
    
    # Enhanced evaluation
    print(f"\n📊 Enhanced Evaluation...")
    test_metrics = trainer.evaluate(test_loader)
    
    print(f"\n🎉 Enhanced sMRI Results:")
    print(f"   Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"   Test Balanced Acc: {test_metrics['balanced_accuracy']:.4f}")
    print(f"   Test AUC: {test_metrics['auc']:.4f}")
    
    # Compare with expectations
    target_acc = 0.60
    current_acc = test_metrics['accuracy']
    improvement = current_acc - 0.55  # Baseline was ~55%
    
    print(f"\n📈 Performance Analysis:")
    print(f"   Target (notebook): {target_acc:.1%}")
    print(f"   Current result: {current_acc:.1%}")
    print(f"   Improvement: {improvement:+.1%}")
    
    if current_acc >= target_acc:
        print(f"   🎯 SUCCESS! Reached target performance!")
    elif current_acc >= 0.58:
        print(f"   ✅ GOOD! Close to target performance!")
    else:
        print(f"   ⚠️  Still room for improvement...")
    
    # Enhanced insights
    print(f"\n💡 Enhanced Insights:")
    print(f"   • Combined F-score + MI feature selection")
    print(f"   • RobustScaler (handles outliers better)")
    print(f"   • Proper working notebook architecture")
    print(f"   • Class weights + label smoothing")
    print(f"   • {config.feature_selection_k} selected features")
    
    return test_metrics

if __name__ == "__main__":
    results = test_enhanced_smri()
    print(f"\n🚀 Enhanced sMRI test completed!")
    print(f"   Final accuracy: {results['accuracy']:.1%}") 
#!/usr/bin/env python3
"""
Test Cross-Attention with ORIGINAL sMRI preprocessing.
This should recover ~63.6% performance if preprocessing mismatch was the issue.

Key changes from enhanced version:
- StandardScaler instead of RobustScaler
- Simple f_classif instead of combined F-score + MI  
- Original CrossAttentionTransformer (no improvements)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import os
import numpy as np
import pandas as pd
import json
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler  # ← ORIGINAL: Not RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif  # ← ORIGINAL: Simple f_classif
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score

from models.cross_attention import CrossAttentionTransformer  # ← ORIGINAL model
from utils.reproducibility import set_seed

def load_and_preprocess_data_original():
    """Load data with ORIGINAL preprocessing pipeline."""
    print("📊 Loading data with ORIGINAL preprocessing...")
    
    # Simulate loading - replace with actual paths
    print("⚠️  SIMULATION MODE - Replace with actual data loading:")
    print("   fmri_data = np.load('/content/drive/MyDrive/processed_fmri_data/features.npy')")
    print("   smri_data = np.load('/content/drive/MyDrive/processed_smri_data/features.npy')")
    print("   labels = np.load('/content/drive/MyDrive/processed_smri_data/labels.npy')")
    
    # For testing purposes, create dummy data
    n_samples = 500
    fmri_dim = 1000
    smri_dim = 400
    
    np.random.seed(42)
    fmri_data = np.random.randn(n_samples, fmri_dim)
    smri_data = np.random.randn(n_samples, smri_dim)
    labels = np.random.randint(0, 2, n_samples)
    
    print(f"Original shapes - fMRI: {fmri_data.shape}, sMRI: {smri_data.shape}")
    
    # ORIGINAL sMRI preprocessing (not enhanced)
    print("🔄 Applying ORIGINAL sMRI preprocessing...")
    
    # 1. Basic cleaning
    smri_data = np.nan_to_num(smri_data, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # 2. ORIGINAL: StandardScaler (not RobustScaler)
    smri_scaler = StandardScaler()
    smri_data_scaled = smri_scaler.fit_transform(smri_data)
    print("   ✓ Applied StandardScaler (original)")
    
    # 3. ORIGINAL: Simple feature selection (not F-score + MI combined)
    selector = SelectKBest(score_func=f_classif, k=min(300, smri_data.shape[1]))
    smri_data_selected = selector.fit_transform(smri_data_scaled, labels)
    print(f"   ✓ Selected {smri_data_selected.shape[1]} features with simple f_classif")
    
    # fMRI preprocessing (standard)
    fmri_scaler = StandardScaler()
    fmri_data_scaled = fmri_scaler.fit_transform(fmri_data)
    
    print(f"Final shapes - fMRI: {fmri_data_scaled.shape}, sMRI: {smri_data_selected.shape}")
    
    return fmri_data_scaled, smri_data_selected, labels

def train_and_evaluate_original():
    """Train cross-attention with original preprocessing."""
    print("🧪 TESTING: Original Preprocessing + Cross-Attention")
    print("=" * 60)
    print("🎯 Expected: Recovery to ~63.6% if preprocessing was the issue")
    print()
    
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data with original preprocessing
    fmri_data, smri_data, labels = load_and_preprocess_data_original()
    
    # ORIGINAL model architecture (no improvements)
    model = CrossAttentionTransformer(
        fmri_dim=fmri_data.shape[1],
        smri_dim=smri_data.shape[1],
        hidden_dim=64,              # Original size
        num_heads=4,                # Original
        num_layers=2,               # Original  
        dropout=0.1                 # Original dropout rate
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,} (original architecture)")
    
    # ORIGINAL training parameters
    criterion = nn.CrossEntropyLoss()  # No label smoothing
    optimizer = optim.Adam(model.parameters(), lr=1e-4)  # Original LR
    
    # Cross-validation
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    results = {'accuracy': [], 'balanced_accuracy': [], 'auc': []}
    
    print("🚀 Running 3-fold CV with original preprocessing...")
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(smri_data, labels)):
        print(f"\n📋 Fold {fold + 1}/3")
        
        # Split data
        fmri_train = torch.FloatTensor(fmri_data[train_idx])
        smri_train = torch.FloatTensor(smri_data[train_idx])
        labels_train = torch.LongTensor(labels[train_idx])
        
        fmri_test = torch.FloatTensor(fmri_data[test_idx])
        smri_test = torch.FloatTensor(smri_data[test_idx])
        labels_test = torch.LongTensor(labels[test_idx])
        
        # Create datasets
        train_dataset = TensorDataset(fmri_train, smri_train, labels_train)
        test_dataset = TensorDataset(fmri_test, smri_test, labels_test)
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)  # Original batch size
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        # Reset model for each fold
        model.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        
        # Training loop
        model.train()
        for epoch in range(30):  # Original epoch count
            epoch_loss = 0
            for batch_fmri, batch_smri, batch_labels in train_loader:
                batch_fmri = batch_fmri.to(device)
                batch_smri = batch_smri.to(device)
                batch_labels = batch_labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(batch_fmri, batch_smri)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                print(f"   Epoch {epoch+1}: Loss = {epoch_loss/len(train_loader):.4f}")
        
        # Evaluation
        model.eval()
        test_preds = []
        test_probs = []
        test_labels_list = []
        
        with torch.no_grad():
            for batch_fmri, batch_smri, batch_labels in test_loader:
                batch_fmri = batch_fmri.to(device)
                batch_smri = batch_smri.to(device)
                
                outputs = model(batch_fmri, batch_smri)
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(outputs, dim=1)
                
                test_preds.extend(preds.cpu().numpy())
                test_probs.extend(probs[:, 1].cpu().numpy())
                test_labels_list.extend(batch_labels.numpy())
        
        # Calculate metrics
        acc = accuracy_score(test_labels_list, test_preds)
        bal_acc = balanced_accuracy_score(test_labels_list, test_preds)
        auc = roc_auc_score(test_labels_list, test_probs)
        
        results['accuracy'].append(acc)
        results['balanced_accuracy'].append(bal_acc)
        results['auc'].append(auc)
        
        print(f"   Results: Acc={acc:.4f}, Bal_Acc={bal_acc:.4f}, AUC={auc:.4f}")
    
    # Summary and diagnosis
    print("\n" + "="*60)
    print("🎯 ORIGINAL PREPROCESSING TEST RESULTS:")
    print(f"   Mean Accuracy:     {np.mean(results['accuracy']):.4f} ± {np.std(results['accuracy']):.4f}")
    print(f"   Mean Bal Accuracy: {np.mean(results['balanced_accuracy']):.4f} ± {np.std(results['balanced_accuracy']):.4f}")
    print(f"   Mean AUC:          {np.mean(results['auc']):.4f} ± {np.std(results['auc']):.4f}")
    print()
    
    mean_acc = np.mean(results['accuracy'])
    
    print("🔍 DIAGNOSIS:")
    if mean_acc > 0.62:  # Close to original 63.6%
        print("   ✅ PREPROCESSING MISMATCH CONFIRMED!")
        print("   → Original preprocessing recovers performance")
        print("   → Enhanced preprocessing caused distribution shift")
        print("   📋 Strategy: Gradually introduce enhancements one by one")
        diagnosis = "preprocessing_mismatch_confirmed"
    elif mean_acc > 0.60:
        print("   ⚠️  PARTIAL RECOVERY")
        print("   → Mixed evidence for preprocessing impact")
        print("   📋 Strategy: Test both preprocessing + hyperparameters")
        diagnosis = "partial_preprocessing_impact"
    else:
        print("   ❌ NO RECOVERY")
        print("   → Issue is NOT primarily preprocessing")
        print("   → Likely training/hyperparameter/architecture problem")
        print("   📋 Strategy: Focus on hyperparameter grid search")
        diagnosis = "not_preprocessing_issue"
    
    print(f"\n🎯 COMPARISON TO TARGETS:")
    print(f"   Original Cross-Attention: 63.6%")
    print(f"   This test:                {mean_acc:.1%}")
    print(f"   Pure fMRI target:         65.0%")
    print(f"   Gap to beat fMRI:         {65.0 - mean_acc*100:.1f} points")
    
    return results, diagnosis

if __name__ == "__main__":
    print("🧪 CROSS-ATTENTION PREPROCESSING MISMATCH TEST")
    print("=" * 60)
    print("Testing if original sMRI preprocessing recovers 63.6% performance")
    print()
    
    results, diagnosis = train_and_evaluate_original()
    
    # Save results
    output = {
        'test_type': 'original_preprocessing_hypothesis',
        'results': {k: [float(x) for x in v] for k, v in results.items()},
        'mean_accuracy': float(np.mean(results['accuracy'])),
        'mean_balanced_accuracy': float(np.mean(results['balanced_accuracy'])),
        'mean_auc': float(np.mean(results['auc'])),
        'diagnosis': diagnosis,
        'timestamp': datetime.now().isoformat(),
        'preprocessing_type': 'original_standardscaler_simple_ftest',
        'model_type': 'original_cross_attention_transformer'
    }
    
    with open('original_preprocessing_test_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✅ Test completed! Results saved to 'original_preprocessing_test_results.json'")
    print(f"📋 Diagnosis: {diagnosis}") 
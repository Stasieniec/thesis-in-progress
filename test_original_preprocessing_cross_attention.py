#!/usr/bin/env python3
"""
Test cross-attention with ORIGINAL sMRI preprocessing.
This should recover ~63.6% performance if preprocessing mismatch was the issue.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

import os
import numpy as np
import pandas as pd
import json
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler  # ORIGINAL: StandardScaler instead of RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif  # ORIGINAL: Simple f_classif only
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score

from data.data_loader import CrossAttentionDataset
from models.cross_attention import CrossAttentionTransformer  # ← ORIGINAL model
from evaluation.metrics import calculate_metrics
from utils.reproducibility import set_seed

def load_data():
    """Load data with ORIGINAL preprocessing pipeline."""
    print("📊 Loading data with ORIGINAL preprocessing...")
    
    # Load fMRI data (this should be fine)
    fmri_data = np.load('/content/drive/MyDrive/processed_fmri_data/features.npy')
    
    # Load sMRI data  
    smri_data = np.load('/content/drive/MyDrive/processed_smri_data/features.npy')
    labels = np.load('/content/drive/MyDrive/processed_smri_data/labels.npy')
    
    print(f"Original shapes - fMRI: {fmri_data.shape}, sMRI: {smri_data.shape}")
    
    # ORIGINAL sMRI preprocessing (not enhanced)
    print("🔄 Applying ORIGINAL sMRI preprocessing...")
    
    # 1. Basic cleaning (minimal)
    smri_data = np.nan_to_num(smri_data, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # 2. ORIGINAL: StandardScaler (not RobustScaler)
    smri_scaler = StandardScaler()
    smri_data_scaled = smri_scaler.fit_transform(smri_data)
    
    # 3. ORIGINAL: Simple feature selection (not F-score + MI combined)
    # Use top 300 features based on simple F-test
    selector = SelectKBest(score_func=f_classif, k=300)
    smri_data_selected = selector.fit_transform(smri_data_scaled, labels)
    
    print(f"After original preprocessing - sMRI: {smri_data_selected.shape}")
    
    # fMRI preprocessing (keep standard)
    fmri_scaler = StandardScaler()
    fmri_data_scaled = fmri_scaler.fit_transform(fmri_data)
    
    return fmri_data_scaled, smri_data_selected, labels

def test_original_preprocessing():
    """Test cross-attention with original preprocessing."""
    print("🧪 TESTING: Original Preprocessing + Cross-Attention")
    print("=" * 60)
    print("🎯 Expected: Recovery to ~63.6% if preprocessing was the issue")
    print()
    
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data with original preprocessing
    fmri_data, smri_data, labels = load_data()
    
    # Original model architecture (no improvements)
    model = CrossAttentionTransformer(
        fmri_dim=fmri_data.shape[1],
        smri_dim=smri_data.shape[1],
        hidden_dim=64,
        num_heads=4,
        num_layers=2,
        dropout=0.1  # Original dropout rate
    ).to(device)
    
    # Training parameters (original)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)  # Original LR
    
    # Cross-validation with original settings
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
        train_dataset = CrossAttentionDataset(fmri_train, smri_train, labels_train)
        test_dataset = CrossAttentionDataset(fmri_test, smri_test, labels_test)
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        # Reset model for each fold
        model.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        
        # Training loop (reduced epochs for quick test)
        model.train()
        for epoch in range(20):  # Quick test
            for batch_fmri, batch_smri, batch_labels in train_loader:
                batch_fmri = batch_fmri.to(device)
                batch_smri = batch_smri.to(device)
                batch_labels = batch_labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(batch_fmri, batch_smri)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()
        
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
        
        print(f"   Accuracy: {acc:.4f}")
        print(f"   Balanced Accuracy: {bal_acc:.4f}")
        print(f"   AUC: {auc:.4f}")
    
    # Summary
    print("\n" + "="*60)
    print("🎯 ORIGINAL PREPROCESSING RESULTS:")
    print(f"   Mean Accuracy: {np.mean(results['accuracy']):.4f} ± {np.std(results['accuracy']):.4f}")
    print(f"   Mean Bal Acc:  {np.mean(results['balanced_accuracy']):.4f} ± {np.std(results['balanced_accuracy']):.4f}")
    print(f"   Mean AUC:      {np.mean(results['auc']):.4f} ± {np.std(results['auc']):.4f}")
    print()
    
    mean_acc = np.mean(results['accuracy'])
    
    print("🔍 DIAGNOSIS:")
    if mean_acc > 0.62:  # Close to original 63.6%
        print("   ✅ PREPROCESSING MISMATCH CONFIRMED!")
        print("   → Original preprocessing recovers performance")
        print("   → Enhanced preprocessing was incompatible")
        print("   📋 Next: Gradually introduce enhancements")
    elif mean_acc > 0.60:
        print("   ⚠️  PARTIAL RECOVERY")
        print("   → Some preprocessing impact + other factors")
        print("   📋 Next: Test hyperparameters + gradual enhancements")
    else:
        print("   ❌ NO RECOVERY")
        print("   → Issue is not preprocessing")
        print("   → Likely training/hyperparameter problem")
        print("   📋 Next: Focus on hyperparameter grid search")
    
    return results

def test_hypothesis():
    """Test the preprocessing mismatch hypothesis."""
    print("🧪 TESTING: Original Preprocessing Hypothesis")
    print("=" * 60)
    print("Theory: Enhanced sMRI preprocessing broke cross-attention compatibility")
    print("Test: Use original StandardScaler + simple feature selection")
    print("Expected: Recovery to ~63.6% if hypothesis is correct")
    print()
    
    print("📊 PREPROCESSING COMPARISON:")
    print("   ORIGINAL (worked at 63.6%):")
    print("     - StandardScaler") 
    print("     - Simple F-test feature selection")
    print("     - Top 300 features")
    print()
    print("   ENHANCED (failed at 57.7%):")
    print("     - RobustScaler")
    print("     - F-score + Mutual Information combined") 
    print("     - Advanced feature selection")
    print()
    
    print("🎯 PREDICTION:")
    print("   IF preprocessing mismatch:")
    print("     → Original: ~63.6% ✅")
    print("     → Enhanced: ~57.7% ❌")
    print("   ")
    print("   IF other issue (hyperparams/architecture):")
    print("     → Both: ~57-60% ⚠️")
    print()
    
    # TODO: Implement actual test when ready
    print("📋 IMPLEMENTATION NEEDED:")
    print("   1. Load data with StandardScaler (not RobustScaler)")
    print("   2. Use simple SelectKBest(f_classif, k=300)")
    print("   3. Test with original CrossAttentionTransformer")
    print("   4. Compare results to enhanced preprocessing")
    
    return {"status": "hypothesis_defined", "needs_implementation": True}

if __name__ == "__main__":
    test_hypothesis() 
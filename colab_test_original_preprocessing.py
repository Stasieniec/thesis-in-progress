#!/usr/bin/env python3
"""
COLAB SCRIPT 1: Test Original Preprocessing Hypothesis
Test if original sMRI preprocessing recovers 63.6% cross-attention performance

Expected outcome: Recovery to ~63.6% if preprocessing mismatch was the issue
"""

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Install dependencies
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 -q
!pip install scikit-learn matplotlib seaborn pandas numpy scipy -q

import os
import sys
import numpy as np
import pandas as pd
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.nn.functional as F

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler  # ← ORIGINAL: Not RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif  # ← ORIGINAL: Simple f_classif

# Set random seeds
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Data paths - UPDATE THESE TO YOUR ACTUAL PATHS
FMRI_DATA_PATH = '/content/drive/MyDrive/processed_fmri_data/features.npy'
SMRI_DATA_PATH = '/content/drive/MyDrive/processed_smri_data/features.npy'
LABELS_PATH = '/content/drive/MyDrive/processed_smri_data/labels.npy'
OUTPUT_PATH = '/content/drive/MyDrive/cross_attention_tests'

# Create output directory
os.makedirs(OUTPUT_PATH, exist_ok=True)

print("🧪 TESTING: Original Preprocessing Hypothesis")
print("=" * 70)
print("🎯 Goal: Test if original sMRI preprocessing recovers 63.6% performance")
print("📊 Expected: If preprocessing mismatch, should recover to ~63.6%")
print()

class CrossAttentionTransformer(nn.Module):
    """ORIGINAL Cross-Attention Transformer (no improvements)"""
    def __init__(self, fmri_dim, smri_dim, hidden_dim=64, num_heads=4, num_layers=2, dropout=0.1):
        super(CrossAttentionTransformer, self).__init__()
        
        self.fmri_projection = nn.Linear(fmri_dim, hidden_dim)
        self.smri_projection = nn.Linear(smri_dim, hidden_dim)
        
        encoder_layer = TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2)
        )
    
    def forward(self, fmri, smri):
        fmri_proj = self.fmri_projection(fmri).unsqueeze(1)  # [B, 1, H]
        smri_proj = self.smri_projection(smri).unsqueeze(1)  # [B, 1, H]
        
        # Cross-attention
        combined = torch.cat([fmri_proj, smri_proj], dim=1)  # [B, 2, H]
        attended = self.transformer(combined)  # [B, 2, H]
        
        # Combine and classify
        combined_features = attended.flatten(1)  # [B, 2*H]
        return self.classifier(combined_features)

def load_and_preprocess_original():
    """Load data with ORIGINAL preprocessing (StandardScaler + simple f_classif)"""
    print("📊 Loading data with ORIGINAL preprocessing...")
    
    # Load data
    fmri_data = np.load(FMRI_DATA_PATH)
    smri_data = np.load(SMRI_DATA_PATH)
    labels = np.load(LABELS_PATH)
    
    print(f"Raw data shapes - fMRI: {fmri_data.shape}, sMRI: {smri_data.shape}")
    print(f"Class distribution: {np.bincount(labels)}")
    
    # ORIGINAL sMRI preprocessing
    print("\n🔄 Applying ORIGINAL sMRI preprocessing...")
    
    # 1. Basic cleaning
    smri_data = np.nan_to_num(smri_data, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # 2. ORIGINAL: StandardScaler (not RobustScaler)
    print("   ✓ Using StandardScaler (original)")
    smri_scaler = StandardScaler()
    smri_data_scaled = smri_scaler.fit_transform(smri_data)
    
    # 3. ORIGINAL: Simple feature selection (not F-score + MI combined)
    print("   ✓ Using simple f_classif feature selection")
    k_features = min(300, smri_data.shape[1])
    selector = SelectKBest(score_func=f_classif, k=k_features)
    smri_data_selected = selector.fit_transform(smri_data_scaled, labels)
    print(f"   ✓ Selected {smri_data_selected.shape[1]} features")
    
    # Standard fMRI preprocessing
    print("   ✓ Standard fMRI preprocessing with StandardScaler")
    fmri_scaler = StandardScaler()
    fmri_data_scaled = fmri_scaler.fit_transform(fmri_data)
    
    print(f"\nFinal shapes - fMRI: {fmri_data_scaled.shape}, sMRI: {smri_data_selected.shape}")
    
    return fmri_data_scaled, smri_data_selected, labels

def train_and_evaluate_fold(model, train_loader, test_loader, device, fold_num):
    """Train and evaluate one fold"""
    
    # ORIGINAL training parameters
    criterion = nn.CrossEntropyLoss()  # No label smoothing
    optimizer = optim.Adam(model.parameters(), lr=1e-4)  # Original LR
    
    print(f"   🚀 Training fold {fold_num}...")
    
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
            print(f"      Epoch {epoch+1}: Loss = {epoch_loss/len(train_loader):.4f}")
    
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
    
    return acc, bal_acc, auc, test_preds, test_labels_list

def run_original_preprocessing_test():
    """Main test function"""
    
    # Load data with original preprocessing
    fmri_data, smri_data, labels = load_and_preprocess_original()
    
    # Cross-validation
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    results = {
        'accuracy': [], 'balanced_accuracy': [], 'auc': [],
        'fold_details': []
    }
    
    print("\n🚀 Running 3-fold cross-validation with ORIGINAL preprocessing...")
    print("-" * 60)
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(smri_data, labels)):
        print(f"\n📋 Fold {fold + 1}/3")
        
        # Split data
        fmri_train = torch.FloatTensor(fmri_data[train_idx])
        smri_train = torch.FloatTensor(smri_data[train_idx])
        labels_train = torch.LongTensor(labels[train_idx])
        
        fmri_test = torch.FloatTensor(fmri_data[test_idx])
        smri_test = torch.FloatTensor(smri_data[test_idx])
        labels_test = torch.LongTensor(labels[test_idx])
        
        print(f"   Train: {len(train_idx)} samples, Test: {len(test_idx)} samples")
        
        # Create datasets and loaders
        train_dataset = TensorDataset(fmri_train, smri_train, labels_train)
        test_dataset = TensorDataset(fmri_test, smri_test, labels_test)
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)  # Original batch size
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        # Initialize ORIGINAL model
        model = CrossAttentionTransformer(
            fmri_dim=fmri_data.shape[1],
            smri_dim=smri_data.shape[1],
            hidden_dim=64,              # Original
            num_heads=4,                # Original
            num_layers=2,               # Original
            dropout=0.1                 # Original dropout
        ).to(device)
        
        # Train and evaluate
        acc, bal_acc, auc, preds, labels_true = train_and_evaluate_fold(
            model, train_loader, test_loader, device, fold + 1
        )
        
        # Store results
        results['accuracy'].append(acc)
        results['balanced_accuracy'].append(bal_acc)
        results['auc'].append(auc)
        results['fold_details'].append({
            'fold': fold + 1,
            'accuracy': float(acc),
            'balanced_accuracy': float(bal_acc),
            'auc': float(auc),
            'train_size': len(train_idx),
            'test_size': len(test_idx)
        })
        
        print(f"   Results: Acc={acc:.4f}, Bal_Acc={bal_acc:.4f}, AUC={auc:.4f}")
    
    return results

def analyze_results(results):
    """Analyze and diagnose results"""
    
    mean_acc = np.mean(results['accuracy'])
    std_acc = np.std(results['accuracy'])
    mean_bal_acc = np.mean(results['balanced_accuracy'])
    mean_auc = np.mean(results['auc'])
    
    print("\n" + "="*70)
    print("🎯 ORIGINAL PREPROCESSING TEST RESULTS:")
    print("-" * 40)
    print(f"   Mean Accuracy:         {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"   Mean Balanced Accuracy: {mean_bal_acc:.4f} ± {np.std(results['balanced_accuracy']):.4f}")
    print(f"   Mean AUC:              {mean_auc:.4f} ± {np.std(results['auc']):.4f}")
    print(f"   Range: [{min(results['accuracy']):.4f}, {max(results['accuracy']):.4f}]")
    print()
    
    # Diagnosis
    print("🔍 DIAGNOSIS:")
    print("-" * 15)
    
    if mean_acc >= 0.62:  # Close to original 63.6%
        diagnosis = "preprocessing_mismatch_confirmed"
        print("   ✅ PREPROCESSING MISMATCH CONFIRMED!")
        print("   → Original preprocessing RECOVERS performance")
        print("   → Enhanced preprocessing caused distribution shift")
        print("   → Root cause: RobustScaler + advanced feature selection incompatible")
        print()
        print("   📋 RECOMMENDED STRATEGY:")
        print("   → Start with original preprocessing (current result)")
        print("   → Gradually introduce enhancements ONE BY ONE")
        print("   → Test each change for distribution compatibility")
        print("   → Goal: Incrementally improve beyond 65%")
        
    elif mean_acc >= 0.58:  # Partial recovery
        diagnosis = "partial_preprocessing_impact"
        print("   ⚠️  PARTIAL RECOVERY")
        print("   → Some preprocessing impact detected")
        print("   → Additional factors involved (hyperparams/training)")
        print()
        print("   📋 RECOMMENDED STRATEGY:")
        print("   → Test both original AND enhanced preprocessing")
        print("   → Hyperparameter grid search on both")
        print("   → Mixed approach: original base + selective enhancements")
        
    else:  # No recovery
        diagnosis = "not_preprocessing_issue"
        print("   ❌ NO RECOVERY - NOT PREPROCESSING ISSUE")
        print("   → Problem is NOT primarily preprocessing")
        print("   → Likely hyperparameter/training/architecture issue")
        print()
        print("   📋 RECOMMENDED STRATEGY:")
        print("   → Focus on hyperparameter optimization")
        print("   → Learning rate grid search: [1e-5, 5e-5, 1e-4, 2e-4]")
        print("   → Batch size testing: [16, 32, 64]")
        print("   → Architecture debugging")
    
    print()
    print("🎯 PERFORMANCE COMPARISON:")
    print(f"   Original Cross-Attention: 63.6%")
    print(f"   Enhanced Cross-Attention: 57.7% (failed)")
    print(f"   This test (original):     {mean_acc:.1%}")
    print(f"   Pure fMRI target:         65.0%")
    print(f"   Gap to beat fMRI:         {65.0 - mean_acc*100:.1f} percentage points")
    
    return diagnosis, mean_acc

# Run the test
if __name__ == "__main__":
    print("Starting original preprocessing hypothesis test...")
    
    try:
        results = run_original_preprocessing_test()
        diagnosis, mean_acc = analyze_results(results)
        
        # Save comprehensive results
        output_data = {
            'test_name': 'original_preprocessing_hypothesis',
            'timestamp': datetime.now().isoformat(),
            'preprocessing_type': 'original_standardscaler_simple_ftest',
            'model_type': 'original_cross_attention_transformer',
            'hyperparameters': {
                'learning_rate': 1e-4,
                'batch_size': 32,
                'epochs': 30,
                'dropout': 0.1,
                'hidden_dim': 64,
                'num_heads': 4,
                'num_layers': 2
            },
            'results': {
                'mean_accuracy': float(mean_acc),
                'mean_balanced_accuracy': float(np.mean(results['balanced_accuracy'])),
                'mean_auc': float(np.mean(results['auc'])),
                'std_accuracy': float(np.std(results['accuracy'])),
                'all_accuracies': [float(x) for x in results['accuracy']],
                'all_balanced_accuracies': [float(x) for x in results['balanced_accuracy']],
                'all_aucs': [float(x) for x in results['auc']],
                'fold_details': results['fold_details']
            },
            'diagnosis': diagnosis,
            'comparison': {
                'original_cross_attention': 63.6,
                'enhanced_cross_attention': 57.7,
                'this_test': float(mean_acc * 100),
                'pure_fmri_target': 65.0
            }
        }
        
        # Save results
        results_path = os.path.join(OUTPUT_PATH, 'original_preprocessing_test_results.json')
        with open(results_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\n✅ Test completed successfully!")
        print(f"📁 Results saved to: {results_path}")
        print(f"🔍 Diagnosis: {diagnosis}")
        print(f"📊 Performance: {mean_acc:.1%}")
        
        print("\n" + "="*70)
        print("🎯 NEXT STEPS:")
        if diagnosis == "preprocessing_mismatch_confirmed":
            print("   → Run SCRIPT 2: Gradual Enhancement Introduction")
        elif diagnosis == "partial_preprocessing_impact":
            print("   → Run SCRIPT 2: Comparative Preprocessing Test")
            print("   → Run SCRIPT 3: Hyperparameter Grid Search")
        else:
            print("   → Run SCRIPT 3: Hyperparameter Grid Search")
            print("   → Focus on training optimization")
        
    except Exception as e:
        print(f"❌ Error during testing: {str(e)}")
        print("Check data paths and dependencies!")
        raise 
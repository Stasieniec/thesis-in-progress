#!/usr/bin/env python3
"""
COLAB SCRIPT 3: Gradual Enhancement Introduction
If preprocessing mismatch is confirmed, gradually introduce enhancements
one by one to find the optimal balance between improvements and compatibility

Run this if Script 1 confirms preprocessing mismatch (recovers to ~63.6%)
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
from torch.utils.data import DataLoader, TensorDataset
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.nn.functional as F

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif

# Set random seeds
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Data paths
FMRI_DATA_PATH = '/content/drive/MyDrive/processed_fmri_data/features.npy'
SMRI_DATA_PATH = '/content/drive/MyDrive/processed_smri_data/features.npy'
LABELS_PATH = '/content/drive/MyDrive/processed_smri_data/labels.npy'
OUTPUT_PATH = '/content/drive/MyDrive/cross_attention_tests'

os.makedirs(OUTPUT_PATH, exist_ok=True)

print("🔄 GRADUAL ENHANCEMENT INTRODUCTION")
print("=" * 70)
print("🎯 Goal: Incrementally improve from original preprocessing to beat 65%")
print("📊 Starting from working baseline, adding enhancements one by one")
print()

class CrossAttentionTransformer(nn.Module):
    """Original Cross-Attention Transformer"""
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
        fmri_proj = self.fmri_projection(fmri).unsqueeze(1)
        smri_proj = self.smri_projection(smri).unsqueeze(1)
        
        combined = torch.cat([fmri_proj, smri_proj], dim=1)
        attended = self.transformer(combined)
        
        combined_features = attended.flatten(1)
        return self.classifier(combined_features)

class WeightedFusionCrossAttention(nn.Module):
    """Cross-Attention with weighted fusion (minimal improvement)"""
    def __init__(self, fmri_dim, smri_dim, hidden_dim=64, num_heads=4, num_layers=2, 
                 dropout=0.1, fmri_weight=0.55):
        super(WeightedFusionCrossAttention, self).__init__()
        
        self.fmri_projection = nn.Linear(fmri_dim, hidden_dim)
        self.smri_projection = nn.Linear(smri_dim, hidden_dim)
        self.fmri_weight = fmri_weight
        self.smri_weight = 1.0 - fmri_weight
        
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
        fmri_proj = self.fmri_projection(fmri).unsqueeze(1)
        smri_proj = self.smri_projection(smri).unsqueeze(1)
        
        # Weighted fusion
        fmri_proj = fmri_proj * self.fmri_weight
        smri_proj = smri_proj * self.smri_weight
        
        combined = torch.cat([fmri_proj, smri_proj], dim=1)
        attended = self.transformer(combined)
        
        combined_features = attended.flatten(1)
        return self.classifier(combined_features)

def load_and_preprocess_data(enhancement_level=0):
    """Load data with gradual enhancement levels"""
    
    enhancements = {
        0: "baseline_original",
        1: "robust_scaler", 
        2: "robust_scaler_f_score_improved",
        3: "robust_scaler_f_score_mi_combined",
        4: "full_enhanced"
    }
    
    print(f"📊 Loading data with enhancement level {enhancement_level}: {enhancements[enhancement_level]}")
    
    fmri_data = np.load(FMRI_DATA_PATH)
    smri_data = np.load(SMRI_DATA_PATH)
    labels = np.load(LABELS_PATH)
    
    print(f"Raw data shapes - fMRI: {fmri_data.shape}, sMRI: {smri_data.shape}")
    
    # Clean data
    smri_data = np.nan_to_num(smri_data, nan=0.0, posinf=1e6, neginf=-1e6)
    
    if enhancement_level == 0:
        # Level 0: Original baseline
        smri_scaler = StandardScaler()
        smri_data_scaled = smri_scaler.fit_transform(smri_data)
        
        selector = SelectKBest(score_func=f_classif, k=min(300, smri_data.shape[1]))
        smri_data_selected = selector.fit_transform(smri_data_scaled, labels)
        print(f"   ✓ Level 0: StandardScaler + f_classif, {smri_data_selected.shape[1]} features")
    
    elif enhancement_level == 1:
        # Level 1: Introduce RobustScaler only
        smri_scaler = RobustScaler()
        smri_data_scaled = smri_scaler.fit_transform(smri_data)
        
        selector = SelectKBest(score_func=f_classif, k=min(300, smri_data.shape[1]))
        smri_data_selected = selector.fit_transform(smri_data_scaled, labels)
        print(f"   ✓ Level 1: RobustScaler + f_classif, {smri_data_selected.shape[1]} features")
    
    elif enhancement_level == 2:
        # Level 2: RobustScaler + improved F-score selection
        smri_scaler = RobustScaler()
        smri_data_scaled = smri_scaler.fit_transform(smri_data)
        
        selector = SelectKBest(score_func=f_classif, k=min(350, smri_data.shape[1]))
        smri_data_selected = selector.fit_transform(smri_data_scaled, labels)
        print(f"   ✓ Level 2: RobustScaler + improved f_classif, {smri_data_selected.shape[1]} features")
    
    elif enhancement_level == 3:
        # Level 3: RobustScaler + F-score + MI combined
        smri_scaler = RobustScaler()
        smri_data_scaled = smri_scaler.fit_transform(smri_data)
        
        f_selector = SelectKBest(score_func=f_classif, k=min(400, smri_data.shape[1]))
        f_selected = f_selector.fit_transform(smri_data_scaled, labels)
        
        mi_selector = SelectKBest(score_func=mutual_info_classif, k=min(300, f_selected.shape[1]))
        smri_data_selected = mi_selector.fit_transform(f_selected, labels)
        print(f"   ✓ Level 3: RobustScaler + F-score+MI, {smri_data_selected.shape[1]} features")
    
    else:  # Level 4
        # Level 4: Full enhanced preprocessing
        smri_scaler = RobustScaler()
        smri_data_scaled = smri_scaler.fit_transform(smri_data)
        
        # Advanced feature selection
        f_selector = SelectKBest(score_func=f_classif, k=min(500, smri_data.shape[1]))
        f_selected = f_selector.fit_transform(smri_data_scaled, labels)
        
        mi_selector = SelectKBest(score_func=mutual_info_classif, k=min(350, f_selected.shape[1]))
        smri_data_selected = mi_selector.fit_transform(f_selected, labels)
        print(f"   ✓ Level 4: Full enhanced preprocessing, {smri_data_selected.shape[1]} features")
    
    # fMRI preprocessing (always standard)
    fmri_scaler = StandardScaler()
    fmri_data_scaled = fmri_scaler.fit_transform(fmri_data)
    
    return fmri_data_scaled, smri_data_selected, labels

def evaluate_enhancement_level(fmri_data, smri_data, labels, enhancement_level, model_type='original'):
    """Evaluate a specific enhancement level"""
    
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    fold_results = []
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(smri_data, labels)):
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
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        # Initialize model
        if model_type == 'weighted':
            model = WeightedFusionCrossAttention(
                fmri_dim=fmri_data.shape[1],
                smri_dim=smri_data.shape[1],
                hidden_dim=64,
                num_heads=4,
                num_layers=2,
                dropout=0.1,
                fmri_weight=0.55  # Slightly favor fMRI
            ).to(device)
        else:
            model = CrossAttentionTransformer(
                fmri_dim=fmri_data.shape[1],
                smri_dim=smri_data.shape[1],
                hidden_dim=64,
                num_heads=4,
                num_layers=2,
                dropout=0.1
            ).to(device)
        
        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
        
        # Training
        model.train()
        for epoch in range(30):
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
        
        fold_results.append({'accuracy': acc, 'balanced_accuracy': bal_acc, 'auc': auc})
    
    # Average across folds
    mean_acc = np.mean([r['accuracy'] for r in fold_results])
    mean_bal_acc = np.mean([r['balanced_accuracy'] for r in fold_results])
    mean_auc = np.mean([r['auc'] for r in fold_results])
    
    return mean_acc, mean_bal_acc, mean_auc, fold_results

def run_gradual_enhancement():
    """Run gradual enhancement introduction"""
    
    enhancement_descriptions = {
        0: "Original baseline (StandardScaler + f_classif)",
        1: "Level 1: RobustScaler only",
        2: "Level 2: RobustScaler + improved F-score",
        3: "Level 3: RobustScaler + F-score + MI",
        4: "Level 4: Full enhanced preprocessing"
    }
    
    results = []
    
    print("🚀 Testing gradual enhancement introduction...")
    print("Strategy: Start from working baseline, add enhancements incrementally")
    print()
    
    # Test each enhancement level with both model types
    for enhancement_level in range(5):
        print(f"\n📋 Enhancement Level {enhancement_level}: {enhancement_descriptions[enhancement_level]}")
        
        # Load data for this enhancement level
        fmri_data, smri_data, labels = load_and_preprocess_data(enhancement_level)
        
        # Test with original model
        print("   🔸 Testing with original cross-attention model...")
        acc_orig, bal_acc_orig, auc_orig, folds_orig = evaluate_enhancement_level(
            fmri_data, smri_data, labels, enhancement_level, model_type='original'
        )
        
        result_orig = {
            'enhancement_level': enhancement_level,
            'description': enhancement_descriptions[enhancement_level],
            'model_type': 'original',
            'mean_accuracy': acc_orig,
            'mean_balanced_accuracy': bal_acc_orig,
            'mean_auc': auc_orig,
            'fold_results': folds_orig,
            'beats_fmri_baseline': acc_orig > 0.65
        }
        results.append(result_orig)
        
        print(f"      Original model: Acc={acc_orig:.4f}, Bal_Acc={bal_acc_orig:.4f}, AUC={auc_orig:.4f}")
        if acc_orig > 0.65:
            print(f"      🎯 BEATS fMRI BASELINE!")
        
        # Test with weighted fusion model (if enhancement level >= 1)
        if enhancement_level >= 1:
            print("   🔸 Testing with weighted fusion model...")
            acc_weighted, bal_acc_weighted, auc_weighted, folds_weighted = evaluate_enhancement_level(
                fmri_data, smri_data, labels, enhancement_level, model_type='weighted'
            )
            
            result_weighted = {
                'enhancement_level': enhancement_level,
                'description': enhancement_descriptions[enhancement_level],
                'model_type': 'weighted_fusion',
                'mean_accuracy': acc_weighted,
                'mean_balanced_accuracy': bal_acc_weighted,
                'mean_auc': auc_weighted,
                'fold_results': folds_weighted,
                'beats_fmri_baseline': acc_weighted > 0.65
            }
            results.append(result_weighted)
            
            print(f"      Weighted model: Acc={acc_weighted:.4f}, Bal_Acc={bal_acc_weighted:.4f}, AUC={auc_weighted:.4f}")
            if acc_weighted > 0.65:
                print(f"      🎯 BEATS fMRI BASELINE!")
            
            # Compare models
            diff = acc_weighted - acc_orig
            if diff > 0.01:
                print(f"      ✅ Weighted fusion improves by {diff*100:.1f} points!")
            elif diff < -0.01:
                print(f"      ❌ Weighted fusion hurts by {abs(diff)*100:.1f} points")
            else:
                print(f"      ⚖️  Similar performance (±{abs(diff)*100:.1f} points)")
    
    return results

def analyze_gradual_results(results):
    """Analyze gradual enhancement results"""
    
    print("\n" + "="*70)
    print("🎯 GRADUAL ENHANCEMENT RESULTS")
    print("-" * 40)
    
    # Find best overall result
    best_result = max(results, key=lambda x: x['mean_accuracy'])
    print(f"📊 BEST OVERALL RESULT:")
    print(f"   Performance: {best_result['mean_accuracy']:.4f} ({best_result['mean_accuracy']:.1%})")
    print(f"   Configuration: {best_result['description']}")
    print(f"   Model: {best_result['model_type']}")
    if best_result['beats_fmri_baseline']:
        print(f"   🎯 BEATS fMRI BASELINE!")
    print()
    
    # Analyze progression by enhancement level
    print("📈 ENHANCEMENT PROGRESSION ANALYSIS:")
    
    # Group by enhancement level
    by_level = {}
    for result in results:
        level = result['enhancement_level']
        if level not in by_level:
            by_level[level] = []
        by_level[level].append(result)
    
    baseline_acc = None
    for level in sorted(by_level.keys()):
        level_results = by_level[level]
        level_desc = level_results[0]['description']
        
        print(f"\n   Level {level}: {level_desc}")
        
        for result in level_results:
            acc = result['mean_accuracy']
            model_type = result['model_type']
            print(f"      {model_type}: {acc:.4f} ({acc:.1%})", end="")
            
            if baseline_acc is not None:
                diff = acc - baseline_acc
                if diff > 0.01:
                    print(f" [+{diff*100:.1f} vs baseline]", end="")
                elif diff < -0.01:
                    print(f" [{diff*100:.1f} vs baseline]", end="")
            
            if result['beats_fmri_baseline']:
                print(" 🎯", end="")
            print()
            
            # Set baseline from level 0 original model
            if level == 0 and model_type == 'original':
                baseline_acc = acc
    
    # Key insights
    print(f"\n🔍 KEY INSIGHTS:")
    
    # Check if any enhancement improves over baseline
    baseline_result = next(r for r in results if r['enhancement_level'] == 0 and r['model_type'] == 'original')
    baseline_acc = baseline_result['mean_accuracy']
    
    successful_enhancements = [r for r in results if r['mean_accuracy'] > baseline_acc + 0.005]
    
    if successful_enhancements:
        print(f"   ✅ {len(successful_enhancements)} enhancement(s) improve over baseline:")
        for result in sorted(successful_enhancements, key=lambda x: x['mean_accuracy'], reverse=True):
            improvement = (result['mean_accuracy'] - baseline_acc) * 100
            print(f"      Level {result['enhancement_level']} ({result['model_type']}): +{improvement:.1f} points")
    else:
        print(f"   ❌ No enhancements improve significantly over baseline")
        print(f"   → Original preprocessing remains optimal")
    
    # Check distribution compatibility
    distribution_compatible = [r for r in results if r['enhancement_level'] <= 1 and r['mean_accuracy'] > baseline_acc - 0.02]
    if len(distribution_compatible) > 1:
        print(f"   ✅ RobustScaler appears compatible (minimal performance drop)")
    else:
        print(f"   ⚠️  RobustScaler may cause distribution issues")
    
    # Success in beating fMRI baseline
    successful_configs = [r for r in results if r['beats_fmri_baseline']]
    if successful_configs:
        print(f"   🎯 {len(successful_configs)} configuration(s) beat fMRI baseline!")
        best_success = max(successful_configs, key=lambda x: x['mean_accuracy'])
        print(f"      Best: Level {best_success['enhancement_level']} ({best_success['model_type']}) at {best_success['mean_accuracy']:.1%}")
    else:
        print(f"   ⚠️  No configuration beats 65% fMRI baseline")
        gap = (0.65 - best_result['mean_accuracy']) * 100
        print(f"   → Best result needs {gap:.1f} more percentage points")
    
    return by_level, best_result

# Run the gradual enhancement
if __name__ == "__main__":
    print("Starting gradual enhancement introduction...")
    
    try:
        results = run_gradual_enhancement()
        by_level, best_result = analyze_gradual_results(results)
        
        # Save comprehensive results
        output_data = {
            'test_name': 'gradual_enhancement_introduction',
            'timestamp': datetime.now().isoformat(),
            'strategy': 'incremental_preprocessing_enhancement',
            'baseline_performance': next(r for r in results if r['enhancement_level'] == 0 and r['model_type'] == 'original')['mean_accuracy'],
            'best_performance': best_result['mean_accuracy'],
            'best_configuration': {
                'enhancement_level': best_result['enhancement_level'],
                'description': best_result['description'],
                'model_type': best_result['model_type']
            },
            'all_results': results,
            'enhancement_levels_tested': 5,
            'model_types_tested': ['original', 'weighted_fusion'],
            'target_baseline': 65.0
        }
        
        results_path = os.path.join(OUTPUT_PATH, 'gradual_enhancement_results.json')
        with open(results_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\n✅ Gradual enhancement testing completed!")
        print(f"📁 Results saved to: {results_path}")
        print(f"🎯 Best performance: {best_result['mean_accuracy']:.1%}")
        print(f"📋 Best config: Level {best_result['enhancement_level']} with {best_result['model_type']}")
        
        if best_result['beats_fmri_baseline']:
            print(f"🏆 SUCCESS! Found enhancement that beats fMRI baseline!")
        else:
            gap = (0.65 - best_result['mean_accuracy']) * 100
            print(f"⚠️  Gap to 65% target: {gap:.1f} percentage points")
            print(f"📋 Consider: More sophisticated architectural improvements")
        
    except Exception as e:
        print(f"❌ Error during gradual enhancement testing: {str(e)}")
        raise 
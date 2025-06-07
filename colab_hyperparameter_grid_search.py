#!/usr/bin/env python3
"""
COLAB SCRIPT 2: Hyperparameter Grid Search
If preprocessing mismatch wasn't confirmed, systematically test hyperparameters
to find configuration that beats 65% fMRI baseline

Run this if Script 1 shows: NO RECOVERY or PARTIAL RECOVERY
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

print("🔍 HYPERPARAMETER GRID SEARCH")
print("=" * 70)
print("🎯 Goal: Find hyperparameters that beat 65% fMRI baseline")
print("📊 Testing critical hyperparameters systematically")
print()

class CrossAttentionTransformer(nn.Module):
    """Flexible Cross-Attention Transformer for hyperparameter search"""
    def __init__(self, fmri_dim, smri_dim, hidden_dim=64, num_heads=4, 
                 num_layers=2, dropout=0.1):
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

def load_and_preprocess_data(preprocessing_type='original'):
    """Load data with specified preprocessing type"""
    print(f"📊 Loading data with {preprocessing_type} preprocessing...")
    
    fmri_data = np.load(FMRI_DATA_PATH)
    smri_data = np.load(SMRI_DATA_PATH)
    labels = np.load(LABELS_PATH)
    
    print(f"Raw data shapes - fMRI: {fmri_data.shape}, sMRI: {smri_data.shape}")
    
    # Clean data
    smri_data = np.nan_to_num(smri_data, nan=0.0, posinf=1e6, neginf=-1e6)
    
    if preprocessing_type == 'original':
        # Original preprocessing
        smri_scaler = StandardScaler()
        smri_data_scaled = smri_scaler.fit_transform(smri_data)
        
        selector = SelectKBest(score_func=f_classif, k=min(300, smri_data.shape[1]))
        smri_data_selected = selector.fit_transform(smri_data_scaled, labels)
        print(f"   ✓ Original: StandardScaler + f_classif, {smri_data_selected.shape[1]} features")
        
    else:  # enhanced
        # Enhanced preprocessing
        smri_scaler = RobustScaler()
        smri_data_scaled = smri_scaler.fit_transform(smri_data)
        
        # Combined F-score + MI
        f_selector = SelectKBest(score_func=f_classif, k=min(400, smri_data.shape[1]))
        f_selected = f_selector.fit_transform(smri_data_scaled, labels)
        
        mi_selector = SelectKBest(score_func=mutual_info_classif, k=min(300, f_selected.shape[1]))
        smri_data_selected = mi_selector.fit_transform(f_selected, labels)
        print(f"   ✓ Enhanced: RobustScaler + F-score+MI, {smri_data_selected.shape[1]} features")
    
    # fMRI preprocessing (standard)
    fmri_scaler = StandardScaler()
    fmri_data_scaled = fmri_scaler.fit_transform(fmri_data)
    
    return fmri_data_scaled, smri_data_selected, labels

def evaluate_config(fmri_data, smri_data, labels, config):
    """Evaluate a single hyperparameter configuration"""
    
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
        
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
        
        # Initialize model
        model = CrossAttentionTransformer(
            fmri_dim=fmri_data.shape[1],
            smri_dim=smri_data.shape[1],
            hidden_dim=config['hidden_dim'],
            num_heads=config['num_heads'],
            num_layers=config['num_layers'],
            dropout=config['dropout']
        ).to(device)
        
        # Training setup
        criterion = nn.CrossEntropyLoss()
        
        if config['optimizer'] == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], 
                                 weight_decay=config['weight_decay'])
        else:
            optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], 
                                  weight_decay=config['weight_decay'])
        
        # Training
        model.train()
        for epoch in range(config['epochs']):
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
    
    return mean_acc, mean_bal_acc, mean_auc

def run_smart_grid_search():
    """Run smart hyperparameter search focusing on most impactful parameters"""
    
    # Priority configurations to test
    test_configs = [
        # Original baseline
        {'preprocessing': 'original', 'learning_rate': 1e-4, 'batch_size': 32, 'hidden_dim': 64, 
         'num_heads': 4, 'num_layers': 2, 'dropout': 0.1, 'epochs': 30, 'optimizer': 'adam',
         'weight_decay': 1e-4},
        
        # Learning rate variations
        {'preprocessing': 'original', 'learning_rate': 5e-5, 'batch_size': 32, 'hidden_dim': 64, 
         'num_heads': 4, 'num_layers': 2, 'dropout': 0.1, 'epochs': 30, 'optimizer': 'adam',
         'weight_decay': 1e-4},
         
        {'preprocessing': 'original', 'learning_rate': 2e-4, 'batch_size': 32, 'hidden_dim': 64, 
         'num_heads': 4, 'num_layers': 2, 'dropout': 0.1, 'epochs': 30, 'optimizer': 'adam',
         'weight_decay': 1e-4},
         
        {'preprocessing': 'original', 'learning_rate': 1e-5, 'batch_size': 32, 'hidden_dim': 64, 
         'num_heads': 4, 'num_layers': 2, 'dropout': 0.1, 'epochs': 40, 'optimizer': 'adam',
         'weight_decay': 1e-4},
        
        # Batch size variations
        {'preprocessing': 'original', 'learning_rate': 1e-4, 'batch_size': 16, 'hidden_dim': 64, 
         'num_heads': 4, 'num_layers': 2, 'dropout': 0.1, 'epochs': 30, 'optimizer': 'adam',
         'weight_decay': 1e-4},
         
        {'preprocessing': 'original', 'learning_rate': 1e-4, 'batch_size': 64, 'hidden_dim': 64, 
         'num_heads': 4, 'num_layers': 2, 'dropout': 0.1, 'epochs': 30, 'optimizer': 'adam',
         'weight_decay': 1e-4},
        
        # Architecture variations
        {'preprocessing': 'original', 'learning_rate': 1e-4, 'batch_size': 32, 'hidden_dim': 128, 
         'num_heads': 8, 'num_layers': 3, 'dropout': 0.15, 'epochs': 30, 'optimizer': 'adamw',
         'weight_decay': 1e-3},
         
        {'preprocessing': 'original', 'learning_rate': 1e-4, 'batch_size': 32, 'hidden_dim': 32, 
         'num_heads': 2, 'num_layers': 1, 'dropout': 0.05, 'epochs': 30, 'optimizer': 'adam',
         'weight_decay': 1e-5},
        
        # Enhanced preprocessing tests
        {'preprocessing': 'enhanced', 'learning_rate': 5e-5, 'batch_size': 32, 'hidden_dim': 64, 
         'num_heads': 4, 'num_layers': 2, 'dropout': 0.15, 'epochs': 30, 'optimizer': 'adamw',
         'weight_decay': 1e-4},
         
        {'preprocessing': 'enhanced', 'learning_rate': 1e-4, 'batch_size': 16, 'hidden_dim': 64, 
         'num_heads': 4, 'num_layers': 2, 'dropout': 0.2, 'epochs': 40, 'optimizer': 'adamw',
         'weight_decay': 5e-4},
        
        # Conservative improvements
        {'preprocessing': 'original', 'learning_rate': 8e-5, 'batch_size': 24, 'hidden_dim': 80, 
         'num_heads': 4, 'num_layers': 2, 'dropout': 0.12, 'epochs': 35, 'optimizer': 'adamw',
         'weight_decay': 3e-4},
         
        {'preprocessing': 'original', 'learning_rate': 1.2e-4, 'batch_size': 28, 'hidden_dim': 96, 
         'num_heads': 6, 'num_layers': 2, 'dropout': 0.08, 'epochs': 25, 'optimizer': 'adam',
         'weight_decay': 2e-4},
    ]
    
    results = []
    best_accuracy = 0
    best_config = None
    
    # Load both preprocessing types
    preprocessing_data = {}
    for prep_type in ['original', 'enhanced']:
        fmri_data, smri_data, labels = load_and_preprocess_data(prep_type)
        preprocessing_data[prep_type] = (fmri_data, smri_data, labels)
    
    print(f"\n🚀 Testing {len(test_configs)} strategic configurations...")
    
    for i, config in enumerate(test_configs):
        print(f"\n📋 Config {i+1}/{len(test_configs)}: {config['preprocessing']} preprocessing")
        print(f"   LR={config['learning_rate']}, BS={config['batch_size']}, HD={config['hidden_dim']}")
        print(f"   Heads={config['num_heads']}, Layers={config['num_layers']}, Dropout={config['dropout']}")
        
        # Get data for this preprocessing type
        fmri_data, smri_data, labels = preprocessing_data[config['preprocessing']]
        
        try:
            mean_acc, mean_bal_acc, mean_auc = evaluate_config(
                fmri_data, smri_data, labels, config
            )
            
            result = {
                'config': config.copy(),
                'mean_accuracy': mean_acc,
                'mean_balanced_accuracy': mean_bal_acc,
                'mean_auc': mean_auc,
                'beats_fmri_baseline': mean_acc > 0.65
            }
            
            results.append(result)
            
            print(f"   Results: Acc={mean_acc:.4f}, Bal_Acc={mean_bal_acc:.4f}, AUC={mean_auc:.4f}")
            if mean_acc > 0.65:
                print(f"   🎯 BEATS fMRI BASELINE! ({mean_acc:.1%} > 65%)")
            
            if mean_acc > best_accuracy:
                best_accuracy = mean_acc
                best_config = config.copy()
                
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
            continue
    
    return results, best_config, best_accuracy

def analyze_results(results, best_config, best_accuracy):
    """Analyze results and provide recommendations"""
    
    print("\n" + "="*70)
    print("🎯 HYPERPARAMETER SEARCH RESULTS")
    print("-" * 40)
    
    # Sort by accuracy
    results_sorted = sorted(results, key=lambda x: x['mean_accuracy'], reverse=True)
    
    print(f"📊 SUMMARY:")
    print(f"   Configurations tested: {len(results)}")
    print(f"   Best accuracy: {best_accuracy:.4f} ({best_accuracy:.1%})")
    beats_baseline = sum(1 for r in results if r['beats_fmri_baseline'])
    print(f"   Configs beating fMRI baseline (>65%): {beats_baseline}")
    print()
    
    print("🏆 TOP 5 CONFIGURATIONS:")
    for i, result in enumerate(results_sorted[:5]):
        config = result['config']
        acc = result['mean_accuracy']
        
        print(f"\n   #{i+1}: {acc:.4f} accuracy ({acc:.1%})")
        print(f"      Preprocessing: {config['preprocessing']}")
        print(f"      LR: {config['learning_rate']}, BS: {config['batch_size']}")
        print(f"      Architecture: HD={config['hidden_dim']}, Heads={config['num_heads']}, Layers={config['num_layers']}")
        print(f"      Dropout: {config['dropout']}, Optimizer: {config['optimizer']}")
        if acc > 0.65:
            print(f"      🎯 BEATS fMRI BASELINE!")
    
    # Analysis by preprocessing type
    original_results = [r for r in results if r['config']['preprocessing'] == 'original']
    enhanced_results = [r for r in results if r['config']['preprocessing'] == 'enhanced']
    
    print(f"\n🔍 PREPROCESSING ANALYSIS:")
    if original_results:
        orig_best = max(original_results, key=lambda x: x['mean_accuracy'])['mean_accuracy']
        orig_avg = np.mean([r['mean_accuracy'] for r in original_results])
        print(f"   Original - Best: {orig_best:.4f}, Average: {orig_avg:.4f}")
    
    if enhanced_results:
        enh_best = max(enhanced_results, key=lambda x: x['mean_accuracy'])['mean_accuracy']
        enh_avg = np.mean([r['mean_accuracy'] for r in enhanced_results])
        print(f"   Enhanced - Best: {enh_best:.4f}, Average: {enh_avg:.4f}")
    
    if original_results and enhanced_results:
        if orig_best > enh_best:
            print(f"   → Original preprocessing performs better (+{(orig_best-enh_best)*100:.1f} points)")
        else:
            print(f"   → Enhanced preprocessing performs better (+{(enh_best-orig_best)*100:.1f} points)")
    
    # Recommendations
    print(f"\n📋 RECOMMENDATIONS:")
    if beats_baseline > 0:
        print(f"   ✅ SUCCESS! Found {beats_baseline} config(s) beating fMRI baseline")
        print(f"   → Use best configuration for final model")
        print(f"   → Consider ensemble of top configurations")
    elif best_accuracy > 0.62:
        print(f"   ⚠️  Close to target (need {(0.65-best_accuracy)*100:.1f} more points)")
        print(f"   → Try longer training (more epochs)")
        print(f"   → Test ensemble methods")
        print(f"   → Consider data augmentation")
    else:
        print(f"   ❌ Significant gap to target ({(0.65-best_accuracy)*100:.1f} points)")
        print(f"   → Architectural changes needed")
        print(f"   → More sophisticated cross-attention")
        print(f"   → Consider different fusion strategies")
    
    return results_sorted

# Run the search
if __name__ == "__main__":
    print("Starting smart hyperparameter grid search...")
    
    try:
        results, best_config, best_accuracy = run_smart_grid_search()
        results_sorted = analyze_results(results, best_config, best_accuracy)
        
        # Save results
        output_data = {
            'test_name': 'smart_hyperparameter_search',
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_configs_tested': len(results),
                'best_accuracy': float(best_accuracy),
                'configs_beating_fmri': sum(1 for r in results if r['beats_fmri_baseline']),
                'best_config': best_config
            },
            'all_results': results_sorted,
            'target_baseline': 65.0
        }
        
        results_path = os.path.join(OUTPUT_PATH, 'hyperparameter_search_results.json')
        with open(results_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\n✅ Hyperparameter search completed!")
        print(f"📁 Results saved to: {results_path}")
        print(f"🎯 Best performance: {best_accuracy:.1%}")
        
        if best_accuracy > 0.65:
            print(f"🏆 SUCCESS! Found configuration beating fMRI baseline!")
        else:
            print(f"⚠️  Gap to 65% target: {(0.65-best_accuracy)*100:.1f} percentage points")
        
    except Exception as e:
        print(f"❌ Error during hyperparameter search: {str(e)}")
        raise 
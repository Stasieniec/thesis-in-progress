#!/usr/bin/env python3
"""
sMRI Ensemble for Final 58%+ Push
=================================

Ensemble the best performing configurations to get the final boost.
"""

import os
import sys
from pathlib import Path

# Add src to path
script_dir = Path(__file__).parent
src_dir = script_dir.parent / "src"
sys.path.insert(0, str(src_dir))

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score

# Project imports
from config import get_config
from utils.subject_matching import get_matched_datasets
from training import Trainer, set_seed
from models.enhanced_smri import EnhancedSMRITransformer
from models import SMRITransformer

import logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def test_ensemble(num_folds: int = 5):
    """Test ensemble of top performing configurations."""
    
    logger.info("🎯 ENSEMBLE TEST FOR 58%+ sMRI ACCURACY")
    logger.info("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data
    if Path('/content/drive').exists():
        config = get_config('cross_attention')
        matched_data = get_matched_datasets(
            fmri_roi_dir=str(config.fmri_roi_dir),
            smri_data_path=str(config.smri_data_path),
            phenotypic_file=str(config.phenotypic_file),
            verbose=False
        )
        smri_data = matched_data['smri_features']
        labels = matched_data['smri_labels']
    else:
        # Mock data
        n_subjects = 100
        smri_data = np.random.randn(n_subjects, 800).astype(np.float32)
        labels = np.random.randint(0, 2, n_subjects)
    
    # Best configs for ensemble
    configs = [
        {
            'd_model': 192, 'n_heads': 8, 'n_layers': 3,
            'dropout': 0.1, 'learning_rate': 0.002,
            'batch_size': 64, 'num_epochs': 80
        },
        {
            'd_model': 256, 'n_heads': 8, 'n_layers': 4,
            'dropout': 0.12, 'learning_rate': 0.0018,
            'batch_size': 48, 'num_epochs': 90
        },
        {
            'd_model': 224, 'n_heads': 8, 'n_layers': 3,
            'dropout': 0.08, 'learning_rate': 0.0015,
            'batch_size': 56, 'num_epochs': 85
        }
    ]
    
    set_seed(42)
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
    fold_results = []
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(smri_data, labels)):
        logger.info(f"\n🔥 Ensemble Fold {fold + 1}/{num_folds}")
        
        X_train, X_test = smri_data[train_idx], smri_data[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]
        
        # Get predictions from each model
        all_preds = []
        
        for i, config in enumerate(configs):
            logger.info(f"   Training model {i+1}/3...")
            
            from sklearn.model_selection import train_test_split
            X_tr, X_val, y_tr, y_val = train_test_split(
                X_train, y_train, test_size=0.2, stratify=y_train, random_state=42+fold
            )
            
            # Preprocess
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_tr)
            X_val = scaler.transform(X_val)
            X_test_scaled = scaler.transform(X_test)
            
            # Create and train model
            model = EnhancedSMRITransformer(
                input_dim=X_tr.shape[1],
                d_model=config['d_model'],
                n_heads=config['n_heads'],
                n_layers=config['n_layers'],
                dropout=config['dropout']
            ).to(device)
            
            # Quick training config
            temp_config = get_config('smri')
            temp_config.learning_rate = config['learning_rate']
            temp_config.batch_size = config['batch_size']
            temp_config.num_epochs = config['num_epochs']
            temp_config.early_stop_patience = 15
            temp_config.output_dir = Path('./temp_ensemble')
            temp_config.output_dir.mkdir(exist_ok=True)
            
            trainer = Trainer(model, device, temp_config, model_type='single')
            
            # Data loaders
            from torch.utils.data import DataLoader, TensorDataset
            train_loader = DataLoader(
                TensorDataset(torch.FloatTensor(X_tr), torch.LongTensor(y_tr)),
                batch_size=config['batch_size'], shuffle=True
            )
            val_loader = DataLoader(
                TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val)),
                batch_size=config['batch_size'], shuffle=False
            )
            
            checkpoint_path = temp_config.output_dir / f'model_{fold}_{i}.pth'
            
            # Train
            trainer.fit(train_loader, val_loader, config['num_epochs'], checkpoint_path, y_tr)
            
            # Predict
            model.eval()
            with torch.no_grad():
                test_tensor = torch.FloatTensor(X_test_scaled).to(device)
                logits = model(test_tensor)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                all_preds.append(probs)
            
            # Cleanup
            try:
                if checkpoint_path.exists():
                    checkpoint_path.unlink()
            except:
                pass
        
        # Ensemble prediction
        ensemble_probs = np.mean(all_preds, axis=0)
        ensemble_preds = np.argmax(ensemble_probs, axis=1)
        
        # Metrics
        accuracy = accuracy_score(y_test, ensemble_preds)
        balanced_acc = balanced_accuracy_score(y_test, ensemble_preds)
        auc = roc_auc_score(y_test, ensemble_probs[:, 1])
        
        fold_results.append({
            'test_accuracy': accuracy,
            'test_balanced_accuracy': balanced_acc,
            'test_auc': auc
        })
        
        logger.info(f"   Fold {fold+1} Result: {accuracy*100:.1f}%")
    
    # Final results
    accuracies = [r['test_accuracy'] for r in fold_results]
    mean_acc = np.mean(accuracies) * 100
    std_acc = np.std(accuracies) * 100
    
    logger.info(f"\n🏆 ENSEMBLE RESULTS: {mean_acc:.1f}% ± {std_acc:.1f}%")
    
    if mean_acc >= 58.0:
        logger.info(f"🎉 SUCCESS! Ensemble achieved 58%+ accuracy!")
    else:
        logger.info(f"📊 Gap: {58.0 - mean_acc:.1f}%")
    
    return mean_acc >= 58.0


if __name__ == "__main__":
    test_ensemble(num_folds=5) 
#!/usr/bin/env python3
"""
Final Push for 58%+ sMRI Accuracy
=================================

Fine-tuned search around the successful High_LR_Enhanced configuration.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add src to path
script_dir = Path(__file__).parent
src_dir = script_dir.parent / "src"
sys.path.insert(0, str(src_dir))

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

# Project imports
from config import get_config
from utils.subject_matching import get_matched_datasets
from training import Trainer, set_seed
from models.enhanced_smri import EnhancedSMRITransformer

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def final_smri_push(num_folds: int = 5, device: str = 'auto'):
    """
    Final push for 58%+ sMRI accuracy with focused hyperparameter search.
    """
    
    logger.info("🎯 FINAL PUSH FOR 58%+ sMRI ACCURACY")
    logger.info("=" * 70)
    logger.info("Building on High_LR_Enhanced success (56.0%)")
    
    # Setup device
    if device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    
    logger.info(f"Using device: {device}")
    
    # Load data
    logger.info("📊 Loading matched subject data...")
    
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
        # Mock data for local testing
        logger.info("🔬 Local testing mode - creating mock data")
        n_subjects = 100
        smri_data = np.random.randn(n_subjects, 800).astype(np.float32)
        labels = np.random.randint(0, 2, n_subjects)
    
    logger.info(f"Data shape: {smri_data.shape}, Labels: {len(labels)}")
    
    # Focused configurations around High_LR_Enhanced winner
    final_configs = [
        # 1. Better regularization + more epochs (BEST CHANCE FOR 58%+)
        {
            'name': 'High_LR_Stable',
            'd_model': 192,
            'n_heads': 8,
            'n_layers': 3,
            'dropout': 0.15,        # Increased from 0.1
            'layer_dropout': 0.08,  # Increased from 0.05
            'learning_rate': 0.002,
            'batch_size': 64,
            'weight_decay': 2e-4,   # Increased from 1e-4
            'num_epochs': 120,      # More epochs
            'patience': 35,         # More patience
            'use_feature_engineering': True,
            'use_positional_encoding': False
        },
        
        # 2. Original High_LR_Enhanced with more epochs
        {
            'name': 'High_LR_Extended',
            'd_model': 192,
            'n_heads': 8,
            'n_layers': 3,
            'dropout': 0.1,
            'layer_dropout': 0.05,
            'learning_rate': 0.002,
            'batch_size': 64,
            'weight_decay': 1e-4,
            'num_epochs': 100,
            'use_feature_engineering': True,
            'use_positional_encoding': False
        },
        
        # 3. Sweet spot learning rate (between 0.002 and 0.003)
        {
            'name': 'High_LR_Sweet_Spot',
            'd_model': 192,
            'n_heads': 8,
            'n_layers': 3,
            'dropout': 0.12,        # Moderate increase
            'layer_dropout': 0.06,
            'learning_rate': 0.0025, # Sweet spot between 0.002 and 0.003
            'batch_size': 64,
            'weight_decay': 1.5e-4,
            'num_epochs': 100,
            'use_feature_engineering': True,
            'use_positional_encoding': False
        },
        
        # 4. Even higher learning rate with more regularization
        {
            'name': 'Ultra_High_LR_Stable',
            'd_model': 192,
            'n_heads': 8,
            'n_layers': 3,
            'dropout': 0.18,        # Higher dropout for higher LR
            'layer_dropout': 0.1,
            'learning_rate': 0.003,
            'batch_size': 64,
            'weight_decay': 2.5e-4, # More weight decay
            'num_epochs': 90,
            'use_feature_engineering': True,
            'use_positional_encoding': False
        }
    ]
    
    results = {}
    best_accuracy = 0.0
    best_config = None
    target_reached = False
    
    for i, config in enumerate(final_configs, 1):
        config_name = config['name']
        
        logger.info(f"\n🔥 Final Push {i}/{len(final_configs)}: {config_name}")
        logger.info(f"   lr={config['learning_rate']}, epochs={config['num_epochs']}")
        logger.info(f"   d_model={config['d_model']}, layers={config['n_layers']}")
        
        try:
            result = test_final_config(smri_data, labels, config, num_folds, device)
            
            accuracy = result['mean_accuracy']
            std = result['std_accuracy']
            
            results[config_name] = {
                'config': config,
                'accuracy': accuracy,
                'std': std,
                'detailed_result': result
            }
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_config = config_name
            
            if accuracy >= 58.0:
                status = "🎉 TARGET REACHED!"
                target_reached = True
            else:
                gap = 58.0 - accuracy
                status = f"📊 Gap: {gap:.1f}%"
            
            logger.info(f"   ✅ Result: {accuracy:.1f}% ± {std:.1f}% {status}")
            
            if target_reached:
                logger.info(f"   🏆 SUCCESS! 58%+ accuracy achieved!")
        
        except Exception as e:
            logger.info(f"   ❌ Failed: {str(e)}")
            results[config_name] = {'config': config, 'error': str(e)}
    
    return results, target_reached


def test_final_config(smri_data, labels, config, num_folds, device):
    """Test a final configuration."""
    
    set_seed(42)
    
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
    fold_results = []
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(smri_data, labels)):
        # Split data
        X_train_fold, X_test = smri_data[train_idx], smri_data[test_idx]
        y_train_fold, y_test = labels[train_idx], labels[test_idx]
        
        # Further split for validation
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_fold, y_train_fold,
            test_size=0.2,
            stratify=y_train_fold,
            random_state=42 + fold
        )
        
        # Preprocess
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)
        
        # Create model
        model = EnhancedSMRITransformer(
            input_dim=X_train.shape[1],
            d_model=config['d_model'],
            n_heads=config['n_heads'],
            n_layers=config['n_layers'],
            dropout=config['dropout'],
            layer_dropout=config['layer_dropout'],
            use_feature_engineering=config.get('use_feature_engineering', True),
            use_positional_encoding=config.get('use_positional_encoding', False)
        ).to(device)
        
        # Create config
        temp_config = get_config('smri')
        temp_config.learning_rate = config['learning_rate']
        temp_config.weight_decay = config['weight_decay']
        temp_config.batch_size = config['batch_size']
        temp_config.num_epochs = config['num_epochs']
        temp_config.early_stop_patience = config.get('patience', 25)  # Use config patience or default
        temp_config.use_class_weights = True
        temp_config.label_smoothing = 0.1
        temp_config.output_dir = Path('./temp_final_push')
        temp_config.output_dir.mkdir(exist_ok=True)
        
        # Train
        trainer = Trainer(model, device, temp_config, model_type='single')
        
        # Data loaders
        from torch.utils.data import DataLoader, TensorDataset
        train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
        val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
        test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
        
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
        
        # Checkpoint path
        checkpoint_path = temp_config.output_dir / f'temp_final_fold_{fold}.pth'
        
        # Train
        history = trainer.fit(
            train_loader, val_loader,
            num_epochs=temp_config.num_epochs,
            checkpoint_path=checkpoint_path,
            y_train=y_train
        )
        
        # Evaluate
        test_metrics = trainer.evaluate_final(test_loader)
        
        fold_results.append({
            'fold': fold,
            'test_accuracy': test_metrics['accuracy'],
            'test_balanced_accuracy': test_metrics['balanced_accuracy'],
            'test_auc': test_metrics['auc']
        })
        
        # Cleanup
        try:
            if checkpoint_path.exists():
                checkpoint_path.unlink()
        except Exception:
            pass
    
    # Aggregate results
    accuracies = [r['test_accuracy'] for r in fold_results]
    balanced_accs = [r['test_balanced_accuracy'] for r in fold_results]
    aucs = [r['test_auc'] for r in fold_results]
    
    return {
        'fold_results': fold_results,
        'mean_accuracy': np.mean(accuracies) * 100,
        'std_accuracy': np.std(accuracies) * 100,
        'mean_balanced_accuracy': np.mean(balanced_accs) * 100,
        'std_balanced_accuracy': np.std(balanced_accs) * 100,
        'mean_auc': np.mean(aucs),
        'std_auc': np.std(aucs)
    }


def main():
    parser = argparse.ArgumentParser(description='Final Push for 58%+ sMRI Accuracy')
    parser.add_argument('--num_folds', type=int, default=5, help='Number of CV folds')
    parser.add_argument('--device', type=str, default='auto', help='Device to use')
    
    args = parser.parse_args()
    
    results, success = final_smri_push(
        num_folds=args.num_folds,
        device=args.device
    )
    
    logger.info("✅ Final push completed!")


if __name__ == "__main__":
    main() 
#!/usr/bin/env python3
"""
Advanced sMRI Hyperparameter Search for 58%+ Accuracy
====================================================

Extended search with enhanced models and training techniques.
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
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score

# Project imports
from config import get_config
from utils.subject_matching import get_matched_datasets
from training import Trainer, set_seed
from models import SMRITransformer
from models.enhanced_smri import EnhancedSMRITransformer, SMRIEnsemble

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def advanced_smri_search(
    num_folds: int = 3,
    num_epochs: int = 60,
    device: str = 'auto'
):
    """
    Advanced sMRI hyperparameter search with enhanced models.
    """
    
    logger.info("🚀 ADVANCED sMRI HYPERPARAMETER SEARCH")
    logger.info("=" * 70)
    logger.info("Goal: Achieve 58%+ accuracy with enhanced techniques")
    
    # Setup device
    if device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    
    logger.info(f"Using device: {device}")
    
    # Load data
    logger.info("📊 Loading matched subject data...")
    
    # Check if we're in Colab
    if Path('/content/drive').exists():
        # Real data
        config = get_config('smri')
        matched_data = get_matched_datasets(
            fmri_roi_dir=str(config.fmri_roi_dir),
            smri_data_path=str(config.smri_data_path),
            phenotypic_file=str(config.phenotypic_file),
            verbose=True
        )
        smri_data = matched_data['smri_data']
        labels = matched_data['labels']
    else:
        # Mock data for local testing
        logger.info("🔬 Local testing mode - creating mock data")
        n_subjects = 100
        smri_data = np.random.randn(n_subjects, 800).astype(np.float32)
        labels = np.random.randint(0, 2, n_subjects)
    
    logger.info(f"Data shape: {smri_data.shape}, Labels: {len(labels)}")
    
    # Advanced parameter configurations
    advanced_configs = [
        # 1. Enhanced Standard Model
        {
            'name': 'Enhanced_Standard',
            'model_class': EnhancedSMRITransformer,
            'd_model': 256,
            'n_heads': 8,
            'n_layers': 4,
            'dropout': 0.1,
            'layer_dropout': 0.05,
            'learning_rate': 0.0005,
            'batch_size': 32,
            'weight_decay': 1e-3,
            'num_epochs': 80,
            'use_feature_engineering': True,
            'use_positional_encoding': True,
            'use_scheduler': True
        },
        
        # 2. Large Enhanced Model
        {
            'name': 'Enhanced_Large',
            'model_class': EnhancedSMRITransformer,
            'd_model': 384,
            'n_heads': 12,
            'n_layers': 6,
            'dropout': 0.15,
            'layer_dropout': 0.08,
            'learning_rate': 0.0003,
            'batch_size': 24,
            'weight_decay': 1e-3,
            'num_epochs': 100,
            'use_feature_engineering': True,
            'use_positional_encoding': True,
            'use_scheduler': True
        },
        
        # 3. Original Model with Extended Training
        {
            'name': 'Original_Extended',
            'model_class': SMRITransformer,
            'd_model': 256,
            'n_heads': 8,
            'n_layers': 4,
            'dropout': 0.1,
            'layer_dropout': 0.05,
            'learning_rate': 0.0008,
            'batch_size': 32,
            'weight_decay': 5e-4,
            'num_epochs': 120,
            'use_scheduler': True,
            'patience': 30
        },
        
        # 4. High Learning Rate Variant
        {
            'name': 'High_LR_Enhanced',
            'model_class': EnhancedSMRITransformer,
            'd_model': 192,
            'n_heads': 8,
            'n_layers': 3,
            'dropout': 0.1,
            'layer_dropout': 0.05,
            'learning_rate': 0.002,  # Much higher
            'batch_size': 64,
            'weight_decay': 1e-4,
            'num_epochs': 60,
            'use_feature_engineering': True,
            'use_positional_encoding': False,
            'use_scheduler': True
        },
        
        # 5. Deep Narrow Model
        {
            'name': 'Deep_Narrow',
            'model_class': EnhancedSMRITransformer,
            'd_model': 128,
            'n_heads': 8,
            'n_layers': 8,  # Very deep
            'dropout': 0.2,  # Higher dropout for deep model
            'layer_dropout': 0.1,
            'learning_rate': 0.0005,
            'batch_size': 48,
            'weight_decay': 1e-3,
            'num_epochs': 100,
            'use_feature_engineering': True,
            'use_positional_encoding': True,
            'use_scheduler': True
        }
    ]
    
    results = {}
    best_accuracy = 0.0
    best_config = None
    
    for i, config in enumerate(advanced_configs, 1):
        config_name = config['name']
        
        logger.info(f"\n🧪 Testing Configuration {i}/{len(advanced_configs)}: {config_name}")
        logger.info(f"   Model: {config['model_class'].__name__}")
        logger.info(f"   d_model={config['d_model']}, n_heads={config['n_heads']}, n_layers={config['n_layers']}")
        logger.info(f"   lr={config['learning_rate']}, batch_size={config['batch_size']}, epochs={config.get('num_epochs', num_epochs)}")
        
        try:
            result = test_advanced_config(
                smri_data, labels, config, num_folds, device
            )
            
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
            
            status = "🎉" if accuracy >= 58.0 else "📊"
            logger.info(f"   ✅ Result: {accuracy:.1f}% ± {std:.1f}% {status}")
            
            if accuracy >= 58.0:
                logger.info(f"   🎯 TARGET ACHIEVED! 58%+ accuracy reached")
        
        except Exception as e:
            logger.info(f"   ❌ Failed: {str(e)}")
            results[config_name] = {
                'config': config,
                'error': str(e)
            }
    
    # Summary
    logger.info(f"\n🏆 ADVANCED SEARCH SUMMARY")
    logger.info("=" * 70)
    logger.info(f"🥇 Best Configuration: {best_config}")
    logger.info(f"🎯 Best Accuracy: {best_accuracy:.1f}%")
    logger.info("")
    logger.info("📈 All Results (sorted by accuracy):")
    
    sorted_results = sorted(
        [(name, r) for name, r in results.items() if 'accuracy' in r],
        key=lambda x: x[1]['accuracy'],
        reverse=True
    )
    
    for rank, (name, result) in enumerate(sorted_results, 1):
        acc = result['accuracy']
        std = result['std']
        status = "🎉" if acc >= 58.0 else "📊"
        logger.info(f"   {rank}. {name}: {acc:.1f}% ± {std:.1f}% {status}")
    
    return results


def test_advanced_config(
    smri_data: np.ndarray,
    labels: np.ndarray,
    config: dict,
    num_folds: int,
    device: torch.device
) -> dict:
    """Test a single advanced configuration."""
    
    set_seed(42)
    
    # Cross-validation
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
        model_class = config['model_class']
        model_kwargs = {
            'input_dim': X_train.shape[1],
            'd_model': config['d_model'],
            'n_heads': config['n_heads'],
            'n_layers': config['n_layers'],
            'dropout': config['dropout']
        }
        
        # Add model-specific parameters
        if 'layer_dropout' in config:
            model_kwargs['layer_dropout'] = config['layer_dropout']
        if 'use_feature_engineering' in config:
            model_kwargs['use_feature_engineering'] = config['use_feature_engineering']
        if 'use_positional_encoding' in config:
            model_kwargs['use_positional_encoding'] = config['use_positional_encoding']
        if 'num_models' in config:  # For ensemble
            model_kwargs['num_models'] = config['num_models']
        
        model = model_class(**model_kwargs).to(device)
        
        # Create config
        temp_config = get_config('smri')
        temp_config.learning_rate = config['learning_rate']
        temp_config.weight_decay = config['weight_decay']
        temp_config.batch_size = config['batch_size']
        temp_config.num_epochs = config.get('num_epochs', 60)
        temp_config.early_stop_patience = config.get('patience', 20)
        temp_config.use_class_weights = True  # Always use for sMRI
        temp_config.label_smoothing = 0.1
        temp_config.output_dir = Path('./temp_advanced_search')
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
        checkpoint_path = temp_config.output_dir / f'temp_advanced_fold_{fold}.pth'
        
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
    parser = argparse.ArgumentParser(description='Advanced sMRI Hyperparameter Search')
    parser.add_argument('--num_folds', type=int, default=3, help='Number of CV folds')
    parser.add_argument('--num_epochs', type=int, default=60, help='Base number of epochs')
    parser.add_argument('--device', type=str, default='auto', help='Device to use')
    
    args = parser.parse_args()
    
    results = advanced_smri_search(
        num_folds=args.num_folds,
        num_epochs=args.num_epochs,
        device=args.device
    )
    
    logger.info("✅ Advanced search completed!")


if __name__ == "__main__":
    main() 
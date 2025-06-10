"""Helper utility functions for ABIDE experiments."""

import torch
import numpy as np
from typing import Tuple, List, Dict, Any
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

from training import Trainer, set_seed, create_data_loaders, create_multimodal_data_loaders
from data import MultiModalPreprocessor


def get_device(device_preference: str = 'auto') -> torch.device:
    """
    Get the appropriate device for training.
    
    Args:
        device_preference: 'auto', 'cuda', or 'cpu'
        
    Returns:
        PyTorch device object
    """
    if device_preference == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_preference)
    
    # Print device info
    print(f"🔧 Using device: {device}")
    if device.type == 'cuda':
        print(f"🔧 GPU: {torch.cuda.get_device_name(0)}")
        print(f"🔧 Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    return device


def run_cross_validation(
    features: np.ndarray,
    labels: np.ndarray,
    model_class: Any,
    config: Any,
    experiment_type: str = 'single',  # 'single' or 'multimodal'
    fmri_features: np.ndarray = None,
    smri_features: np.ndarray = None,
    verbose: bool = True
) -> List[Dict]:
    """
    Run stratified cross-validation with proper preprocessing.
    
    Args:
        features: Feature array (for single modality)
        labels: Label array
        model_class: Model class to instantiate
        config: Configuration object
        experiment_type: Type of experiment ('single' or 'multimodal')
        fmri_features: fMRI features (for multimodal)
        smri_features: sMRI features (for multimodal)
        verbose: Whether to print progress
        
    Returns:
        List of cross-validation results
    """
    # Set random seed
    set_seed(config.seed)
    
    # Get device
    device = get_device(getattr(config, 'device', 'auto'))
    
    # Initialize cross-validation
    kfold = StratifiedKFold(n_splits=config.num_folds, shuffle=True, random_state=config.seed)
    cv_results = []
    
    if verbose:
        print(f"🔄 Starting {config.num_folds}-fold cross-validation...")
        print(f"Total samples: {len(labels)}")
    
    # Handle different experiment types
    if experiment_type == 'multimodal':
        assert fmri_features is not None and smri_features is not None
        iterator = kfold.split(fmri_features, labels)
    else:
        iterator = kfold.split(features, labels)
    
    # Run cross-validation
    for fold, (train_idx, test_idx) in enumerate(iterator, 1):
        if verbose:
            print(f"\n{'='*60}")
            print(f"FOLD {fold}/{config.num_folds}")
            print(f"{'='*60}")
        
        if experiment_type == 'multimodal':
            # Multimodal cross-validation
            fold_results = _run_multimodal_fold(
                fold, train_idx, test_idx,
                fmri_features, smri_features, labels,
                model_class, config, device, verbose
            )
        else:
            # Single modality cross-validation
            fold_results = _run_single_fold(
                fold, train_idx, test_idx,
                features, labels,
                model_class, config, device, verbose
            )
        
        cv_results.append(fold_results)
        
        # Clean up GPU memory
        torch.cuda.empty_cache()
    
    return cv_results


def _run_single_fold(
    fold: int,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    features: np.ndarray,
    labels: np.ndarray,
    model_class: Any,
    config: Any,
    device: torch.device,
    verbose: bool
) -> Dict:
    """Run a single fold for single modality experiments."""
    
    # Split data
    X_train_fold, X_test_fold = features[train_idx], features[test_idx]
    y_train_fold, y_test_fold = labels[train_idx], labels[test_idx]
    
    # Further split training into train/val
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_fold, y_train_fold,
        test_size=config.val_size,
        stratify=y_train_fold,
        random_state=config.seed
    )
    
    # **CRITICAL FIX**: Apply proper preprocessing for all experiments
    # Use StandardScaler for both sMRI and fMRI (simpler and more reliable)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test_fold)
    
    if verbose and hasattr(config, 'feature_selection_k'):
        print(f"📊 sMRI preprocessing: Standardized {X_train.shape[1]} features")
    
    if verbose:
        print(f"📊 Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        print(f"📊 Class distribution - Train: {np.bincount(y_train)}, Test: {np.bincount(y_test_fold)}")
    
    # Create data loaders
    dataset_type = 'smri' if hasattr(config, 'feature_selection_k') else 'base'
    train_loader, val_loader = create_data_loaders(
        X_train, y_train, X_val, y_val,
        batch_size=config.batch_size,
        augment_train=True,
        noise_std=config.noise_std,
        augment_prob=config.augment_prob,
        dataset_type=dataset_type
    )
    
    # Initialize model
    if hasattr(config, 'feature_selection_k'):
        # sMRI model - has feature_selection_k attribute unique to SMRIConfig
        model = model_class(
            input_dim=X_train.shape[1],
            d_model=config.d_model,
            n_heads=config.num_heads,
            n_layers=config.num_layers,
            dropout=config.dropout,
            layer_dropout=config.layer_dropout
        ).to(device)  # **CRITICAL FIX**: Move model to device
    else:
        # fMRI model - has dim_feedforward attribute unique to FMRIConfig
        model = model_class(
            feat_dim=X_train.shape[1],
            d_model=config.d_model,
            dim_feedforward=config.dim_feedforward,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            dropout=config.dropout
        ).to(device)  # **CRITICAL FIX**: Move model to device
    
    # Initialize trainer
    trainer = Trainer(model, device, config, model_type='single')
    
    # Train model
    checkpoint_path = config.output_dir / f'best_model_fold{fold}.pt'
    history = trainer.fit(
        train_loader, val_loader,
        num_epochs=config.num_epochs,
        checkpoint_path=checkpoint_path,
        y_train=y_train
    )
    
    # Evaluate on test set
    test_loader, _ = create_data_loaders(
        X_test, y_test_fold, X_test, y_test_fold,
        batch_size=config.batch_size,
        augment_train=False,
        dataset_type=dataset_type
    )
    
    test_metrics = trainer.evaluate_final(test_loader)
    
    # Store results
    fold_results = {
        'fold': fold,
        'test_accuracy': test_metrics['accuracy'],
        'test_balanced_accuracy': test_metrics['balanced_accuracy'],
        'test_auc': test_metrics['auc'],
        'history': history
    }
    
    if verbose:
        print(f"\n🎯 Fold {fold} Test Results:")
        print(f"   Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"   Balanced Accuracy: {test_metrics['balanced_accuracy']:.4f}")
        print(f"   AUC: {test_metrics['auc']:.4f}")
    
    return fold_results


def _run_multimodal_fold(
    fold: int,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    fmri_features: np.ndarray,
    smri_features: np.ndarray,
    labels: np.ndarray,
    model_class: Any,
    config: Any,
    device: torch.device,
    verbose: bool
) -> Dict:
    """Run a single fold for multimodal experiments."""
    
    # Split data
    fmri_train_fold, fmri_test_fold = fmri_features[train_idx], fmri_features[test_idx]
    smri_train_fold, smri_test_fold = smri_features[train_idx], smri_features[test_idx]
    y_train_fold, y_test_fold = labels[train_idx], labels[test_idx]
    
    # Further split training into train/val
    train_val_split = train_test_split(
        fmri_train_fold, smri_train_fold, y_train_fold,
        test_size=config.val_size,
        stratify=y_train_fold,
        random_state=config.seed
    )
    fmri_train, fmri_val, smri_train, smri_val, y_train, y_val = train_val_split
    
    # Preprocess multimodal data
    preprocessor = MultiModalPreprocessor(smri_feature_selection_k=config.smri_feat_selection)
    preprocessor.fit(fmri_train, smri_train, y_train)
    
    fmri_train, smri_train = preprocessor.transform(fmri_train, smri_train)
    fmri_val, smri_val = preprocessor.transform(fmri_val, smri_val)
    fmri_test, smri_test = preprocessor.transform(fmri_test_fold, smri_test_fold)
    
    if verbose:
        print(f"📊 Train: {len(y_train)}, Val: {len(y_val)}, Test: {len(y_test_fold)}")
        print(f"📊 Class distribution - Train: {np.bincount(y_train)}, Test: {np.bincount(y_test_fold)}")
    
    # Create data loaders
    train_loader, val_loader = create_multimodal_data_loaders(
        fmri_train, smri_train, y_train,
        fmri_val, smri_val, y_val,
        batch_size=config.batch_size,
        augment_train=True,
        noise_std=config.noise_std,
        augment_prob=config.augment_prob
    )
    
    # Initialize model with appropriate parameters
    model_name = model_class.__name__
    
    if 'AdvancedCrossAttention' in model_name or any(name in model_name for name in 
        ['Bidirectional', 'Hierarchical', 'Contrastive', 'Adaptive', 'Ensemble']):
        # Advanced models with specific parameter names
        model_kwargs = {
            'fmri_dim': fmri_train.shape[1],
            'smri_dim': smri_train.shape[1], 
            'd_model': config.d_model,
            'dropout': config.dropout
        }
        
        # Add n_heads if the model accepts it
        import inspect
        sig = inspect.signature(model_class.__init__)
        if 'n_heads' in sig.parameters:
            model_kwargs['n_heads'] = config.num_heads
        if 'n_cross_layers' in sig.parameters:
            model_kwargs['n_cross_layers'] = config.num_cross_layers
        if 'n_ensembles' in sig.parameters:
            model_kwargs['n_ensembles'] = getattr(config, 'n_ensembles', 3)
        if 'temperature' in sig.parameters:
            model_kwargs['temperature'] = getattr(config, 'temperature', 0.1)
            
        model = model_class(**model_kwargs).to(device)  # **CRITICAL FIX**: Move model to device
    else:
        # Original models with legacy parameter names
        model = model_class(
            fmri_dim=fmri_train.shape[1],
            smri_dim=smri_train.shape[1],
            d_model=config.d_model,
            n_heads=config.num_heads,
            n_layers=config.num_layers,
            n_cross_layers=config.num_cross_layers,
            dropout=config.dropout
        ).to(device)  # **CRITICAL FIX**: Move model to device
    
    # Initialize trainer
    trainer = Trainer(model, device, config, model_type='multimodal')
    
    # Train model
    checkpoint_path = config.output_dir / f'best_model_fold{fold}.pt'
    history = trainer.fit(
        train_loader, val_loader,
        num_epochs=config.num_epochs,
        checkpoint_path=checkpoint_path,
        y_train=y_train
    )
    
    # Evaluate on test set
    test_loader, _ = create_multimodal_data_loaders(
        fmri_test, smri_test, y_test_fold,
        fmri_test, smri_test, y_test_fold,
        batch_size=config.batch_size,
        augment_train=False
    )
    
    test_metrics = trainer.evaluate_final(test_loader)
    
    # Store results
    fold_results = {
        'fold': fold,
        'test_accuracy': test_metrics['accuracy'],
        'test_balanced_accuracy': test_metrics['balanced_accuracy'],
        'test_auc': test_metrics['auc'],
        'history': history
    }
    
    if verbose:
        print(f"\n🎯 Fold {fold} Test Results:")
        print(f"   Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"   Balanced Accuracy: {test_metrics['balanced_accuracy']:.4f}")
        print(f"   AUC: {test_metrics['auc']:.4f}")
    
    return fold_results 
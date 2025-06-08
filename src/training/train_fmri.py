"""
fMRI training module for ABIDE experiments.
"""

import sys
import time
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any, Optional, Set

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from data.fmri_processor import FMRIDataProcessor
from models.fmri_transformer import SingleAtlasTransformer
from training.trainer import Trainer
from training.utils import set_seed, create_data_loaders
from evaluation.metrics import calculate_metrics


def run_fmri_training(
    fmri_data_path: str,
    phenotypic_file: str,
    matched_subject_ids: Optional[Set[str]] = None,
    num_epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    patience: int = 10,
    random_seed: int = 42,
    device: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run fMRI training experiment.
    
    Args:
        fmri_data_path: Path to fMRI data directory
        phenotypic_file: Path to phenotypic CSV file
        matched_subject_ids: Optional set of subject IDs to use (for fair comparison)
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        patience: Early stopping patience
        random_seed: Random seed for reproducibility
        device: Device to use ('cuda' or 'cpu', auto-detect if None)
        
    Returns:
        Dictionary containing training results and metrics
    """
    
    # Set random seed for reproducibility
    set_seed(random_seed)
    
    # Set device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"🎯 Device: {device}")
    print(f"🧬 Random Seed: {random_seed}")
    
    # Load and process fMRI data
    print("📊 Loading fMRI data...")
    start_time = time.time()
    
    fmri_processor = FMRIDataProcessor(
        data_path=Path(fmri_data_path),
        pheno_file=Path(phenotypic_file),
        sequence_length=200,  # CC200 atlas
        augment_data=True
    )
    
    # Load all subjects
    fmri_data = fmri_processor.load_all_subjects()
    
    # Filter to matched subjects if provided
    if matched_subject_ids:
        filtered_data = {}
        for sub_id in matched_subject_ids:
            if sub_id in fmri_data:
                filtered_data[sub_id] = fmri_data[sub_id]
        fmri_data = filtered_data
        print(f"🎯 Using {len(fmri_data)} matched subjects")
    else:
        print(f"📊 Using all {len(fmri_data)} available subjects")
    
    # Extract features and labels
    X = np.array([data['features'] for data in fmri_data.values()])
    y = np.array([data['label'] for data in fmri_data.values()])
    subject_ids = list(fmri_data.keys())
    
    data_load_time = time.time() - start_time
    print(f"✅ Data loaded in {data_load_time:.1f}s - Shape: {X.shape}")
    print(f"🧠 ASD: {np.sum(y)}, Control: {len(y) - np.sum(y)}")
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        X=X, 
        y=y,
        batch_size=batch_size,
        random_seed=random_seed
    )
    
    # Initialize model
    print("🏗️ Initializing fMRI Transformer model...")
    model = SingleAtlasTransformer(
        num_features=X.shape[2],     # Time points
        num_classes=2,               # Binary classification
        d_model=128,
        nhead=8,
        num_layers=4,
        dropout=0.1
    ).to(device)
    
    print(f"📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        device=device,
        patience=patience,
        random_seed=random_seed
    )
    
    # Train model
    print(f"🚀 Starting training for {num_epochs} epochs...")
    training_start = time.time()
    
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        learning_rate=learning_rate
    )
    
    training_time = time.time() - training_start
    
    # Evaluate on test set
    print("🧪 Evaluating on test set...")
    test_metrics = trainer.evaluate(test_loader)
    
    # Calculate detailed metrics
    y_true, y_pred, y_prob = trainer.predict(test_loader)
    detailed_metrics = calculate_metrics(y_true, y_pred, y_prob)
    
    # Compile results
    results = {
        'model_type': 'fMRI_Transformer',
        'num_subjects': len(subject_ids),
        'data_shape': X.shape,
        'training_time': training_time,
        'data_load_time': data_load_time,
        'best_epoch': history['best_epoch'],
        'best_accuracy': history['best_val_accuracy'],
        'best_f1': history['best_val_f1'],
        'final_loss': history['val_loss'][-1],
        'test_accuracy': test_metrics['accuracy'],
        'test_f1': test_metrics['f1'],
        'test_auc': test_metrics['auc'],
        'detailed_metrics': detailed_metrics,
        'training_history': history,
        'hyperparameters': {
            'num_epochs': num_epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'patience': patience,
            'random_seed': random_seed,
            'device': device
        },
        'matched_subjects_used': matched_subject_ids is not None
    }
    
    print(f"✅ fMRI training complete!")
    print(f"📊 Best Validation Accuracy: {history['best_val_accuracy']:.3f}")
    print(f"🧪 Test Accuracy: {test_metrics['accuracy']:.3f}")
    print(f"⏱️ Training Time: {training_time/60:.1f} minutes")
    
    return results


if __name__ == "__main__":
    """Command line interface for fMRI training."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train fMRI model for ABIDE classification')
    parser.add_argument('--fmri-data', required=True, help='Path to fMRI data directory')
    parser.add_argument('--phenotypic', required=True, help='Path to phenotypic CSV file')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', choices=['cuda', 'cpu'], help='Device to use')
    
    args = parser.parse_args()
    
    results = run_fmri_training(
        fmri_data_path=args.fmri_data,
        phenotypic_file=args.phenotypic,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        patience=args.patience,
        random_seed=args.seed,
        device=args.device
    )
    
    print("\n📊 Final Results:")
    for key, value in results.items():
        if key not in ['training_history', 'detailed_metrics']:
            print(f"{key}: {value}") 
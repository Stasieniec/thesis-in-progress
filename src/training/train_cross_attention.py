"""
Cross-attention training module for ABIDE experiments.
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
from data.smri_processor import SMRIDataProcessor
from models.cross_attention import CrossAttentionTransformer
from training.trainer import Trainer
from training.utils import set_seed, create_multimodal_data_loaders
from evaluation.metrics import calculate_metrics


def run_cross_attention_training(
    fmri_data_path: str,
    smri_data_path: str,
    phenotypic_file: str,
    matched_subject_ids: Optional[Set[str]] = None,
    num_epochs: int = 75,
    batch_size: int = 32,
    learning_rate: float = 0.0005,
    patience: int = 12,
    random_seed: int = 42,
    device: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run cross-attention training experiment.
    
    Args:
        fmri_data_path: Path to fMRI data directory
        smri_data_path: Path to sMRI data directory
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
    
    # Load and process multimodal data
    print("📊 Loading multimodal data...")
    start_time = time.time()
    
    # Initialize processors
    fmri_processor = FMRIDataProcessor(
        data_path=Path(fmri_data_path),
        pheno_file=Path(phenotypic_file),
        sequence_length=200,  # CC200 atlas
        augment_data=True
    )
    
    smri_processor = SMRIDataProcessor(
        data_path=Path(smri_data_path),
        feature_selection_k=800,  # Use improved feature selection
        scaler_type='robust'
    )
    
    # Load both modalities
    fmri_data = fmri_processor.load_all_subjects()
    smri_data = smri_processor.load_all_subjects(Path(phenotypic_file))
    
    # Find subjects with both modalities available
    if matched_subject_ids:
        # Use provided matched subjects
        common_subjects = set(fmri_data.keys()) & set(smri_data.keys()) & matched_subject_ids
        print(f"🎯 Using {len(common_subjects)} matched subjects (from provided set)")
    else:
        # Find all subjects with both modalities
        common_subjects = set(fmri_data.keys()) & set(smri_data.keys())
        print(f"📊 Using all {len(common_subjects)} subjects with both modalities")
    
    # Extract matched data
    fmri_X = []
    smri_X = []
    y = []
    subject_ids = []
    
    for sub_id in common_subjects:
        fmri_X.append(fmri_data[sub_id]['features'])
        smri_X.append(smri_data[sub_id]['features'])
        y.append(fmri_data[sub_id]['label'])  # Should be same for both modalities
        subject_ids.append(sub_id)
    
    fmri_X = np.array(fmri_X)
    smri_X = np.array(smri_X)
    y = np.array(y)
    
    data_load_time = time.time() - start_time
    print(f"✅ Data loaded in {data_load_time:.1f}s")
    print(f"📊 fMRI Shape: {fmri_X.shape}, sMRI Shape: {smri_X.shape}")
    print(f"🧠 ASD: {np.sum(y)}, Control: {len(y) - np.sum(y)}")
    
    # Create multimodal data loaders
    train_loader, val_loader, test_loader = create_multimodal_data_loaders(
        fmri_X=fmri_X,
        smri_X=smri_X,
        y=y,
        batch_size=batch_size,
        random_seed=random_seed
    )
    
    # Initialize model
    print("🏗️ Initializing Cross-Attention model...")
    model = CrossAttentionTransformer(
        fmri_num_features=fmri_X.shape[2],     # Time points
        smri_num_features=smri_X.shape[1],     # 800 features
        num_classes=2,                         # Binary classification
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
        'model_type': 'Cross_Attention',
        'num_subjects': len(subject_ids),
        'fmri_shape': fmri_X.shape,
        'smri_shape': smri_X.shape,
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
    
    print(f"✅ Cross-Attention training complete!")
    print(f"📊 Best Validation Accuracy: {history['best_val_accuracy']:.3f}")
    print(f"🧪 Test Accuracy: {test_metrics['accuracy']:.3f}")
    print(f"⏱️ Training Time: {training_time/60:.1f} minutes")
    
    return results


if __name__ == "__main__":
    """Command line interface for cross-attention training."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train cross-attention model for ABIDE classification')
    parser.add_argument('--fmri-data', required=True, help='Path to fMRI data directory')
    parser.add_argument('--smri-data', required=True, help='Path to sMRI data directory')
    parser.add_argument('--phenotypic', required=True, help='Path to phenotypic CSV file')
    parser.add_argument('--epochs', type=int, default=75, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0005, help='Learning rate')
    parser.add_argument('--patience', type=int, default=12, help='Early stopping patience')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', choices=['cuda', 'cpu'], help='Device to use')
    
    args = parser.parse_args()
    
    results = run_cross_attention_training(
        fmri_data_path=args.fmri_data,
        smri_data_path=args.smri_data,
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
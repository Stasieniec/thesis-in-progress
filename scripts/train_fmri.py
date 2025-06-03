#!/usr/bin/env python3
"""
fMRI-only training script for ABIDE autism classification.

Usage examples:
  python scripts/train_fmri.py run
  python scripts/train_fmri.py run --num_folds=10 --batch_size=128
  python scripts/train_fmri.py run --output_dir="/path/to/output"
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# Now import everything
import fire
import numpy as np
from config import get_config
from data import FMRIDataProcessor
from models import SingleAtlasTransformer
from utils import run_cross_validation
from evaluation import create_cv_visualizations, save_results


class FMRIExperiment:
    """fMRI-only experiment using Single Atlas Transformer."""
    
    def run(
        self,
        num_folds: int = 5,
        batch_size: int = 256,
        learning_rate: float = 1e-4,
        num_epochs: int = 750,
        d_model: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        output_dir: str = None,
        seed: int = 42,
        device: str = 'auto',
        verbose: bool = True
    ):
        """
        Run fMRI-only experiment with cross-validation.
        
        Args:
            num_folds: Number of cross-validation folds
            batch_size: Training batch size
            learning_rate: Learning rate for optimizer
            num_epochs: Maximum number of training epochs
            d_model: Model embedding dimension
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            dropout: Dropout probability
            output_dir: Output directory (auto-generated if None)
            seed: Random seed for reproducibility
            device: Device to use ('auto', 'cuda', 'cpu')
            verbose: Whether to print detailed progress
        """
        # Get configuration
        config = get_config(
            'fmri',
            num_folds=num_folds,
            batch_size=batch_size,
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            output_dir=Path(output_dir) if output_dir else None,
            seed=seed,
            device=device
        )
        
        if verbose:
            print("🧠 Starting fMRI-only experiment...")
            print(f"📁 Output directory: {config.output_dir}")
            print(f"🔧 Configuration: {config.num_folds}-fold CV, batch={config.batch_size}, lr={config.learning_rate}")
        
        # Load and process fMRI data
        if verbose:
            print("\n📊 Loading fMRI data...")
        
        processor = FMRIDataProcessor(
            roi_dir=config.fmri_roi_dir,
            pheno_file=config.phenotypic_file,
            n_rois=config.n_rois
        )
        
        fc_matrices, labels, subject_ids, skipped_ids = processor.process_all_subjects(verbose=verbose)
        
        if verbose:
            print(f"✅ Loaded {len(fc_matrices)} subjects")
            print(f"📊 Feature dimension: {fc_matrices.shape[1]}")
            print(f"📊 Class distribution: ASD={np.sum(labels)}, Control={len(labels)-np.sum(labels)}")
        
        # Run cross-validation
        if verbose:
            print(f"\n🔄 Starting {config.num_folds}-fold cross-validation...")
        
        cv_results = run_cross_validation(
            features=fc_matrices,
            labels=labels,
            model_class=SingleAtlasTransformer,
            config=config,
            experiment_type='single',
            verbose=verbose
        )
        
        # Generate visualizations and save results
        experiment_name = "fmri_sat"
        
        if verbose:
            print(f"\n📊 Creating visualizations...")
        
        create_cv_visualizations(cv_results, config.output_dir, experiment_name)
        save_results(cv_results, config, config.output_dir, experiment_name)
        
        # Print final summary
        if verbose:
            cv_metrics = {
                'accuracy': [r['test_accuracy'] for r in cv_results],
                'balanced_accuracy': [r['test_balanced_accuracy'] for r in cv_results],
                'auc': [r['test_auc'] for r in cv_results]
            }
            
            print(f"\n🎯 FINAL RESULTS:")
            for metric, values in cv_metrics.items():
                mean_val = np.mean(values)
                std_val = np.std(values)
                print(f"   {metric.upper()}: {mean_val:.4f} ± {std_val:.4f}")
        
        return cv_results

    def quick_test(self, num_folds: int = 2, num_epochs: int = 5, output_dir: str = "./test_fmri_output"):
        """
        Quick test run with minimal epochs for debugging.
        
        Args:
            num_folds: Number of folds for quick test
            num_epochs: Number of epochs for quick test
            output_dir: Output directory for test results
        """
        print("🧪 Running quick test...")
        return self.run(
            num_folds=num_folds,
            num_epochs=num_epochs,
            batch_size=32,
            output_dir=output_dir,
            verbose=True
        )

    def get_config_template(self, output_dir: str = "./test_output"):
        """Print template configuration for reference."""
        config = get_config('fmri', output_dir=Path(output_dir))
        print("📋 fMRI Configuration Template:")
        print("-" * 40)
        for key, value in config.__dict__.items():
            print(f"{key}: {value}")
        return config


if __name__ == '__main__':
    fire.Fire(FMRIExperiment) 
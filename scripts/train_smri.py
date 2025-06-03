#!/usr/bin/env python3
"""
sMRI-only training script for ABIDE autism classification.

Usage examples:
  python scripts/train_smri.py run
  python scripts/train_smri.py run --num_folds=10 --feature_selection_k=500
  python scripts/train_smri.py run --output_dir="/path/to/output"
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import fire
import numpy as np

from src.config import get_config
from src.data import SMRIDataProcessor
from src.models import SMRITransformer
from src.utils import run_cross_validation
from src.evaluation import create_cv_visualizations, save_results


class SMRIExperiment:
    """sMRI-only experiment using transformer model."""
    
    def run(
        self,
        num_folds: int = 5,
        batch_size: int = 16,
        learning_rate: float = 1e-3,
        num_epochs: int = 200,
        d_model: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.3,
        layer_dropout: float = 0.1,
        feature_selection_k: int = 300,
        scaler_type: str = 'robust',
        output_dir: str = None,
        seed: int = 42,
        device: str = 'auto',
        verbose: bool = True
    ):
        """
        Run sMRI-only experiment with cross-validation.
        
        Args:
            num_folds: Number of cross-validation folds
            batch_size: Training batch size
            learning_rate: Learning rate for optimizer
            num_epochs: Maximum number of training epochs
            d_model: Model embedding dimension
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            dropout: Dropout probability
            layer_dropout: Layer dropout probability
            feature_selection_k: Number of features to select
            scaler_type: Type of feature scaler ('robust' or 'standard')
            output_dir: Output directory (auto-generated if None)
            seed: Random seed for reproducibility
            device: Device to use ('auto', 'cuda', 'cpu')
            verbose: Whether to print detailed progress
        """
        # Get configuration
        config = get_config(
            'smri',
            num_folds=num_folds,
            batch_size=batch_size,
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            layer_dropout=layer_dropout,
            feature_selection_k=feature_selection_k,
            scaler_type=scaler_type,
            output_dir=Path(output_dir) if output_dir else None,
            seed=seed,
            device=device
        )
        
        if verbose:
            print("🧠 Starting sMRI-only experiment...")
            print(f"📁 Output directory: {config.output_dir}")
            print(f"🔧 Configuration: {config.num_folds}-fold CV, batch={config.batch_size}, lr={config.learning_rate}")
            print(f"🔧 Feature selection: {config.feature_selection_k} features, scaler={config.scaler_type}")
        
        # Load and process sMRI data
        if verbose:
            print("\n📊 Loading sMRI data...")
        
        processor = SMRIDataProcessor(
            data_path=config.smri_data_path,
            feature_selection_k=config.feature_selection_k,
            scaler_type=config.scaler_type
        )
        
        features, labels, subject_ids = processor.process_all_subjects(
            phenotypic_file=config.phenotypic_file,
            verbose=verbose
        )
        
        if verbose:
            print(f"✅ Loaded {len(features)} subjects")
            print(f"📊 Original feature dimension: {features.shape[1]}")
            print(f"📊 Class distribution: ASD={np.sum(labels)}, Control={len(labels)-np.sum(labels)}")
        
        # Analyze feature importance if verbose
        if verbose:
            print("\n🔍 Analyzing feature importance...")
            feature_importance = processor.analyze_features(features, labels, top_k=20)
        
        # Run cross-validation
        if verbose:
            print(f"\n🔄 Starting {config.num_folds}-fold cross-validation...")
        
        cv_results = run_cross_validation(
            features=features,
            labels=labels,
            model_class=SMRITransformer,
            config=config,
            experiment_type='single',
            verbose=verbose
        )
        
        # Generate visualizations and save results
        experiment_name = "smri_transformer"
        
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

    def quick_test(self, num_folds: int = 2, num_epochs: int = 5):
        """
        Quick test run with minimal epochs for debugging.
        
        Args:
            num_folds: Number of folds for quick test
            num_epochs: Number of epochs for quick test
        """
        print("🧪 Running quick test...")
        return self.run(
            num_folds=num_folds,
            num_epochs=num_epochs,
            batch_size=8,
            feature_selection_k=100,
            verbose=True
        )

    def analyze_features_only(self, top_k: int = 50):
        """
        Only analyze feature importance without training.
        
        Args:
            top_k: Number of top features to analyze
        """
        print("🔍 Analyzing sMRI feature importance...")
        
        config = get_config('smri')
        processor = SMRIDataProcessor(
            data_path=config.smri_data_path,
            feature_selection_k=None,  # Don't select features for analysis
            scaler_type='robust'
        )
        
        features, labels, subject_ids = processor.process_all_subjects(
            phenotypic_file=config.phenotypic_file,
            verbose=True
        )
        
        feature_importance = processor.analyze_features(features, labels, top_k=top_k)
        
        print(f"\n📊 Top {top_k} most important features:")
        print(feature_importance.head(top_k)[['feature_name', 'f_score', 'f_pval']].to_string(index=False))
        
        return feature_importance

    def get_config_template(self):
        """Print template configuration for reference."""
        config = get_config('smri')
        print("📋 sMRI Configuration Template:")
        print("-" * 40)
        for key, value in config.__dict__.items():
            print(f"{key}: {value}")
        return config


if __name__ == '__main__':
    fire.Fire(SMRIExperiment) 
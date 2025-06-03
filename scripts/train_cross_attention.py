#!/usr/bin/env python3
"""
Cross-attention training script for multimodal fMRI-sMRI autism classification.

Usage examples:
  python scripts/train_cross_attention.py run
  python scripts/train_cross_attention.py run --num_folds=5 --batch_size=32
  python scripts/train_cross_attention.py run --output_dir="/path/to/output"
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# Now import everything
import fire
import numpy as np
from config import get_config
from data import FMRIDataProcessor, SMRIDataProcessor, match_multimodal_subjects
from models import CrossAttentionTransformer
from utils import run_cross_validation
from evaluation import create_cv_visualizations, save_results


class CrossAttentionExperiment:
    """Cross-attention experiment for multimodal fMRI-sMRI classification."""
    
    def run(
        self,
        num_folds: int = 5,
        batch_size: int = 32,
        learning_rate: float = 5e-5,
        num_epochs: int = 300,
        d_model: int = 256,
        num_layers: int = 4,
        num_cross_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.2,
        smri_feat_selection: int = 300,
        output_dir: str = None,
        seed: int = 42,
        device: str = 'auto',
        verbose: bool = True
    ):
        """
        Run cross-attention experiment with multimodal data.
        
        Args:
            num_folds: Number of cross-validation folds
            batch_size: Training batch size
            learning_rate: Learning rate for optimizer
            num_epochs: Maximum number of training epochs
            d_model: Model embedding dimension
            num_layers: Number of transformer layers per modality
            num_cross_layers: Number of cross-attention layers
            num_heads: Number of attention heads
            dropout: Dropout probability
            smri_feat_selection: Number of sMRI features to select
            output_dir: Output directory (auto-generated if None)
            seed: Random seed for reproducibility
            device: Device to use ('auto', 'cuda', 'cpu')
            verbose: Whether to print detailed progress
        """
        # Get configuration
        config = get_config(
            'cross_attention',
            num_folds=num_folds,
            batch_size=batch_size,
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            d_model=d_model,
            num_layers=num_layers,
            num_cross_layers=num_cross_layers,
            num_heads=num_heads,
            dropout=dropout,
            smri_feat_selection=smri_feat_selection,
            output_dir=Path(output_dir) if output_dir else None,
            seed=seed,
            device=device
        )
        
        if verbose:
            print("🧠 Starting cross-attention multimodal experiment...")
            print(f"📁 Output directory: {config.output_dir}")
            print(f"🔧 Configuration: {config.num_folds}-fold CV, batch={config.batch_size}, lr={config.learning_rate}")
            print(f"🔧 Architecture: d_model={config.d_model}, cross_layers={config.num_cross_layers}")
        
        # Load fMRI data
        if verbose:
            print("\n📊 Loading fMRI data...")
        
        fmri_processor = FMRIDataProcessor(
            roi_dir=config.fmri_roi_dir,
            pheno_file=config.phenotypic_file,
            n_rois=config.n_rois
        )
        
        fmri_data = fmri_processor.load_all_subjects()
        
        if verbose:
            print(f"✅ Loaded {len(fmri_data)} fMRI subjects")
        
        # Load sMRI data
        if verbose:
            print("📊 Loading sMRI data...")
        
        smri_processor = SMRIDataProcessor(
            data_path=config.smri_data_path,
            feature_selection_k=None,  # Will be handled in preprocessing
            scaler_type='robust'
        )
        
        smri_data = smri_processor.load_all_subjects(config.phenotypic_file)
        
        if verbose:
            print(f"✅ Loaded {len(smri_data)} sMRI subjects")
        
        # Match subjects between modalities
        if verbose:
            print("\n🔗 Matching subjects between modalities...")
        
        matched_fmri_features, matched_smri_features, matched_labels, matched_subject_ids = match_multimodal_subjects(
            fmri_data, smri_data, verbose=verbose
        )
        
        if len(matched_labels) == 0:
            raise ValueError("No subjects with matching labels found between modalities!")
        
        if verbose:
            print(f"✅ Matched {len(matched_labels)} subjects")
            print(f"📊 fMRI feature dim: {matched_fmri_features.shape[1]}")
            print(f"📊 sMRI feature dim: {matched_smri_features.shape[1]}")
            print(f"📊 Class distribution: ASD={np.sum(matched_labels)}, Control={len(matched_labels)-np.sum(matched_labels)}")
        
        # Run cross-validation
        if verbose:
            print(f"\n🔄 Starting {config.num_folds}-fold cross-validation...")
        
        cv_results = run_cross_validation(
            features=None,  # Not used for multimodal
            labels=matched_labels,
            model_class=CrossAttentionTransformer,
            config=config,
            experiment_type='multimodal',
            fmri_features=matched_fmri_features,
            smri_features=matched_smri_features,
            verbose=verbose
        )
        
        # Generate visualizations and save results
        experiment_name = "cross_attention_multimodal"
        
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
            batch_size=16,
            smri_feat_selection=100,
            verbose=True
        )

    def analyze_data_overlap(self):
        """
        Analyze the overlap between fMRI and sMRI subjects.
        """
        print("🔍 Analyzing data overlap between modalities...")
        
        config = get_config('cross_attention')
        
        # Load fMRI data
        fmri_processor = FMRIDataProcessor(
            roi_dir=config.fmri_roi_dir,
            pheno_file=config.phenotypic_file,
            n_rois=config.n_rois
        )
        fmri_data = fmri_processor.load_all_subjects()
        
        # Load sMRI data
        smri_processor = SMRIDataProcessor(
            data_path=config.smri_data_path,
            feature_selection_k=None,
            scaler_type='robust'
        )
        smri_data = smri_processor.load_all_subjects(config.phenotypic_file)
        
        # Analyze overlap
        fmri_subjects = set(fmri_data.keys())
        smri_subjects = set(smri_data.keys())
        
        print(f"\n📊 Data Overlap Analysis:")
        print(f"   fMRI subjects: {len(fmri_subjects)}")
        print(f"   sMRI subjects: {len(smri_subjects)}")
        print(f"   Common subjects: {len(fmri_subjects & smri_subjects)}")
        print(f"   fMRI only: {len(fmri_subjects - smri_subjects)}")
        print(f"   sMRI only: {len(smri_subjects - fmri_subjects)}")
        
        # Check label consistency
        common_subjects = fmri_subjects & smri_subjects
        label_matches = 0
        label_mismatches = 0
        
        for sub_id in common_subjects:
            if fmri_data[sub_id]['label'] == smri_data[sub_id]['label']:
                label_matches += 1
            else:
                label_mismatches += 1
        
        print(f"\n🏷️ Label Consistency:")
        print(f"   Matching labels: {label_matches}")
        print(f"   Mismatched labels: {label_mismatches}")
        
        return {
            'fmri_subjects': len(fmri_subjects),
            'smri_subjects': len(smri_subjects),
            'common_subjects': len(common_subjects),
            'label_matches': label_matches,
            'label_mismatches': label_mismatches
        }

    def get_config_template(self, output_dir: str = "./test_output"):
        """Print template configuration for reference."""
        config = get_config('cross_attention', output_dir=Path(output_dir))
        print("📋 Cross-Attention Configuration Template:")
        print("-" * 40)
        for key, value in config.__dict__.items():
            print(f"{key}: {value}")
        return config


if __name__ == '__main__':
    fire.Fire(CrossAttentionExperiment) 
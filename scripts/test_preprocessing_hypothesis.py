#!/usr/bin/env python3
"""
🔍 Test Preprocessing Mismatch Hypothesis
Test if original sMRI preprocessing recovers the 63.6% cross-attention performance

Usage examples:
  python scripts/test_preprocessing_hypothesis.py run              # Full test
  python scripts/test_preprocessing_hypothesis.py quick_test       # Quick test
  
Google Colab usage:
  !python scripts/test_preprocessing_hypothesis.py run
  !python scripts/test_preprocessing_hypothesis.py quick_test

🎯 HYPOTHESIS: Enhanced sMRI preprocessing breaks cross-attention compatibility
Expected: Recovery to ~63.6% if preprocessing mismatch was the issue
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import fire
import numpy as np
from config import get_config
from data import FMRIDataProcessor, SMRIDataProcessor, match_multimodal_subjects
from models import CrossAttentionTransformer
from utils import run_cross_validation
from evaluation import save_results
import json
from datetime import datetime


class PreprocessingHypothesisTest:
    """🔍 Test if original sMRI preprocessing recovers cross-attention performance.
    
    HYPOTHESIS: Enhanced sMRI preprocessing (RobustScaler + advanced features) 
    creates distribution shift that breaks cross-attention compatibility.
    
    TEST: Use original preprocessing (StandardScaler + simple f_classif)
    EXPECTED: Recovery to ~63.6% if hypothesis is correct
    """
    
    def run(
        self,
        num_folds: int = 3,
        batch_size: int = 32,
        learning_rate: float = 1e-4,
        num_epochs: int = 30,
        d_model: int = 64,
        num_layers: int = 2,
        num_cross_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        smri_feat_selection: int = 300,
        output_dir: str = None,
        seed: int = 42,
        device: str = 'auto',
        verbose: bool = True
    ):
        """
        Test original preprocessing hypothesis with cross-attention.
        
        🎯 USES ORIGINAL PREPROCESSING:
        - StandardScaler (not RobustScaler)
        - Simple f_classif feature selection
        - Original CrossAttentionTransformer architecture
        - Original hyperparameters
        
        Expected: Recovery to ~63.6% if preprocessing mismatch was the issue
        """
        # Get configuration with original settings
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
            print("🔍 TESTING: Original Preprocessing Hypothesis")
            print("=" * 60)
            print("🎯 Hypothesis: Enhanced sMRI preprocessing breaks cross-attention")
            print("🧪 Test: Use ORIGINAL preprocessing (StandardScaler + f_classif)")
            print("📊 Expected: Recovery to ~63.6% if hypothesis correct")
            print(f"📁 Output directory: {config.output_dir}")
            print()
        
        # Load fMRI data (standard)
        if verbose:
            print("📊 Loading fMRI data...")
        
        fmri_processor = FMRIDataProcessor(
            roi_dir=config.fmri_roi_dir,
            pheno_file=config.phenotypic_file,
            n_rois=config.n_rois
        )
        
        fmri_data = fmri_processor.load_all_subjects()
        
        if verbose:
            print(f"✅ Loaded {len(fmri_data)} fMRI subjects")
        
        # Load sMRI data with ORIGINAL preprocessing
        if verbose:
            print("📊 Loading sMRI data with ORIGINAL preprocessing...")
            print("   🔧 Using: StandardScaler + simple f_classif (NOT enhanced)")
        
        smri_processor = SMRIDataProcessor(
            data_path=config.smri_data_path,
            feature_selection_k=config.smri_feat_selection,
            scaler_type='standard'  # ORIGINAL: StandardScaler (not robust)
        )
        
        smri_data = smri_processor.load_all_subjects(config.phenotypic_file)
        
        if verbose:
            print(f"✅ Loaded {len(smri_data)} sMRI subjects")
            print("   ✓ Used StandardScaler (original)")
            print("   ✓ Used simple f_classif selection (original)")
        
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
        
        # Run cross-validation with ORIGINAL model
        if verbose:
            print(f"\n🔄 Starting {config.num_folds}-fold cross-validation with ORIGINAL model...")
            print("   🔧 Original CrossAttentionTransformer (no improvements)")
            print("   🔧 Original hyperparameters")
        
        cv_results = run_cross_validation(
            features=None,
            labels=matched_labels,
            model_class=CrossAttentionTransformer,
            config=config,
            experiment_type='multimodal',
            fmri_features=matched_fmri_features,
            smri_features=matched_smri_features,
            verbose=verbose
        )
        
        # Analyze results and diagnose
        if verbose:
            print(f"\n📊 Analyzing results and diagnosing...")
        
        cv_metrics = {
            'accuracy': [r['test_accuracy'] for r in cv_results],
            'balanced_accuracy': [r['test_balanced_accuracy'] for r in cv_results],
            'auc': [r['test_auc'] for r in cv_results]
        }
        
        mean_acc = np.mean(cv_metrics['accuracy'])
        mean_bal_acc = np.mean(cv_metrics['balanced_accuracy'])
        mean_auc = np.mean(cv_metrics['auc'])
        
        # Diagnosis
        print("\n" + "="*60)
        print("🎯 ORIGINAL PREPROCESSING TEST RESULTS")
        print("-" * 40)
        
        for metric, values in cv_metrics.items():
            mean_val = np.mean(values)
            std_val = np.std(values)
            print(f"{metric.upper()}:")
            print(f"  Mean ± Std: {mean_val:.4f} ± {std_val:.4f}")
            print(f"  Range: [{np.min(values):.4f}, {np.max(values):.4f}]")
            print()
        
        print("🔍 DIAGNOSIS:")
        print("-" * 15)
        
        if mean_acc >= 0.62:  # Close to original 63.6%
            diagnosis = "preprocessing_mismatch_confirmed"
            print("   ✅ PREPROCESSING MISMATCH CONFIRMED!")
            print("   → Original preprocessing RECOVERS performance")
            print("   → Enhanced preprocessing caused distribution shift")
            print("   → Root cause: RobustScaler + advanced features incompatible")
            print()
            print("   📋 RECOMMENDED NEXT STEPS:")
            print("   → Run gradual enhancement script")
            print("   → Start with original preprocessing (current result)")
            print("   → Add enhancements ONE BY ONE")
            print("   → Goal: Beat 65% fMRI baseline incrementally")
            
        elif mean_acc >= 0.58:  # Partial recovery
            diagnosis = "partial_preprocessing_impact"
            print("   ⚠️  PARTIAL RECOVERY")
            print("   → Some preprocessing impact detected")
            print("   → Additional factors involved (hyperparams/training)")
            print()
            print("   📋 RECOMMENDED NEXT STEPS:")
            print("   → Test both original AND enhanced preprocessing")
            print("   → Run hyperparameter search script")
            print("   → Mixed approach: original base + selective enhancements")
            
        else:  # No recovery
            diagnosis = "not_preprocessing_issue"
            print("   ❌ NO RECOVERY - NOT PREPROCESSING ISSUE")
            print("   → Problem is NOT primarily preprocessing")
            print("   → Likely hyperparameter/training/architecture issue")
            print()
            print("   📋 RECOMMENDED NEXT STEPS:")
            print("   → Run hyperparameter search script")
            print("   → Focus on learning rate, batch size, architecture")
            print("   → Consider training procedure changes")
        
        print()
        print("🎯 PERFORMANCE COMPARISON:")
        print(f"   Original Cross-Attention: 63.6%")
        print(f"   Enhanced Cross-Attention: 57.7% (failed)")
        print(f"   This test (original):     {mean_acc:.1%}")
        print(f"   Pure fMRI target:         65.0%")
        print(f"   Gap to beat fMRI:         {65.0 - mean_acc*100:.1f} percentage points")
        
        # Save comprehensive results
        experiment_name = "preprocessing_hypothesis_test"
        save_results(cv_results, config, config.output_dir, experiment_name)
        
        # Save detailed analysis
        analysis_results = {
            'test_name': 'original_preprocessing_hypothesis',
            'timestamp': datetime.now().isoformat(),
            'preprocessing_type': 'original_standardscaler_simple_ftest',
            'model_type': 'original_cross_attention_transformer',
            'hyperparameters': {
                'learning_rate': learning_rate,
                'batch_size': batch_size,
                'epochs': num_epochs,
                'dropout': dropout,
                'hidden_dim': d_model,
                'num_heads': num_heads,
                'num_layers': num_layers
            },
            'results': {
                'mean_accuracy': float(mean_acc),
                'mean_balanced_accuracy': float(mean_bal_acc),
                'mean_auc': float(mean_auc),
                'std_accuracy': float(np.std(cv_metrics['accuracy'])),
                'all_accuracies': [float(x) for x in cv_metrics['accuracy']],
                'all_balanced_accuracies': [float(x) for x in cv_metrics['balanced_accuracy']],
                'all_aucs': [float(x) for x in cv_metrics['auc']]
            },
            'diagnosis': diagnosis,
            'comparison': {
                'original_cross_attention': 63.6,
                'enhanced_cross_attention': 57.7,
                'this_test': float(mean_acc * 100),
                'pure_fmri_target': 65.0
            }
        }
        
        analysis_path = config.output_dir / 'preprocessing_hypothesis_analysis.json'
        with open(analysis_path, 'w') as f:
            json.dump(analysis_results, f, indent=2)
        
        if verbose:
            print(f"\n✅ Test completed!")
            print(f"📁 Detailed analysis saved to: {analysis_path}")
            print(f"🔍 Diagnosis: {diagnosis}")
            
            if diagnosis == "preprocessing_mismatch_confirmed":
                print(f"\n🚀 NEXT STEP: Run gradual enhancement script")
                print(f"   !python scripts/gradual_enhancement.py run")
            elif diagnosis == "partial_preprocessing_impact":
                print(f"\n🚀 NEXT STEPS: Run both enhancement and hyperparameter scripts")
                print(f"   !python scripts/gradual_enhancement.py run")
                print(f"   !python scripts/hyperparameter_search.py run")
            else:
                print(f"\n🚀 NEXT STEP: Run hyperparameter search script")
                print(f"   !python scripts/hyperparameter_search.py run")
        
        return cv_results, diagnosis

    def quick_test(self, num_folds: int = 2, num_epochs: int = 5, output_dir: str = "./test_preprocessing_output"):
        """
        🧪 Quick test of preprocessing hypothesis.
        """
        print("🧪 Running preprocessing hypothesis quick test...")
        print("🎯 Testing if original preprocessing recovers 63.6% performance")
        return self.run(
            num_folds=num_folds,
            num_epochs=num_epochs,
            batch_size=16,
            smri_feat_selection=200,
            output_dir=output_dir,
            verbose=True
        )


if __name__ == '__main__':
    fire.Fire(PreprocessingHypothesisTest) 
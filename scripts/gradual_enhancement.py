#!/usr/bin/env python3
"""
🔄 Gradual Enhancement Introduction for Cross-Attention
If preprocessing mismatch is confirmed, gradually introduce enhancements
one by one to find optimal balance between improvements and compatibility

Usage examples:
  python scripts/gradual_enhancement.py run                       # Full gradual test
  python scripts/gradual_enhancement.py quick_test                # Quick test
  
Google Colab usage:
  !python scripts/gradual_enhancement.py run
  !python scripts/gradual_enhancement.py quick_test

🎯 STRATEGY: Start from working baseline, add enhancements incrementally
Run this if preprocessing mismatch test shows RECOVERY to ~63.6%
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import fire
import numpy as np
from config import get_config
from data import FMRIDataProcessor, SMRIDataProcessor, match_multimodal_subjects
from models import CrossAttentionTransformer, MinimalImprovedCrossAttentionTransformer
from utils import run_cross_validation
from evaluation import save_results
import json
from datetime import datetime


class GradualEnhancement:
    """🔄 Gradually introduce enhancements from working baseline.
    
    STRATEGY: Start with original preprocessing (63.6% performance)
    Add enhancements ONE BY ONE to find optimal combination
    
    Enhancement Levels:
    0: Original baseline (StandardScaler + f_classif)
    1: RobustScaler only
    2: RobustScaler + improved F-score selection
    3: RobustScaler + F-score + MI combined
    4: Full enhanced preprocessing
    """
    
    def run(
        self,
        num_folds: int = 3,
        num_epochs: int = 30,
        output_dir: str = None,
        seed: int = 42,
        device: str = 'auto',
        verbose: bool = True
    ):
        """
        Run gradual enhancement introduction.
        
        🎯 GRADUAL ENHANCEMENT LEVELS:
        Level 0: Original baseline (StandardScaler + f_classif)
        Level 1: RobustScaler only
        Level 2: RobustScaler + improved F-score
        Level 3: RobustScaler + F-score + MI
        Level 4: Full enhanced preprocessing
        """
        
        # Get base configuration
        config = get_config(
            'cross_attention',
            num_folds=num_folds,
            num_epochs=num_epochs,
            output_dir=Path(output_dir) if output_dir else None,
            seed=seed,
            device=device
        )
        
        if verbose:
            print("🔄 GRADUAL ENHANCEMENT INTRODUCTION")
            print("=" * 60)
            print("🎯 Goal: Incrementally improve from original preprocessing to beat 65%")
            print("📊 Starting from working baseline, adding enhancements one by one")
            print(f"📁 Output directory: {config.output_dir}")
            print()
        
        # Define enhancement levels
        enhancement_descriptions = {
            0: "Original baseline (StandardScaler + f_classif)",
            1: "Level 1: RobustScaler only",
            2: "Level 2: RobustScaler + improved F-score",
            3: "Level 3: RobustScaler + F-score + MI",
            4: "Level 4: Full enhanced preprocessing"
        }
        
        results = []
        
        if verbose:
            print("🚀 Testing gradual enhancement introduction...")
            print("Strategy: Start from working baseline, add enhancements incrementally")
            print()
        
        # Test each enhancement level
        for enhancement_level in range(5):
            if verbose:
                print(f"\n📋 Enhancement Level {enhancement_level}: {enhancement_descriptions[enhancement_level]}")
            
            # Load data for this enhancement level
            fmri_data, smri_data, matched_labels, matched_ids = self._load_data_for_level(
                enhancement_level, config, verbose
            )
            
            # Test with original model
            if verbose:
                print("   🔸 Testing with original cross-attention model...")
            
            cv_results_orig = run_cross_validation(
                features=None,
                labels=matched_labels,
                model_class=CrossAttentionTransformer,
                config=config,
                experiment_type='multimodal',
                fmri_features=fmri_data,
                smri_features=smri_data,
                verbose=False
            )
            
            mean_acc_orig = np.mean([r['test_accuracy'] for r in cv_results_orig])
            mean_bal_acc_orig = np.mean([r['test_balanced_accuracy'] for r in cv_results_orig])
            mean_auc_orig = np.mean([r['test_auc'] for r in cv_results_orig])
            
            result_orig = {
                'enhancement_level': enhancement_level,
                'description': enhancement_descriptions[enhancement_level],
                'model_type': 'original',
                'mean_accuracy': mean_acc_orig,
                'mean_balanced_accuracy': mean_bal_acc_orig,
                'mean_auc': mean_auc_orig,
                'beats_fmri_baseline': mean_acc_orig > 0.65
            }
            results.append(result_orig)
            
            if verbose:
                print(f"      Original model: Acc={mean_acc_orig:.4f}, Bal_Acc={mean_bal_acc_orig:.4f}, AUC={mean_auc_orig:.4f}")
                if mean_acc_orig > 0.65:
                    print(f"      🎯 BEATS fMRI BASELINE!")
            
            # Test with minimal improved model (if level >= 1)
            if enhancement_level >= 1:
                if verbose:
                    print("   🔸 Testing with minimal improved model...")
                
                cv_results_improved = run_cross_validation(
                    features=None,
                    labels=matched_labels,
                    model_class=MinimalImprovedCrossAttentionTransformer,
                    config=config,
                    experiment_type='multimodal',
                    fmri_features=fmri_data,
                    smri_features=smri_data,
                    verbose=False
                )
                
                mean_acc_improved = np.mean([r['test_accuracy'] for r in cv_results_improved])
                mean_bal_acc_improved = np.mean([r['test_balanced_accuracy'] for r in cv_results_improved])
                mean_auc_improved = np.mean([r['test_auc'] for r in cv_results_improved])
                
                result_improved = {
                    'enhancement_level': enhancement_level,
                    'description': enhancement_descriptions[enhancement_level],
                    'model_type': 'minimal_improved',
                    'mean_accuracy': mean_acc_improved,
                    'mean_balanced_accuracy': mean_bal_acc_improved,
                    'mean_auc': mean_auc_improved,
                    'beats_fmri_baseline': mean_acc_improved > 0.65
                }
                results.append(result_improved)
                
                if verbose:
                    print(f"      Improved model: Acc={mean_acc_improved:.4f}, Bal_Acc={mean_bal_acc_improved:.4f}, AUC={mean_auc_improved:.4f}")
                    if mean_acc_improved > 0.65:
                        print(f"      🎯 BEATS fMRI BASELINE!")
                    
                    # Compare models
                    diff = mean_acc_improved - mean_acc_orig
                    if diff > 0.01:
                        print(f"      ✅ Improved model gains {diff*100:.1f} points!")
                    elif diff < -0.01:
                        print(f"      ❌ Improved model loses {abs(diff)*100:.1f} points")
                    else:
                        print(f"      ⚖️  Similar performance (±{abs(diff)*100:.1f} points)")
        
        # Analyze results
        if verbose:
            best_result = self._analyze_gradual_results(results)
        
        # Save results
        self._save_gradual_results(results, config.output_dir, enhancement_descriptions)
        
        return results

    def _load_data_for_level(self, enhancement_level, config, verbose):
        """Load data with specified enhancement level."""
        
        # Load fMRI data (always the same)
        fmri_processor = FMRIDataProcessor(
            roi_dir=config.fmri_roi_dir,
            pheno_file=config.phenotypic_file,
            n_rois=config.n_rois
        )
        fmri_data = fmri_processor.load_all_subjects()
        
        # Load sMRI data with enhancement level
        if enhancement_level == 0:
            # Level 0: Original baseline
            scaler_type = 'standard'
            feature_selection_k = 300
            if verbose:
                print(f"   ✓ Level 0: StandardScaler + f_classif, {feature_selection_k} features")
                
        elif enhancement_level == 1:
            # Level 1: RobustScaler only
            scaler_type = 'robust'
            feature_selection_k = 300
            if verbose:
                print(f"   ✓ Level 1: RobustScaler + f_classif, {feature_selection_k} features")
                
        elif enhancement_level == 2:
            # Level 2: RobustScaler + improved F-score
            scaler_type = 'robust'
            feature_selection_k = 350
            if verbose:
                print(f"   ✓ Level 2: RobustScaler + improved f_classif, {feature_selection_k} features")
                
        elif enhancement_level == 3:
            # Level 3: RobustScaler + F-score + MI
            scaler_type = 'robust'
            feature_selection_k = 300  # Will use combined selection in processor
            if verbose:
                print(f"   ✓ Level 3: RobustScaler + F-score+MI, combined selection")
                
        else:  # Level 4
            # Level 4: Full enhanced
            scaler_type = 'robust'
            feature_selection_k = 350  # Enhanced selection
            if verbose:
                print(f"   ✓ Level 4: Full enhanced preprocessing, {feature_selection_k} features")
        
        # Create sMRI processor
        smri_processor = SMRIDataProcessor(
            data_path=config.smri_data_path,
            feature_selection_k=feature_selection_k,
            scaler_type=scaler_type
        )
        
        smri_data = smri_processor.load_all_subjects(config.phenotypic_file)
        
        # Match subjects
        matched_fmri, matched_smri, matched_labels, matched_ids = match_multimodal_subjects(
            fmri_data, smri_data, verbose=False
        )
        
        return matched_fmri, matched_smri, matched_labels, matched_ids

    def _analyze_gradual_results(self, results):
        """Analyze gradual enhancement results."""
        
        print("\n" + "="*60)
        print("🎯 GRADUAL ENHANCEMENT RESULTS")
        print("-" * 40)
        
        # Find best overall result
        best_result = max(results, key=lambda x: x['mean_accuracy'])
        print(f"📊 BEST OVERALL RESULT:")
        print(f"   Performance: {best_result['mean_accuracy']:.4f} ({best_result['mean_accuracy']:.1%})")
        print(f"   Configuration: {best_result['description']}")
        print(f"   Model: {best_result['model_type']}")
        if best_result['beats_fmri_baseline']:
            print(f"   🎯 BEATS fMRI BASELINE!")
        print()
        
        # Key insights
        print(f"🔍 KEY INSIGHTS:")
        
        # Check if any enhancement improves over baseline
        baseline_result = next(r for r in results if r['enhancement_level'] == 0 and r['model_type'] == 'original')
        baseline_acc = baseline_result['mean_accuracy']
        
        successful_enhancements = [r for r in results if r['mean_accuracy'] > baseline_acc + 0.005]
        
        if successful_enhancements:
            print(f"   ✅ {len(successful_enhancements)} enhancement(s) improve over baseline:")
            for result in sorted(successful_enhancements, key=lambda x: x['mean_accuracy'], reverse=True):
                improvement = (result['mean_accuracy'] - baseline_acc) * 100
                print(f"      Level {result['enhancement_level']} ({result['model_type']}): +{improvement:.1f} points")
        else:
            print(f"   ❌ No enhancements improve significantly over baseline")
            print(f"   → Original preprocessing remains optimal")
        
        # Success in beating fMRI baseline
        successful_configs = [r for r in results if r['beats_fmri_baseline']]
        if successful_configs:
            print(f"   🎯 {len(successful_configs)} configuration(s) beat fMRI baseline!")
            best_success = max(successful_configs, key=lambda x: x['mean_accuracy'])
            print(f"      Best: Level {best_success['enhancement_level']} ({best_success['model_type']}) at {best_success['mean_accuracy']:.1%}")
        else:
            print(f"   ⚠️  No configuration beats 65% fMRI baseline")
            gap = (0.65 - best_result['mean_accuracy']) * 100
            print(f"   → Best result needs {gap:.1f} more percentage points")
        
        return best_result

    def _save_gradual_results(self, results, output_dir, enhancement_descriptions):
        """Save gradual enhancement results."""
        
        # Find best result
        best_result = max(results, key=lambda x: x['mean_accuracy'])
        baseline_result = next(r for r in results if r['enhancement_level'] == 0 and r['model_type'] == 'original')
        
        output_data = {
            'test_name': 'gradual_enhancement_introduction',
            'timestamp': datetime.now().isoformat(),
            'strategy': 'incremental_preprocessing_enhancement',
            'baseline_performance': float(baseline_result['mean_accuracy']),
            'best_performance': float(best_result['mean_accuracy']),
            'best_configuration': {
                'enhancement_level': best_result['enhancement_level'],
                'description': best_result['description'],
                'model_type': best_result['model_type']
            },
            'all_results': results,
            'enhancement_descriptions': enhancement_descriptions,
            'enhancement_levels_tested': 5,
            'model_types_tested': ['original', 'minimal_improved'],
            'target_baseline': 65.0
        }
        
        results_path = output_dir / 'gradual_enhancement_results.json'
        with open(results_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\n✅ Gradual enhancement testing completed!")
        print(f"📁 Results saved to: {results_path}")
        print(f"🎯 Best performance: {best_result['mean_accuracy']:.1%}")
        print(f"📋 Best config: Level {best_result['enhancement_level']} with {best_result['model_type']}")
        
        if best_result['beats_fmri_baseline']:
            print(f"🏆 SUCCESS! Found enhancement that beats fMRI baseline!")
        else:
            gap = (0.65 - best_result['mean_accuracy']) * 100
            print(f"⚠️  Gap to 65% target: {gap:.1f} percentage points")
            print(f"📋 Consider: More sophisticated architectural improvements")

    def quick_test(self, num_folds: int = 2, num_epochs: int = 10, output_dir: str = "./test_gradual_enhancement_output"):
        """
        🧪 Quick gradual enhancement test.
        """
        print("🧪 Running gradual enhancement quick test...")
        print("🔄 Testing incremental preprocessing improvements")
        
        return self.run(
            num_folds=num_folds,
            num_epochs=num_epochs,
            output_dir=output_dir,
            verbose=True
        )


if __name__ == '__main__':
    fire.Fire(GradualEnhancement) 
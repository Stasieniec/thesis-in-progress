#!/usr/bin/env python3
"""
⚙️ Hyperparameter Search for Cross-Attention
Systematic search to find hyperparameters that beat 65% fMRI baseline

Usage examples:
  python scripts/hyperparameter_search.py run                     # Full search
  python scripts/hyperparameter_search.py quick_test              # Quick search
  
Google Colab usage:
  !python scripts/hyperparameter_search.py run
  !python scripts/hyperparameter_search.py quick_test

🎯 GOAL: Find hyperparameter configuration that beats 65% fMRI baseline
Run this if preprocessing mismatch test shows NO RECOVERY or PARTIAL RECOVERY
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


class HyperparameterSearch:
    """⚙️ Systematic hyperparameter search for cross-attention.
    
    GOAL: Find hyperparameters that beat 65% fMRI baseline
    STRATEGY: Test critical hyperparameters systematically
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
        Run strategic hyperparameter search.
        
        🎯 TESTS CRITICAL HYPERPARAMETERS:
        - Learning rates: [1e-5, 5e-5, 1e-4, 2e-4]
        - Batch sizes: [16, 32, 64]
        - Architecture variations
        - Both original and enhanced preprocessing
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
            print("⚙️ HYPERPARAMETER SEARCH FOR CROSS-ATTENTION")
            print("=" * 60)
            print("🎯 Goal: Find configuration that beats 65% fMRI baseline")
            print(f"📁 Output directory: {config.output_dir}")
            print()
        
        # Define strategic configurations
        configs = self._get_strategic_configs()
            
        if verbose:
            print(f"🔍 Testing {len(configs)} strategic configurations...")
            print()
        
        # Load data for both preprocessing types
        preprocessing_data = {}
        for prep_type in ['original', 'enhanced']:
            if verbose:
                print(f"📊 Loading data with {prep_type} preprocessing...")
            
            # Load fMRI data
            fmri_processor = FMRIDataProcessor(
                roi_dir=config.fmri_roi_dir,
                pheno_file=config.phenotypic_file,
                n_rois=config.n_rois
            )
            fmri_data = fmri_processor.load_all_subjects()
            
            # Load sMRI data with specified preprocessing
            scaler_type = 'standard' if prep_type == 'original' else 'robust'
            smri_processor = SMRIDataProcessor(
                data_path=config.smri_data_path,
                feature_selection_k=300,
                scaler_type=scaler_type
            )
            smri_data = smri_processor.load_all_subjects(config.phenotypic_file)
            
            # Match subjects
            matched_fmri, matched_smri, matched_labels, matched_ids = match_multimodal_subjects(
                fmri_data, smri_data, verbose=False
            )
            
            preprocessing_data[prep_type] = {
                'fmri_features': matched_fmri,
                'smri_features': matched_smri,
                'labels': matched_labels,
                'subject_ids': matched_ids
            }
            
            if verbose:
                print(f"   ✅ {prep_type}: {len(matched_labels)} subjects, sMRI dim: {matched_smri.shape[1]}")
        
        # Run hyperparameter search
        results = []
        best_accuracy = 0
        best_config = None
        
        for i, test_config in enumerate(configs):
            if verbose:
                print(f"\n📋 Config {i+1}/{len(configs)}: {test_config['preprocessing']} preprocessing")
                print(f"   LR={test_config['learning_rate']}, BS={test_config['batch_size']}")
                print(f"   HD={test_config['d_model']}, Heads={test_config['num_heads']}")
                print(f"   Dropout={test_config['dropout']}")
            
            try:
                # Get data for this preprocessing type
                data = preprocessing_data[test_config['preprocessing']]
                
                # Create config for this test
                test_config_obj = get_config(
                    'cross_attention',
                    num_folds=num_folds,
                    batch_size=test_config['batch_size'],
                    learning_rate=test_config['learning_rate'],
                    num_epochs=num_epochs,
                    d_model=test_config['d_model'],
                    num_layers=test_config['num_layers'],
                    num_cross_layers=test_config['num_cross_layers'],
                    num_heads=test_config['num_heads'],
                    dropout=test_config['dropout'],
                    output_dir=config.output_dir / f"config_{i+1}",
                    seed=seed,
                    device=device
                )
                
                # Run cross-validation
                cv_results = run_cross_validation(
                    features=None,
                    labels=data['labels'],
                    model_class=CrossAttentionTransformer,
                    config=test_config_obj,
                    experiment_type='multimodal',
                    fmri_features=data['fmri_features'],
                    smri_features=data['smri_features'],
                    verbose=False
                )
                
                # Calculate metrics
                mean_acc = np.mean([r['test_accuracy'] for r in cv_results])
                mean_bal_acc = np.mean([r['test_balanced_accuracy'] for r in cv_results])
                mean_auc = np.mean([r['test_auc'] for r in cv_results])
                
                result = {
                    'config_id': i + 1,
                    'config': test_config.copy(),
                    'mean_accuracy': mean_acc,
                    'mean_balanced_accuracy': mean_bal_acc,
                    'mean_auc': mean_auc,
                    'std_accuracy': np.std([r['test_accuracy'] for r in cv_results]),
                    'beats_fmri_baseline': mean_acc > 0.65
                }
                
                results.append(result)
                
                if verbose:
                    print(f"   Results: Acc={mean_acc:.4f}, Bal_Acc={mean_bal_acc:.4f}, AUC={mean_auc:.4f}")
                    if mean_acc > 0.65:
                        print(f"   🎯 BEATS fMRI BASELINE! ({mean_acc:.1%} > 65%)")
                
                if mean_acc > best_accuracy:
                    best_accuracy = mean_acc
                    best_config = test_config.copy()
                    best_config['config_id'] = i + 1
                    
            except Exception as e:
                if verbose:
                    print(f"   ❌ Error: {str(e)}")
                continue
        
        # Analyze results
        if verbose:
            self._analyze_results(results, best_config, best_accuracy)
        
        # Save results
        self._save_search_results(results, best_config, best_accuracy, config.output_dir)
        
        return results, best_config, best_accuracy

    def _get_strategic_configs(self):
        """Get strategic configurations focusing on most impactful parameters."""
        return [
            # Original baseline
            {'preprocessing': 'original', 'learning_rate': 1e-4, 'batch_size': 32, 'd_model': 64, 
             'num_heads': 4, 'num_layers': 2, 'num_cross_layers': 2, 'dropout': 0.1},
            
            # Learning rate variations
            {'preprocessing': 'original', 'learning_rate': 5e-5, 'batch_size': 32, 'd_model': 64, 
             'num_heads': 4, 'num_layers': 2, 'num_cross_layers': 2, 'dropout': 0.1},
             
            {'preprocessing': 'original', 'learning_rate': 2e-4, 'batch_size': 32, 'd_model': 64, 
             'num_heads': 4, 'num_layers': 2, 'num_cross_layers': 2, 'dropout': 0.1},
             
            {'preprocessing': 'original', 'learning_rate': 1e-5, 'batch_size': 32, 'd_model': 64, 
             'num_heads': 4, 'num_layers': 2, 'num_cross_layers': 2, 'dropout': 0.1},
            
            # Batch size variations
            {'preprocessing': 'original', 'learning_rate': 1e-4, 'batch_size': 16, 'd_model': 64, 
             'num_heads': 4, 'num_layers': 2, 'num_cross_layers': 2, 'dropout': 0.1},
             
            {'preprocessing': 'original', 'learning_rate': 1e-4, 'batch_size': 64, 'd_model': 64, 
             'num_heads': 4, 'num_layers': 2, 'num_cross_layers': 2, 'dropout': 0.1},
            
            # Architecture variations
            {'preprocessing': 'original', 'learning_rate': 1e-4, 'batch_size': 32, 'd_model': 128, 
             'num_heads': 8, 'num_layers': 3, 'num_cross_layers': 3, 'dropout': 0.15},
             
            {'preprocessing': 'original', 'learning_rate': 1e-4, 'batch_size': 32, 'd_model': 32, 
             'num_heads': 2, 'num_layers': 1, 'num_cross_layers': 1, 'dropout': 0.05},
            
            # Enhanced preprocessing tests
            {'preprocessing': 'enhanced', 'learning_rate': 5e-5, 'batch_size': 32, 'd_model': 64, 
             'num_heads': 4, 'num_layers': 2, 'num_cross_layers': 2, 'dropout': 0.15},
             
            {'preprocessing': 'enhanced', 'learning_rate': 1e-4, 'batch_size': 16, 'd_model': 64, 
             'num_heads': 4, 'num_layers': 2, 'num_cross_layers': 2, 'dropout': 0.2},
            
            # Conservative improvements
            {'preprocessing': 'original', 'learning_rate': 8e-5, 'batch_size': 24, 'd_model': 80, 
             'num_heads': 4, 'num_layers': 2, 'num_cross_layers': 2, 'dropout': 0.12},
             
            {'preprocessing': 'original', 'learning_rate': 1.2e-4, 'batch_size': 28, 'd_model': 96, 
             'num_heads': 6, 'num_layers': 2, 'num_cross_layers': 2, 'dropout': 0.08},
        ]

    def _analyze_results(self, results, best_config, best_accuracy):
        """Analyze search results."""
        
        print("\n" + "="*60)
        print("🎯 HYPERPARAMETER SEARCH RESULTS")
        print("-" * 40)
        
        # Sort by accuracy
        results_sorted = sorted(results, key=lambda x: x['mean_accuracy'], reverse=True)
        
        print(f"📊 SUMMARY:")
        print(f"   Configurations tested: {len(results)}")
        print(f"   Best accuracy: {best_accuracy:.4f} ({best_accuracy:.1%})")
        beats_baseline = sum(1 for r in results if r['beats_fmri_baseline'])
        print(f"   Configs beating fMRI baseline (>65%): {beats_baseline}")
        print()
        
        print("🏆 TOP 5 CONFIGURATIONS:")
        for i, result in enumerate(results_sorted[:5]):
            config = result['config']
            acc = result['mean_accuracy']
            
            print(f"\n   #{i+1}: {acc:.4f} accuracy ({acc:.1%})")
            print(f"      Preprocessing: {config['preprocessing']}")
            print(f"      LR: {config['learning_rate']}, BS: {config['batch_size']}")
            print(f"      Architecture: d_model={config['d_model']}, heads={config['num_heads']}")
            print(f"      Dropout: {config['dropout']:.3f}")
            if acc > 0.65:
                print(f"      🎯 BEATS fMRI BASELINE!")
        
        # Recommendations
        print(f"\n📋 RECOMMENDATIONS:")
        if beats_baseline > 0:
            print(f"   ✅ SUCCESS! Found {beats_baseline} config(s) beating fMRI baseline")
            print(f"   → Use best configuration for final model")
        elif best_accuracy > 0.62:
            print(f"   ⚠️  Close to target (need {(0.65-best_accuracy)*100:.1f} more points)")
            print(f"   → Try longer training or ensemble methods")
        else:
            print(f"   ❌ Significant gap to target ({(0.65-best_accuracy)*100:.1f} points)")
            print(f"   → Architectural changes needed")

    def _save_search_results(self, results, best_config, best_accuracy, output_dir):
        """Save search results."""
        
        output_data = {
            'test_name': 'hyperparameter_search',
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_configs_tested': len(results),
                'best_accuracy': float(best_accuracy),
                'configs_beating_fmri': sum(1 for r in results if r['beats_fmri_baseline']),
                'best_config': best_config
            },
            'all_results': results,
            'target_baseline': 65.0
        }
        
        results_path = output_dir / 'hyperparameter_search_results.json'
        with open(results_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\n✅ Hyperparameter search completed!")
        print(f"📁 Results saved to: {results_path}")
        print(f"🎯 Best performance: {best_accuracy:.1%}")

    def quick_test(self, num_folds: int = 2, num_epochs: int = 10, output_dir: str = "./test_hyperparameter_output"):
        """
        🧪 Quick hyperparameter test.
        """
        print("🧪 Running hyperparameter search quick test...")
        print("🎯 Testing key hyperparameters to find improvements")
        
        return self.run(
            num_folds=num_folds,
            num_epochs=num_epochs,
            output_dir=output_dir,
            verbose=True
        )


if __name__ == '__main__':
    fire.Fire(HyperparameterSearch) 
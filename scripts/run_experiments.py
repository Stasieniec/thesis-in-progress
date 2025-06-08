#!/usr/bin/env python3
"""
🧠 ABIDE Cross-Attention Experiments Runner

Single entry point for running all three experiments with matched subjects.
Ensures fair comparison by using identical subject sets across experiments.
"""

import sys
import os
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from utils.subject_matching import get_matched_datasets


def run_fmri_experiment(
    fmri_data_path: str,
    phenotypic_file: str,
    matched_subject_ids: set,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """Run fMRI experiment with matched subjects."""
    print("🧠 Running fMRI Experiment")
    print("=" * 50)
    
    # Import here to avoid circular imports
    from training.train_fmri import run_fmri_training
    
    results = run_fmri_training(
        fmri_data_path=fmri_data_path,
        phenotypic_file=phenotypic_file,
        matched_subject_ids=matched_subject_ids,
        **config.get('fmri', {})
    )
    
    print(f"✅ fMRI Experiment Complete - Best Accuracy: {results.get('best_accuracy', 0):.2%}")
    return results


def run_smri_experiment(
    smri_data_path: str,
    phenotypic_file: str,
    matched_subject_ids: set,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """Run sMRI experiment with matched subjects."""
    print("\n🧠 Running sMRI Experiment")
    print("=" * 50)
    
    # Import here to avoid circular imports
    from training.train_smri import run_smri_training
    
    results = run_smri_training(
        smri_data_path=smri_data_path,
        phenotypic_file=phenotypic_file,
        matched_subject_ids=matched_subject_ids,
        **config.get('smri', {})
    )
    
    print(f"✅ sMRI Experiment Complete - Best Accuracy: {results.get('best_accuracy', 0):.2%}")
    return results


def run_cross_attention_experiment(
    fmri_data_path: str,
    smri_data_path: str,
    phenotypic_file: str,
    matched_subject_ids: set,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """Run cross-attention experiment with matched subjects."""
    print("\n🧠 Running Cross-Attention Experiment")
    print("=" * 50)
    
    # Import here to avoid circular imports
    from training.train_cross_attention import run_cross_attention_training
    
    results = run_cross_attention_training(
        fmri_data_path=fmri_data_path,
        smri_data_path=smri_data_path,
        phenotypic_file=phenotypic_file,
        matched_subject_ids=matched_subject_ids,
        **config.get('cross_attention', {})
    )
    
    print(f"✅ Cross-Attention Experiment Complete - Best Accuracy: {results.get('best_accuracy', 0):.2%}")
    return results


def load_experiment_config(config_file: Optional[str] = None) -> Dict[str, Any]:
    """Load experiment configuration."""
    if config_file and Path(config_file).exists():
        with open(config_file, 'r') as f:
            return json.load(f)
    
    # Default configuration
    return {
        'fmri': {
            'num_epochs': 50,
            'batch_size': 32,
            'learning_rate': 0.001,
            'patience': 10,
            'random_seed': 42
        },
        'smri': {
            'num_epochs': 100,
            'batch_size': 64,
            'learning_rate': 0.001,
            'patience': 15,
            'feature_selection_k': 800,
            'random_seed': 42
        },
        'cross_attention': {
            'num_epochs': 75,
            'batch_size': 32,
            'learning_rate': 0.0005,
            'patience': 12,
            'random_seed': 42
        }
    }


def save_results_summary(results: Dict[str, Dict[str, Any]], output_file: str):
    """Save a summary of all experiment results."""
    summary = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'experiments': {},
        'comparison': {}
    }
    
    # Extract key metrics
    for exp_name, exp_results in results.items():
        summary['experiments'][exp_name] = {
            'best_accuracy': exp_results.get('best_accuracy', 0),
            'best_f1': exp_results.get('best_f1', 0),
            'final_loss': exp_results.get('final_loss', 0),
            'num_subjects': exp_results.get('num_subjects', 0),
            'training_time': exp_results.get('training_time', 0)
        }
    
    # Create comparison
    accuracies = {name: res.get('best_accuracy', 0) for name, res in results.items()}
    best_model = max(accuracies.keys(), key=lambda k: accuracies[k])
    
    summary['comparison'] = {
        'best_model': best_model,
        'best_accuracy': accuracies[best_model],
        'accuracy_ranking': sorted(accuracies.items(), key=lambda x: x[1], reverse=True)
    }
    
    # Save to file
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n📊 Results Summary saved to: {output_file}")
    return summary


def run_all_experiments(
    fmri_data_path: str,
    smri_data_path: str,
    phenotypic_file: str,
    use_matched_subjects: bool = True,
    config_file: Optional[str] = None,
    output_dir: str = "results",
    experiments_to_run: Optional[list] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Run all ABIDE experiments with matched subjects.
    
    Args:
        fmri_data_path: Path to fMRI data directory
        smri_data_path: Path to sMRI data directory  
        phenotypic_file: Path to phenotypic CSV file
        use_matched_subjects: Whether to use matched subjects (recommended)
        config_file: Optional path to JSON config file
        output_dir: Directory to save results
        experiments_to_run: List of experiments ['fmri', 'smri', 'cross_attention'] or None for all
        
    Returns:
        Dictionary of results for each experiment
    """
    
    print("🧠 ABIDE Cross-Attention Experiments")
    print("=" * 60)
    print(f"📁 fMRI Data: {fmri_data_path}")
    print(f"📁 sMRI Data: {smri_data_path}")
    print(f"📄 Phenotypic: {phenotypic_file}")
    print(f"🔗 Matched Subjects: {'Yes' if use_matched_subjects else 'No'}")
    print("=" * 60)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Load configuration
    config = load_experiment_config(config_file)
    
    # Determine which experiments to run
    if experiments_to_run is None:
        experiments_to_run = ['fmri', 'smri', 'cross_attention']
    
    # Get matched datasets if requested
    matched_subject_ids = None
    if use_matched_subjects:
        print("🔗 Finding matched subjects across modalities...")
        try:
            matched_data = get_matched_datasets(
                fmri_data_path=fmri_data_path,
                smri_data_path=smri_data_path,
                phenotypic_file=phenotypic_file
            )
            matched_subject_ids = matched_data['matched_subject_ids']
            print(f"✅ Found {len(matched_subject_ids)} matched subjects")
        except Exception as e:
            print(f"❌ Error finding matched subjects: {e}")
            print("⚠️ Proceeding without subject matching...")
            use_matched_subjects = False
    
    # Run experiments
    results = {}
    start_time = time.time()
    
    try:
        if 'fmri' in experiments_to_run:
            results['fmri'] = run_fmri_experiment(
                fmri_data_path=fmri_data_path,
                phenotypic_file=phenotypic_file,
                matched_subject_ids=matched_subject_ids,
                config=config
            )
        
        if 'smri' in experiments_to_run:
            results['smri'] = run_smri_experiment(
                smri_data_path=smri_data_path,
                phenotypic_file=phenotypic_file,
                matched_subject_ids=matched_subject_ids,
                config=config
            )
        
        if 'cross_attention' in experiments_to_run:
            results['cross_attention'] = run_cross_attention_experiment(
                fmri_data_path=fmri_data_path,
                smri_data_path=smri_data_path,
                phenotypic_file=phenotypic_file,
                matched_subject_ids=matched_subject_ids,
                config=config
            )
            
    except Exception as e:
        print(f"\n❌ Experiment failed: {e}")
        raise
    
    # Calculate total time
    total_time = time.time() - start_time
    
    # Save results summary
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    results_file = output_path / f"experiment_results_{timestamp}.json"
    summary = save_results_summary(results, str(results_file))
    
    # Print final summary
    print("\n" + "=" * 60)
    print("🏆 EXPERIMENT RESULTS SUMMARY")
    print("=" * 60)
    
    for exp_name, exp_results in results.items():
        accuracy = exp_results.get('best_accuracy', 0)
        subjects = exp_results.get('num_subjects', 0)
        print(f"{exp_name.upper():>15}: {accuracy:>6.2%} accuracy ({subjects} subjects)")
    
    print(f"\n⏱️ Total Time: {total_time/60:.1f} minutes")
    print(f"🏆 Best Model: {summary['comparison']['best_model']}")
    print(f"📊 Results: {results_file}")
    
    if use_matched_subjects:
        print("✅ All experiments used matched subjects for fair comparison")
    else:
        print("⚠️ Experiments did not use matched subjects")
    
    return results


def main():
    """Command line interface."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run ABIDE cross-attention experiments')
    parser.add_argument('--fmri-data', required=True, help='Path to fMRI data directory')
    parser.add_argument('--smri-data', required=True, help='Path to sMRI data directory')
    parser.add_argument('--phenotypic', required=True, help='Path to phenotypic CSV file')
    parser.add_argument('--config', help='Path to JSON config file')
    parser.add_argument('--output-dir', default='results', help='Output directory for results')
    parser.add_argument('--no-matching', action='store_true', help='Disable subject matching')
    parser.add_argument('--experiments', nargs='+', 
                       choices=['fmri', 'smri', 'cross_attention'],
                       help='Which experiments to run (default: all)')
    
    args = parser.parse_args()
    
    results = run_all_experiments(
        fmri_data_path=args.fmri_data,
        smri_data_path=args.smri_data,
        phenotypic_file=args.phenotypic,
        use_matched_subjects=not args.no_matching,
        config_file=args.config,
        output_dir=args.output_dir,
        experiments_to_run=args.experiments
    )
    
    return results


if __name__ == "__main__":
    main() 
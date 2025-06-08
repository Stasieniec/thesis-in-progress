#!/usr/bin/env python3
"""
Comprehensive Experimental Framework for Thesis Results
======================================================

This script runs a comprehensive evaluation of all model configurations
on both regular cross-validation and leave-site-out cross-validation.

Perfect for generating publication-quality results for thesis work.

Usage Examples:
  # Run all experiments (recommended for thesis)
  python scripts/comprehensive_experiments.py run_all
  
  # Run specific experiments
  python scripts/comprehensive_experiments.py run_selected --experiments smri_basic fmri_basic cross_attention_basic
  
  # Quick test (reduced epochs, fewer folds)
  python scripts/comprehensive_experiments.py quick_test
  
  # Generate plots only (if results already exist)
  python scripts/comprehensive_experiments.py generate_plots --results_dir results_20240101_120000

Google Colab Usage:
  !python scripts/comprehensive_experiments.py run_all
  !python scripts/comprehensive_experiments.py quick_test
"""

import sys
import fire
import json
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from evaluation.experiment_framework import ComprehensiveExperimentFramework, ExperimentRegistry

# Try different import paths for result analyzer
try:
    from evaluation.result_analyzer import ThesisPlotter, ResultAnalyzer
except ImportError:
    try:
        from src.evaluation.result_analyzer import ThesisPlotter, ResultAnalyzer
    except ImportError:
        print("⚠️ ThesisPlotter not available - plots will be skipped")
        ThesisPlotter = None
        ResultAnalyzer = None


class ComprehensiveExperiments:
    """Main class for running comprehensive experiments."""
    
    def __init__(self):
        """Initialize the comprehensive experiments."""
        self.registry = ExperimentRegistry()
        print("🚀 Comprehensive Experimental Framework")
        print("=" * 50)
        print("Available experiments:")
        for exp_name in self.registry.list_experiments():
            exp = self.registry.get_experiment(exp_name)
            print(f"  - {exp_name}: {exp['name']}")
        print()
    
    def run_all(
        self,
        cv_folds: int = 5,
        include_advanced: bool = True,
        output_dir: Optional[str] = None,
        data_paths: Optional[Dict[str, str]] = None,
        seed: int = 42,
        verbose: bool = True
    ):
        """
        Run all experiments - the main function for thesis results.
        
        Args:
            cv_folds: Number of folds for regular cross-validation
            include_advanced: Include advanced cross-attention models
            output_dir: Output directory (auto-generated if None)
            data_paths: Custom data paths (uses Google Colab defaults if None)
            seed: Random seed for reproducibility
            verbose: Verbose output
        """
        print("🎯 RUNNING COMPREHENSIVE EVALUATION FOR THESIS")
        print("=" * 60)
        
        # Initialize framework
        framework = ComprehensiveExperimentFramework(
            data_paths=data_paths,
            output_dir=output_dir,
            seed=seed
        )
        
        # Load data
        print("\n📊 Loading matched multimodal data...")
        try:
            matched_data = framework.load_data(verbose=verbose)
            print(f"✅ Successfully loaded {matched_data['num_matched_subjects']} matched subjects")
        except Exception as e:
            print(f"❌ Failed to load data: {e}")
            print("\nTroubleshooting:")
            print("1. Make sure Google Drive is mounted (if using Colab)")
            print("2. Check that data paths exist")
            print("3. Run: from google.colab import drive; drive.mount('/content/drive')")
            return
        
        # Run comprehensive evaluation
        print(f"\n🧠 Starting comprehensive evaluation...")
        print(f"   CV folds: {cv_folds}")
        print(f"   Include advanced models: {include_advanced}")
        print(f"   Random seed: {seed}")
        
        start_time = datetime.now()
        
        results = framework.run_comprehensive_evaluation(
            cv_folds=cv_folds,
            include_advanced=include_advanced,
            verbose=verbose
        )
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        # Generate analysis and plots
        print(f"\n📈 Generating thesis plots and analysis...")
        try:
            framework.generate_thesis_plots()
            
            # Generate statistical report (if available)
            if ResultAnalyzer is not None:
                analyzer = ResultAnalyzer(results)
                analyzer.generate_statistical_report(
                    framework.output_dir / 'statistical_report.txt'
                )
            
            print(f"✅ Analysis complete!")
            
        except Exception as e:
            print(f"⚠️ Plot generation failed: {e}")
        
        # Summary
        print(f"\n📋 COMPREHENSIVE EVALUATION COMPLETE")
        print("=" * 50)
        print(f"⏱️  Total time: {duration}")
        print(f"📁 Results saved to: {framework.output_dir}")
        print(f"📊 Experiments run: {len(results)}")
        print(f"✅ Successful: {sum(1 for r in results.values() if 'error' not in r)}")
        print(f"❌ Failed: {sum(1 for r in results.values() if 'error' in r)}")
        
        # Key files for thesis
        print(f"\n📄 Key files for thesis:")
        print(f"   - Summary table: {framework.output_dir}/thesis_summary_table.csv")
        print(f"   - Raw results: {framework.output_dir}/comprehensive_results.json")
        print(f"   - Statistical report: {framework.output_dir}/statistical_report.txt")
        print(f"   - Plots: {framework.output_dir}/plots/")
        
        return results
    
    def run_selected(
        self,
        experiments: List[str],
        cv_folds: int = 5,
        output_dir: Optional[str] = None,
        seed: int = 42,
        verbose: bool = True
    ):
        """
        Run selected experiments only.
        
        Args:
            experiments: List of experiment names to run
            cv_folds: Number of CV folds
            output_dir: Output directory
            seed: Random seed
            verbose: Verbose output
        """
        print(f"🎯 RUNNING SELECTED EXPERIMENTS")
        print("=" * 40)
        print(f"Selected: {experiments}")
        
        # Validate experiment names
        available = self.registry.list_experiments()
        invalid = [e for e in experiments if e not in available]
        if invalid:
            print(f"❌ Invalid experiments: {invalid}")
            print(f"Available: {available}")
            return
        
        # Initialize framework
        framework = ComprehensiveExperimentFramework(
            output_dir=output_dir,
            seed=seed
        )
        
        # Load data
        framework.load_data(verbose=verbose)
        
        # Run selected experiments
        results = framework.run_comprehensive_evaluation(
            experiments=experiments,
            cv_folds=cv_folds,
            verbose=verbose
        )
        
        # Generate plots
        try:
            framework.generate_thesis_plots()
        except Exception as e:
            print(f"⚠️ Plot generation failed: {e}")
        
        print(f"✅ Selected experiments complete!")
        print(f"📁 Results: {framework.output_dir}")
        
        return results
    
    def quick_test(
        self,
        cv_folds: int = 2,
        num_epochs: int = 5,
        output_dir: Optional[str] = None,
        seed: int = 42
    ):
        """
        Quick test with reduced parameters for fast validation.
        
        Args:
            cv_folds: Number of CV folds (reduced)
            num_epochs: Number of epochs (reduced)
            output_dir: Output directory
            seed: Random seed
        """
        print("⚡ QUICK TEST MODE")
        print("=" * 30)
        print(f"⚠️  Reduced parameters: {cv_folds} folds, {num_epochs} epochs")
        print("   Use run_all() for full thesis results!")
        
        # Run basic experiments only for quick test
        basic_experiments = [
            'smri_basic',
            'fmri_basic', 
            'cross_attention_basic'
        ]
        
        # Override epochs for quick test
        framework = ComprehensiveExperimentFramework(
            output_dir=output_dir or "quick_test_results",
            seed=seed
        )
        
        # Modify experiment configs for quick test
        for exp_name in basic_experiments:
            exp = framework.registry.get_experiment(exp_name)
            if 'config_overrides' in exp:
                exp['config_overrides']['num_epochs'] = num_epochs
        
        # Load data
        framework.load_data(verbose=True)
        
        # Run experiments
        results = framework.run_comprehensive_evaluation(
            experiments=basic_experiments,
            cv_folds=cv_folds,
            verbose=True
        )
        
        # Generate basic plots
        try:
            framework.generate_thesis_plots()
        except Exception as e:
            print(f"⚠️ Plot generation skipped in quick test: {e}")
        
        print(f"⚡ Quick test complete!")
        print(f"📁 Results: {framework.output_dir}")
        
        return results
    
    def generate_plots(
        self,
        results_dir: str,
        output_dir: Optional[str] = None
    ):
        """
        Generate plots from existing results.
        
        Args:
            results_dir: Directory containing comprehensive_results.json
            output_dir: Output directory for plots (default: results_dir/plots)
        """
        print("📈 GENERATING PLOTS FROM EXISTING RESULTS")
        print("=" * 50)
        
        results_path = Path(results_dir) / 'comprehensive_results.json'
        
        if not results_path.exists():
            print(f"❌ Results file not found: {results_path}")
            return
        
        # Load results
        with open(results_path, 'r') as f:
            results = json.load(f)
        
        print(f"📊 Loaded {len(results)} experiment results")
        
        # Set output directory
        if output_dir is None:
            output_dir = Path(results_dir)
        else:
            output_dir = Path(output_dir)
        
        # Generate plots
        if ThesisPlotter is not None:
            plotter = ThesisPlotter(results, output_dir)
            plotter.create_all_plots()
        else:
            print("⚠️ ThesisPlotter not available")
        
        # Generate statistical report
        if ResultAnalyzer is not None:
            analyzer = ResultAnalyzer(results)
            analyzer.generate_statistical_report(output_dir / 'statistical_report.txt')
        else:
            print("⚠️ ResultAnalyzer not available")
        
        print(f"✅ Plots generated!")
        print(f"📁 Output: {output_dir}/plots/")
    
    def list_experiments(self):
        """List all available experiments with descriptions."""
        print("📋 AVAILABLE EXPERIMENTS")
        print("=" * 40)
        
        by_type = {}
        for exp_name in self.registry.list_experiments():
            exp = self.registry.get_experiment(exp_name)
            exp_type = exp['type']
            if exp_type not in by_type:
                by_type[exp_type] = []
            by_type[exp_type].append((exp_name, exp))
        
        for exp_type, experiments in by_type.items():
            print(f"\n{exp_type.upper()}:")
            for exp_name, exp in experiments:
                print(f"  {exp_name}")
                print(f"    Name: {exp['name']}")
                print(f"    Description: {exp['description']}")
                if exp.get('leave_site_out_only'):
                    print(f"    Note: Leave-site-out CV only")
                print()
    
    def validate_setup(self):
        """Validate that the experimental setup is ready."""
        print("🔍 VALIDATING EXPERIMENTAL SETUP")
        print("=" * 40)
        
        checks = []
        
        # Check data paths
        try:
            framework = ComprehensiveExperimentFramework()
            matched_data = framework.load_data(verbose=False)
            checks.append(("✅ Data loading", "OK"))
            print(f"✅ Data loading: {matched_data['num_matched_subjects']} subjects")
        except Exception as e:
            checks.append(("❌ Data loading", str(e)))
            print(f"❌ Data loading: {e}")
        
        # Check model imports
        try:
            from models.fmri_transformer import SingleAtlasTransformer
            from models.smri_transformer import SMRITransformer
            from models.cross_attention import CrossAttentionTransformer
            checks.append(("✅ Basic models", "OK"))
            print("✅ Basic models: Available")
        except Exception as e:
            checks.append(("❌ Basic models", str(e)))
            print(f"❌ Basic models: {e}")
        
        # Check advanced models
        try:
            from advanced_cross_attention_experiments import (
                BidirectionalCrossAttentionTransformer
            )
            checks.append(("✅ Advanced models", "OK"))
            print("✅ Advanced models: Available")
        except Exception as e:
            checks.append(("⚠️ Advanced models", "Not available"))
            print("⚠️ Advanced models: Not available (basic models will be used)")
        
        # Check plotting dependencies
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            checks.append(("✅ Plotting", "OK"))
            print("✅ Plotting libraries: Available")
        except Exception as e:
            checks.append(("❌ Plotting", str(e)))
            print(f"❌ Plotting libraries: {e}")
        
        print(f"\n📋 Setup validation complete!")
        
        # Show recommendations
        failed_checks = [c for c in checks if c[0].startswith("❌")]
        if failed_checks:
            print(f"\n⚠️ Issues found:")
            for check, error in failed_checks:
                print(f"   {check}: {error}")
            print(f"\nRecommendations:")
            print(f"1. Ensure Google Drive is mounted (if using Colab)")
            print(f"2. Check data paths and file permissions")
            print(f"3. Install missing dependencies")
        else:
            print(f"🎉 All checks passed! Ready for experiments.")


def main():
    """Main entry point."""
    fire.Fire(ComprehensiveExperiments)


if __name__ == "__main__":
    main() 
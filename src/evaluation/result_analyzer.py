"""
Result Analyzer and Thesis Plotter
=================================

Utilities for analyzing experimental results and generating publication-quality plots.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set publication-quality plotting style
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except OSError:
    try:
        plt.style.use('seaborn-whitegrid')
    except OSError:
        plt.style.use('default')
        print("⚠️ Seaborn style not available, using default matplotlib style")

try:
    sns.set_palette("husl")
except:
    print("⚠️ Could not set seaborn palette")


class ThesisPlotter:
    """Creates publication-quality plots for thesis results."""
    
    def __init__(self, results: Dict, output_dir: Path):
        """
        Initialize the thesis plotter.
        
        Args:
            results: Results dictionary from ComprehensiveExperimentFramework
            output_dir: Directory to save plots
        """
        self.results = results
        self.output_dir = Path(output_dir) / 'plots'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract data for plotting
        self.plot_data = self._extract_plot_data()
    
    def _extract_plot_data(self) -> pd.DataFrame:
        """Extract data into a structured DataFrame for plotting."""
        data = []
        
        for exp_name, result in self.results.items():
            if 'error' in result:
                continue
            
            base_row = {
                'experiment': exp_name,
                'name': result.get('name', exp_name),
                'type': result.get('type', 'unknown'),
                'description': result.get('description', '')
            }
            
            # Regular CV data
            regular_cv = result.get('regular_cv', {})
            if regular_cv and 'mean_accuracy' in regular_cv:
                cv_row = base_row.copy()
                cv_row.update({
                    'cv_type': 'Regular CV',
                    'accuracy': regular_cv['mean_accuracy'],
                    'accuracy_std': regular_cv['std_accuracy'],
                    'balanced_accuracy': regular_cv.get('mean_balanced_accuracy', 0),
                    'auc': regular_cv.get('mean_auc', 0)
                })
                data.append(cv_row)
            
            # Leave-site-out CV data
            lso_cv = result.get('leave_site_out_cv', {})
            if lso_cv and not lso_cv.get('error'):
                lso_row = base_row.copy()
                summary = lso_cv.get('summary', lso_cv)
                lso_row.update({
                    'cv_type': 'Leave-Site-Out CV',
                    'accuracy': summary.get('mean_accuracy', 0),
                    'accuracy_std': summary.get('std_accuracy', 0),
                    'balanced_accuracy': summary.get('mean_balanced_accuracy', 0),
                    'auc': summary.get('mean_auc', 0)
                })
                data.append(lso_row)
        
        return pd.DataFrame(data)
    
    def create_all_plots(self):
        """Create all thesis plots."""
        if self.plot_data.empty:
            print("⚠️ No data available for plotting")
            return
        
        print("📈 Generating thesis plots...")
        
        # Main comparison plot
        self.plot_accuracy_comparison()
        
        # Generalization gap analysis
        self.plot_generalization_gap()
        
        # Model type comparison
        self.plot_by_model_type()
        
        # Statistical significance heatmap
        self.plot_statistical_significance()
        
        # Performance vs complexity trade-off
        self.plot_performance_complexity()
        
        print(f"✅ All plots saved to: {self.output_dir}")
    
    def plot_accuracy_comparison(self):
        """Main accuracy comparison plot."""
        plt.figure(figsize=(14, 8))
        
        # Prepare data for plotting
        plot_df = self.plot_data.copy()
        
        # Create grouped bar plot
        experiments = plot_df['name'].unique()
        cv_types = plot_df['cv_type'].unique()
        
        x = np.arange(len(experiments))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(16, 8))
        
        for i, cv_type in enumerate(cv_types):
            cv_data = plot_df[plot_df['cv_type'] == cv_type]
            
            # Ensure we have data for all experiments
            accuracies = []
            errors = []
            
            for exp in experiments:
                exp_data = cv_data[cv_data['name'] == exp]
                if not exp_data.empty:
                    accuracies.append(exp_data['accuracy'].iloc[0])
                    errors.append(exp_data['accuracy_std'].iloc[0])
                else:
                    accuracies.append(0)
                    errors.append(0)
            
            bars = ax.bar(x + i*width, accuracies, width, 
                         yerr=errors, capsize=5,
                         label=cv_type, alpha=0.8)
            
            # Add value labels on bars
            for bar, acc in zip(bars, accuracies):
                if acc > 0:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{acc:.3f}', ha='center', va='bottom', fontsize=9)
        
        # Customize plot
        ax.set_xlabel('Model', fontsize=12, fontweight='bold')
        ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        ax.set_title('Model Performance: Regular CV vs Leave-Site-Out CV', 
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x + width/2)
        ax.set_xticklabels(experiments, rotation=45, ha='right')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.0)
        
        # Add baseline lines
        ax.axhline(y=0.6, color='red', linestyle='--', alpha=0.7, 
                  label='fMRI Baseline (60%)')
        ax.axhline(y=0.58, color='orange', linestyle='--', alpha=0.7,
                  label='sMRI Baseline (58%)')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'accuracy_comparison.png', 
                   dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'accuracy_comparison.pdf', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_generalization_gap(self):
        """Plot generalization gap analysis."""
        # Calculate generalization gaps
        gaps = []
        names = []
        
        for exp_name, result in self.results.items():
            if 'error' in result or 'generalization_gap' not in result:
                continue
            
            gap = result['generalization_gap']
            if gap is not None:
                gaps.append(gap)
                names.append(result.get('name', exp_name))
        
        if not gaps:
            return
        
        plt.figure(figsize=(12, 6))
        
        # Create horizontal bar plot
        y_pos = np.arange(len(names))
        bars = plt.barh(y_pos, gaps, alpha=0.7)
        
        # Color bars: green for negative gaps (better generalization), red for positive
        for bar, gap in zip(bars, gaps):
            if gap < 0:
                bar.set_color('green')
            else:
                bar.set_color('red')
        
        plt.yticks(y_pos, names)
        plt.xlabel('Generalization Gap (Regular CV - Leave-Site-Out CV)', 
                  fontsize=12, fontweight='bold')
        plt.title('Generalization Gap Analysis\n(Negative = Better Generalization)', 
                 fontsize=14, fontweight='bold')
        plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        plt.grid(True, alpha=0.3)
        
        # Add value labels
        for i, gap in enumerate(gaps):
            plt.text(gap + 0.005 if gap >= 0 else gap - 0.005, i,
                    f'{gap:.3f}', va='center', 
                    ha='left' if gap >= 0 else 'right')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'generalization_gap.png', 
                   dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'generalization_gap.pdf',
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_by_model_type(self):
        """Plot results grouped by model type."""
        if self.plot_data.empty:
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Group by model type
        type_groups = self.plot_data.groupby(['type', 'cv_type'])['accuracy'].agg(['mean', 'std']).reset_index()
        
        # Plot 1: Mean accuracy by type
        ax1 = axes[0]
        for cv_type in type_groups['cv_type'].unique():
            data = type_groups[type_groups['cv_type'] == cv_type]
            ax1.bar(data['type'], data['mean'], 
                   yerr=data['std'], capsize=5,
                   alpha=0.7, label=cv_type)
        
        ax1.set_title('Average Accuracy by Model Type', fontweight='bold')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Individual model performances
        ax2 = axes[1]
        
        # Create scatter plot
        for model_type in self.plot_data['type'].unique():
            type_data = self.plot_data[self.plot_data['type'] == model_type]
            
            regular_data = type_data[type_data['cv_type'] == 'Regular CV']
            lso_data = type_data[type_data['cv_type'] == 'Leave-Site-Out CV']
            
            # Match experiments between CV types
            for exp in regular_data['name'].unique():
                reg_acc = regular_data[regular_data['name'] == exp]['accuracy']
                lso_acc = lso_data[lso_data['name'] == exp]['accuracy']
                
                if not reg_acc.empty and not lso_acc.empty:
                    ax2.scatter(reg_acc.iloc[0], lso_acc.iloc[0], 
                              s=100, alpha=0.7, label=model_type)
        
        # Add diagonal line (perfect generalization)
        ax2.plot([0.4, 1.0], [0.4, 1.0], 'k--', alpha=0.5, 
                label='Perfect Generalization')
        
        ax2.set_xlabel('Regular CV Accuracy')
        ax2.set_ylabel('Leave-Site-Out CV Accuracy')
        ax2.set_title('Generalization Performance', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0.4, 1.0)
        ax2.set_ylim(0.4, 1.0)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'by_model_type.png', 
                   dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'by_model_type.pdf',
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_statistical_significance(self):
        """Plot statistical significance matrix."""
        # Extract p-values
        p_values = {}
        for exp_name, result in self.results.items():
            if 'statistical_test' in result and 'p_value' in result['statistical_test']:
                p_values[result.get('name', exp_name)] = result['statistical_test']['p_value']
        
        if not p_values:
            return
        
        # Create significance indicators
        names = list(p_values.keys())
        p_vals = list(p_values.values())
        significance = ['***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns' 
                       for p in p_vals]
        
        plt.figure(figsize=(10, 6))
        
        # Create bar plot with significance annotations
        bars = plt.bar(names, p_vals, alpha=0.7)
        
        # Color bars by significance
        colors = ['red' if p < 0.05 else 'orange' if p < 0.1 else 'gray' for p in p_vals]
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        # Add significance level line
        plt.axhline(y=0.05, color='red', linestyle='--', 
                   label='α = 0.05', alpha=0.7)
        
        # Add significance annotations
        for i, (bar, sig) in enumerate(zip(bars, significance)):
            plt.text(bar.get_x() + bar.get_width()/2., 
                    bar.get_height() + 0.005,
                    sig, ha='center', va='bottom', fontweight='bold')
        
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('P-value')
        plt.title('Statistical Significance of CV Type Differences\n(*** p<0.001, ** p<0.01, * p<0.05, ns = not significant)', 
                 fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'statistical_significance.png', 
                   dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'statistical_significance.pdf',
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_performance_complexity(self):
        """Plot performance vs model complexity trade-off."""
        if self.plot_data.empty:
            return
        
        # Estimate model complexity (this is simplified)
        complexity_map = {
            'smri': 1,  # Simplest
            'fmri': 2,
            'cross_attention': 3,
            'cross_attention_advanced': 4  # Most complex
        }
        
        plt.figure(figsize=(10, 8))
        
        # Create scatter plot
        for cv_type in self.plot_data['cv_type'].unique():
            cv_data = self.plot_data[self.plot_data['cv_type'] == cv_type]
            
            complexities = [complexity_map.get(t, 3) for t in cv_data['type']]
            accuracies = cv_data['accuracy'].values
            
            plt.scatter(complexities, accuracies, 
                       s=100, alpha=0.7, label=cv_type)
            
            # Add model names as annotations
            for i, name in enumerate(cv_data['name']):
                plt.annotate(name, (complexities[i], accuracies[i]),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=8, alpha=0.8)
        
        plt.xlabel('Model Complexity\n(1=sMRI, 2=fMRI, 3=Cross-Attention, 4=Advanced)', 
                  fontsize=12, fontweight='bold')
        plt.ylabel('Accuracy', fontsize=12, fontweight='bold')
        plt.title('Performance vs Model Complexity Trade-off', 
                 fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks([1, 2, 3, 4], ['sMRI', 'fMRI', 'Cross-Att', 'Advanced'])
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'performance_complexity.png', 
                   dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'performance_complexity.pdf',
                   dpi=300, bbox_inches='tight')
        plt.close()


class ResultAnalyzer:
    """Statistical analysis of experimental results."""
    
    def __init__(self, results: Dict):
        """Initialize with results dictionary."""
        self.results = results
    
    def generate_statistical_report(self, output_file: Path):
        """Generate comprehensive statistical report."""
        with open(output_file, 'w') as f:
            f.write("STATISTICAL ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            # Overall statistics
            self._write_overall_stats(f)
            
            # Model comparisons
            self._write_model_comparisons(f)
            
            # Generalization analysis
            self._write_generalization_analysis(f)
            
            # Recommendations
            self._write_recommendations(f)
    
    def _write_overall_stats(self, f):
        """Write overall statistics."""
        f.write("OVERALL STATISTICS\n")
        f.write("-" * 30 + "\n")
        
        total_experiments = len(self.results)
        successful = sum(1 for r in self.results.values() if 'error' not in r)
        
        f.write(f"Total experiments: {total_experiments}\n")
        f.write(f"Successful experiments: {successful}\n")
        f.write(f"Success rate: {successful/total_experiments:.1%}\n\n")
        
        # Performance ranges
        regular_accs = []
        lso_accs = []
        
        for result in self.results.values():
            if 'error' in result:
                continue
            
            regular_cv = result.get('regular_cv', {})
            if 'mean_accuracy' in regular_cv:
                regular_accs.append(regular_cv['mean_accuracy'])
            
            lso_cv = result.get('leave_site_out_cv', {})
            if lso_cv and not lso_cv.get('error'):
                summary = lso_cv.get('summary', lso_cv)
                acc = summary.get('mean_accuracy', 0)
                if acc > 0:
                    lso_accs.append(acc)
        
        if regular_accs:
            f.write(f"Regular CV accuracy range: {min(regular_accs):.3f} - {max(regular_accs):.3f}\n")
            f.write(f"Regular CV mean: {np.mean(regular_accs):.3f} ± {np.std(regular_accs):.3f}\n")
        
        if lso_accs:
            f.write(f"Leave-site-out CV range: {min(lso_accs):.3f} - {max(lso_accs):.3f}\n")
            f.write(f"Leave-site-out CV mean: {np.mean(lso_accs):.3f} ± {np.std(lso_accs):.3f}\n")
        
        f.write("\n")
    
    def _write_model_comparisons(self, f):
        """Write model comparison analysis."""
        f.write("MODEL COMPARISONS\n")
        f.write("-" * 30 + "\n")
        
        # Group by model type
        by_type = {}
        for exp_name, result in self.results.items():
            if 'error' in result:
                continue
            
            model_type = result.get('type', 'unknown')
            if model_type not in by_type:
                by_type[model_type] = []
            by_type[model_type].append(result)
        
        for model_type, results in by_type.items():
            f.write(f"\n{model_type.upper()}:\n")
            
            for result in results:
                name = result.get('name', 'Unknown')
                
                regular_cv = result.get('regular_cv', {})
                if 'mean_accuracy' in regular_cv:
                    f.write(f"  {name}: {regular_cv['mean_accuracy']:.3f} ± {regular_cv['std_accuracy']:.3f} (Regular CV)\n")
                
                lso_cv = result.get('leave_site_out_cv', {})
                if lso_cv and not lso_cv.get('error'):
                    summary = lso_cv.get('summary', lso_cv)
                    acc = summary.get('mean_accuracy', 0)
                    std = summary.get('std_accuracy', 0)
                    if acc > 0:
                        f.write(f"  {name}: {acc:.3f} ± {std:.3f} (Leave-Site-Out CV)\n")
        
        f.write("\n")
    
    def _write_generalization_analysis(self, f):
        """Write generalization gap analysis."""
        f.write("GENERALIZATION ANALYSIS\n")
        f.write("-" * 30 + "\n")
        
        gaps = []
        for result in self.results.values():
            gap = result.get('generalization_gap')
            if gap is not None:
                gaps.append(gap)
                name = result.get('name', 'Unknown')
                f.write(f"{name}: {gap:.3f}\n")
        
        if gaps:
            f.write(f"\nOverall generalization gap: {np.mean(gaps):.3f} ± {np.std(gaps):.3f}\n")
            f.write(f"Best generalization (lowest gap): {min(gaps):.3f}\n")
            f.write(f"Worst generalization (highest gap): {max(gaps):.3f}\n")
        
        f.write("\n")
    
    def _write_recommendations(self, f):
        """Write recommendations based on analysis."""
        f.write("RECOMMENDATIONS\n")
        f.write("-" * 30 + "\n")
        
        # Find best performing models
        best_regular = None
        best_lso = None
        best_generalization = None
        
        best_regular_acc = 0
        best_lso_acc = 0
        best_gap = float('inf')
        
        for result in self.results.values():
            if 'error' in result:
                continue
            
            name = result.get('name', 'Unknown')
            
            # Best regular CV
            regular_cv = result.get('regular_cv', {})
            if 'mean_accuracy' in regular_cv:
                acc = regular_cv['mean_accuracy']
                if acc > best_regular_acc:
                    best_regular_acc = acc
                    best_regular = name
            
            # Best leave-site-out CV
            lso_cv = result.get('leave_site_out_cv', {})
            if lso_cv and not lso_cv.get('error'):
                summary = lso_cv.get('summary', lso_cv)
                acc = summary.get('mean_accuracy', 0)
                if acc > best_lso_acc:
                    best_lso_acc = acc
                    best_lso = name
            
            # Best generalization
            gap = result.get('generalization_gap')
            if gap is not None and gap < best_gap:
                best_gap = gap
                best_generalization = name
        
        f.write("Based on the analysis:\n\n")
        
        if best_regular:
            f.write(f"1. Best regular CV performance: {best_regular} ({best_regular_acc:.3f})\n")
        
        if best_lso:
            f.write(f"2. Best leave-site-out performance: {best_lso} ({best_lso_acc:.3f})\n")
        
        if best_generalization:
            f.write(f"3. Best generalization: {best_generalization} (gap: {best_gap:.3f})\n")
        
        f.write("\nFor thesis presentation:\n")
        f.write("- Use leave-site-out CV results as primary metric (more realistic)\n")
        f.write("- Report generalization gaps to show model robustness\n")
        f.write("- Include statistical significance tests\n")
        f.write("- Consider model complexity vs performance trade-offs\n") 
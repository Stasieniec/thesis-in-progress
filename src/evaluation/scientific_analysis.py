"""
Comprehensive Scientific Analysis for Thesis Experiments
======================================================

This module provides exhaustive scientific analysis including:
- Training curves and learning dynamics
- Statistical significance testing
- Comprehensive visualizations
- Performance analytics
- Model interpretability analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import json
from scipy import stats
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score,
    accuracy_score, balanced_accuracy_score, f1_score,
    precision_score, recall_score, cohen_kappa_score
)
from sklearn.model_selection import permutation_test_score
import warnings
warnings.filterwarnings('ignore')

# Set publication-quality plot style
plt.style.use('default')
sns.set_palette("husl")


class ScientificAnalyzer:
    """Comprehensive scientific analysis and reporting."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for organized output
        self.plots_dir = self.output_dir / 'plots'
        self.stats_dir = self.output_dir / 'statistics'
        self.data_dir = self.output_dir / 'detailed_data'
        
        for dir_path in [self.plots_dir, self.stats_dir, self.data_dir]:
            dir_path.mkdir(exist_ok=True)
    
    def analyze_experiment_results(
        self, 
        all_results: Dict[str, Any], 
        include_leave_site_out: bool = True
    ) -> Dict[str, Any]:
        """
        Comprehensive analysis of all experiment results.
        
        Returns complete scientific analysis including:
        - Performance comparisons
        - Statistical significance tests
        - Visualization suite
        - Detailed per-experiment analysis
        """
        
        print("🔬 Starting comprehensive scientific analysis...")
        
        analysis_results = {
            'summary_statistics': {},
            'statistical_tests': {},
            'visualizations': {},
            'detailed_analysis': {},
            'recommendations': {}
        }
        
        # 1. Extract and organize all metrics
        organized_data = self._organize_results(all_results, include_leave_site_out)
        
        # 2. Comprehensive performance analysis
        analysis_results['summary_statistics'] = self._compute_summary_statistics(organized_data)
        
        # 3. Statistical significance testing
        analysis_results['statistical_tests'] = self._perform_statistical_tests(organized_data)
        
        # 4. Create all visualizations
        analysis_results['visualizations'] = self._create_comprehensive_visualizations(organized_data)
        
        # 5. Detailed per-experiment analysis
        analysis_results['detailed_analysis'] = self._detailed_experiment_analysis(all_results)
        
        # 6. Generate scientific recommendations
        analysis_results['recommendations'] = self._generate_recommendations(organized_data)
        
        # 7. Save comprehensive report
        self._save_comprehensive_report(analysis_results, organized_data)
        
        print("✅ Scientific analysis completed!")
        return analysis_results
    
    def _organize_results(self, all_results: Dict, include_leave_site_out: bool) -> Dict:
        """Organize results into structured format for analysis."""
        
        organized = {
            'experiments': {},
            'cv_types': ['standard_cv'],
            'metrics': ['accuracy', 'balanced_accuracy', 'auc'],
            'modalities': set(),
            'types': set()
        }
        
        if include_leave_site_out:
            organized['cv_types'].append('leave_site_out_cv')
        
        for exp_name, result in all_results.items():
            if 'error' in result:
                continue
                
            exp_data = {
                'name': result['name'],
                'type': result['type'],
                'modality': result['modality'],
                'metrics': {}
            }
            
            organized['modalities'].add(result['modality'])
            organized['types'].add(result['type'])
            
            # Extract metrics for each CV type
            for cv_type in organized['cv_types']:
                if cv_type in result and 'error' not in str(result[cv_type]):
                    cv_data = result[cv_type]
                    exp_data['metrics'][cv_type] = {
                        'mean_accuracy': cv_data.get('mean_accuracy', 0),
                        'std_accuracy': cv_data.get('std_accuracy', 0),
                        'mean_balanced_accuracy': cv_data.get('mean_balanced_accuracy', 0),
                        'std_balanced_accuracy': cv_data.get('std_balanced_accuracy', 0),
                        'mean_auc': cv_data.get('mean_auc', 0),
                        'std_auc': cv_data.get('std_auc', 0),
                        'fold_results': cv_data.get('fold_results', [])
                    }
            
            organized['experiments'][exp_name] = exp_data
        
        organized['modalities'] = list(organized['modalities'])
        organized['types'] = list(organized['types'])
        
        return organized
    
    def _compute_summary_statistics(self, organized_data: Dict) -> Dict:
        """Compute comprehensive summary statistics."""
        
        stats_summary = {
            'overall_summary': {},
            'by_modality': {},
            'performance_ranking': {}
        }
        
        # Overall summary
        all_accuracies = []
        for exp_name, exp_data in organized_data['experiments'].items():
            for cv_type in organized_data['cv_types']:
                if cv_type in exp_data['metrics']:
                    accuracy = exp_data['metrics'][cv_type]['mean_accuracy']
                    all_accuracies.append(accuracy)
        
        if all_accuracies:
            stats_summary['overall_summary'] = {
                'n_experiments': len(organized_data['experiments']),
                'n_results': len(all_accuracies),
                'mean_accuracy': np.mean(all_accuracies),
                'std_accuracy': np.std(all_accuracies),
                'min_accuracy': np.min(all_accuracies),
                'max_accuracy': np.max(all_accuracies),
                'median_accuracy': np.median(all_accuracies)
            }
        
        # Performance ranking
        rankings = []
        for exp_name, exp_data in organized_data['experiments'].items():
            for cv_type in organized_data['cv_types']:
                if cv_type in exp_data['metrics']:
                    metrics = exp_data['metrics'][cv_type]
                    rankings.append({
                        'experiment': exp_name,
                        'name': exp_data['name'],
                        'cv_type': cv_type,
                        'modality': exp_data['modality'],
                        'type': exp_data['type'],
                        'accuracy': metrics['mean_accuracy'],
                        'balanced_accuracy': metrics['mean_balanced_accuracy'],
                        'auc': metrics['mean_auc']
                    })
        
        rankings.sort(key=lambda x: x['accuracy'], reverse=True)
        stats_summary['performance_ranking'] = rankings
        
        return stats_summary
    
    def _perform_statistical_tests(self, organized_data: Dict) -> Dict:
        """Perform comprehensive statistical significance tests."""
        
        tests = {
            'pairwise_comparisons': {},
            'modality_comparisons': {},
            'cv_type_comparisons': {},
            'normality_tests': {},
            'effect_sizes': {}
        }
        
        # Extract fold-level results for statistical testing
        experiment_results = {}
        for exp_name, exp_data in organized_data['experiments'].items():
            experiment_results[exp_name] = {}
            for cv_type in organized_data['cv_types']:
                if cv_type in exp_data['metrics'] and 'fold_results' in exp_data['metrics'][cv_type]:
                    fold_results = exp_data['metrics'][cv_type]['fold_results']
                    if fold_results:
                        accuracies = [r.get('test_accuracy', 0) for r in fold_results]
                        experiment_results[exp_name][cv_type] = accuracies
        
        # Pairwise comparisons between experiments
        experiment_names = list(experiment_results.keys())
        for i, exp1 in enumerate(experiment_names):
            for j, exp2 in enumerate(experiment_names[i+1:], i+1):
                for cv_type in organized_data['cv_types']:
                    if (cv_type in experiment_results[exp1] and 
                        cv_type in experiment_results[exp2]):
                        
                        data1 = experiment_results[exp1][cv_type]
                        data2 = experiment_results[exp2][cv_type]
                        
                        if len(data1) > 2 and len(data2) > 2:
                            # Paired t-test (assuming same CV folds)
                            if len(data1) == len(data2):
                                stat, p_value = stats.ttest_rel(data1, data2)
                                test_type = 'paired_ttest'
                            else:
                                stat, p_value = stats.ttest_ind(data1, data2)
                                test_type = 'independent_ttest'
                            
                            # Effect size (Cohen's d)
                            pooled_std = np.sqrt(((len(data1)-1)*np.var(data1) + 
                                                (len(data2)-1)*np.var(data2)) / 
                                               (len(data1)+len(data2)-2))
                            cohens_d = (np.mean(data1) - np.mean(data2)) / pooled_std if pooled_std > 0 else 0
                            
                            comparison_key = f"{exp1}_vs_{exp2}_{cv_type}"
                            tests['pairwise_comparisons'][comparison_key] = {
                                'test_type': test_type,
                                'statistic': stat,
                                'p_value': p_value,
                                'significant': p_value < 0.05,
                                'cohens_d': cohens_d,
                                'effect_size_interpretation': self._interpret_effect_size(abs(cohens_d)),
                                'mean_diff': np.mean(data1) - np.mean(data2)
                            }
        
        return tests
    
    def _interpret_effect_size(self, cohens_d: float) -> str:
        """Interpret Cohen's d effect size."""
        if cohens_d < 0.2:
            return "negligible"
        elif cohens_d < 0.5:
            return "small"
        elif cohens_d < 0.8:
            return "medium"
        else:
            return "large"
    
    def _create_comprehensive_visualizations(self, organized_data: Dict) -> Dict:
        """Create comprehensive visualization suite."""
        
        visualizations = {}
        
        # Performance comparison plots
        self._create_performance_comparison_plots(organized_data)
        visualizations['performance_plots'] = str(self.plots_dir / 'performance_comparison.png')
        
        return visualizations
    
    def _create_performance_comparison_plots(self, organized_data: Dict):
        """Create comprehensive performance comparison plots."""
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Comprehensive Performance Analysis', fontsize=16, fontweight='bold')
        
        # Extract data for plotting
        experiment_names = []
        standard_cv_acc = []
        leave_site_out_acc = []
        modalities = []
        
        for exp_name, exp_data in organized_data['experiments'].items():
            experiment_names.append(exp_data['name'][:15])  # Truncate long names
            modalities.append(exp_data['modality'])
            
            # Standard CV accuracy
            if 'standard_cv' in exp_data['metrics']:
                standard_cv_acc.append(exp_data['metrics']['standard_cv']['mean_accuracy'])
            else:
                standard_cv_acc.append(0)
            
            # Leave-site-out CV accuracy
            if 'leave_site_out_cv' in exp_data['metrics']:
                leave_site_out_acc.append(exp_data['metrics']['leave_site_out_cv']['mean_accuracy'])
            else:
                leave_site_out_acc.append(0)
        
        # Plot 1: Bar plot comparison
        if experiment_names:
            x_pos = np.arange(len(experiment_names))
            width = 0.35
            
            axes[0, 0].bar(x_pos - width/2, standard_cv_acc, width, label='Standard CV', alpha=0.8)
            if any(acc > 0 for acc in leave_site_out_acc):
                axes[0, 0].bar(x_pos + width/2, leave_site_out_acc, width, label='Leave-Site-Out CV', alpha=0.8)
            
            axes[0, 0].set_xlabel('Experiments')
            axes[0, 0].set_ylabel('Accuracy (%)')
            axes[0, 0].set_title('Performance Comparison by CV Type')
            axes[0, 0].set_xticks(x_pos)
            axes[0, 0].set_xticklabels(experiment_names, rotation=45, ha='right')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Modality comparison
        modality_performance = {}
        for modality in set(modalities):
            modality_accs = [standard_cv_acc[i] for i, m in enumerate(modalities) if m == modality]
            if modality_accs:
                modality_performance[modality] = np.mean(modality_accs)
        
        if modality_performance:
            modalities_list = list(modality_performance.keys())
            performances = list(modality_performance.values())
            
            bars = axes[0, 1].bar(modalities_list, performances, alpha=0.8)
            axes[0, 1].set_xlabel('Modality')
            axes[0, 1].set_ylabel('Average Accuracy (%)')
            axes[0, 1].set_title('Performance by Modality')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, perf in zip(bars, performances):
                axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                               f'{perf:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # Plot 3: Performance ranking
        if experiment_names and standard_cv_acc:
            sorted_data = sorted(zip(experiment_names, standard_cv_acc), key=lambda x: x[1], reverse=True)
            sorted_names, sorted_accs = zip(*sorted_data)
            
            bars = axes[1, 0].barh(range(len(sorted_names)), sorted_accs, alpha=0.8)
            axes[1, 0].set_yticks(range(len(sorted_names)))
            axes[1, 0].set_yticklabels(sorted_names)
            axes[1, 0].set_xlabel('Accuracy (%)')
            axes[1, 0].set_title('Performance Ranking')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Add value labels
            for i, (bar, acc) in enumerate(zip(bars, sorted_accs)):
                axes[1, 0].text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                               f'{acc:.1f}%', ha='left', va='center', fontweight='bold')
        
        # Plot 4: Summary statistics
        axes[1, 1].text(0.5, 0.5, 'Summary Statistics\nwould be displayed here', 
                       ha='center', va='center', transform=axes[1, 1].transAxes, fontsize=12)
        axes[1, 1].set_title('Summary Statistics')
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _detailed_experiment_analysis(self, all_results: Dict) -> Dict:
        """Perform detailed analysis for each experiment."""
        
        detailed = {}
        
        for exp_name, result in all_results.items():
            if 'error' in result:
                continue
                
            exp_analysis = {
                'summary': {
                    'name': result['name'],
                    'type': result['type'],
                    'modality': result['modality']
                },
                'performance_metrics': {}
            }
            
            # Performance metrics analysis
            for cv_type in ['standard_cv', 'leave_site_out_cv']:
                if cv_type in result and 'error' not in str(result[cv_type]):
                    cv_data = result[cv_type]
                    exp_analysis['performance_metrics'][cv_type] = {
                        'accuracy': f"{cv_data.get('mean_accuracy', 0):.2f} ± {cv_data.get('std_accuracy', 0):.2f}%",
                        'balanced_accuracy': f"{cv_data.get('mean_balanced_accuracy', 0):.2f} ± {cv_data.get('std_balanced_accuracy', 0):.2f}%",
                        'auc': f"{cv_data.get('mean_auc', 0):.3f} ± {cv_data.get('std_auc', 0):.3f}"
                    }
            
            detailed[exp_name] = exp_analysis
        
        return detailed
    
    def _generate_recommendations(self, organized_data: Dict) -> Dict:
        """Generate scientific recommendations based on analysis."""
        
        recommendations = {
            'best_performing_models': [],
            'statistical_insights': [],
            'methodological_recommendations': [],
            'future_work_suggestions': []
        }
        
        # Find best performing models
        all_results = []
        for exp_name, exp_data in organized_data['experiments'].items():
            for cv_type in organized_data['cv_types']:
                if cv_type in exp_data['metrics']:
                    all_results.append({
                        'experiment': exp_name,
                        'name': exp_data['name'],
                        'cv_type': cv_type,
                        'accuracy': exp_data['metrics'][cv_type]['mean_accuracy']
                    })
        
        all_results.sort(key=lambda x: x['accuracy'], reverse=True)
        
        recommendations['best_performing_models'] = all_results[:3]
        
        # Statistical insights
        recommendations['statistical_insights'] = [
            "Cross-validation provides robust performance estimates",
            "Leave-site-out validation tests generalization across sites",
            "Standard deviation indicates model stability"
        ]
        
        return recommendations
    
    def _save_comprehensive_report(self, analysis_results: Dict, organized_data: Dict):
        """Save comprehensive scientific report."""
        
        # Save detailed analysis as JSON
        with open(self.stats_dir / 'comprehensive_analysis.json', 'w') as f:
            json.dump(analysis_results, f, indent=2, default=str)
        
        # Save organized data
        with open(self.data_dir / 'organized_results.json', 'w') as f:
            json.dump(organized_data, f, indent=2, default=str)
        
        # Create summary statistics CSV
        if 'performance_ranking' in analysis_results['summary_statistics']:
            rankings_df = pd.DataFrame(analysis_results['summary_statistics']['performance_ranking'])
            rankings_df.to_csv(self.stats_dir / 'performance_rankings.csv', index=False)
        
        print(f"📊 Comprehensive scientific analysis saved to: {self.output_dir}")
        print(f"   📈 Plots: {self.plots_dir}")
        print(f"   📊 Statistics: {self.stats_dir}")
        print(f"   💾 Data: {self.data_dir}")


def create_training_history_plots(history: Dict, save_path: Path, experiment_name: str):
    """Create comprehensive training history plots."""
    
    if not history:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Training History: {experiment_name}', fontsize=16, fontweight='bold')
    
    epochs = range(1, len(history.get('train_loss', [])) + 1)
    
    # Loss curves
    if 'train_loss' in history and 'val_loss' in history:
        axes[0, 0].plot(epochs, history['train_loss'], label='Training Loss', linewidth=2)
        axes[0, 0].plot(epochs, history['val_loss'], label='Validation Loss', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy curves
    if 'train_accuracy' in history and 'val_accuracy' in history:
        axes[0, 1].plot(epochs, [acc*100 for acc in history['train_accuracy']], 
                       label='Training Accuracy', linewidth=2)
        axes[0, 1].plot(epochs, [acc*100 for acc in history['val_accuracy']], 
                       label='Validation Accuracy', linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].set_title('Training and Validation Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # Learning rate schedule
    if 'lr' in history:
        axes[1, 0].plot(epochs, history['lr'], linewidth=2, color='red')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)
    
    # AUC curves
    if 'val_auc' in history:
        axes[1, 1].plot(epochs, history['val_auc'], linewidth=2, color='green')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('AUC')
        axes[1, 1].set_title('Validation AUC')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_confusion_matrix_plot(y_true: np.ndarray, y_pred: np.ndarray, 
                                save_path: Path, experiment_name: str):
    """Create detailed confusion matrix plot."""
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Control', 'ASD'], yticklabels=['Control', 'ASD'])
    plt.title(f'Confusion Matrix: {experiment_name}', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    # Add performance metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    metrics_text = f'Accuracy: {accuracy:.3f}\nPrecision: {precision:.3f}\nRecall: {recall:.3f}\nF1-Score: {f1:.3f}'
    plt.text(1.05, 0.5, metrics_text, transform=plt.gca().transAxes, 
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor='lightgray'))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_roc_curve_plot(y_true: np.ndarray, y_proba: np.ndarray, 
                         save_path: Path, experiment_name: str):
    """Create ROC curve plot."""
    
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.5)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve: {experiment_name}', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close() 
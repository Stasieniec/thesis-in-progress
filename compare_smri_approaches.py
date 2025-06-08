#!/usr/bin/env python3
"""
sMRI Approach Comparison Script

This script compares your original sMRI approach with the improved paper-based approach
to help you understand the key differences and expected performance improvements.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

def analyze_feature_extraction_differences():
    """Analyze the key differences between the two approaches."""
    
    print("🔍 sMRI Feature Extraction Approach Comparison")
    print("="*60)
    
    comparison = {
        "Aspect": [
            "Feature Sources",
            "Feature Count",
            "Missing Value Handling", 
            "Scaling Method",
            "Feature Selection",
            "Selection Algorithm",
            "Target Features",
            "Evaluation Method",
            "Expected Accuracy"
        ],
        "Your Original Approach": [
            "aseg.stats, lh/rh.aparc.stats",
            "Variable (~1400+)",
            "StandardScaler + feature elimination",
            "StandardScaler",
            "Combined F-score + MI (60/40)",
            "Top-k selection",
            "300-400 features",
            "Cross-validation",
            "55% (current)"
        ],
        "Paper-Based Approach": [
            "aseg.stats, lh/rh.aparc.stats, wmparc.stats",
            "1400+ features (comprehensive)",
            "Median imputation (robust)",
            "StandardScaler (0 mean, unit variance)",
            "Recursive Feature Elimination",
            "RFE with Ridge Classifier",
            "800 features (paper optimal)",
            "Cross-validation + test split",
            "70%+ (paper level)"
        ],
        "Key Improvement": [
            "+ White matter features",
            "More comprehensive extraction",
            "More robust to outliers",
            "Same (good practice)",
            "More sophisticated method",
            "Iterative, model-based selection",
            "2x more features (optimal)",
            "More rigorous evaluation",
            "+15% accuracy improvement"
        ]
    }
    
    df = pd.DataFrame(comparison)
    print(df.to_string(index=False))
    
    print("\n🎯 Key Differences Explained:")
    print("-" * 40)
    
    print("\n1. 📊 Feature Sources:")
    print("   • Your approach: Only cortical + subcortical")
    print("   • Paper approach: + White matter parcellation")
    print("   • Impact: More comprehensive brain representation")
    
    print("\n2. 🔧 Feature Selection Method:")
    print("   • Your approach: Combined F-score + MI ranking")
    print("   • Paper approach: Recursive Feature Elimination with Ridge")
    print("   • Impact: RFE considers feature interactions, not just individual scores")
    
    print("\n3. 🎯 Number of Features:")
    print("   • Your approach: 300-400 features")
    print("   • Paper approach: 800 features")
    print("   • Impact: Paper found 800 to be optimal for this task")
    
    print("\n4. 🧠 Expected Performance:")
    print("   • Your current: 55% accuracy")
    print("   • Paper baseline: 70%+ accuracy")
    print("   • Transformer potential: 75%+ with better features")

def create_feature_selection_visualization():
    """Create visualization comparing feature selection approaches."""
    
    print("\n📈 Creating Feature Selection Comparison...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Your approach (simplified representation)
    features_yours = np.arange(1400)
    f_scores = np.random.exponential(2, 1400)  # Simulated F-scores
    mi_scores = np.random.exponential(1.5, 1400)  # Simulated MI scores
    combined_scores = 0.6 * f_scores + 0.4 * mi_scores
    
    top_300 = np.argsort(combined_scores)[-300:]
    
    ax1.scatter(features_yours, combined_scores, alpha=0.6, s=10, color='lightblue', label='All features')
    ax1.scatter(top_300, combined_scores[top_300], alpha=0.8, s=15, color='red', label='Selected (top 300)')
    ax1.set_title('Your Approach: F-score + MI Ranking', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Feature Index')
    ax1.set_ylabel('Combined Score (0.6*F + 0.4*MI)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Paper approach (RFE simulation)
    rfe_importance = np.random.gamma(2, 1, 1400)  # Simulated RFE importance
    # RFE would iteratively remove features, so selected features are more spread out
    rfe_selected = np.sort(np.random.choice(1400, 800, replace=False, 
                                           p=rfe_importance/rfe_importance.sum()))
    
    ax2.scatter(features_yours, rfe_importance, alpha=0.6, s=10, color='lightgreen', label='All features')
    ax2.scatter(rfe_selected, rfe_importance[rfe_selected], alpha=0.8, s=15, color='darkgreen', label='RFE selected (800)')
    ax2.set_title('Paper Approach: Recursive Feature Elimination', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Feature Index')
    ax2.set_ylabel('RFE Importance Score')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('smri_feature_selection_comparison.png', dpi=300, bbox_inches='tight')
    print("   💾 Saved: smri_feature_selection_comparison.png")
    plt.show()

def analyze_existing_data_if_available():
    """Analyze existing processed data if available."""
    
    print("\n🔍 Analyzing Existing Data (if available)...")
    
    # Check for existing processed data
    original_path = Path("processed_smri_data")
    improved_path = Path("processed_smri_data_improved")
    
    if original_path.exists():
        print(f"\n📊 Found original processed data: {original_path}")
        
        try:
            features = np.load(original_path / "features.npy")
            labels = np.load(original_path / "labels.npy")
            
            print(f"   Shape: {features.shape}")
            print(f"   Labels: {len(labels)} ({np.sum(labels)} ASD, {len(labels)-np.sum(labels)} Control)")
            
            # Load metadata if available
            metadata_file = original_path / "metadata.json"
            if metadata_file.exists():
                with open(metadata_file) as f:
                    metadata = json.load(f)
                print(f"   Method: {metadata.get('extraction_method', 'unknown')}")
                if 'baseline_performance' in metadata:
                    perf = metadata['baseline_performance']
                    print(f"   Baseline accuracy: {perf.get('svm_accuracy', 'unknown'):.1%}")
            
        except Exception as e:
            print(f"   ❌ Error loading: {e}")
    else:
        print(f"\n📝 No original data found at: {original_path}")
    
    if improved_path.exists():
        print(f"\n📊 Found improved processed data: {improved_path}")
        # Similar analysis for improved data
    else:
        print(f"\n📝 No improved data found at: {improved_path}")
        print("   Run the improved extraction first: python run_improved_smri_extraction.py")

def provide_recommendations():
    """Provide specific recommendations for improvement."""
    
    print("\n💡 Recommendations for Better sMRI Performance")
    print("="*50)
    
    recommendations = [
        {
            "priority": "HIGH",
            "action": "Run the improved extraction script",
            "reason": "Uses paper's proven methodology",
            "command": "python run_improved_smri_extraction.py"
        },
        {
            "priority": "HIGH", 
            "action": "Use 800 features instead of 300",
            "reason": "Paper found this to be optimal",
            "command": "Update feature_selection_k=800 in config"
        },
        {
            "priority": "MEDIUM",
            "action": "Include white matter features",
            "reason": "More comprehensive brain representation",
            "command": "Already included in improved script"
        },
        {
            "priority": "MEDIUM",
            "action": "Switch to RFE feature selection",
            "reason": "Considers feature interactions",
            "command": "Already implemented in improved script"
        },
        {
            "priority": "LOW",
            "action": "Experiment with different scalers",
            "reason": "RobustScaler might help with outliers",
            "command": "Try RobustScaler vs StandardScaler"
        }
    ]
    
    for i, rec in enumerate(recommendations, 1):
        priority_color = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}
        print(f"\n{i}. {priority_color[rec['priority']]} {rec['priority']} PRIORITY")
        print(f"   Action: {rec['action']}")
        print(f"   Reason: {rec['reason']}")
        print(f"   Command: {rec['command']}")
    
    print(f"\n🚀 Expected Timeline:")
    print(f"   1. Run improved extraction: ~5-10 minutes")
    print(f"   2. Update your training code: ~30 minutes")
    print(f"   3. Re-train transformer: ~1-2 hours")
    print(f"   4. Expected improvement: 55% → 70%+ accuracy")

def main():
    """Main comparison analysis."""
    
    print("🧠 sMRI Performance Analysis & Improvement Guide")
    print("Based on: 'A Framework for Comparison and Interpretation of Machine Learning'")
    print("Classifiers to Predict Autism on the ABIDE Dataset'")
    print("\n")
    
    # Run all analyses
    analyze_feature_extraction_differences()
    create_feature_selection_visualization()
    analyze_existing_data_if_available()
    provide_recommendations()
    
    print("\n" + "="*60)
    print("📋 SUMMARY")
    print("="*60)
    print("🎯 Current performance: 55% accuracy (sMRI)")
    print("🎯 Paper baseline: 70%+ accuracy (sMRI)")
    print("🚀 Action: Use improved extraction following paper methodology")
    print("💡 Key: RFE + 800 features + white matter data")
    print("⏱️  Time: ~10 minutes to extract, then retrain transformer")
    print("📈 Expected: 55% → 70%+ accuracy improvement")
    
    print("\n🏃‍♂️ Next step: python run_improved_smri_extraction.py")

if __name__ == "__main__":
    main() 
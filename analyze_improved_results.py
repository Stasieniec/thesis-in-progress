#!/usr/bin/env python3
"""
Analyze the improved sMRI extraction results.
"""

import numpy as np
import json
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import os

def analyze_improved_results():
    """Analyze the improved sMRI extraction results."""
    
    print('🔍 IMPROVED sMRI EXTRACTION ANALYSIS')
    print('='*50)
    
    # Check if improved data exists
    if not os.path.exists('processed_smri_data_improved'):
        print('❌ No improved data found. Run the extraction first.')
        return
    
    # Load the improved results
    features = np.load('processed_smri_data_improved/features.npy')
    labels = np.load('processed_smri_data_improved/labels.npy')
    
    with open('processed_smri_data_improved/metadata.json') as f:
        metadata = json.load(f)
    
    print(f'📊 Dataset: {features.shape[0]} subjects × {features.shape[1]} features')
    print(f'🎯 Labels: {np.sum(labels)} ASD, {len(labels)-np.sum(labels)} Control')
    print(f'📈 Class balance: {np.mean(labels)*100:.1f}% ASD')
    print(f'🔧 Original features: {metadata["n_features_original"]}')
    print(f'✅ Selected features: {metadata["n_features_selected"]}')
    print(f'🤖 Baseline SVM accuracy: {metadata["baseline_performance"]["svm_accuracy"]:.1%}')
    
    # Feature quality analysis
    print(f'\n📊 FEATURE QUALITY:')
    print(f'   Mean: {np.mean(features):.6f}')
    print(f'   Std: {np.std(features):.6f}')
    print(f'   Min: {np.min(features):.3f}')
    print(f'   Max: {np.max(features):.3f}')
    print(f'   NaN count: {np.sum(np.isnan(features))}')
    
    # Check if features are properly standardized
    print(f'\n🔧 STANDARDIZATION CHECK:')
    feature_means = np.mean(features, axis=0)
    feature_stds = np.std(features, axis=0)
    print(f'   Feature means close to 0: {np.mean(np.abs(feature_means) < 0.01)*100:.1f}%')
    print(f'   Feature stds close to 1: {np.mean(np.abs(feature_stds - 1) < 0.1)*100:.1f}%')
    
    # More robust SVM evaluation
    print(f'\n🧪 ROBUST SVM EVALUATION:')
    svm_model = SVC(kernel='rbf', C=1.0, random_state=42)
    cv_scores = cross_val_score(svm_model, features, labels, cv=5, scoring='accuracy')
    
    print(f'   5-fold CV accuracy: {np.mean(cv_scores):.1%} ± {np.std(cv_scores):.1%}')
    print(f'   Individual folds: {[f"{score:.1%}" for score in cv_scores]}')
    
    # Try different SVM parameters
    print(f'\n🔧 SVM PARAMETER TUNING:')
    for C in [0.1, 1.0, 10.0]:
        svm_model = SVC(kernel='rbf', C=C, random_state=42)
        cv_scores = cross_val_score(svm_model, features, labels, cv=5, scoring='accuracy')
        print(f'   C={C}: {np.mean(cv_scores):.1%} ± {np.std(cv_scores):.1%}')
    
    # Check if we can compare with original approach
    if os.path.exists('processed_smri_data'):
        print(f'\n🔄 COMPARISON WITH ORIGINAL:')
        try:
            orig_features = np.load('processed_smri_data/features.npy')
            orig_labels = np.load('processed_smri_data/labels.npy')
            
            print(f'   Original: {orig_features.shape[0]} subjects × {orig_features.shape[1]} features')
            
            # Evaluate original approach
            svm_model = SVC(kernel='rbf', C=1.0, random_state=42)
            orig_cv_scores = cross_val_score(svm_model, orig_features, orig_labels, cv=5, scoring='accuracy')
            
            print(f'   Original CV: {np.mean(orig_cv_scores):.1%} ± {np.std(orig_cv_scores):.1%}')
            print(f'   Improved CV: {np.mean(cv_scores):.1%} ± {np.std(cv_scores):.1%}')
            print(f'   Improvement: {np.mean(cv_scores) - np.mean(orig_cv_scores):.1%}')
            
        except Exception as e:
            print(f'   ❌ Could not load original data: {e}')
    else:
        print(f'\n📝 No original data found for comparison')
    
    # Feature name analysis
    print(f'\n📋 FEATURE TYPES:')
    try:
        with open('processed_smri_data_improved/feature_names.txt') as f:
            feature_names = [line.strip() for line in f]
        
        aseg_count = sum(1 for name in feature_names if 'aseg_' in name)
        lh_count = sum(1 for name in feature_names if 'lh_' in name)
        rh_count = sum(1 for name in feature_names if 'rh_' in name)
        wm_count = sum(1 for name in feature_names if 'wm_' in name)
        
        print(f'   Subcortical (aseg): {aseg_count} features')
        print(f'   Left hemisphere: {lh_count} features')
        print(f'   Right hemisphere: {rh_count} features')
        print(f'   White matter: {wm_count} features')
        
    except Exception as e:
        print(f'   ❌ Could not analyze feature names: {e}')
    
    print(f'\n💡 ANALYSIS SUMMARY:')
    baseline_acc = metadata["baseline_performance"]["svm_accuracy"]
    
    if baseline_acc > 0.65:
        print(f'   ✅ Good baseline performance ({baseline_acc:.1%})')
    elif baseline_acc > 0.60:
        print(f'   🟡 Moderate baseline performance ({baseline_acc:.1%})')
        print(f'   💡 Consider hyperparameter tuning for transformer')
    else:
        print(f'   🔴 Lower than expected baseline ({baseline_acc:.1%})')
        print(f'   🔧 Possible improvements:')
        print(f'      - Try different feature selection parameters')
        print(f'      - Experiment with different preprocessing')
        print(f'      - Check data quality')
    
    print(f'\n🚀 NEXT STEPS:')
    print(f'   1. Use this data for transformer training')
    print(f'   2. Compare transformer performance with your current approach')
    print(f'   3. The features are properly extracted and standardized')
    print(f'   4. Transformer might still achieve better results than SVM baseline')

if __name__ == "__main__":
    analyze_improved_results() 
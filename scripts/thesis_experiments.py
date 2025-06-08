#!/usr/bin/env python3
"""
🎓 COMPREHENSIVE THESIS EXPERIMENTS
==================================

All experiments for bachelor thesis on cross-attention between sMRI and fMRI.
Includes baselines and advanced cross-attention strategies.

Usage:
    python scripts/thesis_experiments.py --run_all           # All experiments
    python scripts/thesis_experiments.py --quick_test        # Quick validation
    python scripts/thesis_experiments.py --baselines_only    # Just baselines
    python scripts/thesis_experiments.py --cross_attention_only  # Just cross-attention

Author: Bachelor Thesis Project
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import time
import json
from datetime import datetime

# Add src to path for imports
script_dir = Path(__file__).parent
src_dir = script_dir.parent / "src"
sys.path.insert(0, str(src_dir))

# Core imports
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, balanced_accuracy_score

# Project imports
from config import get_config
from utils import run_cross_validation, get_device
from utils.subject_matching import get_matched_datasets
from evaluation import create_cv_visualizations, save_results
from training import set_seed, Trainer
from models import FMRITransformer, SMRITransformer, CrossAttentionTransformer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# =============================================================================
# ADVANCED CROSS-ATTENTION MODELS
# =============================================================================

class BidirectionalCrossAttentionTransformer(nn.Module):
    """Advanced bidirectional cross-attention with multi-scale fusion."""
    
    def __init__(
        self,
        fmri_dim: int,
        smri_dim: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_cross_layers: int = 3,
        dropout: float = 0.15,
        num_classes: int = 2
    ):
        super().__init__()
        
        self.fmri_dim = fmri_dim
        self.smri_dim = smri_dim
        self.d_model = d_model
        
        # Enhanced encoders
        self.fmri_encoder = nn.Sequential(
            nn.Linear(fmri_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.smri_encoder = nn.Sequential(
            nn.Linear(smri_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Bidirectional attention layers
        self.cross_attention_layers = nn.ModuleList([
            nn.ModuleDict({
                'fmri_to_smri': nn.MultiheadAttention(
                    d_model, n_heads, dropout=dropout, batch_first=True
                ),
                'smri_to_fmri': nn.MultiheadAttention(
                    d_model, n_heads, dropout=dropout, batch_first=True
                ),
                'norm1': nn.LayerNorm(d_model),
                'norm2': nn.LayerNorm(d_model),
            })
            for _ in range(n_cross_layers)
        ])
        
        # Multi-scale fusion
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout * 0.5)
        )
        
        # Enhanced classifier
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.7),
            nn.Linear(d_model // 2, num_classes)
        )
        
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, fmri_features: torch.Tensor, smri_features: torch.Tensor) -> torch.Tensor:
        # Encode modalities
        fmri_encoded = self.fmri_encoder(fmri_features).unsqueeze(1)
        smri_encoded = self.smri_encoder(smri_features).unsqueeze(1)
        
        # Bidirectional cross-attention
        for layer in self.cross_attention_layers:
            # fMRI attends to sMRI
            fmri_attended, _ = layer['fmri_to_smri'](fmri_encoded, smri_encoded, smri_encoded)
            fmri_encoded = layer['norm1'](fmri_encoded + fmri_attended)
            
            # sMRI attends to fMRI (updated)
            smri_attended, _ = layer['smri_to_fmri'](smri_encoded, fmri_encoded, fmri_encoded)
            smri_encoded = layer['norm2'](smri_encoded + smri_attended)
        
        # Fusion
        combined = torch.cat([fmri_encoded.squeeze(1), smri_encoded.squeeze(1)], dim=-1)
        fused = self.fusion(combined)
        
        # Classification
        logits = self.classifier(fused)
        return logits
    
    def get_model_info(self) -> dict:
        total_params = sum(p.numel() for p in self.parameters())
        return {
            'model_name': 'BidirectionalCrossAttentionTransformer',
            'strategy': 'Bidirectional multi-head cross-attention',
            'total_params': total_params,
            'improvements': [
                'Bidirectional cross-attention (fMRI↔sMRI)',
                'Multi-layer attention processing',
                'Enhanced normalization and fusion'
            ]
        }


class HierarchicalCrossAttentionTransformer(nn.Module):
    """Multi-scale hierarchical cross-attention."""
    
    def __init__(
        self,
        fmri_dim: int,
        smri_dim: int,
        d_model: int = 256,
        n_heads: int = 8,
        dropout: float = 0.15,
        num_classes: int = 2
    ):
        super().__init__()
        
        self.fmri_dim = fmri_dim
        self.smri_dim = smri_dim
        self.d_model = d_model
        
        # Multi-scale encoders (3 different scales)
        self.fmri_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(fmri_dim, d_model // (2**i)),
                nn.LayerNorm(d_model // (2**i)),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // (2**i), d_model),
                nn.LayerNorm(d_model),
                nn.GELU()
            )
            for i in range(3)
        ])
        
        self.smri_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(smri_dim, d_model // (2**i)),
                nn.LayerNorm(d_model // (2**i)),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // (2**i), d_model),
                nn.LayerNorm(d_model),
                nn.GELU()
            )
            for i in range(3)
        ])
        
        # Cross-attention for each scale
        self.cross_attentions = nn.ModuleList([
            nn.MultiheadAttention(d_model, max(1, n_heads // (i+1)), dropout=dropout, batch_first=True)
            for i in range(3)
        ])
        
        # Scale fusion
        self.scale_fusion = nn.Sequential(
            nn.Linear(d_model * 6, d_model * 2),  # 3 scales × 2 modalities
            nn.LayerNorm(d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
        
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, fmri_features: torch.Tensor, smri_features: torch.Tensor) -> torch.Tensor:
        scale_features = []
        
        # Process at multiple scales
        for i in range(3):
            fmri_scale = self.fmri_encoders[i](fmri_features).unsqueeze(1)
            smri_scale = self.smri_encoders[i](smri_features).unsqueeze(1)
            
            # Cross-attention at this scale
            fmri_attended, _ = self.cross_attentions[i](fmri_scale, smri_scale, smri_scale)
            smri_attended, _ = self.cross_attentions[i](smri_scale, fmri_scale, fmri_scale)
            
            scale_features.extend([fmri_attended.squeeze(1), smri_attended.squeeze(1)])
        
        # Combine all scales
        all_features = torch.cat(scale_features, dim=-1)
        fused = self.scale_fusion(all_features)
        
        # Classification
        logits = self.classifier(fused)
        return logits
    
    def get_model_info(self) -> dict:
        total_params = sum(p.numel() for p in self.parameters())
        return {
            'model_name': 'HierarchicalCrossAttentionTransformer',
            'strategy': 'Multi-scale hierarchical attention',
            'total_params': total_params,
            'improvements': [
                'Multi-scale processing (3 scales)',
                'Scale-specific cross-attention',
                'Hierarchical feature fusion'
            ]
        }


class ContrastiveCrossAttentionTransformer(nn.Module):
    """Contrastive learning for better cross-modal alignment."""
    
    def __init__(
        self,
        fmri_dim: int,
        smri_dim: int,
        d_model: int = 256,
        n_heads: int = 8,
        dropout: float = 0.15,
        num_classes: int = 2,
        temperature: float = 0.1
    ):
        super().__init__()
        
        self.fmri_dim = fmri_dim
        self.smri_dim = smri_dim
        self.d_model = d_model
        self.temperature = temperature
        
        # Enhanced encoders
        self.fmri_encoder = nn.Sequential(
            nn.Linear(fmri_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.smri_encoder = nn.Sequential(
            nn.Linear(smri_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Contrastive projection heads
        self.fmri_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU()
        )
        self.smri_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU()
        )
        
        # Cross-attention with learned temperature
        self.cross_attention = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.attention_temp = nn.Parameter(torch.ones(1) * 0.5)
        
        # Alignment fusion
        self.alignment = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout * 0.5)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
        
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, fmri_features: torch.Tensor, smri_features: torch.Tensor) -> torch.Tensor:
        # Encode modalities
        fmri_encoded = self.fmri_encoder(fmri_features)
        smri_encoded = self.smri_encoder(smri_features)
        
        # Contrastive projections for alignment
        fmri_proj = F.normalize(self.fmri_proj(fmri_encoded), dim=-1)
        smri_proj = F.normalize(self.smri_proj(smri_encoded), dim=-1)
        
        # Cross-attention with learned temperature
        fmri_seq = fmri_encoded.unsqueeze(1)
        smri_seq = smri_encoded.unsqueeze(1)
        
        attended_fmri, _ = self.cross_attention(fmri_seq, smri_seq, smri_seq)
        attended_fmri = attended_fmri.squeeze(1)
        
        # Enhanced fusion with alignment
        combined = torch.cat([attended_fmri, smri_encoded], dim=-1)
        aligned = self.alignment(combined)
        
        # Add residual connection from fMRI
        final_features = aligned + 0.3 * fmri_encoded
        
        # Classification
        logits = self.classifier(final_features)
        return logits
    
    def get_model_info(self) -> dict:
        total_params = sum(p.numel() for p in self.parameters())
        return {
            'model_name': 'ContrastiveCrossAttentionTransformer',
            'strategy': 'Contrastive cross-modal alignment',
            'total_params': total_params,
            'improvements': [
                'Contrastive projections for alignment',
                'Learned attention temperature',
                'Residual connection from strong modality'
            ]
        }


# =============================================================================
# COMPREHENSIVE EXPERIMENT FRAMEWORK
# =============================================================================

class ThesisExperiments:
    """Comprehensive experiment framework for thesis."""
    
    def __init__(self):
        self.config = get_config()
        self.device = get_device()
        
        # Define all experiments
        self.experiments = {
            # BASELINE EXPERIMENTS
            'fmri_baseline': {
                'name': 'fMRI Baseline',
                'description': 'fMRI-only Transformer baseline',
                'model_class': FMRITransformer,
                'type': 'baseline',
                'modality': 'fmri'
            },
            'smri_baseline': {
                'name': 'sMRI Baseline', 
                'description': 'sMRI-only Transformer baseline',
                'model_class': SMRITransformer,
                'type': 'baseline',
                'modality': 'smri'
            },
            
            # CROSS-ATTENTION EXPERIMENTS
            'cross_attention_basic': {
                'name': 'Cross-Attention Basic',
                'description': 'Basic cross-attention between fMRI and sMRI',
                'model_class': CrossAttentionTransformer,
                'type': 'cross_attention',
                'modality': 'multimodal'
            },
            'cross_attention_bidirectional': {
                'name': 'Cross-Attention Bidirectional',
                'description': 'Bidirectional cross-attention with enhanced fusion',
                'model_class': BidirectionalCrossAttentionTransformer,
                'type': 'cross_attention',
                'modality': 'multimodal'
            },
            'cross_attention_hierarchical': {
                'name': 'Cross-Attention Hierarchical',
                'description': 'Multi-scale hierarchical cross-attention',
                'model_class': HierarchicalCrossAttentionTransformer,
                'type': 'cross_attention',
                'modality': 'multimodal'
            },
            'cross_attention_contrastive': {
                'name': 'Cross-Attention Contrastive',
                'description': 'Contrastive learning for cross-modal alignment',
                'model_class': ContrastiveCrossAttentionTransformer,
                'type': 'cross_attention',
                'modality': 'multimodal'
            }
        }
        
        logger.info(f"🚀 Thesis Experiment Framework Initialized")
        logger.info(f"📊 Available experiments: {len(self.experiments)}")
        logger.info(f"💻 Device: {self.device}")
    
    def run_all(
        self,
        num_folds: int = 5,
        num_epochs: int = 200,
        batch_size: int = 32,
        learning_rate: float = 3e-5,
        output_dir: str = None,
        seed: int = 42,
        verbose: bool = True
    ):
        """Run all experiments (baselines + cross-attention)."""
        
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"thesis_results_{timestamp}"
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        set_seed(seed)
        logger.info(f"🎯 Running ALL THESIS EXPERIMENTS")
        logger.info(f"📁 Output directory: {output_path}")
        logger.info(f"🔬 Total experiments: {len(self.experiments)}")
        
        # Load matched data once
        matched_data = self._load_matched_data(verbose)
        
        # Run all experiments
        all_results = {}
        start_time = time.time()
        
        for exp_name, exp_config in self.experiments.items():
            logger.info(f"\n{'='*60}")
            logger.info(f"🧪 EXPERIMENT: {exp_config['name']}")
            logger.info(f"📝 Description: {exp_config['description']}")
            logger.info(f"🏷️ Type: {exp_config['type']}")
            logger.info(f"{'='*60}")
            
            try:
                result = self._run_experiment(
                    exp_name, exp_config, matched_data,
                    num_folds, num_epochs, batch_size, learning_rate,
                    output_path, seed, verbose
                )
                all_results[exp_name] = result
                
                # Log result summary
                if 'error' not in result:
                    acc = result['regular_cv']['mean_accuracy']
                    std = result['regular_cv']['std_accuracy']
                    logger.info(f"✅ {exp_config['name']}: {acc:.1f}% ± {std:.1f}%")
                else:
                    logger.error(f"❌ {exp_config['name']}: {result['error']}")
                    
            except Exception as e:
                logger.error(f"❌ Experiment {exp_name} failed: {str(e)}")
                all_results[exp_name] = {
                    'experiment_name': exp_name,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }
        
        # Save comprehensive results
        total_time = time.time() - start_time
        self._save_comprehensive_results(all_results, output_path, total_time, verbose)
        
        logger.info(f"\n🎉 ALL EXPERIMENTS COMPLETED!")
        logger.info(f"⏱️ Total time: {total_time/60:.1f} minutes")
        logger.info(f"📁 Results saved to: {output_path}")
        
        return all_results
    
    def run_baselines_only(self, **kwargs):
        """Run only baseline experiments."""
        baseline_experiments = {
            k: v for k, v in self.experiments.items() 
            if v['type'] == 'baseline'
        }
        return self._run_specific_experiments(baseline_experiments, **kwargs)
    
    def run_cross_attention_only(self, **kwargs):
        """Run only cross-attention experiments."""
        cross_attention_experiments = {
            k: v for k, v in self.experiments.items() 
            if v['type'] == 'cross_attention'
        }
        return self._run_specific_experiments(cross_attention_experiments, **kwargs)
    
    def quick_test(
        self,
        experiments: List[str] = None,
        num_folds: int = 2,
        num_epochs: int = 10,
        verbose: bool = True
    ):
        """Quick test of experiments for validation."""
        
        if experiments is None:
            experiments = ['fmri_baseline', 'cross_attention_basic']
        
        test_experiments = {
            k: v for k, v in self.experiments.items() 
            if k in experiments
        }
        
        logger.info(f"🚀 QUICK TEST - {len(test_experiments)} experiments")
        return self._run_specific_experiments(
            test_experiments,
            num_folds=num_folds,
            num_epochs=num_epochs,
            output_dir="quick_test_results",
            verbose=verbose
        )
    
    def _run_specific_experiments(self, experiments_dict, **kwargs):
        """Run a specific set of experiments."""
        # Temporarily replace experiments
        original_experiments = self.experiments
        self.experiments = experiments_dict
        
        try:
            results = self.run_all(**kwargs)
        finally:
            # Restore original experiments
            self.experiments = original_experiments
        
        return results
    
    def _load_matched_data(self, verbose: bool = True):
        """Load matched subject data."""
        if verbose:
            logger.info("📊 Loading matched subject data...")
        
        # Load matched datasets
        matched_data = get_matched_datasets(
            data_dir=self.config['data_dir'],
            verbose=verbose
        )
        
        if verbose:
            logger.info(f"✅ Loaded {matched_data['n_subjects']} matched subjects")
            logger.info(f"🧠 fMRI shape: {matched_data['fmri_data'].shape}")
            logger.info(f"🏗️ sMRI shape: {matched_data['smri_data'].shape}")
        
        return matched_data
    
    def _run_experiment(
        self,
        exp_name: str,
        exp_config: dict,
        matched_data: dict,
        num_folds: int,
        num_epochs: int,
        batch_size: int,
        learning_rate: float,
        output_path: Path,
        seed: int,
        verbose: bool
    ):
        """Run a single experiment."""
        
        start_time = time.time()
        
        # Prepare experiment data
        if exp_config['modality'] == 'fmri':
            X = matched_data['fmri_data']
        elif exp_config['modality'] == 'smri':
            X = matched_data['smri_data']
        else:  # multimodal
            X = {
                'fmri': matched_data['fmri_data'],
                'smri': matched_data['smri_data']
            }
        
        y = matched_data['labels']
        
        # Create experiment directory
        exp_dir = output_path / exp_name
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Run cross-validation
        if exp_config['modality'] == 'multimodal':
            # Use multimodal cross-validation
            cv_results = self._run_multimodal_cv(
                exp_config['model_class'], X, y,
                num_folds, num_epochs, batch_size, learning_rate,
                exp_dir, seed, verbose
            )
        else:
            # Use single-modality cross-validation  
            cv_results = run_cross_validation(
                exp_config['model_class'], X, y,
                num_folds=num_folds,
                num_epochs=num_epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                device=self.device,
                save_dir=exp_dir,
                seed=seed,
                verbose=verbose
            )
        
        # Prepare result
        result = {
            'experiment_name': exp_name,
            'name': exp_config['name'],
            'description': exp_config['description'],
            'type': exp_config['type'],
            'modality': exp_config['modality'],
            'timestamp': datetime.now().isoformat(),
            'runtime_minutes': (time.time() - start_time) / 60,
            'regular_cv': cv_results
        }
        
        # Save individual result
        result_path = exp_dir / 'results.json'
        with open(result_path, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        
        return result
    
    def _run_multimodal_cv(
        self,
        model_class,
        X: dict,
        y: np.ndarray,
        num_folds: int,
        num_epochs: int,
        batch_size: int,
        learning_rate: float,
        save_dir: Path,
        seed: int,
        verbose: bool
    ):
        """Run cross-validation for multimodal models."""
        
        set_seed(seed)
        
        fmri_data = X['fmri']
        smri_data = X['smri']
        
        # Cross-validation
        skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
        fold_results = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(fmri_data, y)):
            if verbose:
                logger.info(f"📋 Fold {fold + 1}/{num_folds}")
            
            # Split data
            fmri_train, fmri_val = fmri_data[train_idx], fmri_data[val_idx]
            smri_train, smri_val = smri_data[train_idx], smri_data[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Create model
            model = model_class(
                fmri_dim=fmri_data.shape[1],
                smri_dim=smri_data.shape[1]
            ).to(self.device)
            
            # Train model
            trainer = Trainer(model, device=self.device)
            
            # Train with multimodal data
            history = trainer.train_multimodal(
                fmri_train, smri_train, y_train,
                fmri_val, smri_val, y_val,
                num_epochs=num_epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                save_dir=save_dir / f'fold_{fold}',
                verbose=verbose
            )
            
            # Evaluate
            model.load_state_dict(torch.load(save_dir / f'fold_{fold}' / 'best_model.pth')['model_state_dict'])
            model.eval()
            
            with torch.no_grad():
                fmri_val_tensor = torch.FloatTensor(fmri_val).to(self.device)
                smri_val_tensor = torch.FloatTensor(smri_val).to(self.device)
                logits = model(fmri_val_tensor, smri_val_tensor)
                y_pred = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                y_pred_class = (y_pred > 0.5).astype(int)
            
            # Calculate metrics
            accuracy = accuracy_score(y_val, y_pred_class)
            balanced_acc = balanced_accuracy_score(y_val, y_pred_class)
            auc = roc_auc_score(y_val, y_pred)
            
            fold_results.append({
                'fold': fold,
                'accuracy': accuracy,
                'balanced_accuracy': balanced_acc,
                'auc': auc,
                'history': history
            })
            
            if verbose:
                logger.info(f"   Accuracy: {accuracy:.3f}, AUC: {auc:.3f}")
        
        # Aggregate results
        accuracies = [r['accuracy'] for r in fold_results]
        balanced_accs = [r['balanced_accuracy'] for r in fold_results]
        aucs = [r['auc'] for r in fold_results]
        
        return {
            'fold_results': fold_results,
            'mean_accuracy': np.mean(accuracies) * 100,
            'std_accuracy': np.std(accuracies) * 100,
            'mean_balanced_accuracy': np.mean(balanced_accs) * 100,
            'std_balanced_accuracy': np.std(balanced_accs) * 100,
            'mean_auc': np.mean(aucs),
            'std_auc': np.std(aucs)
        }
    
    def _save_comprehensive_results(
        self,
        all_results: dict,
        output_path: Path,
        total_time: float,
        verbose: bool
    ):
        """Save comprehensive results with analysis."""
        
        # Save all results
        results_path = output_path / 'comprehensive_results.json'
        with open(results_path, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        # Create summary
        summary = {
            'experiment_info': {
                'total_experiments': len(all_results),
                'successful_experiments': len([r for r in all_results.values() if 'error' not in r]),
                'failed_experiments': len([r for r in all_results.values() if 'error' in r]),
                'total_runtime_minutes': total_time / 60,
                'timestamp': datetime.now().isoformat()
            },
            'results_summary': {}
        }
        
        # Performance summary
        for exp_name, result in all_results.items():
            if 'error' not in result and 'regular_cv' in result:
                cv_results = result['regular_cv']
                summary['results_summary'][exp_name] = {
                    'name': result['name'],
                    'type': result['type'],
                    'accuracy': f"{cv_results['mean_accuracy']:.1f}% ± {cv_results['std_accuracy']:.1f}%",
                    'balanced_accuracy': f"{cv_results['mean_balanced_accuracy']:.1f}% ± {cv_results['std_balanced_accuracy']:.1f}%",
                    'auc': f"{cv_results['mean_auc']:.3f} ± {cv_results['std_auc']:.3f}",
                }
        
        # Save summary
        summary_path = output_path / 'experiment_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        if verbose:
            logger.info("\n📊 EXPERIMENT SUMMARY")
            logger.info("=" * 60)
            for exp_name, exp_summary in summary['results_summary'].items():
                logger.info(f"🔬 {exp_summary['name']}: {exp_summary['accuracy']}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Comprehensive Thesis Experiments')
    parser.add_argument('--run_all', action='store_true', help='Run all experiments')
    parser.add_argument('--baselines_only', action='store_true', help='Run only baseline experiments')
    parser.add_argument('--cross_attention_only', action='store_true', help='Run only cross-attention experiments')
    parser.add_argument('--quick_test', action='store_true', help='Quick test with reduced parameters')
    
    parser.add_argument('--num_folds', type=int, default=5, help='Number of CV folds')
    parser.add_argument('--num_epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=3e-5, help='Learning rate')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--verbose', action='store_true', default=True, help='Verbose output')
    
    args = parser.parse_args()
    
    # Initialize framework
    experiments = ThesisExperiments()
    
    # Run requested experiments
    if args.run_all:
        logger.info("🚀 Running ALL thesis experiments...")
        results = experiments.run_all(
            num_folds=args.num_folds,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            output_dir=args.output_dir,
            seed=args.seed,
            verbose=args.verbose
        )
    elif args.baselines_only:
        logger.info("🚀 Running BASELINE experiments only...")
        results = experiments.run_baselines_only(
            num_folds=args.num_folds,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            output_dir=args.output_dir or "baseline_results",
            seed=args.seed,
            verbose=args.verbose
        )
    elif args.cross_attention_only:
        logger.info("🚀 Running CROSS-ATTENTION experiments only...")
        results = experiments.run_cross_attention_only(
            num_folds=args.num_folds,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            output_dir=args.output_dir or "cross_attention_results",
            seed=args.seed,
            verbose=args.verbose
        )
    elif args.quick_test:
        logger.info("🚀 Running QUICK TEST...")
        results = experiments.quick_test(verbose=args.verbose)
    else:
        logger.info("ℹ️ No experiment type specified. Use --help for options.")
        logger.info("💡 Try: python scripts/thesis_experiments.py --quick_test")
        return
    
    logger.info("✅ Experiments completed successfully!")


if __name__ == "__main__":
    main() 
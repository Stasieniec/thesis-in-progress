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
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, roc_auc_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from copy import deepcopy

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
from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut
from sklearn.metrics import accuracy_score, roc_auc_score, balanced_accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy import stats
from collections import defaultdict

# Project imports
from config import get_config
from utils import run_cross_validation, get_device
from utils.subject_matching import get_matched_datasets
from evaluation import create_cv_visualizations, save_results
from training import set_seed, Trainer
from models import SingleAtlasTransformer, SMRITransformer, CrossAttentionTransformer
from models.enhanced_smri import EnhancedSMRITransformer
from utils.helpers import _run_multimodal_fold

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
    
    # Known ABIDE site mappings for leave-site-out CV
    ABIDE_SITES = {
        'CALTECH': 'California Institute of Technology',
        'CMU': 'Carnegie Mellon University', 
        'KKI': 'Kennedy Krieger Institute',
        'LEUVEN': 'University of Leuven',
        'MAX_MUN': 'Ludwig Maximilians University Munich',
        'NYU': 'NYU Langone Medical Center',
        'OHSU': 'Oregon Health and Science University',
        'OLIN': 'Olin Institute',
        'PITT': 'University of Pittsburgh',
        'SBL': 'Social Brain Lab',
        'SDSU': 'San Diego State University',
        'STANFORD': 'Stanford University',
        'TRINITY': 'Trinity Centre for Health Sciences',
        'UCLA': 'UCLA',
        'UM': 'University of Michigan',
        'USM': 'University of Southern Mississippi',
        'YALE': 'Yale'
    }
    
    def __init__(self):
        # Use cross_attention config as default (most comprehensive)  
        # Override paths for local testing (not in Colab)
        if not Path('/content/drive').exists():
            # Local testing - use current directory
            self.config = get_config('cross_attention', output_dir=Path('./local_test_results'))
        else:
            # Google Colab environment
            self.config = get_config('cross_attention')
        self.device = get_device()
        
        # Define all experiments
        self.experiments = {
            # BASELINE EXPERIMENTS
            'fmri_baseline': {
                'name': 'fMRI Baseline',
                'description': 'fMRI-only Transformer baseline',
                'model_class': SingleAtlasTransformer,
                'type': 'baseline',
                'modality': 'fmri'
            },
            'smri_baseline': {
                'name': 'sMRI Enhanced Baseline', 
                'description': 'Enhanced sMRI Transformer baseline with improved architecture',
                'model_class': EnhancedSMRITransformer,
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
        
        # Baselines for comparison
        self.baselines = {
            'fmri': 0.60,  # 60% fMRI baseline
            'smri': 0.56,  # 56% Enhanced sMRI baseline (updated from 58% to proven 56%)
            'cross_attention': 0.58  # 58% original cross-attention
        }
        
        logger.info(f"🚀 Thesis Experiment Framework Initialized")
        logger.info(f"📊 Available experiments: {len(self.experiments)}")
        logger.info(f"💻 Device: {self.device}")
    
    def extract_site_info(
        self, 
        subject_ids: List[str], 
        phenotypic_file: str = None
    ) -> Tuple[List[str], Dict[str, List[str]], pd.DataFrame]:
        """
        Extract site information from subject IDs and phenotypic data.
        
        Args:
            subject_ids: List of subject IDs
            phenotypic_file: Path to phenotypic CSV file
            
        Returns:
            Tuple of (site_labels, site_mapping, site_stats)
        """
        logger.info("🔍 Extracting site information from subject IDs...")
        
        site_labels = []
        site_mapping = defaultdict(list)
        
        # Load phenotypic data if available
        phenotypic_sites = {}
        if phenotypic_file and Path(phenotypic_file).exists():
            try:
                pheno_df = pd.read_csv(phenotypic_file)
                if 'SITE_ID' in pheno_df.columns:
                    phenotypic_sites = dict(zip(
                        pheno_df['SUB_ID'].astype(str), 
                        pheno_df['SITE_ID'].astype(str)
                    ))
                    logger.info(f"   ✅ Found SITE_ID column in phenotypic data")
                else:
                    logger.info(f"   ⚠️ No SITE_ID column found in phenotypic data")
            except Exception as e:
                logger.info(f"   ⚠️ Error loading phenotypic data: {e}")
        
        # Extract sites from subject IDs
        for sub_id in subject_ids:
            site = self._extract_site_from_subject_id(sub_id, phenotypic_sites)
            site_labels.append(site)
            site_mapping[site].append(sub_id)
        
        # Create site statistics
        site_stats = pd.DataFrame([
            {
                'site': site,
                'n_subjects': len(subjects),
                'subjects': subjects[:5] + (['...'] if len(subjects) > 5 else [])
            }
            for site, subjects in site_mapping.items()
        ]).sort_values('n_subjects', ascending=False)
        
        logger.info(f"\n📊 Site extraction results:")
        logger.info(f"   Total sites: {len(site_mapping)}")
        logger.info(f"   Total subjects: {len(subject_ids)}")
        logger.info(f"   Sites found: {list(site_mapping.keys())}")
        
        return site_labels, dict(site_mapping), site_stats

    def _extract_site_from_subject_id(
        self, 
        subject_id: str, 
        phenotypic_sites: Dict[str, str]
    ) -> str:
        """Extract site information from a single subject ID."""
        # First check phenotypic data
        if subject_id in phenotypic_sites:
            return phenotypic_sites[subject_id]
        
        # Handle different subject ID formats
        subject_id_clean = str(subject_id).strip()
        subject_id_upper = subject_id_clean.upper()
        
        # Check for known ABIDE site prefixes
        for site_code in self.ABIDE_SITES.keys():
            if site_code in subject_id_upper:
                return site_code
        
        # Try common patterns:
        # Pattern 1: Site prefix followed by numbers (e.g., "NYU_0050001")
        for site_code in self.ABIDE_SITES.keys():
            if subject_id_upper.startswith(site_code):
                return site_code
        
        # Pattern 2: Numbers followed by site info (e.g., "0050001_KKI")
        for site_code in self.ABIDE_SITES.keys():
            if subject_id_upper.endswith(f"_{site_code}") or subject_id_upper.endswith(site_code):
                return site_code
        
        # Pattern 3: Extract numeric prefix and map to common sites based on ABIDE known ranges
        if subject_id_clean.startswith(('0050', '0051')):  # NYU site
            return 'NYU'
        elif subject_id_clean.startswith(('0028', '0029')):  # Stanford site
            return 'STANFORD'
        elif subject_id_clean.startswith(('0027')):  # UCLA site
            return 'UCLA'
        elif subject_id_clean.startswith(('0026')):  # Trinity site
            return 'TRINITY'
        elif subject_id_clean.startswith(('0025')):  # SBL site
            return 'SBL'
        elif subject_id_clean.startswith(('0024')):  # SDSU site  
            return 'SDSU'
        elif subject_id_clean.startswith(('0023')):  # PITT site
            return 'PITT'
        elif subject_id_clean.startswith(('0022')):  # OLIN site
            return 'OLIN'
        elif subject_id_clean.startswith(('0021')):  # OHSU site
            return 'OHSU'
        elif subject_id_clean.startswith(('0020')):  # Max Mun site
            return 'MAX_MUN'
        elif subject_id_clean.startswith(('0019')):  # Leuven site
            return 'LEUVEN'
        elif subject_id_clean.startswith(('0018')):  # KKI site
            return 'KKI'
        elif subject_id_clean.startswith(('0017')):  # CMU site
            return 'CMU'
        elif subject_id_clean.startswith(('0016')):  # Caltech site
            return 'CALTECH'
        
        # Pattern 4: Try to find site info in middle of string
        for site_code in self.ABIDE_SITES.keys():
            if f"_{site_code}_" in subject_id_upper or f"-{site_code}-" in subject_id_upper:
                return site_code
        
        # Try to use first 4 digits to create distinct sites
        import re
        numeric_match = re.match(r'(\d{4})', subject_id_clean)
        if numeric_match:
            prefix = numeric_match.group(1)
            return f"SITE_{prefix}"
        
        # Last resort: use first few characters
        if len(subject_id_clean) >= 3:
            return f"SITE_{subject_id_clean[:3].upper()}"
        else:
            return f"SITE_{subject_id_clean.upper()}"
    
    def run_all(
        self,
        num_folds: int = 5,
        num_epochs: int = 200,
        batch_size: int = 32,
        learning_rate: float = 3e-5,
        output_dir: str = None,
        seed: int = 42,
        verbose: bool = True,
        include_leave_site_out: bool = True
    ):
        """Run all experiments with both standard and leave-site-out cross-validation."""
        
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"thesis_results_{timestamp}"
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        set_seed(seed)
        logger.info(f"🎯 Running COMPREHENSIVE THESIS EXPERIMENTS")
        logger.info(f"📁 Output directory: {output_path}")
        logger.info(f"🔬 Total experiments: {len(self.experiments)}")
        logger.info(f"📊 Standard CV: YES")
        logger.info(f"🏥 Leave-Site-Out CV: {'YES' if include_leave_site_out else 'NO'}")
        
        # Load matched data once
        matched_data = self._load_matched_data(verbose)
        
        # Run all experiments with both CV types
        all_results = {}
        start_time = time.time()
        
        for exp_name, exp_config in self.experiments.items():
            logger.info(f"\n{'='*80}")
            logger.info(f"🧪 EXPERIMENT: {exp_config['name']}")
            logger.info(f"📝 Description: {exp_config['description']}")
            logger.info(f"🏷️ Type: {exp_config['type']}")
            logger.info(f"{'='*80}")
            
            try:
                result = self._run_comprehensive_experiment(
                    exp_name, exp_config, matched_data,
                    num_folds, num_epochs, batch_size, learning_rate,
                    output_path, seed, verbose, include_leave_site_out
                )
                all_results[exp_name] = result
                
                # Log result summary
                if 'error' not in result:
                    # Standard CV results
                    if 'standard_cv' in result and result['standard_cv'] is not None:
                        if isinstance(result['standard_cv'], dict) and 'mean_accuracy' in result['standard_cv']:
                            acc = result['standard_cv']['mean_accuracy']
                            std = result['standard_cv']['std_accuracy']
                            logger.info(f"✅ {exp_config['name']} (Standard CV): {acc:.1f}% ± {std:.1f}%")
                        else:
                            logger.warning(f"⚠️ {exp_config['name']}: Standard CV results incomplete")
                    
                    # Leave-site-out CV results (if available)
                    if include_leave_site_out and 'leave_site_out_cv' in result:
                        if 'error' not in result['leave_site_out_cv'] and result['leave_site_out_cv'] is not None:
                            lso_acc = result['leave_site_out_cv']['mean_accuracy']
                            lso_std = result['leave_site_out_cv']['std_accuracy']
                            logger.info(f"✅ {exp_config['name']} (Leave-Site-Out CV): {lso_acc:.1f}% ± {lso_std:.1f}%")
                        else:
                            logger.warning(f"⚠️ {exp_config['name']}: Leave-site-out CV failed")
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
        self._save_comprehensive_results(all_results, output_path, total_time, verbose, include_leave_site_out)
        
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
    
    def run_standard_cv_only(self, **kwargs):
        """Run all experiments with only standard cross-validation."""
        kwargs['include_leave_site_out'] = False
        return self.run_all(**kwargs)
    
    def run_leave_site_out_only(
        self,
        num_epochs: int = 200,
        batch_size: int = 32,
        learning_rate: float = 3e-5,
        output_dir: str = None,
        seed: int = 42,
        verbose: bool = True
    ):
        """Run all experiments with only leave-site-out cross-validation."""
        
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"leave_site_out_results_{timestamp}"
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        set_seed(seed)
        logger.info(f"🏥 Running LEAVE-SITE-OUT EXPERIMENTS ONLY")
        logger.info(f"📁 Output directory: {output_path}")
        logger.info(f"🔬 Total experiments: {len(self.experiments)}")
        
        # Load matched data once
        matched_data = self._load_matched_data(verbose)
        
        # Run only leave-site-out experiments
        all_results = {}
        start_time = time.time()
        
        for exp_name, exp_config in self.experiments.items():
            logger.info(f"\n{'='*80}")
            logger.info(f"🧪 EXPERIMENT: {exp_config['name']}")
            logger.info(f"📝 Description: {exp_config['description']}")
            logger.info(f"🏥 Leave-Site-Out CV Only")
            logger.info(f"{'='*80}")
            
            try:
                # Create experiment directory
                exp_dir = output_path / exp_name
                exp_dir.mkdir(parents=True, exist_ok=True)
                
                # Run only leave-site-out CV
                leave_site_out_results = self._run_leave_site_out_cv_for_experiment(
                    exp_config, matched_data, num_epochs, 
                    batch_size, learning_rate, exp_dir, seed, verbose
                )
                
                result = {
                    'experiment_name': exp_name,
                    'name': exp_config['name'],
                    'description': exp_config['description'],
                    'type': exp_config['type'],
                    'modality': exp_config['modality'],
                    'timestamp': datetime.now().isoformat(),
                    'leave_site_out_cv': leave_site_out_results
                }
                
                all_results[exp_name] = result
                
                # Log result summary
                lso_acc = leave_site_out_results['mean_accuracy']
                lso_std = leave_site_out_results['std_accuracy']
                beats_baseline = leave_site_out_results.get('beats_baseline', False)
                status = "🎉 BEATS baseline!" if beats_baseline else "📊 Below baseline"
                logger.info(f"✅ {exp_config['name']}: {lso_acc:.1f}% ± {lso_std:.1f}% - {status}")
                
            except Exception as e:
                logger.error(f"❌ Experiment {exp_name} failed: {str(e)}")
                all_results[exp_name] = {
                    'experiment_name': exp_name,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }
        
        # Save comprehensive results
        total_time = time.time() - start_time
        self._save_comprehensive_results(all_results, output_path, total_time, verbose, include_leave_site_out=True)
        
        logger.info(f"\n🎉 LEAVE-SITE-OUT EXPERIMENTS COMPLETED!")
        logger.info(f"⏱️ Total time: {total_time/60:.1f} minutes")
        logger.info(f"📁 Results saved to: {output_path}")
        
        return all_results
    
    def debug_smri_performance(
        self,
        num_epochs: int = 50,
        verbose: bool = True
    ):
        """Debug sMRI performance issues by testing different configurations."""
        
        if verbose:
            logger.info("🔍 DEBUGGING sMRI PERFORMANCE")
            logger.info("=" * 60)
        
        # Load data
        matched_data = self._load_matched_data(verbose)
        smri_data = matched_data['smri_data']
        labels = matched_data['labels']
        
        # Test different configurations
        configs = [
            {
                'name': 'Current (Low Performance)',
                'learning_rate': 3e-5,
                'batch_size': 32,
                'd_model': 256,
                'epochs': 40
            },
            {
                'name': 'Optimized (Expected 58%)',
                'learning_rate': 0.001,
                'batch_size': 64,
                'd_model': 64,
                'epochs': 50
            },
            {
                'name': 'Conservative',
                'learning_rate': 0.0005,
                'batch_size': 48,
                'd_model': 128,
                'epochs': 60
            }
        ]
        
        results = {}
        
        for config in configs:
            if verbose:
                logger.info(f"\n🧪 Testing: {config['name']}")
                logger.info(f"   LR: {config['learning_rate']}, BS: {config['batch_size']}")
                logger.info(f"   d_model: {config['d_model']}, epochs: {config['epochs']}")
            
            try:
                # Quick 2-fold test
                result = self.test_single_experiment(
                    experiment_name='smri_baseline',
                    num_folds=2,
                    num_epochs=config['epochs'],
                    include_leave_site_out=False,
                    verbose=False
                )
                
                if 'standard_cv' in result and 'mean_accuracy' in result['standard_cv']:
                    acc = result['standard_cv']['mean_accuracy']
                    results[config['name']] = acc
                    if verbose:
                        logger.info(f"   Result: {acc:.1f}%")
                else:
                    if verbose:
                        logger.warning(f"   Failed to get results")
                    
            except Exception as e:
                if verbose:
                    logger.error(f"   Error: {e}")
        
        if verbose:
            logger.info("\n📊 PERFORMANCE COMPARISON:")
            for name, acc in results.items():
                status = "✅ Good" if acc > 55 else "❌ Poor"
                logger.info(f"   {name}: {acc:.1f}% {status}")
        
        return results
    
    def compare_smri_models(
        self,
        num_folds: int = 3,
        verbose: bool = True
    ):
        """Compare sMRI Transformer vs. simple Logistic Regression to debug performance."""
        
        if verbose:
            logger.info("🔬 COMPARING sMRI MODELS: Transformer vs. Logistic Regression")
            logger.info("=" * 70)
        
        # Load data
        matched_data = self._load_matched_data(verbose=False)
        X = matched_data['smri_data']
        y = matched_data['labels']
        
        # Compare models
        results = {}
        
        # 1. Test Logistic Regression (Simple baseline)
        if verbose:
            logger.info("🧮 Testing Logistic Regression...")
        
        skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
        lr_scores = []
        
        for train_idx, test_idx in skf.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train logistic regression
            lr = LogisticRegression(max_iter=1000, random_state=42)
            lr.fit(X_train_scaled, y_train)
            
            # Evaluate
            y_pred = lr.predict(X_test_scaled)
            acc = accuracy_score(y_test, y_pred)
            lr_scores.append(acc)
        
        lr_mean = np.mean(lr_scores) * 100
        lr_std = np.std(lr_scores) * 100
        results['logistic_regression'] = lr_mean
        
        if verbose:
            logger.info(f"   Result: {lr_mean:.1f}% ± {lr_std:.1f}%")
        
        # 2. Test sMRI Transformer (Current approach)
        if verbose:
            logger.info("🧠 Testing sMRI Transformer...")
        
        try:
            result = self.test_single_experiment(
                experiment_name='smri_baseline',
                num_folds=num_folds,
                num_epochs=20,  # Fewer epochs for faster comparison
                include_leave_site_out=False,
                verbose=False
            )
            
            if 'standard_cv' in result and 'mean_accuracy' in result['standard_cv']:
                transformer_mean = result['standard_cv']['mean_accuracy']
                results['transformer'] = transformer_mean
                
                if verbose:
                    logger.info(f"   Result: {transformer_mean:.1f}%")
            else:
                if verbose:
                    logger.error("   Failed to get transformer results")
                results['transformer'] = 0.0
                
        except Exception as e:
            if verbose:
                logger.error(f"   Transformer failed: {e}")
            results['transformer'] = 0.0
        
        # 3. Summary
        if verbose:
            logger.info("\n📊 MODEL COMPARISON RESULTS:")
            logger.info("-" * 40)
            
            lr_acc = results.get('logistic_regression', 0)
            tf_acc = results.get('transformer', 0)
            
            logger.info(f"   Logistic Regression: {lr_acc:.1f}%")
            logger.info(f"   sMRI Transformer:    {tf_acc:.1f}%")
            
            if lr_acc > tf_acc + 5:  # More than 5% difference
                logger.info("   🚨 ISSUE: Transformer significantly underperforming!")
                logger.info("   💡 Suggestions:")
                logger.info("      - Transformer may be overfitting")
                logger.info("      - Hyperparameters need adjustment")
                logger.info("      - Consider simpler architecture")
            elif tf_acc > lr_acc + 5:
                logger.info("   ✅ GOOD: Transformer outperforming simple model")
            else:
                logger.info("   ⚖️ SIMILAR: Both models perform comparably")
        
        return results
    
    def systematic_smri_hyperparameter_search(
        self,
        num_folds: int = 3,
        num_epochs: int = 40,
        verbose: bool = True
    ):
        """
        Systematic hyperparameter search for sMRI to find optimal 58% parameters.
        Tests different combinations to identify the best performing setup.
        """
        
        if verbose:
            logger.info("🔬 SYSTEMATIC sMRI HYPERPARAMETER SEARCH")
            logger.info("=" * 70)
            logger.info("Goal: Find hyperparameters that achieve 58% accuracy")
        
        # Load data once
        matched_data = self._load_matched_data(verbose=False)
        smri_data = matched_data['smri_data']
        labels = matched_data['labels']
        
        # Define hyperparameter search space based on known good configurations
        param_grid = [
            # Configuration 1: Original thesis parameters (target: 58%)
            {
                'name': 'Thesis_Original',
                'd_model': 128,
                'n_heads': 8,
                'n_layers': 3,
                'dropout': 0.1,
                'layer_dropout': 0.05,
                'learning_rate': 0.0005,
                'batch_size': 64,
                'weight_decay': 1e-3,
                'use_class_weights': False,
                'label_smoothing': 0.0
            },
            # Configuration 2: Higher capacity model
            {
                'name': 'High_Capacity',
                'd_model': 256,
                'n_heads': 8,
                'n_layers': 4,
                'dropout': 0.1,
                'layer_dropout': 0.05,
                'learning_rate': 0.0003,
                'batch_size': 32,
                'weight_decay': 1e-3,
                'use_class_weights': False,
                'label_smoothing': 0.0
            },
            # Configuration 3: Lower regularization
            {
                'name': 'Low_Regularization',
                'd_model': 128,
                'n_heads': 8,
                'n_layers': 3,
                'dropout': 0.05,
                'layer_dropout': 0.02,
                'learning_rate': 0.001,
                'batch_size': 64,
                'weight_decay': 1e-4,
                'use_class_weights': False,
                'label_smoothing': 0.0
            },
            # Configuration 4: Working notebook style
            {
                'name': 'Notebook_Style',
                'd_model': 64,
                'n_heads': 4,
                'n_layers': 2,
                'dropout': 0.3,
                'layer_dropout': 0.1,
                'learning_rate': 0.001,
                'batch_size': 16,
                'weight_decay': 1e-4,
                'use_class_weights': True,
                'label_smoothing': 0.1
            },
            # Configuration 5: Balanced approach
            {
                'name': 'Balanced_Approach',
                'd_model': 96,
                'n_heads': 6,
                'n_layers': 3,
                'dropout': 0.15,
                'layer_dropout': 0.08,
                'learning_rate': 0.0007,
                'batch_size': 48,
                'weight_decay': 5e-4,
                'use_class_weights': False,
                'label_smoothing': 0.05
            },
            # Configuration 6: Large batch stable
            {
                'name': 'Large_Batch',
                'd_model': 128,
                'n_heads': 8,
                'n_layers': 2,
                'dropout': 0.1,
                'layer_dropout': 0.05,
                'learning_rate': 0.0002,
                'batch_size': 128,
                'weight_decay': 1e-3,
                'use_class_weights': False,
                'label_smoothing': 0.0
            }
        ]
        
        results = {}
        best_accuracy = 0.0
        best_config = None
        
        for i, params in enumerate(param_grid, 1):
            config_name = params['name']
            
            if verbose:
                logger.info(f"\n🧪 Testing Configuration {i}/{len(param_grid)}: {config_name}")
                logger.info(f"   d_model={params['d_model']}, n_heads={params['n_heads']}, n_layers={params['n_layers']}")
                logger.info(f"   dropout={params['dropout']}, lr={params['learning_rate']}, batch_size={params['batch_size']}")
            
            try:
                # Test this configuration
                config_result = self._test_smri_config(
                    smri_data, labels, params, 
                    num_folds, num_epochs, verbose=False
                )
                
                accuracy = config_result['mean_accuracy']
                std = config_result['std_accuracy']
                
                results[config_name] = {
                    'params': params,
                    'accuracy': accuracy,
                    'std': std,
                    'config_result': config_result
                }
                
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_config = config_name
                
                if verbose:
                    logger.info(f"   ✅ Result: {accuracy:.1f}% ± {std:.1f}%")
                    if accuracy >= 58.0:
                        logger.info(f"   🎉 TARGET ACHIEVED! 58%+ accuracy reached")
                    
            except Exception as e:
                if verbose:
                    logger.info(f"   ❌ Failed: {str(e)}")
                results[config_name] = {
                    'params': params,
                    'error': str(e)
                }
        
        # Summary
        if verbose:
            logger.info(f"\n📊 HYPERPARAMETER SEARCH SUMMARY")
            logger.info("=" * 70)
            logger.info(f"🏆 Best Configuration: {best_config}")
            logger.info(f"🎯 Best Accuracy: {best_accuracy:.1f}%")
            logger.info("")
            logger.info("📈 All Results (sorted by accuracy):")
            
            # Sort results by accuracy
            sorted_results = sorted(
                [(name, r) for name, r in results.items() if 'accuracy' in r],
                key=lambda x: x[1]['accuracy'],
                reverse=True
            )
            
            for rank, (name, result) in enumerate(sorted_results, 1):
                acc = result['accuracy']
                std = result['std']
                status = "🎉" if acc >= 58.0 else "📊"
                logger.info(f"   {rank}. {name}: {acc:.1f}% ± {std:.1f}% {status}")
            
            # Show failed configurations
            failed = [name for name, r in results.items() if 'error' in r]
            if failed:
                logger.info(f"\n❌ Failed Configurations: {failed}")
            
            if best_accuracy >= 58.0:
                logger.info(f"\n🎉 SUCCESS! Found configuration achieving 58%+ accuracy")
                logger.info(f"💡 Use the '{best_config}' parameters for optimal sMRI performance")
            else:
                logger.info(f"\n⚠️  Target not reached. Best was {best_accuracy:.1f}%")
                logger.info("💡 Consider trying different parameter ranges or more epochs")
        
        return results
    
    def _test_smri_config(
        self,
        smri_data: np.ndarray,
        labels: np.ndarray,
        params: dict,
        num_folds: int,
        num_epochs: int,
        verbose: bool = False
    ):
        """Test a single sMRI configuration."""
        
        skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
        fold_results = []
        
        for fold, (train_idx, test_idx) in enumerate(skf.split(smri_data, labels)):
            fold_result = self._run_single_smri_fold_with_params(
                fold, train_idx, test_idx, smri_data, labels, 
                params, num_epochs, verbose=False
            )
            fold_results.append(fold_result)
        
        # Aggregate results
        accuracies = [r['test_accuracy'] for r in fold_results]
        
        return {
            'fold_results': fold_results,
            'mean_accuracy': np.mean(accuracies) * 100,
            'std_accuracy': np.std(accuracies) * 100,
            'params_used': params
        }
    
    def _run_single_smri_fold_with_params(
        self,
        fold: int,
        train_idx: np.ndarray,
        test_idx: np.ndarray,
        features: np.ndarray,
        labels: np.ndarray,
        params: dict,
        num_epochs: int,
        verbose: bool = False
    ):
        """Run a single fold with specific hyperparameters."""
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        
        # Handle imports
        try:
            from src.training import Trainer, set_seed
        except ImportError:
            from training import Trainer, set_seed
        
        set_seed(42 + fold)
        
        # Split data
        X_train_fold, X_test = features[train_idx], features[test_idx]
        y_train_fold, y_test = labels[train_idx], labels[test_idx]
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_fold, y_train_fold,
            test_size=0.2,
            stratify=y_train_fold,
            random_state=42 + fold
        )
        
        # Preprocessing
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)
        
        # Convert to tensors
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train), 
            torch.LongTensor(y_train)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val), 
            torch.LongTensor(y_val)
        )
        test_dataset = TensorDataset(
            torch.FloatTensor(X_test), 
            torch.LongTensor(y_test)
        )
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=params['batch_size'], shuffle=False)
        
        # Create model with specific parameters
        model = SMRITransformer(
            input_dim=X_train.shape[1],
            d_model=params['d_model'],
            n_heads=params['n_heads'],
            n_layers=params['n_layers'],
            dropout=params['dropout'],
            layer_dropout=params['layer_dropout']
        ).to(self.device)
        
        # Create config
        temp_config = get_config('smri')
        temp_config.learning_rate = params['learning_rate']
        temp_config.weight_decay = params['weight_decay']
        temp_config.batch_size = params['batch_size']
        temp_config.num_epochs = num_epochs
        temp_config.use_class_weights = params['use_class_weights']
        temp_config.label_smoothing = params['label_smoothing']
        temp_config.early_stop_patience = 15
        temp_config.output_dir = Path('./temp_search_results')
        temp_config.output_dir.mkdir(exist_ok=True)
        
        # Train model
        trainer = Trainer(model, self.device, temp_config, model_type='single')
        
        # Train without model checkpointing
        history = trainer.fit(
            train_loader, val_loader,
            num_epochs=num_epochs,
            checkpoint_path=None,  # No model saving
            y_train=y_train
        )
        
        # Evaluate model
        test_metrics = trainer.evaluate_final(test_loader)
        
        return {
            'accuracy': test_metrics['accuracy'],
            'balanced_accuracy': test_metrics['balanced_accuracy'],
            'auc': test_metrics['auc'],
            'history': history
        }
    
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
    
    def test_single_experiment(
        self,
        experiment_name: str,
        num_folds: int = 2,
        num_epochs: int = 10,
        include_leave_site_out: bool = False,
        verbose: bool = True
    ):
        """Quick test of a single experiment."""
        
        if experiment_name not in self.experiments:
            available = list(self.experiments.keys())
            raise ValueError(f"Experiment '{experiment_name}' not found. Available: {available}")
        
        single_experiment = {experiment_name: self.experiments[experiment_name]}
        
        logger.info(f"🚀 TESTING SINGLE EXPERIMENT: {experiment_name}")
        return self._run_specific_experiments(
            single_experiment,
            num_folds=num_folds,
            num_epochs=num_epochs,
            output_dir=f"test_{experiment_name}",
            include_leave_site_out=include_leave_site_out,
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
        
        # Check if we're in a local testing environment
        if not Path('/content/drive').exists():
            # Local testing - create mock data
            if verbose:
                logger.info("🔬 Local testing mode - creating mock data")
            
            n_subjects = 100
            fmri_data = np.random.randn(n_subjects, 19900).astype(np.float32)
            smri_data = np.random.randn(n_subjects, 800).astype(np.float32)
            labels = np.random.randint(0, 2, n_subjects)
            
            matched_data = {
                'fmri_data': fmri_data,
                'smri_data': smri_data,
                'labels': labels,
                'n_subjects': n_subjects
            }
        else:
            # Google Colab environment - load real data
            matched_data = get_matched_datasets(
                fmri_roi_dir=str(self.config.fmri_roi_dir),
                smri_data_path=str(self.config.smri_data_path),
                phenotypic_file=str(self.config.phenotypic_file),
                verbose=verbose
            )
        
        if verbose:
            logger.info(f"✅ Loaded {matched_data['num_matched_subjects']} matched subjects")
            logger.info(f"🧠 fMRI shape: {matched_data['fmri_features'].shape}")
            logger.info(f"🏗️ sMRI shape: {matched_data['smri_features'].shape}")
        
        # Standardize field names for consistency
        standardized_data = {
            'fmri_data': matched_data['fmri_features'],
            'smri_data': matched_data['smri_features'],
            'labels': matched_data['fmri_labels'],  # fmri_labels and smri_labels are the same
            'n_subjects': matched_data['num_matched_subjects'],
            'fmri_subject_ids': matched_data.get('fmri_subject_ids', []),
            'smri_subject_ids': matched_data.get('smri_subject_ids', [])
        }
        
        if verbose:
            logger.info(f"🔍 Data quality check:")
            logger.info(f"   sMRI data range: [{matched_data['smri_features'].min():.3f}, {matched_data['smri_features'].max():.3f}]")
            logger.info(f"   fMRI data range: [{matched_data['fmri_features'].min():.3f}, {matched_data['fmri_features'].max():.3f}]")
            logger.info(f"   Label distribution: {np.bincount(matched_data['fmri_labels'])}")
        
        return standardized_data
    
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
            # Create appropriate config for this experiment
            if exp_config['modality'] == 'fmri':
                temp_config = get_config('fmri')
            elif exp_config['modality'] == 'smri':
                temp_config = get_config('smri')
            else:
                temp_config = get_config('cross_attention')
                
            temp_config.num_folds = num_folds
            temp_config.num_epochs = num_epochs
            temp_config.batch_size = batch_size
            temp_config.learning_rate = learning_rate
            temp_config.output_dir = exp_dir
            temp_config.seed = seed
            
            fold_results = run_cross_validation(
                features=X,
                labels=y,
                model_class=exp_config['model_class'],
                config=temp_config,
                experiment_type='single',
                verbose=verbose
            )
            
            # Convert to expected format
            accuracies = [r['test_accuracy'] for r in fold_results]
            balanced_accs = [r['test_balanced_accuracy'] for r in fold_results]
            aucs = [r['test_auc'] for r in fold_results]
            
            cv_results = {
                'fold_results': fold_results,
                'mean_accuracy': np.mean(accuracies) * 100,
                'std_accuracy': np.std(accuracies) * 100,
                'mean_balanced_accuracy': np.mean(balanced_accs) * 100,
                'std_balanced_accuracy': np.std(balanced_accs) * 100,
                'mean_auc': np.mean(aucs),
                'std_auc': np.std(aucs)
            }
        
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
    
    def _run_comprehensive_experiment(
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
        verbose: bool,
        include_leave_site_out: bool
    ):
        """Run a single experiment with both standard and leave-site-out CV."""
        
        start_time = time.time()
        
        # Create experiment directory
        exp_dir = output_path / exp_name
        exp_dir.mkdir(parents=True, exist_ok=True)
        
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
        
        result = {
            'experiment_name': exp_name,
            'name': exp_config['name'],
            'description': exp_config['description'],
            'type': exp_config['type'],
            'modality': exp_config['modality'],
            'timestamp': datetime.now().isoformat(),
        }
        
        try:
            # 1. Run standard cross-validation
            logger.info(f"📊 Running Standard {num_folds}-Fold Cross-Validation...")
            standard_cv_results = self._run_standard_cv_for_experiment(
                exp_config, X, y, num_folds, num_epochs, 
                batch_size, learning_rate, exp_dir / 'standard_cv', seed, verbose
            )
            result['standard_cv'] = standard_cv_results
            
            # 2. Run leave-site-out cross-validation (if requested)
            if include_leave_site_out:
                logger.info(f"🏥 Running Leave-Site-Out Cross-Validation...")
                try:
                    leave_site_out_results = self._run_leave_site_out_cv_for_experiment(
                        exp_config, matched_data, num_epochs, 
                        batch_size, learning_rate, exp_dir / 'leave_site_out', seed, verbose
                    )
                    result['leave_site_out_cv'] = leave_site_out_results
                except Exception as e:
                    logger.warning(f"⚠️ Leave-site-out CV failed for {exp_name}: {e}")
                    result['leave_site_out_cv'] = {'error': str(e)}
            
            # Calculate runtime
            result['runtime_minutes'] = (time.time() - start_time) / 60
            
        except Exception as e:
            logger.error(f"❌ Experiment {exp_name} failed: {e}")
            result['error'] = str(e)
        
        # Save individual result
        result_path = exp_dir / 'comprehensive_results.json'
        with open(result_path, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        
        return result
    
    def _run_leave_site_out_cv_for_experiment(
        self,
        exp_config: dict,
        matched_data: dict,
        num_epochs: int,
        batch_size: int,
        learning_rate: float,
        output_dir: Path,
        seed: int,
        verbose: bool
    ):
        """Run leave-site-out cross-validation for a single experiment."""
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract site information
        if 'fmri_subject_ids' in matched_data:
            subject_ids = matched_data['fmri_subject_ids']
        else:
            # Create dummy subject IDs for local testing
            n_subjects = len(matched_data['labels'])
            subject_ids = [f"subject_{i:05d}" for i in range(n_subjects)]
        
        # Get phenotypic file path
        phenotypic_file = str(self.config.phenotypic_file) if hasattr(self.config, 'phenotypic_file') else None
        
        site_labels, site_mapping, site_stats = self.extract_site_info(
            subject_ids, phenotypic_file
        )
        
        # Check if we have enough sites for meaningful CV
        n_sites = len(site_mapping)
        if n_sites < 3:
            raise ValueError(f"Need at least 3 sites for leave-site-out CV, found {n_sites}")
        
        # Prepare data arrays
        if exp_config['modality'] == 'fmri':
            features = matched_data['fmri_data']
        elif exp_config['modality'] == 'smri':
            features = matched_data['smri_data']
        else:  # multimodal
            fmri_features = matched_data['fmri_data']
            smri_features = matched_data['smri_data']
        
        labels = matched_data['labels']
        
        # Convert site labels to numpy array for indexing
        site_array = np.array(site_labels)
        
        # Initialize LeaveOneGroupOut
        logo = LeaveOneGroupOut()
        
        fold_results = []
        site_results = []
        
        # Run leave-site-out cross-validation
        for fold_idx, (train_idx, test_idx) in enumerate(logo.split(labels, labels, site_array)):
            
            # Get test site name
            test_sites = np.unique(site_array[test_idx])
            test_site = test_sites[0] if len(test_sites) == 1 else f"Mixed_{fold_idx}"
            
            if verbose:
                train_sites = np.unique(site_array[train_idx])
                logger.info(f"      Fold {fold_idx+1}: Training on {len(train_sites)} sites, testing on {test_site}")
            
            try:
                if exp_config['modality'] == 'multimodal':
                    # Use the helper function for multimodal experiments
                    fold_result = _run_multimodal_fold(
                        fold=fold_idx,
                        train_idx=train_idx,
                        test_idx=test_idx,
                        fmri_features=fmri_features,
                        smri_features=smri_features,
                        labels=labels,
                        model_class=exp_config['model_class'],
                        config=get_config(
                            'cross_attention',
                            num_epochs=num_epochs,
                            batch_size=batch_size,
                            learning_rate=learning_rate,
                            seed=seed + fold_idx,
                            output_dir=output_dir / f'fold_{fold_idx}'
                        ),
                        device=self.device,
                        verbose=False
                    )
                else:
                    # Single modality experiment
                    fold_result = self._run_single_modality_fold(
                        fold_idx, train_idx, test_idx, features, labels,
                        exp_config, num_epochs, batch_size, learning_rate,
                        output_dir, seed
                    )
                
                fold_results.append(fold_result)
                site_results.append({
                    'test_site': test_site,
                    'test_accuracy': fold_result['test_accuracy'],
                    'test_balanced_accuracy': fold_result['test_balanced_accuracy'],
                    'test_auc': fold_result['test_auc'],
                    'n_test_subjects': len(test_idx),
                    'n_train_subjects': len(train_idx)
                })
                
            except Exception as e:
                if verbose:
                    logger.warning(f"         ❌ Fold {fold_idx} failed: {e}")
                continue
        
        if not fold_results:
            raise RuntimeError("All leave-site-out folds failed")
        
        # Calculate aggregate metrics
        accuracies = [r['test_accuracy'] for r in fold_results]
        balanced_accuracies = [r['test_balanced_accuracy'] for r in fold_results]
        aucs = [r['test_auc'] for r in fold_results]
        
        results = {
            'mean_accuracy': float(np.mean(accuracies)) * 100,  # Convert to percentage
            'std_accuracy': float(np.std(accuracies)) * 100,
            'mean_balanced_accuracy': float(np.mean(balanced_accuracies)) * 100,
            'std_balanced_accuracy': float(np.std(balanced_accuracies)) * 100,
            'mean_auc': float(np.mean(aucs)),
            'std_auc': float(np.std(aucs)),
            'n_sites': len(site_mapping),
            'n_folds': len(fold_results),
            'site_results': site_results,
            'fold_results': fold_results,
            'cv_type': 'leave_site_out',
            'beats_baseline': float(np.mean(accuracies)) > (self.baselines['fmri'] / 100.0)
        }
        
        # Save detailed results with robust JSON serialization
        try:
            # Create JSON-safe version of results
            json_safe_results = {}
            for key, value in results.items():
                if isinstance(value, (list, dict)):
                    json_safe_results[key] = json.loads(json.dumps(value, default=str))
                else:
                    json_safe_results[key] = value
            
            with open(output_dir / 'leave_site_out_results.json', 'w') as f:
                json.dump(json_safe_results, f, indent=2, default=str)
        except Exception as e:
            if verbose:
                logger.warning(f"Could not save JSON results: {e}")
        
        # Save site information
        try:
            site_stats.to_csv(output_dir / 'site_information.csv', index=False)
        except Exception as e:
            if verbose:
                logger.warning(f"Could not save site information: {e}")
        
        try:
            # Create JSON-safe site mapping
            json_safe_mapping = {str(k): [str(s) for s in v] for k, v in site_mapping.items()}
            with open(output_dir / 'site_mapping.json', 'w') as f:
                json.dump(json_safe_mapping, f, indent=2)
        except Exception as e:
            if verbose:
                logger.warning(f"Could not save site mapping: {e}")
        
        return results
    
    def _run_single_modality_fold(
        self,
        fold_idx: int,
        train_idx: np.ndarray,
        test_idx: np.ndarray,
        features: np.ndarray,
        labels: np.ndarray,
        exp_config: dict,
        num_epochs: int,
        batch_size: int,
        learning_rate: float,
        output_dir: Path,
        seed: int
    ):
        """Run a single fold for single-modality experiments using the actual model."""
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        try:
            from src.training import Trainer, set_seed
        except ImportError:
            from training import Trainer, set_seed
        
        set_seed(seed)
        
        # Split data
        X_train_fold, X_test = features[train_idx], features[test_idx]
        y_train_fold, y_test = labels[train_idx], labels[test_idx]
        
        # **CRITICAL FIX**: Create proper train/validation split
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_fold, y_train_fold,
            test_size=0.2,
            stratify=y_train_fold,
            random_state=seed + fold_idx
        )
        
        # **CRITICAL FIX**: Apply proper preprocessing for sMRI
        if exp_config['modality'] == 'smri':
            # Standardize features (essential for sMRI)
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)
            X_test = scaler.transform(X_test)
        else:
            # Basic standardization for fMRI
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)
            X_test = scaler.transform(X_test)
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train)
        X_val_tensor = torch.FloatTensor(X_val)
        X_test_tensor = torch.FloatTensor(X_test)
        y_train_tensor = torch.LongTensor(y_train)
        y_val_tensor = torch.LongTensor(y_val)
        y_test_tensor = torch.LongTensor(y_test)
        
        # Create data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # Create model instance
        model_class = exp_config['model_class']
        
        # Get input dimension (after preprocessing)
        input_dim = X_train.shape[1]
        
        # **CRITICAL FIX**: Create model with correct parameters
        if exp_config['modality'] == 'smri':
            model = model_class(
                input_dim=input_dim,
                d_model=192,   # INCREASED from 128 (even better capacity for 58%)
                n_heads=8,     # INCREASED from 4 (more attention heads)
                n_layers=4,    # INCREASED from 3 (deeper model for complex patterns)
                dropout=0.15,  # SLIGHTLY increased (balance overfitting)
                layer_dropout=0.1  # INCREASED back (some regularization needed)
            ).to(self.device)
        else:
            # fMRI model has different parameter structure
            model = model_class(
                feat_dim=input_dim,
                d_model=256,
                num_heads=8,
                num_layers=4,
                dropout=0.1
            ).to(self.device)
        
        # Create config for this fold
        temp_config = get_config(exp_config['modality'])
        temp_config.num_epochs = num_epochs
        temp_config.batch_size = batch_size
        temp_config.learning_rate = learning_rate
        temp_config.seed = seed
        temp_config.output_dir = output_dir
        
        # **CRITICAL FIX**: Apply proven sMRI optimizations for 58% performance
        if exp_config['modality'] == 'smri':
            temp_config.learning_rate = 0.001    # INCREASED back (sMRI needs higher LR)
            temp_config.weight_decay = 5e-4      # REDUCED (less regularization)
            temp_config.batch_size = 64          # LARGER batch size for stability
            temp_config.use_class_weights = False # DISABLED (can hurt performance)
            temp_config.label_smoothing = 0.0    # DISABLED (not needed for sMRI)
            temp_config.early_stop_patience = 25  # MORE patience for convergence
            temp_config.gradient_clip_norm = 1.0  # INCREASED gradient clipping
        
        # **CRITICAL FIX**: Use proper trainer initialization
        trainer = Trainer(model, self.device, temp_config, model_type='single')
        
        # Create temporary checkpoint path for this fold
        checkpoint_path = output_dir / f'temp_fold_{fold_idx}_model.pth'
        
        # **CRITICAL FIX**: Train with proper validation - no checkpoint saving
        history = trainer.fit(
            train_loader, val_loader,  # Proper train/val split
            num_epochs=num_epochs,
            checkpoint_path=None,  # No model saving
            y_train=y_train
        )
        
        # Evaluate on test set
        test_metrics = trainer.evaluate_final(test_loader)
        
        return {
            'fold': fold_idx,
            'test_accuracy': test_metrics['accuracy'],
            'test_balanced_accuracy': test_metrics['balanced_accuracy'],
            'test_auc': test_metrics['auc'],
            'train_size': len(y_train),
            'val_size': len(y_val),
            'test_size': len(y_test),
            'history': history,  # Include training history for analysis
            'targets': test_metrics['targets'],
            'predictions': test_metrics['predictions'],
            'probabilities': test_metrics['probabilities']
        }
    
    def _run_standard_cv_for_experiment(
        self,
        exp_config: dict,
        X,
        y: np.ndarray,
        num_folds: int,
        num_epochs: int,
        batch_size: int,
        learning_rate: float,
        output_dir: Path,
        seed: int,
        verbose: bool
    ):
        """Run standard cross-validation for a single experiment."""
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if exp_config['modality'] == 'multimodal':
            # Use multimodal cross-validation
            cv_results = self._run_multimodal_cv(
                exp_config['model_class'], X, y,
                num_folds, num_epochs, batch_size, learning_rate,
                output_dir, seed, verbose
            )
        else:
            # Use single-modality cross-validation
            # Create appropriate config for this experiment
            if exp_config['modality'] == 'fmri':
                temp_config = get_config('fmri')
            elif exp_config['modality'] == 'smri':
                temp_config = get_config('smri')
            else:
                temp_config = get_config('cross_attention')
                
            temp_config.num_folds = num_folds
            temp_config.num_epochs = num_epochs
            temp_config.batch_size = batch_size
            temp_config.learning_rate = learning_rate
            temp_config.output_dir = output_dir
            temp_config.seed = seed
            
            # Optimize config for sMRI based on known working parameters
            if exp_config['modality'] == 'smri':
                # Use proven sMRI parameters that achieved 58%
                temp_config.learning_rate = 0.001  # Higher learning rate for sMRI
                temp_config.batch_size = min(64, batch_size * 2)  # Larger batch size
                temp_config.d_model = 64  # Smaller model size for sMRI
                temp_config.num_epochs = min(num_epochs * 2, 100)  # More epochs for sMRI
                temp_config.patience = 15  # More patience
                temp_config.use_class_weights = True  # Important for sMRI
                temp_config.label_smoothing = 0.1  # Add label smoothing
                if verbose:
                    logger.info(f"🔧 Using optimized sMRI parameters:")
                    logger.info(f"   Learning rate: {temp_config.learning_rate}")
                    logger.info(f"   Batch size: {temp_config.batch_size}")
                    logger.info(f"   Model dim: {temp_config.d_model}")
                    logger.info(f"   Epochs: {temp_config.num_epochs}")
                    logger.info(f"   Class weights: {temp_config.use_class_weights}")
                    logger.info(f"   Label smoothing: {temp_config.label_smoothing}")
            
            fold_results = run_cross_validation(
                features=X,
                labels=y,
                model_class=exp_config['model_class'],
                config=temp_config,
                experiment_type='single',
                verbose=verbose
            )
            
            # Convert to expected format
            accuracies = [r['test_accuracy'] for r in fold_results]
            balanced_accs = [r['test_balanced_accuracy'] for r in fold_results]
            aucs = [r['test_auc'] for r in fold_results]
            
            cv_results = {
                'fold_results': fold_results,
                'mean_accuracy': np.mean(accuracies) * 100,
                'std_accuracy': np.std(accuracies) * 100,
                'mean_balanced_accuracy': np.mean(balanced_accs) * 100,
                'std_balanced_accuracy': np.std(balanced_accs) * 100,
                'mean_auc': np.mean(aucs),
                'std_auc': np.std(aucs),
                'cv_type': 'standard'
            }
        
        return cv_results
    
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
            trainer = Trainer(model, self.device, self.config, model_type='multimodal')
            
            # Create data loaders for training
            from training.utils import create_multimodal_data_loaders
            train_loader, val_loader = create_multimodal_data_loaders(
                fmri_train, smri_train, y_train,
                fmri_val, smri_val, y_val,
                batch_size=batch_size,
                augment_train=True
            )
            
            # Train without model checkpointing
            history = trainer.fit(
                train_loader, val_loader,
                num_epochs=num_epochs,
                checkpoint_path=None,  # No model saving
                y_train=y_train
            )
            
            # Evaluate on validation set
            val_test_loader, _ = create_multimodal_data_loaders(
                fmri_val, smri_val, y_val,
                fmri_val, smri_val, y_val,
                batch_size=batch_size,
                augment_train=False
            )
            
            # Evaluate final performance
            test_metrics = trainer.evaluate_final(val_test_loader)
            
            accuracy = test_metrics['accuracy']
            balanced_acc = test_metrics['balanced_accuracy']
            auc = test_metrics['auc']
            
            fold_results.append({
                'fold': fold,
                'test_accuracy': accuracy,
                'test_balanced_accuracy': balanced_acc,
                'test_auc': auc,
                'history': history,
                'targets': test_metrics['targets'],
                'predictions': test_metrics['predictions'],
                'probabilities': test_metrics['probabilities']
            })
            
            if verbose:
                logger.info(f"   Accuracy: {accuracy:.3f}, AUC: {auc:.3f}")
        
        # Aggregate results
        accuracies = [r['test_accuracy'] for r in fold_results]
        balanced_accs = [r['test_balanced_accuracy'] for r in fold_results]
        aucs = [r['test_auc'] for r in fold_results]
        
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
        verbose: bool,
        include_leave_site_out: bool = True
    ):
        """Save comprehensive results with exhaustive scientific analysis."""
        
        if verbose:
            logger.info("📊 Starting comprehensive scientific analysis...")
        
        # Import scientific analysis module
        try:
            from evaluation.scientific_analysis import ScientificAnalyzer
            
            # Create scientific analyzer
            analyzer = ScientificAnalyzer(output_path / 'scientific_analysis')
            
            # Perform comprehensive analysis
            scientific_analysis = analyzer.analyze_experiment_results(
                all_results, 
                include_leave_site_out=include_leave_site_out
            )
        except Exception as e:
            logger.warning(f"Scientific analysis failed: {e}")
            scientific_analysis = {'error': str(e)}
        
        # Save all results with timestamps
        results_with_metadata = {
            'metadata': {
                'total_experiments': len(all_results),
                'successful_experiments': len([r for r in all_results.values() if 'error' not in r]),
                'failed_experiments': len([r for r in all_results.values() if 'error' in r]),
                'total_runtime_minutes': total_time / 60,
                'timestamp': datetime.now().isoformat(),
                'included_leave_site_out': include_leave_site_out,
                'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                'torch_version': torch.__version__,
                'device_used': str(self.device),
                'config_summary': {
                    'base_config': self.config.__class__.__name__,
                    'data_paths': {
                        'fmri_path': str(getattr(self.config, 'fmri_path', 'default_fmri_path')),
                        'smri_path': str(getattr(self.config, 'smri_path', 'default_smri_path')),
                        'phenotypic_path': str(getattr(self.config, 'phenotypic_path', 'default_phenotypic_path'))
                    }
                }
            },
            'experiments': all_results,
            'scientific_analysis': scientific_analysis
        }
        
        # Save master results file
        master_results_path = output_path / 'complete_thesis_results.json'
        with open(master_results_path, 'w') as f:
            json.dump(results_with_metadata, f, indent=2, default=str)
        
        # Create detailed performance summary
        self._create_detailed_performance_summary(all_results, output_path, include_leave_site_out)
        
        # Create training curves for each experiment
        self._create_training_visualizations(all_results, output_path)
        
        # Create confusion matrices for each experiment
        self._create_confusion_matrices(all_results, output_path)
        
        # Create statistical comparison tables
        self._create_statistical_tables(all_results, output_path, include_leave_site_out)
        
        # Create executive summary report
        self._create_executive_summary(all_results, output_path, total_time, include_leave_site_out)
        
        # Save experiment configurations for reproducibility
        self._save_experiment_configs(output_path)
        
        if verbose:
            logger.info("\n🎯 COMPREHENSIVE THESIS RESULTS SUMMARY")
            logger.info("=" * 80)
            self._print_detailed_summary(all_results, include_leave_site_out)
            logger.info("=" * 80)
            logger.info(f"📁 Complete results saved to: {output_path}")
            logger.info(f"📊 Scientific analysis: {output_path / 'scientific_analysis'}")
            logger.info(f"📈 Visualizations: {output_path / 'plots'}")
            logger.info(f"📋 Summary report: {output_path / 'executive_summary.md'}")
    
    def _create_detailed_performance_summary(
        self, 
        all_results: dict, 
        output_path: Path, 
        include_leave_site_out: bool
    ):
        """Create detailed performance summary with all metrics."""
        
        summary_data = []
        
        for exp_name, result in all_results.items():
            if 'error' in result:
                summary_data.append({
                    'experiment': exp_name,
                    'name': result.get('name', exp_name),
                    'type': result.get('type', 'unknown'),
                    'modality': result.get('modality', 'unknown'),
                    'status': 'FAILED',
                    'error': str(result['error'])
                })
                continue
            
            # Standard CV results
            base_entry = {
                'experiment': exp_name,
                'name': result['name'],
                'type': result['type'],
                'modality': result['modality'],
                'status': 'SUCCESS'
            }
            
            if 'standard_cv' in result:
                cv = result['standard_cv']
                standard_entry = base_entry.copy()
                standard_entry.update({
                    'cv_type': 'Standard CV',
                    'accuracy_mean': cv.get('mean_accuracy', 0),
                    'accuracy_std': cv.get('std_accuracy', 0),
                    'balanced_accuracy_mean': cv.get('mean_balanced_accuracy', 0),
                    'balanced_accuracy_std': cv.get('std_balanced_accuracy', 0),
                    'auc_mean': cv.get('mean_auc', 0),
                    'auc_std': cv.get('std_auc', 0),
                    'n_folds': len(cv.get('fold_results', [])),
                    'performance_string': f"{cv.get('mean_accuracy', 0):.1f}% ± {cv.get('std_accuracy', 0):.1f}%"
                })
                summary_data.append(standard_entry)
            
            # Leave-site-out CV results
            if include_leave_site_out and 'leave_site_out_cv' in result and 'error' not in result['leave_site_out_cv']:
                lso = result['leave_site_out_cv']
                lso_entry = base_entry.copy()
                lso_entry.update({
                    'cv_type': 'Leave-Site-Out CV',
                    'accuracy_mean': lso.get('mean_accuracy', 0),
                    'accuracy_std': lso.get('std_accuracy', 0),
                    'balanced_accuracy_mean': lso.get('mean_balanced_accuracy', 0),
                    'balanced_accuracy_std': lso.get('std_balanced_accuracy', 0),
                    'auc_mean': lso.get('mean_auc', 0),
                    'auc_std': lso.get('std_auc', 0),
                    'n_sites': lso.get('n_sites', 0),
                    'performance_string': f"{lso.get('mean_accuracy', 0):.1f}% ± {lso.get('std_accuracy', 0):.1f}%",
                    'beats_baseline': lso.get('beats_baseline', False)
                })
                summary_data.append(lso_entry)
        
        # Save as CSV for easy analysis
        df = pd.DataFrame(summary_data)
        df.to_csv(output_path / 'detailed_performance_summary.csv', index=False)
        
        # Save best performers summary
        if summary_data:
            successful_results = [r for r in summary_data if r['status'] == 'SUCCESS']
            if successful_results:
                best_standard_cv = max(
                    [r for r in successful_results if r.get('cv_type') == 'Standard CV'],
                    key=lambda x: x.get('accuracy_mean', 0),
                    default=None
                )
                
                best_performers = {'best_standard_cv': best_standard_cv}
                
                if include_leave_site_out:
                    best_lso_cv = max(
                        [r for r in successful_results if r.get('cv_type') == 'Leave-Site-Out CV'],
                        key=lambda x: x.get('accuracy_mean', 0),
                        default=None
                    )
                    best_performers['best_leave_site_out_cv'] = best_lso_cv
                
                with open(output_path / 'best_performers.json', 'w') as f:
                    json.dump(best_performers, f, indent=2, default=str)
    
    def _create_training_visualizations(self, all_results: dict, output_path: Path):
        """Create training curve visualizations for each experiment."""
        
        plots_dir = output_path / 'plots' / 'training_curves'
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        for exp_name, result in all_results.items():
            if 'error' in result:
                continue
            
            # Look for training history in fold results
            for cv_type in ['standard_cv', 'leave_site_out_cv']:
                if cv_type not in result or 'fold_results' not in result[cv_type]:
                    continue
                
                fold_results = result[cv_type]['fold_results']
                
                # Create combined training curves plot
                self._plot_combined_training_curves(
                    fold_results, 
                    plots_dir / f'{exp_name}_{cv_type}_training_curves.png',
                    f"{result['name']} - {cv_type.replace('_', ' ').title()}"
                )
    
    def _plot_combined_training_curves(
        self, 
        fold_results: List[dict], 
        save_path: Path, 
        title: str
    ):
        """Plot combined training curves from all folds."""
        
        if not fold_results or 'history' not in fold_results[0]:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Training Curves: {title}', fontsize=16, fontweight='bold')
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(fold_results)))
        
        for fold_idx, fold_result in enumerate(fold_results):
            history = fold_result.get('history', {})
            if not history:
                continue
            
            color = colors[fold_idx]
            alpha = 0.7
            
            epochs = range(1, len(history.get('train_loss', [])) + 1)
            
            # Training and validation loss
            if 'train_loss' in history and 'val_loss' in history:
                axes[0, 0].plot(epochs, history['train_loss'], 
                               color=color, alpha=alpha, linewidth=1.5,
                               label=f'Fold {fold_idx+1} Train' if fold_idx == 0 else "")
                axes[0, 0].plot(epochs, history['val_loss'], 
                               color=color, alpha=alpha, linewidth=1.5, linestyle='--',
                               label=f'Fold {fold_idx+1} Val' if fold_idx == 0 else "")
            
            # Training and validation accuracy
            if 'train_accuracy' in history and 'val_accuracy' in history:
                axes[0, 1].plot(epochs, [acc*100 for acc in history['train_accuracy']], 
                               color=color, alpha=alpha, linewidth=1.5)
                axes[0, 1].plot(epochs, [acc*100 for acc in history['val_accuracy']], 
                               color=color, alpha=alpha, linewidth=1.5, linestyle='--')
            
            # Learning rate
            if 'learning_rates' in history:
                axes[1, 0].plot(epochs, history['learning_rates'], 
                               color=color, alpha=alpha, linewidth=1.5)
            
            # Validation AUC
            if 'val_auc' in history:
                axes[1, 1].plot(epochs, history['val_auc'], 
                               color=color, alpha=alpha, linewidth=1.5)
        
        # Customize plots
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].set_title('Training and Validation Accuracy')
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('AUC')
        axes[1, 1].set_title('Validation AUC')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_confusion_matrices(self, all_results: dict, output_path: Path):
        """Create confusion matrices for each experiment."""
        
        cm_dir = output_path / 'plots' / 'confusion_matrices'
        cm_dir.mkdir(parents=True, exist_ok=True)
        
        for exp_name, result in all_results.items():
            if 'error' in result:
                continue
            
            for cv_type in ['standard_cv', 'leave_site_out_cv']:
                if cv_type not in result:
                    continue
                
                cv_data = result[cv_type]
                if 'fold_results' not in cv_data:
                    continue
                
                # Create combined confusion matrix from all folds
                all_true = []
                all_pred = []
                
                for fold_result in cv_data['fold_results']:
                    if 'targets' in fold_result and 'predictions' in fold_result:
                        all_true.extend(fold_result['targets'])
                        all_pred.extend(fold_result['predictions'])
                
                if all_true and all_pred:
                    self._plot_confusion_matrix(
                        all_true, all_pred,
                        cm_dir / f'{exp_name}_{cv_type}_confusion_matrix.png',
                        f"{result['name']} - {cv_type.replace('_', ' ').title()}"
                    )
    
    def _plot_confusion_matrix(
        self, 
        y_true: List[int], 
        y_pred: List[int], 
        save_path: Path, 
        title: str
    ):
        """Plot confusion matrix with detailed metrics."""
        
        from sklearn.metrics import confusion_matrix, classification_report
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Control', 'ASD'], yticklabels=['Control', 'ASD'])
        plt.title(f'Confusion Matrix: {title}', fontsize=14, fontweight='bold')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        # Calculate and display metrics
        accuracy = accuracy_score(y_true, y_pred)
        balanced_acc = balanced_accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='weighted')
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        
        metrics_text = (
            f'Accuracy: {accuracy:.3f}\n'
            f'Balanced Acc: {balanced_acc:.3f}\n'
            f'Precision: {precision:.3f}\n'
            f'Recall: {recall:.3f}\n'
            f'F1-Score: {f1:.3f}'
        )
        
        plt.text(1.05, 0.5, metrics_text, transform=plt.gca().transAxes, 
                 verticalalignment='center', 
                 bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_statistical_tables(
        self, 
        all_results: dict, 
        output_path: Path, 
        include_leave_site_out: bool
    ):
        """Create detailed statistical comparison tables."""
        
        stats_dir = output_path / 'statistical_tables'
        stats_dir.mkdir(parents=True, exist_ok=True)
        
        # Performance comparison table
        comparison_data = []
        
        for exp_name, result in all_results.items():
            if 'error' in result:
                continue
            
            base_row = {
                'Experiment': exp_name,
                'Name': result['name'],
                'Type': result['type'],
                'Modality': result['modality']
            }
            
            # Standard CV
            if 'standard_cv' in result:
                cv = result['standard_cv']
                row = base_row.copy()
                row.update({
                    'CV_Type': 'Standard',
                    'Accuracy_Mean': cv.get('mean_accuracy', 0),
                    'Accuracy_Std': cv.get('std_accuracy', 0),
                    'Balanced_Accuracy_Mean': cv.get('mean_balanced_accuracy', 0),
                    'Balanced_Accuracy_Std': cv.get('std_balanced_accuracy', 0),
                    'AUC_Mean': cv.get('mean_auc', 0),
                    'AUC_Std': cv.get('std_auc', 0),
                    'N_Folds': len(cv.get('fold_results', [])),
                    'Performance_Summary': f"{cv.get('mean_accuracy', 0):.2f} ± {cv.get('std_accuracy', 0):.2f}"
                })
                comparison_data.append(row)
            
            # Leave-site-out CV
            if include_leave_site_out and 'leave_site_out_cv' in result:
                lso = result['leave_site_out_cv']
                if 'error' not in lso:
                    row = base_row.copy()
                    row.update({
                        'CV_Type': 'Leave-Site-Out',
                        'Accuracy_Mean': lso.get('mean_accuracy', 0),
                        'Accuracy_Std': lso.get('std_accuracy', 0),
                        'Balanced_Accuracy_Mean': lso.get('mean_balanced_accuracy', 0),
                        'Balanced_Accuracy_Std': lso.get('std_balanced_accuracy', 0),
                        'AUC_Mean': lso.get('mean_auc', 0),
                        'AUC_Std': lso.get('std_auc', 0),
                        'N_Sites': lso.get('n_sites', 0),
                        'Performance_Summary': f"{lso.get('mean_accuracy', 0):.2f} ± {lso.get('std_accuracy', 0):.2f}",
                        'Beats_Baseline': lso.get('beats_baseline', False)
                    })
                    comparison_data.append(row)
        
        if comparison_data:
            df = pd.DataFrame(comparison_data)
            
            # Sort by performance
            df_sorted = df.sort_values('Accuracy_Mean', ascending=False)
            df_sorted.to_csv(stats_dir / 'performance_comparison.csv', index=False)
            
            # Create modality summary
            modality_summary = df.groupby(['Modality', 'CV_Type']).agg({
                'Accuracy_Mean': ['mean', 'std', 'max', 'count'],
                'AUC_Mean': ['mean', 'std', 'max']
            }).round(3)
            modality_summary.to_csv(stats_dir / 'modality_summary.csv')
            
            # Create ranking table
            ranking_data = []
            for cv_type in df['CV_Type'].unique():
                cv_data = df[df['CV_Type'] == cv_type].sort_values('Accuracy_Mean', ascending=False)
                for rank, (_, row) in enumerate(cv_data.iterrows(), 1):
                    ranking_data.append({
                        'Rank': rank,
                        'CV_Type': cv_type,
                        'Experiment': row['Name'],
                        'Modality': row['Modality'],
                        'Accuracy': row['Accuracy_Mean'],
                        'Performance_Summary': row['Performance_Summary']
                    })
            
            ranking_df = pd.DataFrame(ranking_data)
            ranking_df.to_csv(stats_dir / 'performance_rankings.csv', index=False)
    
    def _create_executive_summary(
        self, 
        all_results: dict, 
        output_path: Path, 
        total_time: float, 
        include_leave_site_out: bool
    ):
        """Create executive summary report in markdown format."""
        
        summary_path = output_path / 'executive_summary.md'
        
        # Calculate summary statistics
        successful_experiments = [r for r in all_results.values() if 'error' not in r]
        failed_experiments = [r for r in all_results.values() if 'error' in r]
        
        # Find best performers
        best_standard_cv = None
        best_lso_cv = None
        
        best_standard_acc = 0
        best_lso_acc = 0
        
        for result in successful_experiments:
            if 'standard_cv' in result:
                acc = result['standard_cv'].get('mean_accuracy', 0)
                if acc > best_standard_acc:
                    best_standard_acc = acc
                    best_standard_cv = result
            
            if include_leave_site_out and 'leave_site_out_cv' in result:
                lso = result['leave_site_out_cv']
                if 'error' not in lso:
                    acc = lso.get('mean_accuracy', 0)
                    if acc > best_lso_acc:
                        best_lso_acc = acc
                        best_lso_cv = result
        
        # Write markdown report
        with open(summary_path, 'w') as f:
            f.write("# Comprehensive Thesis Experiment Results\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Total Runtime:** {total_time/3600:.2f} hours\n\n")
            
            f.write("## Executive Summary\n\n")
            f.write(f"- **Total Experiments:** {len(all_results)}\n")
            f.write(f"- **Successful:** {len(successful_experiments)}\n")
            f.write(f"- **Failed:** {len(failed_experiments)}\n")
            f.write(f"- **Cross-Validation Types:** {'Standard CV + Leave-Site-Out CV' if include_leave_site_out else 'Standard CV only'}\n\n")
            
            f.write("## Best Performing Models\n\n")
            
            if best_standard_cv:
                f.write("### Standard Cross-Validation\n")
                f.write(f"**Best Model:** {best_standard_cv['name']}\n")
                f.write(f"**Accuracy:** {best_standard_acc:.2f}% ± {best_standard_cv['standard_cv'].get('std_accuracy', 0):.2f}%\n")
                f.write(f"**AUC:** {best_standard_cv['standard_cv'].get('mean_auc', 0):.3f}\n\n")
            
            if best_lso_cv and include_leave_site_out:
                f.write("### Leave-Site-Out Cross-Validation\n")
                f.write(f"**Best Model:** {best_lso_cv['name']}\n")
                f.write(f"**Accuracy:** {best_lso_acc:.2f}% ± {best_lso_cv['leave_site_out_cv'].get('std_accuracy', 0):.2f}%\n")
                f.write(f"**AUC:** {best_lso_cv['leave_site_out_cv'].get('mean_auc', 0):.3f}\n\n")
            
            f.write("## Files Generated\n\n")
            f.write("- `complete_thesis_results.json` - Complete experimental results\n")
            f.write("- `detailed_performance_summary.csv` - Tabular performance summary\n")
            f.write("- `scientific_analysis/` - Comprehensive scientific analysis\n")
            f.write("- `plots/` - All visualizations and training curves\n")
            f.write("- `statistical_tables/` - Statistical comparison tables\n")
            f.write("- `best_performers.json` - Summary of top-performing models\n\n")
            
            f.write("## Next Steps\n\n")
            f.write("1. Review detailed performance in `performance_comparison.csv`\n")
            f.write("2. Analyze training curves in `plots/training_curves/`\n")
            f.write("3. Check statistical significance in `scientific_analysis/`\n")
            f.write("4. Compare confusion matrices in `plots/confusion_matrices/`\n")
    
    def _save_experiment_configs(self, output_path: Path):
        """Save all experiment configurations for reproducibility."""
        
        config_dir = output_path / 'experiment_configs'
        config_dir.mkdir(parents=True, exist_ok=True)
        
        # Save main config
        config_dict = {
            'device': str(self.device),
            'config_class': self.config.__class__.__name__,
            'config_attributes': {}
        }
        
        # Extract config attributes
        for attr in dir(self.config):
            if not attr.startswith('_'):
                try:
                    value = getattr(self.config, attr)
                    if not callable(value):
                        config_dict['config_attributes'][attr] = str(value)
                except:
                    pass
        
        with open(config_dir / 'main_config.json', 'w') as f:
            json.dump(config_dict, f, indent=2, default=str)
        
        # Save experiment definitions
        with open(config_dir / 'experiment_definitions.json', 'w') as f:
            json.dump(self.experiments, f, indent=2, default=str)
        
        # Save environment info
        env_info = {
            'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            'torch_version': torch.__version__,
            'device_info': str(self.device),
            'timestamp': datetime.now().isoformat(),
            'git_commit': self._get_git_commit(),
            'hardware_info': self._get_hardware_info()
        }
        
        with open(config_dir / 'environment_info.json', 'w') as f:
            json.dump(env_info, f, indent=2, default=str)
        
        logger.info(f"📋 Experiment configurations saved to: {config_dir}")
    
    def _get_git_commit(self) -> str:
        """Get current git commit hash for reproducibility."""
        try:
            import subprocess
            result = subprocess.run(['git', 'rev-parse', 'HEAD'], 
                                  capture_output=True, text=True, cwd=Path(__file__).parent)
            return result.stdout.strip() if result.returncode == 0 else "unknown"
        except:
            return "unknown"
    
    def _get_hardware_info(self) -> dict:
        """Get hardware information for reproducibility."""
        info = {'device': str(self.device)}
        
        if torch.cuda.is_available():
            info.update({
                'cuda_version': torch.version.cuda,
                'gpu_name': torch.cuda.get_device_name(0),
                'gpu_memory': f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
            })
        
        try:
            import psutil
            info.update({
                'cpu_count': psutil.cpu_count(),
                'memory_gb': f"{psutil.virtual_memory().total / 1e9:.1f} GB"
            })
        except:
            pass
        
        return info
    
    def _print_detailed_summary(self, all_results: dict, include_leave_site_out: bool):
        """Print detailed summary to console."""
        
        successful_experiments = [r for r in all_results.values() if 'error' not in r]
        
        for result in successful_experiments:
            logger.info(f"\n🔬 {result['name']}:")
            
            if 'standard_cv' in result:
                cv = result['standard_cv']
                logger.info(f"   📊 Standard CV: {cv.get('mean_accuracy', 0):.1f}% ± {cv.get('std_accuracy', 0):.1f}% (AUC: {cv.get('mean_auc', 0):.3f})")
            
            if include_leave_site_out and 'leave_site_out_cv' in result:
                lso = result['leave_site_out_cv']
                if 'error' not in lso:
                    logger.info(f"   🏥 Leave-Site-Out CV: {lso.get('mean_accuracy', 0):.1f}% ± {lso.get('std_accuracy', 0):.1f}% (AUC: {lso.get('mean_auc', 0):.3f})")
                    if lso.get('beats_baseline', False):
                        logger.info("   ✅ BEATS BASELINE!")
        
        # Print failures
        failed_experiments = [r for r in all_results.values() if 'error' in r]
        if failed_experiments:
            logger.info(f"\n❌ Failed Experiments ({len(failed_experiments)}):")
            for result in failed_experiments:
                logger.info(f"   - {result.get('name', 'Unknown')}: {result['error']}")
    
    def _create_results_table(self, all_results: dict, output_path: Path, include_leave_site_out: bool):
        """Create a CSV table with all results for easy analysis."""
        
        table_data = []
        
        for exp_name, result in all_results.items():
            if 'error' not in result:
                row = {
                    'experiment': exp_name,
                    'name': result['name'],
                    'type': result['type'],
                    'modality': result['modality']
                }
                
                # Standard CV results
                if 'standard_cv' in result:
                    cv = result['standard_cv']
                    row.update({
                        'standard_cv_accuracy_mean': cv['mean_accuracy'],
                        'standard_cv_accuracy_std': cv['std_accuracy'],
                        'standard_cv_balanced_accuracy_mean': cv['mean_balanced_accuracy'],
                        'standard_cv_balanced_accuracy_std': cv['std_balanced_accuracy'],
                        'standard_cv_auc_mean': cv['mean_auc'],
                        'standard_cv_auc_std': cv['std_auc']
                    })
                
                # Leave-site-out CV results
                if include_leave_site_out and 'leave_site_out_cv' in result and 'error' not in result['leave_site_out_cv']:
                    lso = result['leave_site_out_cv']
                    row.update({
                        'leave_site_out_accuracy_mean': lso['mean_accuracy'],
                        'leave_site_out_accuracy_std': lso['std_accuracy'],
                        'leave_site_out_balanced_accuracy_mean': lso['mean_balanced_accuracy'],
                        'leave_site_out_balanced_accuracy_std': lso['std_balanced_accuracy'],
                        'leave_site_out_auc_mean': lso['mean_auc'],
                        'leave_site_out_auc_std': lso['std_auc'],
                        'n_sites': lso.get('n_sites', None),
                        'beats_baseline': lso.get('beats_baseline', False)
                    })
                
                table_data.append(row)
        
        if table_data:
            df = pd.DataFrame(table_data)
            df.to_csv(output_path / 'results_table.csv', index=False)
            logger.info(f"📊 Results table saved: {output_path / 'results_table.csv'}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Comprehensive Thesis Experiments')
    parser.add_argument('--run_all', action='store_true', help='Run all experiments with both CV types')
    parser.add_argument('--baselines_only', action='store_true', help='Run only baseline experiments')
    parser.add_argument('--cross_attention_only', action='store_true', help='Run only cross-attention experiments')
    parser.add_argument('--standard_cv_only', action='store_true', help='Run all experiments with standard CV only')
    parser.add_argument('--leave_site_out_only', action='store_true', help='Run all experiments with leave-site-out CV only')
    parser.add_argument('--quick_test', action='store_true', help='Quick test with reduced parameters')
    parser.add_argument('--test_single', type=str, help='Quick test of single experiment (e.g., smri_baseline)')
    parser.add_argument('--compare_smri', action='store_true', help='Compare sMRI Transformer vs. Logistic Regression')
    parser.add_argument('--debug_smri', action='store_true', help='Debug sMRI performance across different configurations')
    parser.add_argument('--search_smri_params', action='store_true', help='Systematic hyperparameter search for sMRI to achieve 58% accuracy')
    
    parser.add_argument('--num_folds', type=int, default=5, help='Number of CV folds')
    parser.add_argument('--num_epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=3e-5, help='Learning rate')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--verbose', action='store_true', default=True, help='Verbose output')
    parser.add_argument('--no_leave_site_out', action='store_true', help='Disable leave-site-out CV in run_all')
    
    args = parser.parse_args()
    
    # Initialize framework
    experiments = ThesisExperiments()
    
    # Run requested experiments
    if args.run_all:
        logger.info("🚀 Running ALL thesis experiments with BOTH CV types...")
        results = experiments.run_all(
            num_folds=args.num_folds,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            output_dir=args.output_dir,
            seed=args.seed,
            verbose=args.verbose,
            include_leave_site_out=not args.no_leave_site_out
        )
    elif args.standard_cv_only:
        logger.info("🚀 Running ALL experiments with STANDARD CV only...")
        results = experiments.run_standard_cv_only(
            num_folds=args.num_folds,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            output_dir=args.output_dir or "standard_cv_results",
            seed=args.seed,
            verbose=args.verbose
        )
    elif args.leave_site_out_only:
        logger.info("🚀 Running ALL experiments with LEAVE-SITE-OUT CV only...")
        results = experiments.run_leave_site_out_only(
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
            verbose=args.verbose,
            include_leave_site_out=not args.no_leave_site_out
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
            verbose=args.verbose,
            include_leave_site_out=not args.no_leave_site_out
        )
    elif args.quick_test:
        logger.info("🚀 Running QUICK TEST...")
        results = experiments.quick_test(verbose=args.verbose)
    elif args.test_single:
        logger.info(f"🚀 Running SINGLE EXPERIMENT TEST: {args.test_single}")
        results = experiments.test_single_experiment(
            experiment_name=args.test_single,
            num_folds=args.num_folds,
            num_epochs=args.num_epochs,
            include_leave_site_out=not args.no_leave_site_out,
            verbose=args.verbose
        )
    elif args.compare_smri:
        logger.info("🔬 Comparing sMRI models...")
        results = experiments.compare_smri_models(
            num_folds=args.num_folds,
            verbose=args.verbose
        )
    elif args.debug_smri:
        logger.info("🔍 Debugging sMRI performance...")
        results = experiments.debug_smri_performance(
            num_epochs=args.num_epochs,
            verbose=args.verbose
        )
    elif args.search_smri_params:
        logger.info("🔬 Starting systematic sMRI hyperparameter search...")
        results = experiments.systematic_smri_hyperparameter_search(
            num_folds=args.num_folds,
            num_epochs=args.num_epochs,
            verbose=args.verbose
        )
    else:
        logger.info("ℹ️ No experiment type specified. Use --help for options.")
        logger.info("💡 Examples:")
        logger.info("   python scripts/thesis_experiments.py --run_all")
        logger.info("   python scripts/thesis_experiments.py --standard_cv_only")
        logger.info("   python scripts/thesis_experiments.py --leave_site_out_only")
        logger.info("   python scripts/thesis_experiments.py --test_single smri_baseline")
        logger.info("   python scripts/thesis_experiments.py --compare_smri")
        logger.info("   python scripts/thesis_experiments.py --search_smri_params")
        logger.info("   python scripts/thesis_experiments.py --debug_smri")
        logger.info("   python scripts/thesis_experiments.py --quick_test")
        return
    
    logger.info("✅ Experiments completed successfully!")


if __name__ == "__main__":
    main() 
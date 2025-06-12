#!/usr/bin/env python3
"""
Tokenization Strategy Experiments for Cross-Attention Models
===========================================================

This script explores different ways to tokenize fMRI and sMRI modalities for cross-attention.
Current approach: Each modality = 1 token
New approaches: Create multiple tokens per modality

Usage:
    python scripts/tokenization_experiments.py --quick_test
    python scripts/tokenization_experiments.py --full_experiment  
    python scripts/tokenization_experiments.py --compare_strategies single_token,smri_grouped,hemisphere_tokens
"""

import os
import sys
import time
import warnings
import argparse
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import json
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / 'src'))

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🎯 Using device: {device}")

# Try to import project modules
try:
    from utils.subject_matching import get_matched_datasets
    from data.fmri_processor import FMRIDataProcessor  
    from data.smri_processor import SMRIDataProcessor
    PROJECT_MODULES_AVAILABLE = True
    print("✅ Project modules loaded successfully")
except ImportError as e:
    PROJECT_MODULES_AVAILABLE = False
    print(f"⚠️ Project modules not available: {e}")
    print("Will use synthetic data for demonstration")

# =============================================================================
# TOKENIZATION STRATEGIES ANALYSIS
# =============================================================================

def analyze_data_for_tokenization():
    """Analyze which modality is better for different tokenization strategies."""
    print("\n🔍 TOKENIZATION FEASIBILITY ANALYSIS")
    print("=" * 50)
    
    # Based on your current setup:
    print("📊 Current Data Dimensions:")
    print(f"   fMRI: 19,900 features (CC200 connectivity: 200×199/2)")
    print(f"   sMRI: 800 features (FreeSurfer after RFE selection)")
    
    print("\n🎯 Tokenization Strategy Recommendations:")
    
    print("\n1. 🧠 fMRI Tokenization Options:")
    print("   ✅ ROI-based tokens: 200 ROIs → 200 tokens (each with 199 connections)")
    print("   ✅ Network-based tokens: Group ROIs by brain networks (8-12 tokens)")
    print("   ✅ Hierarchical tokens: Multi-scale (regions → networks → hemispheres)")
    print("   ✅ PCA tokens: Dimensionality reduction to create semantic tokens")
    print("   ⚠️ Pros: Rich spatial structure, natural ROI boundaries")
    print("   ⚠️ Cons: High dimensionality, potential overfitting")
    
    print("\n2. 🏗️ sMRI Tokenization Options:")
    print("   ✅ Anatomical tokens: Group by brain structures (cortical, subcortical)")
    print("   ✅ Feature-type tokens: Volume, thickness, surface area groups")
    print("   ✅ Hemisphere tokens: Left vs right brain features")  
    print("   ✅ Statistical tokens: Cluster similar features")
    print("   ⚠️ Pros: Already feature-selected, more stable")
    print("   ⚠️ Cons: Lower dimensional, less natural structure")
    
    print("\n🏆 RECOMMENDATION:")
    print("   Start with sMRI tokenization - easier to implement and debug")
    print("   sMRI has cleaner feature structure after RFE selection")
    print("   Lower risk of overfitting with 800 vs 19,900 dimensions")
    
    return {
        'fmri_dimensions': 19900,
        'smri_dimensions': 800,
        'recommended_start': 'smri',
        'fmri_tokenization_options': ['roi_based', 'network_based', 'hierarchical', 'pca_based'],
        'smri_tokenization_options': ['anatomical', 'feature_type', 'hemisphere', 'statistical']
    }

# =============================================================================
# TOKENIZATION STRATEGIES
# =============================================================================

class TokenizationStrategy:
    """Base class for tokenization strategies."""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
    
    def tokenize_fmri(self, fmri_features: np.ndarray) -> np.ndarray:
        """Convert fMRI features to tokens. Should return (batch, n_tokens, token_dim)"""
        raise NotImplementedError
    
    def tokenize_smri(self, smri_features: np.ndarray) -> np.ndarray:
        """Convert sMRI features to tokens. Should return (batch, n_tokens, token_dim)"""
        raise NotImplementedError
    
    def get_info(self) -> Dict[str, Any]:
        """Return tokenization strategy information."""
        return {
            'name': self.name,
            'description': self.description
        }

class SingleTokenStrategy(TokenizationStrategy):
    """Current approach: Each modality = 1 token (baseline)."""
    
    def __init__(self):
        super().__init__(
            name="single_token",
            description="Current approach: Each modality as one token (baseline)"
        )
    
    def tokenize_fmri(self, fmri_features: np.ndarray) -> np.ndarray:
        # Shape: (batch, 19900) -> (batch, 1, 19900)
        return fmri_features[:, np.newaxis, :]
    
    def tokenize_smri(self, smri_features: np.ndarray) -> np.ndarray:
        # Shape: (batch, 800) -> (batch, 1, 800)  
        return smri_features[:, np.newaxis, :]

class sMRITokenStrategy(TokenizationStrategy):
    """Focus on sMRI tokenization - easier to start with."""
    
    def __init__(self, n_smri_tokens: int = 8):
        super().__init__(
            name="smri_focused",
            description=f"fMRI: 1 token, sMRI: {n_smri_tokens} semantic tokens"
        )
        self.n_smri_tokens = n_smri_tokens
    
    def tokenize_fmri(self, fmri_features: np.ndarray) -> np.ndarray:
        # Keep fMRI as single token to focus on sMRI tokenization
        return fmri_features[:, np.newaxis, :]
    
    def tokenize_smri(self, smri_features: np.ndarray) -> np.ndarray:
        """Split sMRI into semantic groups."""
        batch_size, n_features = smri_features.shape
        features_per_token = n_features // self.n_smri_tokens
        
        # Simple grouping - could be improved with anatomical knowledge
        tokens = []
        for i in range(self.n_smri_tokens):
            start_idx = i * features_per_token
            if i == self.n_smri_tokens - 1:
                # Last token gets remaining features
                end_idx = n_features
            else:
                end_idx = start_idx + features_per_token
            
            token = smri_features[:, start_idx:end_idx]
            tokens.append(token)
        
        # Stack tokens: (batch, n_tokens, features_per_token)
        max_len = max(token.shape[1] for token in tokens)
        padded_tokens = []
        for token in tokens:
            if token.shape[1] < max_len:
                padding = np.zeros((batch_size, max_len - token.shape[1]))
                token = np.concatenate([token, padding], axis=1)
            padded_tokens.append(token[:, np.newaxis, :])
        
        result = np.concatenate(padded_tokens, axis=1)
        return result

class PCATokenStrategy(TokenizationStrategy):
    """Use PCA to create meaningful tokens from features."""
    
    def __init__(self, fmri_tokens: int = 1, smri_tokens: int = 5, 
                 smri_components_per_token: int = 20):
        super().__init__(
            name="pca_tokens", 
            description=f"PCA-based: fMRI 1 token, sMRI {smri_tokens}×{smri_components_per_token} PCA tokens"
        )
        self.fmri_tokens = fmri_tokens
        self.smri_tokens = smri_tokens
        self.smri_components_per_token = smri_components_per_token
        
        # PCA transformers (will be fitted during first use)
        self.smri_pcas = []
        self.fitted = False
    
    def _fit_pcas(self, fmri_features: np.ndarray, smri_features: np.ndarray):
        """Fit PCA transformers on training data."""
        print(f"🔄 Fitting PCA for {self.name} strategy...")
        
        # Fit separate PCAs for sMRI tokens
        smri_split_size = smri_features.shape[1] // self.smri_tokens
        for i in range(self.smri_tokens):
            start_idx = i * smri_split_size
            end_idx = start_idx + smri_split_size if i < self.smri_tokens - 1 else smri_features.shape[1]
            
            pca = PCA(n_components=min(self.smri_components_per_token, end_idx - start_idx))
            pca.fit(smri_features[:, start_idx:end_idx])
            self.smri_pcas.append(pca)
        
        self.fitted = True
        print("✅ PCA fitting completed")
    
    def tokenize_fmri(self, fmri_features: np.ndarray) -> np.ndarray:
        # Keep fMRI as single token for simplicity
        return fmri_features[:, np.newaxis, :]
    
    def tokenize_smri(self, smri_features: np.ndarray) -> np.ndarray:
        batch_size = smri_features.shape[0]
        tokens = []
        
        smri_split_size = smri_features.shape[1] // self.smri_tokens
        for i, pca in enumerate(self.smri_pcas):
            start_idx = i * smri_split_size  
            end_idx = start_idx + smri_split_size if i < self.smri_tokens - 1 else smri_features.shape[1]
            
            token = pca.transform(smri_features[:, start_idx:end_idx])
            tokens.append(token[:, np.newaxis, :])
        
        result = np.concatenate(tokens, axis=1)
        return result

class ClusterTokenStrategy(TokenizationStrategy):
    """Use clustering to create feature groups as tokens."""
    
    def __init__(self, smri_clusters: int = 4):
        super().__init__(
            name="cluster_tokens",
            description=f"Clustering-based: fMRI 1 token, sMRI {smri_clusters} cluster tokens"
        )
        self.smri_clusters = smri_clusters
        
        # Feature cluster assignments (will be computed during first use)
        self.smri_cluster_labels = None
        self.fitted = False
    
    def _fit_clusters(self, fmri_features: np.ndarray, smri_features: np.ndarray):
        """Cluster features to create token groupings."""
        print(f"🔄 Computing feature clusters for {self.name} strategy...")
        
        # Cluster sMRI features based on correlation
        smri_corr = np.corrcoef(smri_features.T)
        kmeans_smri = KMeans(n_clusters=self.smri_clusters, random_state=42, n_init=10) 
        self.smri_cluster_labels = kmeans_smri.fit_predict(smri_corr)
        
        self.fitted = True
        print("✅ Feature clustering completed")
    
    def tokenize_fmri(self, fmri_features: np.ndarray) -> np.ndarray:
        # Keep fMRI as single token
        return fmri_features[:, np.newaxis, :]
    
    def tokenize_smri(self, smri_features: np.ndarray) -> np.ndarray:
        batch_size = smri_features.shape[0]
        
        tokens = []
        for cluster_id in range(self.smri_clusters):
            cluster_features = smri_features[:, self.smri_cluster_labels == cluster_id]
            # Each token = concatenated features in that cluster
            tokens.append(cluster_features[:, np.newaxis, :])
        
        # Pad to same size
        max_features = max(token.shape[2] for token in tokens)
        padded_tokens = []
        for token in tokens:
            if token.shape[2] < max_features:
                padding = np.zeros((batch_size, 1, max_features - token.shape[2]))
                token = np.concatenate([token, padding], axis=2)
            padded_tokens.append(token)
        
        result = np.concatenate(padded_tokens, axis=1)
        return result

class HemisphereTokenStrategy(TokenizationStrategy):
    """Create tokens based on brain hemispheres (for sMRI)."""
    
    def __init__(self):
        super().__init__(
            name="hemisphere_tokens",
            description="Anatomical: fMRI 1 token, sMRI left/right hemisphere tokens"
        )
    
    def tokenize_fmri(self, fmri_features: np.ndarray) -> np.ndarray:
        return fmri_features[:, np.newaxis, :]
    
    def tokenize_smri(self, smri_features: np.ndarray) -> np.ndarray:
        """Split sMRI features into hemispheres + subcortical."""
        batch_size, n_features = smri_features.shape
        
        # Simple heuristic: assume features are ordered by hemisphere
        # In real FreeSurfer data, you'd use actual anatomical labels
        left_hem_size = n_features // 3
        right_hem_size = n_features // 3  
        subcortical_size = n_features - left_hem_size - right_hem_size
        
        left_token = smri_features[:, :left_hem_size]
        right_token = smri_features[:, left_hem_size:left_hem_size + right_hem_size]
        subcortical_token = smri_features[:, left_hem_size + right_hem_size:]
        
        # Pad to same size
        max_size = max(left_token.shape[1], right_token.shape[1], subcortical_token.shape[1])
        
        def pad_token(token, target_size):
            if token.shape[1] < target_size:
                padding = np.zeros((batch_size, target_size - token.shape[1]))
                return np.concatenate([token, padding], axis=1)
            return token
        
        left_padded = pad_token(left_token, max_size)[:, np.newaxis, :]
        right_padded = pad_token(right_token, max_size)[:, np.newaxis, :] 
        subcortical_padded = pad_token(subcortical_token, max_size)[:, np.newaxis, :]
        
        result = np.concatenate([left_padded, right_padded, subcortical_padded], axis=1)
        return result

# =============================================================================
# CROSS-ATTENTION MODEL WITH TOKENIZATION
# =============================================================================

class TokenizedCrossAttentionTransformer(nn.Module):
    """Cross-attention transformer that works with tokenized inputs."""
    
    def __init__(
        self,
        fmri_token_dim: int,
        smri_token_dim: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        num_classes: int = 2
    ):
        super().__init__()
        
        self.d_model = d_model
        
        # Token projections
        self.fmri_projection = nn.Linear(fmri_token_dim, d_model)
        self.smri_projection = nn.Linear(smri_token_dim, d_model)
        
        # Modality encoders
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.fmri_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.smri_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Cross-attention
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        
        # Fusion and classification
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
        
        # Learnable [CLS] tokens for global representation
        self.fmri_cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.smri_cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
    def forward(self, fmri_tokens: torch.Tensor, smri_tokens: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with tokenized inputs.
        
        Args:
            fmri_tokens: (batch, n_fmri_tokens, fmri_token_dim)
            smri_tokens: (batch, n_smri_tokens, smri_token_dim)
        """
        batch_size = fmri_tokens.size(0)
        
        # Project tokens to model dimension
        fmri_projected = self.fmri_projection(fmri_tokens)  # (batch, n_fmri_tokens, d_model)
        smri_projected = self.smri_projection(smri_tokens)  # (batch, n_smri_tokens, d_model)
        
        # Add [CLS] tokens
        fmri_cls = self.fmri_cls_token.expand(batch_size, -1, -1)
        smri_cls = self.smri_cls_token.expand(batch_size, -1, -1)
        
        fmri_with_cls = torch.cat([fmri_cls, fmri_projected], dim=1)
        smri_with_cls = torch.cat([smri_cls, smri_projected], dim=1)
        
        # Self-attention within modalities
        fmri_encoded = self.fmri_encoder(fmri_with_cls)
        smri_encoded = self.smri_encoder(smri_with_cls)
        
        # Cross-attention: fMRI attends to sMRI
        fmri_attended, _ = self.cross_attn(fmri_encoded, smri_encoded, smri_encoded)
        
        # Extract [CLS] tokens for classification
        fmri_cls_final = fmri_attended[:, 0]  # (batch, d_model)
        smri_cls_final = smri_encoded[:, 0]   # (batch, d_model)
        
        # Fuse modalities
        fused = torch.cat([fmri_cls_final, smri_cls_final], dim=-1)
        fused = self.fusion(fused)
        
        # Classification
        logits = self.classifier(fused)
        return logits

# =============================================================================
# EXPERIMENT FRAMEWORK
# =============================================================================

class TokenizationExperiment:
    """Framework for running tokenization experiments."""
    
    def __init__(self, output_dir: str = None):
        self.device = device
        self.output_dir = Path(output_dir) if output_dir else Path(f"/content/drive/MyDrive/tokenization_experiments_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Focus on sMRI tokenization strategies (easier to start with)
        self.strategies = {
            'single_token': SingleTokenStrategy(),
            'smri_grouped': sMRITokenStrategy(n_smri_tokens=4),
            'smri_detailed': sMRITokenStrategy(n_smri_tokens=8),
            'pca_tokens': PCATokenStrategy(smri_tokens=5),
            'cluster_tokens': ClusterTokenStrategy(smri_clusters=4),
            'hemisphere_tokens': HemisphereTokenStrategy()
        }
        
        self.results = {}
    
    def load_data(self) -> Dict[str, np.ndarray]:
        """Load fMRI and sMRI data."""
        print("📊 Loading multimodal data...")
        
        if PROJECT_MODULES_AVAILABLE:
            # Load real data
            try:
                matched_data = get_matched_datasets(
                    fmri_roi_dir="/content/drive/MyDrive/b_data/ABIDE_pcp/cpac/filt_noglobal/rois_cc200",
                    smri_data_path="/content/drive/MyDrive/processed_smri_data_improved",
                    phenotypic_file="/content/drive/MyDrive/b_data/ABIDE_pcp/Phenotypic_V1_0b_preprocessed1.csv",
                    verbose=True
                )
                
                return {
                    'fmri_data': matched_data['fmri_features'],
                    'smri_data': matched_data['smri_features'], 
                    'labels': matched_data['fmri_labels'],
                    'subject_ids': matched_data.get('fmri_subject_ids', [])
                }
            except Exception as e:
                print(f"⚠️ Failed to load real data: {e}")
                print("Falling back to synthetic data...")
        
        # Synthetic data for testing
        print("🎲 Generating synthetic data for demonstration...")
        n_subjects = 200
        
        # Synthetic data matching real dimensions
        fmri_data = np.random.randn(n_subjects, 19900).astype(np.float32)
        smri_data = np.random.randn(n_subjects, 800).astype(np.float32)
        labels = np.random.randint(0, 2, n_subjects)
        
        # Add some structure to make it more realistic
        fmri_data = StandardScaler().fit_transform(fmri_data)
        smri_data = StandardScaler().fit_transform(smri_data)
        
        return {
            'fmri_data': fmri_data,
            'smri_data': smri_data,
            'labels': labels,
            'subject_ids': [f'synthetic_{i:04d}' for i in range(n_subjects)]
        }
    
    def create_data_splits(self, data: Dict[str, np.ndarray], test_size: float = 0.2, 
                          val_size: float = 0.1, random_state: int = 42) -> Dict[str, Dict[str, np.ndarray]]:
        """Create train/validation/test splits."""
        from sklearn.model_selection import train_test_split
        
        X_fmri, X_smri, y = data['fmri_data'], data['smri_data'], data['labels']
        
        # Train/test split
        X_fmri_train, X_fmri_test, X_smri_train, X_smri_test, y_train, y_test = train_test_split(
            X_fmri, X_smri, y, test_size=test_size, stratify=y, random_state=random_state
        )
        
        # Train/validation split
        X_fmri_train, X_fmri_val, X_smri_train, X_smri_val, y_train, y_val = train_test_split(
            X_fmri_train, X_smri_train, y_train, test_size=val_size/(1-test_size), 
            stratify=y_train, random_state=random_state
        )
        
        return {
            'train': {'fmri': X_fmri_train, 'smri': X_smri_train, 'labels': y_train},
            'val': {'fmri': X_fmri_val, 'smri': X_smri_val, 'labels': y_val},
            'test': {'fmri': X_fmri_test, 'smri': X_smri_test, 'labels': y_test}
        }
    
    def train_strategy(self, strategy_name: str, splits: Dict[str, Dict[str, np.ndarray]], 
                      num_epochs: int = 30, batch_size: int = 32, learning_rate: float = 1e-3,
                      patience: int = 8) -> Dict[str, Any]:
        """Train and evaluate a specific tokenization strategy."""
        print(f"\n🚀 Training {strategy_name} strategy...")
        
        strategy = self.strategies[strategy_name]
        
        # Fit strategy-specific components (PCA, clustering, etc.)
        if hasattr(strategy, '_fit_pcas') and not strategy.fitted:
            strategy._fit_pcas(splits['train']['fmri'], splits['train']['smri'])
        elif hasattr(strategy, '_fit_clusters') and not strategy.fitted:
            strategy._fit_clusters(splits['train']['fmri'], splits['train']['smri'])
        
        # Tokenize data
        train_fmri_tokens = strategy.tokenize_fmri(splits['train']['fmri'])
        train_smri_tokens = strategy.tokenize_smri(splits['train']['smri'])
        val_fmri_tokens = strategy.tokenize_fmri(splits['val']['fmri'])
        val_smri_tokens = strategy.tokenize_smri(splits['val']['smri'])
        
        print(f"📊 Token shapes - fMRI: {train_fmri_tokens.shape}, sMRI: {train_smri_tokens.shape}")
        
        # Create model
        model = TokenizedCrossAttentionTransformer(
            fmri_token_dim=train_fmri_tokens.shape[2],
            smri_token_dim=train_smri_tokens.shape[2],
            d_model=128,
            n_heads=4,
            n_layers=2,
            dropout=0.1
        ).to(self.device)
        
        # Create data loaders
        train_dataset = TensorDataset(
            torch.FloatTensor(train_fmri_tokens),
            torch.FloatTensor(train_smri_tokens),
            torch.LongTensor(splits['train']['labels'])
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(val_fmri_tokens),
            torch.FloatTensor(val_smri_tokens),
            torch.LongTensor(splits['val']['labels'])
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Training setup
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=patience//2)
        
        # Training loop
        best_val_acc = 0
        patience_counter = 0
        train_losses, val_accs = [], []
        
        for epoch in range(num_epochs):
            # Training
            model.train()
            train_loss = 0
            for fmri_batch, smri_batch, labels_batch in train_loader:
                fmri_batch = fmri_batch.to(self.device)
                smri_batch = smri_batch.to(self.device)
                labels_batch = labels_batch.to(self.device)
                
                optimizer.zero_grad()
                logits = model(fmri_batch, smri_batch)
                loss = criterion(logits, labels_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            model.eval()
            val_preds, val_true = [], []
            with torch.no_grad():
                for fmri_batch, smri_batch, labels_batch in val_loader:
                    fmri_batch = fmri_batch.to(self.device)
                    smri_batch = smri_batch.to(self.device)
                    
                    logits = model(fmri_batch, smri_batch)
                    preds = torch.argmax(logits, dim=1)
                    
                    val_preds.extend(preds.cpu().numpy())
                    val_true.extend(labels_batch.numpy())
            
            val_acc = accuracy_score(val_true, val_preds)
            train_losses.append(train_loss / len(train_loader))
            val_accs.append(val_acc)
            
            scheduler.step(val_acc)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                # Save best model
                torch.save(model.state_dict(), self.output_dir / f'{strategy_name}_best_model.pth')
            else:
                patience_counter += 1
            
            if epoch % 5 == 0 or patience_counter >= patience:
                print(f"Epoch {epoch}: Train Loss={train_loss/len(train_loader):.4f}, Val Acc={val_acc:.4f}")
            
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
        
        # Test evaluation
        test_fmri_tokens = strategy.tokenize_fmri(splits['test']['fmri'])
        test_smri_tokens = strategy.tokenize_smri(splits['test']['smri'])
        
        test_dataset = TensorDataset(
            torch.FloatTensor(test_fmri_tokens),
            torch.FloatTensor(test_smri_tokens),
            torch.LongTensor(splits['test']['labels'])
        )
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # Load best model for testing
        model.load_state_dict(torch.load(self.output_dir / f'{strategy_name}_best_model.pth'))
        model.eval()
        
        test_preds, test_true = [], []
        with torch.no_grad():
            for fmri_batch, smri_batch, labels_batch in test_loader:
                fmri_batch = fmri_batch.to(self.device)
                smri_batch = smri_batch.to(self.device)
                
                logits = model(fmri_batch, smri_batch)
                preds = torch.argmax(logits, dim=1)
                
                test_preds.extend(preds.cpu().numpy())
                test_true.extend(labels_batch.numpy())
        
        test_acc = accuracy_score(test_true, test_preds)
        
        print(f"✅ {strategy_name} - Best Val Acc: {best_val_acc:.4f}, Test Acc: {test_acc:.4f}")
        
        return {
            'strategy': strategy_name,
            'strategy_info': strategy.get_info(),
            'best_val_acc': best_val_acc,
            'test_acc': test_acc,
            'train_losses': train_losses,
            'val_accs': val_accs,
            'fmri_tokens_shape': train_fmri_tokens.shape,
            'smri_tokens_shape': train_smri_tokens.shape,
            'model_params': sum(p.numel() for p in model.parameters()),
            'test_predictions': test_preds,
            'test_true': test_true
        }
    
    def run_experiment(self, strategies: List[str] = None, quick_test: bool = False) -> Dict[str, Any]:
        """Run experiments for selected tokenization strategies."""
        print("🧪 TOKENIZATION STRATEGY EXPERIMENTS")
        print("=" * 60)
        
        # Analyze data for tokenization
        analysis = analyze_data_for_tokenization()
        
        # Load data
        data = self.load_data()
        splits = self.create_data_splits(data)
        
        print(f"\n📊 Data splits:")
        print(f"   Train: {len(splits['train']['labels'])} samples")
        print(f"   Val: {len(splits['val']['labels'])} samples") 
        print(f"   Test: {len(splits['test']['labels'])} samples")
        print(f"   Class distribution: {np.bincount(data['labels'])}")
        
        # Select strategies to run
        if strategies is None:
            if quick_test:
                strategies = ['single_token', 'smri_grouped']  # Quick test
            else:
                strategies = list(self.strategies.keys())  # All strategies
        
        # Run experiments
        num_epochs = 15 if quick_test else 30
        results = {}
        
        for strategy_name in strategies:
            try:
                result = self.train_strategy(
                    strategy_name, splits,
                    num_epochs=num_epochs,
                    batch_size=32,
                    learning_rate=1e-3,
                    patience=5 if quick_test else 8
                )
                results[strategy_name] = result
                
            except Exception as e:
                print(f"❌ Error training {strategy_name}: {e}")
                results[strategy_name] = {'error': str(e)}
        
        # Save results
        self.results = results
        self._save_results()
        self._create_summary()
        
        return results
    
    def _save_results(self):
        """Save detailed results to JSON."""
        # Prepare results for JSON serialization
        json_results = {}
        for strategy, result in self.results.items():
            if 'error' not in result:
                json_results[strategy] = {
                    'strategy_info': result['strategy_info'],
                    'best_val_acc': float(result['best_val_acc']),
                    'test_acc': float(result['test_acc']),
                    'model_params': int(result['model_params']),
                    'fmri_tokens_shape': result['fmri_tokens_shape'],
                    'smri_tokens_shape': result['smri_tokens_shape']
                }
            else:
                json_results[strategy] = result
        
        with open(self.output_dir / 'tokenization_results.json', 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"💾 Results saved to {self.output_dir / 'tokenization_results.json'}")
    
    def _create_summary(self):
        """Create summary report."""
        print("\n📈 TOKENIZATION EXPERIMENT SUMMARY")
        print("=" * 60)
        
        # Create DataFrame for easy comparison
        summary_data = []
        for strategy, result in self.results.items():
            if 'error' not in result:
                summary_data.append({
                    'Strategy': strategy,
                    'Description': result['strategy_info']['description'],
                    'Val Accuracy': f"{result['best_val_acc']:.4f}",
                    'Test Accuracy': f"{result['test_acc']:.4f}",
                    'fMRI Tokens': f"{result['fmri_tokens_shape'][1]}×{result['fmri_tokens_shape'][2]}",
                    'sMRI Tokens': f"{result['smri_tokens_shape'][1]}×{result['smri_tokens_shape'][2]}",
                    'Model Params': f"{result['model_params']:,}"
                })
            else:
                summary_data.append({
                    'Strategy': strategy,
                    'Description': 'ERROR',
                    'Val Accuracy': 'N/A',
                    'Test Accuracy': 'N/A',
                    'fMRI Tokens': 'N/A',
                    'sMRI Tokens': 'N/A',
                    'Model Params': 'N/A'
                })
        
        df = pd.DataFrame(summary_data)
        print(df.to_string(index=False))
        
        # Save summary
        df.to_csv(self.output_dir / 'tokenization_summary.csv', index=False)
        
        # Find best strategy
        valid_results = {k: v for k, v in self.results.items() if 'error' not in v}
        if valid_results:
            best_strategy = max(valid_results.items(), key=lambda x: x[1]['test_acc'])
            print(f"\n🏆 BEST STRATEGY: {best_strategy[0]}")
            print(f"   Test Accuracy: {best_strategy[1]['test_acc']:.4f}")
            print(f"   Description: {best_strategy[1]['strategy_info']['description']}")
        
        print(f"\n📁 All results saved to: {self.output_dir}")

# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Tokenization Strategy Experiments for Cross-Attention Models')
    
    # Experiment modes
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--quick_test', action='store_true',
                      help='Quick test with 2 strategies (15 epochs each)')
    group.add_argument('--full_experiment', action='store_true',
                      help='Full experiment with all strategies (30 epochs each)')
    group.add_argument('--compare_strategies', type=str,
                      help='Compare specific strategies (comma-separated list)')
    
    # Optional parameters
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for results')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of epochs (overrides mode defaults)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                       help='Learning rate')
    
    args = parser.parse_args()
    
    print("🔬 Tokenization Strategy Experiments for Cross-Attention Models")
    print("=" * 70)
    
    # Show analysis of tokenization feasibility
    analyze_data_for_tokenization()
    
    # Initialize experiment
    experiment = TokenizationExperiment(output_dir=args.output_dir)
    
    # Determine strategies and parameters
    if args.quick_test:
        print("\n🧪 QUICK TEST MODE")
        strategies = ['single_token', 'smri_grouped']
        quick_test = True
        
    elif args.full_experiment:
        print("\n🧪 FULL EXPERIMENT MODE")
        strategies = None  # All strategies
        quick_test = False
        
    elif args.compare_strategies:
        print(f"\n🧪 COMPARING STRATEGIES: {args.compare_strategies}")
        strategies = [s.strip() for s in args.compare_strategies.split(',')]
        quick_test = False
        
        # Validate strategy names
        available_strategies = list(experiment.strategies.keys())
        invalid_strategies = [s for s in strategies if s not in available_strategies]
        if invalid_strategies:
            print(f"❌ Invalid strategies: {invalid_strategies}")
            print(f"Available strategies: {available_strategies}")
            return
    
    # Run experiments
    try:
        results = experiment.run_experiment(strategies=strategies, quick_test=quick_test)
        
        print(f"\n🎉 EXPERIMENTS COMPLETED!")
        print(f"📁 Results saved to: {experiment.output_dir}")
        
        # Show quick summary
        valid_results = {k: v for k, v in results.items() if 'error' not in v}
        if valid_results:
            print(f"\n📊 Quick Results Summary:")
            for strategy, result in valid_results.items():
                print(f"   {strategy}: {result['test_acc']:.4f} test accuracy")
        
    except Exception as e:
        print(f"❌ Experiment failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 
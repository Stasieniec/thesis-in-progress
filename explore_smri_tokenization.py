#!/usr/bin/env python3
"""
Experiment: Exploring Tokenization Techniques for sMRI Data
===========================================================

This script investigates how to tokenize sMRI features by brain region
and evaluates the impact on transformer performance.

Usage:
    python explore_smri_tokenization.py analyze_data          # Explore data structure
    python explore_smri_tokenization.py test_tokenization     # Test tokenization methods
    python explore_smri_tokenization.py run_experiments       # Compare tokenized vs flat
    python explore_smri_tokenization.py visualize_results     # Visualize findings
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import json
import fire
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import re

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Try to import project modules, but make script work without them
try:
    from config import get_config
    from data import SMRIDataProcessor
    from models import SMRITransformer
    from evaluation import create_cv_visualizations, save_results
    PROJECT_MODULES_AVAILABLE = True
except ImportError:
    PROJECT_MODULES_AVAILABLE = False
    print("⚠️ Project modules not available - using standalone mode")


class SMRITokenizationExplorer:
    """Explore and evaluate tokenization strategies for sMRI data."""
    
    def __init__(self, data_path: str = None):
        """Initialize the explorer with data paths."""
        if data_path is None:
            # Try to use improved data first
            improved_path = "/content/drive/MyDrive/processed_smri_data_improved"
            original_path = "/content/drive/MyDrive/processed_smri_data"
            
            if os.path.exists(improved_path):
                self.data_path = improved_path
                print(f"✅ Using improved sMRI data (800 features)")
            elif os.path.exists(original_path):
                self.data_path = original_path
                print(f"📊 Using original sMRI data")
            else:
                raise FileNotFoundError(
                    f"No sMRI data found at:\n"
                    f"  - {improved_path}\n"
                    f"  - {original_path}\n"
                    f"Please check that Google Drive is mounted and data is uploaded."
                )
        else:
            self.data_path = data_path
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"Data path not found: {data_path}")
            
        self.features = None
        self.labels = None
        self.feature_names = None
        self.subject_ids = None
        self.metadata = None
        
    def analyze_data(self):
        """
        Step 1: Analyze the sMRI data structure and feature names.
        This helps us understand how to tokenize the data.
        """
        print("=" * 70)
        print("🔍 ANALYZING sMRI DATA STRUCTURE")
        print("=" * 70)
        
        # Load the data
        print(f"\n📁 Loading data from: {self.data_path}")
        
        # Load features and labels
        self.features = np.load(os.path.join(self.data_path, 'features.npy'))
        self.labels = np.load(os.path.join(self.data_path, 'labels.npy'))
        
        print(f"✅ Features shape: {self.features.shape}")
        print(f"✅ Labels shape: {self.labels.shape}")
        print(f"   - Class distribution: ASD={np.sum(self.labels)}, Control={len(self.labels)-np.sum(self.labels)}")
        
        # Load feature names
        feature_names_file = os.path.join(self.data_path, 'feature_names.txt')
        if os.path.exists(feature_names_file):
            with open(feature_names_file, 'r') as f:
                self.feature_names = [line.strip() for line in f.readlines()]
            print(f"✅ Loaded {len(self.feature_names)} feature names")
        else:
            print("⚠️ No feature_names.txt found - will use generic names")
            self.feature_names = [f"feature_{i}" for i in range(self.features.shape[1])]
        
        # Load metadata
        metadata_file = os.path.join(self.data_path, 'metadata.json')
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                self.metadata = json.load(f)
            print(f"✅ Loaded metadata")
        
        # Analyze feature names to understand structure
        print("\n📊 ANALYZING FEATURE NAMES...")
        self._analyze_feature_structure()
        
        # Identify brain regions
        print("\n🧠 IDENTIFYING BRAIN REGIONS...")
        self._identify_brain_regions()
        
        return self.features, self.labels, self.feature_names
    
    def _analyze_feature_structure(self):
        """Analyze the structure of feature names to understand naming conventions."""
        if not self.feature_names:
            return
            
        # Sample first 10 feature names
        print("\n📝 Sample feature names:")
        for i, name in enumerate(self.feature_names[:10]):
            print(f"   {i}: {name}")
        
        # Look for common patterns
        patterns = {
            'volume': 0,
            'area': 0,
            'thickness': 0,
            'mean': 0,
            'std': 0,
            'left': 0,
            'right': 0,
            'cortical': 0,
            'subcortical': 0,
            'white_matter': 0
        }
        
        for name in self.feature_names:
            name_lower = name.lower()
            for pattern in patterns:
                if pattern in name_lower:
                    patterns[pattern] += 1
        
        print("\n📊 Feature type distribution:")
        for pattern, count in sorted(patterns.items(), key=lambda x: x[1], reverse=True):
            if count > 0:
                print(f"   {pattern}: {count} features ({count/len(self.feature_names)*100:.1f}%)")
    
    def _identify_brain_regions(self):
        """Identify unique brain regions from feature names."""
        # Common FreeSurfer region patterns
        region_patterns = {
            # Subcortical structures
            'hippocampus': [],
            'amygdala': [],
            'thalamus': [],
            'caudate': [],
            'putamen': [],
            'pallidum': [],
            'accumbens': [],
            'ventricle': [],
            
            # Cortical regions (examples)
            'frontal': [],
            'temporal': [],
            'parietal': [],
            'occipital': [],
            'cingulate': [],
            'insula': [],
            
            # White matter
            'corpus_callosum': [],
            'white_matter': [],
            'wm': []
        }
        
        # Match features to regions
        for idx, name in enumerate(self.feature_names):
            name_lower = name.lower()
            for region, indices in region_patterns.items():
                if region.replace('_', '') in name_lower.replace('_', '') or \
                   region.replace('_', '-') in name_lower:
                    indices.append(idx)
        
        # Report findings
        print("\n🧠 Brain regions identified:")
        regions_found = {k: v for k, v in region_patterns.items() if v}
        for region, indices in sorted(regions_found.items(), key=lambda x: len(x[1]), reverse=True):
            print(f"   {region}: {len(indices)} features")
            
        self.region_feature_mapping = regions_found
        return regions_found
    
    def test_tokenization(self):
        """
        Step 2: Test different tokenization strategies.
        """
        print("\n" + "=" * 70)
        print("🧪 TESTING TOKENIZATION STRATEGIES")
        print("=" * 70)
        
        if self.features is None:
            print("⚠️ Loading data first...")
            self.analyze_data()
        
        # Strategy 1: Simple anatomical grouping
        print("\n📍 Strategy 1: Anatomical Region Grouping")
        tokens_v1 = self._tokenize_by_anatomy_simple()
        
        # Strategy 2: Hemisphere-aware tokenization
        print("\n📍 Strategy 2: Hemisphere-Aware Tokenization")
        tokens_v2 = self._tokenize_by_hemisphere()
        
        # Strategy 3: Feature-type grouping
        print("\n📍 Strategy 3: Feature-Type Grouping")
        tokens_v3 = self._tokenize_by_feature_type()
        
        # Strategy 4: Data-driven clustering
        print("\n📍 Strategy 4: Data-Driven Clustering")
        tokens_v4 = self._tokenize_by_clustering()
        
        return {
            'anatomical': tokens_v1,
            'hemisphere': tokens_v2,
            'feature_type': tokens_v3,
            'clustering': tokens_v4
        }
    
    def _tokenize_by_anatomy_simple(self) -> Dict[str, List[int]]:
        """Group features by major anatomical structures."""
        # Define major anatomical groups
        anatomy_groups = {
            'subcortical': ['hippocampus', 'amygdala', 'thalamus', 'caudate', 
                           'putamen', 'pallidum', 'accumbens'],
            'frontal': ['frontal', 'precentral', 'orbitofrontal', 'motor'],
            'temporal': ['temporal', 'hippocampal', 'entorhinal', 'parahippocampal'],
            'parietal': ['parietal', 'postcentral', 'precuneus', 'supramarginal'],
            'occipital': ['occipital', 'calcarine', 'cuneus', 'lingual'],
            'cingulate': ['cingulate', 'cingulum'],
            'white_matter': ['white_matter', 'wm', 'corpus_callosum', 'tract'],
            'ventricles': ['ventricle', 'csf'],
            'global': ['total', 'mean', 'whole_brain']
        }
        
        # Assign features to groups
        tokens = defaultdict(list)
        unassigned = []
        
        for idx, name in enumerate(self.feature_names):
            name_lower = name.lower()
            assigned = False
            
            for group, keywords in anatomy_groups.items():
                for keyword in keywords:
                    if keyword in name_lower:
                        tokens[group].append(idx)
                        assigned = True
                        break
                if assigned:
                    break
            
            if not assigned:
                unassigned.append(idx)
        
        # Add unassigned features to 'other' group
        if unassigned:
            tokens['other'] = unassigned
        
        # Report tokenization results
        print(f"\n✅ Created {len(tokens)} anatomical tokens:")
        for token_name, indices in sorted(tokens.items(), key=lambda x: len(x[1]), reverse=True):
            print(f"   {token_name}: {len(indices)} features")
            
        return dict(tokens)
    
    def _tokenize_by_hemisphere(self) -> Dict[str, List[int]]:
        """Group features by hemisphere and structure."""
        tokens = defaultdict(list)
        
        for idx, name in enumerate(self.feature_names):
            name_lower = name.lower()
            
            # Determine hemisphere
            if 'left' in name_lower or '_l_' in name_lower or name_lower.startswith('l_'):
                hemisphere = 'left'
            elif 'right' in name_lower or '_r_' in name_lower or name_lower.startswith('r_'):
                hemisphere = 'right'
            else:
                hemisphere = 'bilateral'
            
            # Determine structure type
            if any(x in name_lower for x in ['volume', 'area', 'thickness']):
                structure = 'morphometry'
            elif any(x in name_lower for x in ['mean', 'std', 'intensity']):
                structure = 'intensity'
            else:
                structure = 'other'
            
            token_name = f"{hemisphere}_{structure}"
            tokens[token_name].append(idx)
        
        print(f"\n✅ Created {len(tokens)} hemisphere-based tokens:")
        for token_name, indices in sorted(tokens.items()):
            print(f"   {token_name}: {len(indices)} features")
            
        return dict(tokens)
    
    def _tokenize_by_feature_type(self) -> Dict[str, List[int]]:
        """Group features by measurement type."""
        feature_types = {
            'volume': ['volume', 'vol'],
            'area': ['area', 'surfarea'],
            'thickness': ['thickness', 'thick'],
            'curvature': ['curvature', 'curv'],
            'intensity': ['mean', 'std', 'intensity'],
            'shape': ['folding', 'sulcal', 'gyrification']
        }
        
        tokens = defaultdict(list)
        unassigned = []
        
        for idx, name in enumerate(self.feature_names):
            name_lower = name.lower()
            assigned = False
            
            for feat_type, keywords in feature_types.items():
                for keyword in keywords:
                    if keyword in name_lower:
                        tokens[feat_type].append(idx)
                        assigned = True
                        break
                if assigned:
                    break
            
            if not assigned:
                unassigned.append(idx)
        
        if unassigned:
            tokens['other_measures'] = unassigned
        
        print(f"\n✅ Created {len(tokens)} feature-type tokens:")
        for token_name, indices in sorted(tokens.items(), key=lambda x: len(x[1]), reverse=True):
            print(f"   {token_name}: {len(indices)} features")
            
        return dict(tokens)
    
    def _tokenize_by_clustering(self, n_clusters: int = 20) -> Dict[str, List[int]]:
        """Use data-driven clustering to create tokens."""
        
        # Transpose features to cluster features (not subjects)
        feature_data = self.features.T  # Shape: (n_features, n_subjects)
        
        # Standardize
        scaler = StandardScaler()
        feature_data_scaled = scaler.fit_transform(feature_data)
        
        # Cluster features
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(feature_data_scaled)
        
        # Create tokens from clusters
        tokens = defaultdict(list)
        for idx, cluster in enumerate(cluster_labels):
            tokens[f'cluster_{cluster}'].append(idx)
        
        print(f"\n✅ Created {len(tokens)} data-driven tokens:")
        for token_name, indices in sorted(tokens.items(), key=lambda x: int(x[0].split('_')[1])):
            print(f"   {token_name}: {len(indices)} features")
            
        return dict(tokens)
    
    def run_experiments(self, num_folds: int = 3, num_epochs: int = 50):
        """
        Step 3: Compare different tokenization strategies with transformers.
        """
        print("\n" + "=" * 70)
        print("🔬 RUNNING TOKENIZATION EXPERIMENTS")
        print("=" * 70)
        
        if self.features is None:
            print("⚠️ Loading data first...")
            self.analyze_data()
        
        # Get tokenization strategies
        tokenization_strategies = self.test_tokenization()
        
        # Add baseline (no tokenization)
        tokenization_strategies['baseline'] = None
        
        results = {}
        
        for strategy_name, token_mapping in tokenization_strategies.items():
            print(f"\n{'='*50}")
            print(f"🧪 Testing strategy: {strategy_name.upper()}")
            print(f"{'='*50}")
            
            if strategy_name == 'baseline':
                # Run standard transformer
                accuracy = self._run_baseline_experiment(num_folds, num_epochs)
            else:
                # Run tokenized transformer
                accuracy = self._run_tokenized_experiment(
                    token_mapping, strategy_name, num_folds, num_epochs
                )
            
            results[strategy_name] = accuracy
        
        # Compare results
        print("\n" + "=" * 70)
        print("📊 EXPERIMENT RESULTS SUMMARY")
        print("=" * 70)
        
        for strategy, accuracy in sorted(results.items(), key=lambda x: x[1], reverse=True):
            improvement = (accuracy - results['baseline']) * 100 if 'baseline' in results else 0
            print(f"{strategy:20s}: {accuracy:.4f} ({improvement:+.1f}% vs baseline)")
        
        return results
    
    def _run_baseline_experiment(self, num_folds: int, num_epochs: int) -> float:
        """Run standard transformer without tokenization."""
        
        accuracies = []
        skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(self.features, self.labels)):
            print(f"\nFold {fold+1}/{num_folds}")
            
            # Split data
            X_train, X_val = self.features[train_idx], self.features[val_idx]
            y_train, y_val = self.labels[train_idx], self.labels[val_idx]
            
            # Standardize
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)
            
            # Create simple model
            model = nn.Sequential(
                nn.Linear(X_train.shape[1], 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, 2)
            )
            
            # Train
            acc = self._train_simple_model(model, X_train, y_train, X_val, y_val, num_epochs)
            accuracies.append(acc)
        
        return np.mean(accuracies)
    
    def _run_tokenized_experiment(self, token_mapping: Dict[str, List[int]], 
                                 strategy_name: str, num_folds: int, num_epochs: int) -> float:
        """Run transformer with tokenized features."""
        
        accuracies = []
        skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(self.features, self.labels)):
            print(f"\nFold {fold+1}/{num_folds}")
            
            # Split data
            X_train, X_val = self.features[train_idx], self.features[val_idx]
            y_train, y_val = self.labels[train_idx], self.labels[val_idx]
            
            # Standardize
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)
            
            # Tokenize features
            X_train_tokenized = self._create_token_sequences(X_train, token_mapping)
            X_val_tokenized = self._create_token_sequences(X_val, token_mapping)
            
            # Create tokenized transformer model
            model = self._create_tokenized_transformer(
                num_tokens=len(token_mapping),
                token_dims=[len(indices) for indices in token_mapping.values()],
                hidden_dim=128,
                num_heads=4,
                num_layers=2
            )
            
            # Train
            acc = self._train_tokenized_model(
                model, X_train_tokenized, y_train, X_val_tokenized, y_val, num_epochs
            )
            accuracies.append(acc)
        
        return np.mean(accuracies)
    
    def _create_token_sequences(self, X: np.ndarray, token_mapping: Dict[str, List[int]]) -> torch.Tensor:
        """Convert flat features to token sequences."""
        batch_size = X.shape[0]
        num_tokens = len(token_mapping)
        
        # Create list to store token tensors
        token_list = []
        
        for token_name, feature_indices in token_mapping.items():
            # Extract features for this token
            token_features = X[:, feature_indices]  # (batch_size, token_dim)
            token_list.append(torch.FloatTensor(token_features))
        
        # Stack tokens: (batch_size, num_tokens, max_token_dim)
        # Pad tokens to same dimension
        max_dim = max(token.shape[1] for token in token_list)
        padded_tokens = []
        
        for token in token_list:
            if token.shape[1] < max_dim:
                padding = torch.zeros(batch_size, max_dim - token.shape[1])
                token = torch.cat([token, padding], dim=1)
            padded_tokens.append(token.unsqueeze(1))
        
        return torch.cat(padded_tokens, dim=1)
    
    def _create_tokenized_transformer(self, num_tokens: int, token_dims: List[int],
                                    hidden_dim: int, num_heads: int, num_layers: int) -> nn.Module:
        """Create a transformer that processes tokenized input."""
        
        class TokenizedTransformer(nn.Module):
            def __init__(self):
                super().__init__()
                
                # Token embedding layers (one per token type)
                max_token_dim = max(token_dims)
                self.token_embed = nn.Linear(max_token_dim, hidden_dim)
                
                # Positional encoding for tokens
                self.pos_embed = nn.Parameter(torch.randn(1, num_tokens, hidden_dim))
                
                # Transformer encoder
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=hidden_dim,
                    nhead=num_heads,
                    dim_feedforward=hidden_dim * 4,
                    dropout=0.1,
                    batch_first=True
                )
                self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
                
                # Classification head
                self.classifier = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(hidden_dim // 2, 2)
                )
                
            def forward(self, x):
                # x shape: (batch_size, num_tokens, max_token_dim)
                
                # Embed tokens
                x = self.token_embed(x)  # (batch_size, num_tokens, hidden_dim)
                
                # Add positional encoding
                x = x + self.pos_embed
                
                # Apply transformer
                x = self.transformer(x)  # (batch_size, num_tokens, hidden_dim)
                
                # Global pooling (mean over tokens)
                x = x.mean(dim=1)  # (batch_size, hidden_dim)
                
                # Classify
                return self.classifier(x)
        
        return TokenizedTransformer()
    
    def _train_simple_model(self, model, X_train, y_train, X_val, y_val, num_epochs):
        """Train a simple model and return validation accuracy."""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        # Convert to tensors
        X_train = torch.FloatTensor(X_train).to(device)
        y_train = torch.LongTensor(y_train).to(device)
        X_val = torch.FloatTensor(X_val).to(device)
        y_val = torch.LongTensor(y_val).to(device)
        
        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        # Train
        model.train()
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            outputs = model(X_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()
        
        # Evaluate
        model.eval()
        with torch.no_grad():
            outputs = model(X_val)
            _, predicted = torch.max(outputs.data, 1)
            accuracy = (predicted == y_val).float().mean().item()
        
        return accuracy
    
    def _train_tokenized_model(self, model, X_train, y_train, X_val, y_val, num_epochs):
        """Train a tokenized transformer model."""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        # Move data to device
        X_train = X_train.to(device)
        y_train = torch.LongTensor(y_train).to(device)
        X_val = X_val.to(device)
        y_val = torch.LongTensor(y_val).to(device)
        
        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        # Train
        model.train()
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            outputs = model(X_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 10 == 0:
                print(f"   Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")
        
        # Evaluate
        model.eval()
        with torch.no_grad():
            outputs = model(X_val)
            _, predicted = torch.max(outputs.data, 1)
            accuracy = (predicted == y_val).float().mean().item()
        
        return accuracy
    
    def visualize_results(self):
        """
        Step 4: Visualize tokenization strategies and results.
        """
        print("\n" + "=" * 70)
        print("📊 VISUALIZING TOKENIZATION ANALYSIS")
        print("=" * 70)
        
        if self.features is None:
            print("⚠️ Loading data first...")
            self.analyze_data()
        
        # Get tokenization strategies
        tokenization_strategies = self.test_tokenization()
        
        # Create visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()
        
        for idx, (strategy_name, token_mapping) in enumerate(list(tokenization_strategies.items())[:4]):
            ax = axes[idx]
            
            # Visualize token sizes
            token_sizes = [len(indices) for indices in token_mapping.values()]
            token_names = list(token_mapping.keys())
            
            # Create bar plot
            bars = ax.bar(range(len(token_sizes)), token_sizes)
            ax.set_xlabel('Token')
            ax.set_ylabel('Number of Features')
            ax.set_title(f'{strategy_name.capitalize()} Tokenization')
            ax.set_xticks(range(len(token_names)))
            ax.set_xticklabels(token_names, rotation=45, ha='right')
            
            # Color bars by size
            colors = plt.cm.viridis(np.array(token_sizes) / max(token_sizes))
            for bar, color in zip(bars, colors):
                bar.set_color(color)
        
        plt.tight_layout()
        plt.savefig('smri_tokenization_strategies.png', dpi=300, bbox_inches='tight')
        print(f"✅ Saved visualization to: smri_tokenization_strategies.png")
        
        # Create correlation heatmap for anatomical tokenization
        if 'anatomical' in tokenization_strategies:
            self._visualize_token_correlations(tokenization_strategies['anatomical'])


    def _visualize_token_correlations(self, token_mapping: Dict[str, List[int]]):
        """Visualize correlations between tokens."""
        # Calculate mean features per token
        token_means = {}
        for token_name, indices in token_mapping.items():
            token_means[token_name] = self.features[:, indices].mean(axis=1)
        
        # Create correlation matrix
        token_names = list(token_means.keys())
        n_tokens = len(token_names)
        corr_matrix = np.zeros((n_tokens, n_tokens))
        
        for i, token1 in enumerate(token_names):
            for j, token2 in enumerate(token_names):
                corr_matrix[i, j] = np.corrcoef(token_means[token1], token_means[token2])[0, 1]
        
        # Plot heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, 
                   xticklabels=token_names,
                   yticklabels=token_names,
                   cmap='coolwarm',
                   center=0,
                   annot=True,
                   fmt='.2f')
        plt.title('Token Correlation Matrix (Anatomical Tokenization)')
        plt.tight_layout()
        plt.savefig('token_correlation_heatmap.png', dpi=300, bbox_inches='tight')
        print(f"✅ Saved correlation heatmap to: token_correlation_heatmap.png")


class SMRITokenizationCLI:
    """CLI interface for the tokenization explorer."""
    
    def analyze_data(self, data_path: str = None):
        """Analyze sMRI data structure and feature names."""
        explorer = SMRITokenizationExplorer(data_path)
        explorer.analyze_data()
    
    def test_tokenization(self, data_path: str = None):
        """Test different tokenization strategies."""
        explorer = SMRITokenizationExplorer(data_path)
        explorer.test_tokenization()
    
    def run_experiments(self, data_path: str = None, num_folds: int = 3, num_epochs: int = 50):
        """Run experiments comparing tokenization strategies."""
        explorer = SMRITokenizationExplorer(data_path)
        explorer.run_experiments(num_folds, num_epochs)
    
    def visualize_results(self, data_path: str = None):
        """Visualize tokenization analysis results."""
        explorer = SMRITokenizationExplorer(data_path)
        explorer.visualize_results()
    
    def full_analysis(self, data_path: str = None):
        """Run complete analysis pipeline."""
        print("🚀 RUNNING FULL TOKENIZATION ANALYSIS")
        print("=" * 70)
        
        explorer = SMRITokenizationExplorer(data_path)
        
        # Step 1: Analyze data
        explorer.analyze_data()
        
        # Step 2: Test tokenization
        explorer.test_tokenization()
        
        # Step 3: Run experiments
        results = explorer.run_experiments(num_folds=3, num_epochs=50)
        
        # Step 4: Visualize
        explorer.visualize_results()
        
        print("\n✅ ANALYSIS COMPLETE!")
        print("=" * 70)
        
        return results


if __name__ == '__main__':
    fire.Fire(SMRITokenizationCLI) 
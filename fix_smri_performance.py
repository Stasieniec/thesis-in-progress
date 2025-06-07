#!/usr/bin/env python3
"""
Comprehensive sMRI performance fix combining all working notebook insights.
This script applies every optimization from the working notebook to achieve 60% accuracy.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
import json
import warnings
warnings.filterwarnings('ignore')

# Set seeds for reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

from models.smri_transformer_working import WorkingNotebookSMRITransformer
from data import SMRIDataset

class WorkingNotebookPreprocessor:
    """Exact preprocessing pipeline from working notebook."""
    
    def __init__(self, feature_selection_k=300, f_score_ratio=0.6):
        self.feature_selection_k = feature_selection_k
        self.f_score_ratio = f_score_ratio
        self.scaler = None
        self.feature_selector = None
        
    def fit(self, X, y):
        """Fit preprocessing exactly as in working notebook."""
        # Handle NaN/inf values
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # RobustScaler (from working notebook)
        self.scaler = RobustScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Combined feature selection (F-score + Mutual Info)
        if self.feature_selection_k and self.feature_selection_k < X.shape[1]:
            print(f"Selecting {self.feature_selection_k} features using combined F-score + MI...")
            
            # Calculate F-scores
            f_scores, _ = f_classif(X_scaled, y)
            
            # Calculate mutual information
            mi_scores = mutual_info_classif(X_scaled, y, random_state=SEED)
            
            # Combine scores: 60% F-score + 40% MI (from data creation script)
            combined_scores = (self.f_score_ratio * f_scores + 
                             (1 - self.f_score_ratio) * mi_scores)
            
            # Select top features
            top_indices = np.argsort(combined_scores)[-self.feature_selection_k:]
            self.selected_features = np.zeros(X.shape[1], dtype=bool)
            self.selected_features[top_indices] = True
            
            X_selected = X_scaled[:, self.selected_features]
            print(f"Selected {self.feature_selection_k} best features out of {X.shape[1]}")
            return X_selected
        else:
            return X_scaled
    
    def transform(self, X):
        """Transform new data."""
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        X_scaled = self.scaler.transform(X)
        
        if hasattr(self, 'selected_features'):
            return X_scaled[:, self.selected_features]
        return X_scaled

class WorkingNotebookTrainer:
    """Complete training strategy from working notebook."""
    
    def __init__(self, model, device, config=None):
        self.model = model.to(device)
        self.device = device
        
        # Working notebook training parameters
        self.num_epochs = config.get('num_epochs', 100)  # Reduced for testing
        self.learning_rate = config.get('learning_rate', 1e-3)
        self.weight_decay = config.get('weight_decay', 1e-4)
        self.warmup_epochs = config.get('warmup_epochs', 10)
        self.patience = config.get('patience', 20)
        
    def train(self, train_loader, val_loader, y_train):
        """Train with complete working notebook strategy."""
        print(f"🎯 Training with working notebook strategy...")
        print(f"   Epochs: {self.num_epochs}, LR: {self.learning_rate}, Patience: {self.patience}")
        
        # Class weights for imbalanced data
        class_counts = np.bincount(y_train.astype(int))
        class_weights = torch.FloatTensor(len(y_train) / (len(class_counts) * class_counts)).to(self.device)
        print(f"   Class weights: {class_weights}")
        
        # Loss with class weights and label smoothing
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
        
        # AdamW optimizer with weight decay
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999)
        )
        
        # Learning rate scheduler with warmup
        def lr_lambda(epoch):
            if epoch < self.warmup_epochs:
                return (epoch + 1) / self.warmup_epochs
            return 0.95 ** (epoch - self.warmup_epochs)
        
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        # Training tracking
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'lr': []}
        best_val_acc = 0
        best_model_state = None
        patience_counter = 0
        
        print("-" * 80)
        
        for epoch in range(self.num_epochs):
            # Training epoch
            train_loss, train_acc = self._train_epoch(train_loader, criterion, optimizer)
            
            # Validation epoch
            val_loss, val_acc, _, _, _ = self._evaluate_epoch(val_loader, criterion)
            
            # Update learning rate
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            
            # Save history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['lr'].append(current_lr)
            
            # Print progress
            if (epoch + 1) % 10 == 0 or epoch < 10:
                print(f'Epoch [{epoch+1:3d}/{self.num_epochs}] '
                      f'Train: {train_acc:.4f}, Val: {val_acc:.4f}, '
                      f'Loss: {val_loss:.4f}, LR: {current_lr:.6f}')
            
            # Early stopping with patience
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = self.model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= self.patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
        
        # Load best model
        if best_model_state:
            self.model.load_state_dict(best_model_state)
        
        print("-" * 80)
        print(f"Training completed! Best validation accuracy: {best_val_acc:.4f}")
        
        return history, best_val_acc
    
    def _train_epoch(self, dataloader, criterion, optimizer):
        """Train for one epoch with gradient clipping."""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for features, labels in dataloader:
            features, labels = features.to(self.device), labels.to(self.device)
            
            optimizer.zero_grad()
            outputs = self.model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping (from working notebook)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        return total_loss / len(dataloader), correct / total
    
    def _evaluate_epoch(self, dataloader, criterion):
        """Evaluate for one epoch."""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for features, labels in dataloader:
                features, labels = features.to(self.device), labels.to(self.device)
                
                outputs = self.model(features)
                loss = criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                probs = torch.softmax(outputs, dim=1)
                
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())
        
        return total_loss / len(dataloader), correct / total, all_preds, all_labels, all_probs

def load_real_smri_data():
    """Try to load real sMRI data."""
    try:
        # Try multiple possible data locations
        possible_paths = [
            "data/processed_smri_features.npy",
            "test_smri_output/processed_features.npy",
            "../processed_smri_features.npy"
        ]
        
        for features_path in possible_paths:
            if Path(features_path).exists():
                labels_path = features_path.replace('features', 'labels')
                if Path(labels_path).exists():
                    features = np.load(features_path)
                    labels = np.load(labels_path)
                    print(f"✅ Loaded real data from {features_path}")
                    return features, labels
        
        print("⚠️ No real data found, creating realistic synthetic data...")
        return create_realistic_smri_data()
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return create_realistic_smri_data()

def create_realistic_smri_data():
    """Create realistic sMRI data that mimics FreeSurfer features."""
    print("🔬 Creating realistic FreeSurfer-like synthetic data...")
    
    n_samples = 870  # Realistic ABIDE size
    n_features = 694  # FreeSurfer feature count
    
    # Create realistic feature patterns
    np.random.seed(SEED)
    
    # Base features with realistic FreeSurfer-like distributions
    features = np.random.randn(n_samples, n_features)
    
    # Add realistic feature scaling (some volume, some area, some thickness)
    volume_features = features[:, :200] * 1000 + 5000  # Volume-like
    area_features = features[:, 200:400] * 500 + 2000   # Area-like
    thickness_features = features[:, 400:600] * 0.5 + 2.5  # Thickness-like
    other_features = features[:, 600:]  # Other measures
    
    features = np.column_stack([volume_features, area_features, thickness_features, other_features])
    
    # Add some realistic group differences (subtle but detectable)
    n_asd = 403
    n_control = 467
    
    # Create subtle but realistic group differences
    asd_effect = np.random.randn(n_features) * 0.1  # Small effect sizes
    
    # Apply group differences to first n_asd samples
    features[:n_asd] += asd_effect
    
    # Create labels
    labels = np.concatenate([np.ones(n_asd), np.zeros(n_control)])
    
    # Shuffle to mix groups
    indices = np.random.permutation(len(labels))
    features = features[indices]
    labels = labels[indices]
    
    print(f"   Created {n_samples} subjects with {n_features} features")
    print(f"   ASD: {np.sum(labels==1)}, Control: {np.sum(labels==0)}")
    
    return features, labels

def comprehensive_smri_test():
    """Comprehensive test with all working notebook optimizations."""
    print("🚀 Comprehensive sMRI Performance Fix")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"📱 Device: {device}")
    
    # Load data
    features, labels = load_real_smri_data()
    print(f"📊 Data: {features.shape[0]} subjects, {features.shape[1]} features")
    
    # Configuration from working notebook
    config = {
        'num_epochs': 100,
        'learning_rate': 1e-3,
        'weight_decay': 1e-4,
        'warmup_epochs': 10,
        'patience': 20,
        'batch_size': 16,
        'feature_selection_k': 300
    }
    
    print(f"📋 Working Notebook Configuration:")
    for key, value in config.items():
        print(f"   {key}: {value}")
    
    # Cross-validation for robust evaluation
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    cv_results = []
    
    print(f"\n🔄 3-Fold Cross-Validation...")
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(features, labels)):
        print(f"\n{'='*20} FOLD {fold+1}/3 {'='*20}")
        
        # Split data
        X_train_fold, X_test_fold = features[train_idx], features[test_idx]
        y_train_fold, y_test_fold = labels[train_idx], labels[test_idx]
        
        # Create validation split
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_fold, y_train_fold, test_size=0.2, random_state=SEED, stratify=y_train_fold
        )
        
        print(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test_fold.shape[0]}")
        
        # Preprocessing with working notebook strategy
        preprocessor = WorkingNotebookPreprocessor(
            feature_selection_k=config['feature_selection_k'],
            f_score_ratio=0.6  # From data creation script
        )
        
        X_train_proc = preprocessor.fit(X_train, y_train)
        X_val_proc = preprocessor.transform(X_val)
        X_test_proc = preprocessor.transform(X_test_fold)
        
        # Create datasets with working notebook parameters
        train_dataset = SMRIDataset(X_train_proc, y_train, augment=True, noise_factor=0.005)
        val_dataset = SMRIDataset(X_val_proc, y_val, augment=False)
        test_dataset = SMRIDataset(X_test_proc, y_test_fold, augment=False)
        
        # Data loaders with weighted sampling
        class_counts = np.bincount(y_train.astype(int))
        class_weights = 1.0 / class_counts
        sample_weights = class_weights[y_train.astype(int)]
        sampler = WeightedRandomSampler(
            weights=sample_weights, num_samples=len(sample_weights), replacement=True
        )
        
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], sampler=sampler)
        val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
        
        # Create working notebook model
        model = WorkingNotebookSMRITransformer(
            input_dim=X_train_proc.shape[1],
            d_model=64,
            n_heads=4,
            n_layers=2,
            dropout=0.3,
            layer_dropout=0.1
        )
        
        # Train with working notebook strategy
        trainer = WorkingNotebookTrainer(model, device, config)
        history, best_val_acc = trainer.train(train_loader, val_loader, y_train)
        
        # Final test evaluation
        test_loss, test_acc, test_preds, test_labels, test_probs = trainer._evaluate_epoch(
            test_loader, nn.CrossEntropyLoss()
        )
        
        test_auc = roc_auc_score(test_labels, test_probs)
        
        cv_results.append({
            'fold': fold + 1,
            'test_acc': test_acc,
            'test_auc': test_auc,
            'val_acc': best_val_acc
        })
        
        print(f"Fold {fold+1} Results: Test Acc: {test_acc:.4f}, Test AUC: {test_auc:.4f}")
    
    # Final results
    mean_test_acc = np.mean([r['test_acc'] for r in cv_results])
    std_test_acc = np.std([r['test_acc'] for r in cv_results])
    mean_test_auc = np.mean([r['test_auc'] for r in cv_results])
    
    print(f"\n🎯 FINAL COMPREHENSIVE RESULTS:")
    print("=" * 60)
    print(f"Mean Test Accuracy: {mean_test_acc:.4f} ± {std_test_acc:.4f}")
    print(f"Mean Test AUC: {mean_test_auc:.4f}")
    print(f"Target Accuracy: 0.6000")
    print(f"Performance Gap: {0.6000 - mean_test_acc:+.4f}")
    
    # Performance assessment
    if mean_test_acc >= 0.60:
        print(f"\n🎯 SUCCESS! Achieved target performance!")
        status = "TARGET_ACHIEVED"
    elif mean_test_acc >= 0.58:
        print(f"\n✅ VERY GOOD! Close to target performance!")
        status = "CLOSE_TO_TARGET"
    elif mean_test_acc >= 0.55:
        print(f"\n✅ GOOD! Reasonable improvement!")
        status = "GOOD_IMPROVEMENT"
    else:
        print(f"\n⚠️ Still below target, but progress made!")
        status = "NEEDS_MORE_WORK"
    
    # Save results
    results = {
        'mean_test_accuracy': float(mean_test_acc),
        'std_test_accuracy': float(std_test_acc),
        'mean_test_auc': float(mean_test_auc),
        'target_accuracy': 0.60,
        'performance_gap': float(0.60 - mean_test_acc),
        'status': status,
        'cv_results': cv_results,
        'config': config
    }
    
    with open('comprehensive_smri_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💡 Applied Optimizations:")
    print(f"   ✅ Working notebook architecture")
    print(f"   ✅ Combined F-score + MI feature selection")
    print(f"   ✅ RobustScaler preprocessing")
    print(f"   ✅ Class weights + label smoothing")
    print(f"   ✅ Learning rate warmup + decay")
    print(f"   ✅ Gradient clipping")
    print(f"   ✅ Early stopping with patience")
    print(f"   ✅ Data augmentation")
    print(f"   ✅ Cross-validation evaluation")
    
    return results

if __name__ == "__main__":
    results = comprehensive_smri_test()
    print(f"\n🚀 Comprehensive sMRI fix completed!")
    print(f"   Final accuracy: {results['mean_test_accuracy']:.1%}")
    print(f"   Status: {results['status']}") 
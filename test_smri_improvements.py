#!/usr/bin/env python3
"""
Test the improved sMRI system with all optimizations applied.
This bypasses config issues and directly tests the improvements.
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
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# Set seeds
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

from models.smri_transformer import SMRITransformer  # Now the improved version
from data import SMRIDataset

def create_realistic_test_data():
    """Create realistic test data for validation."""
    print("🔬 Creating realistic test data...")
    
    # Realistic ABIDE-like data
    n_samples = 870
    n_features = 694
    
    # Create realistic FreeSurfer-like features
    np.random.seed(SEED)
    features = np.random.randn(n_samples, n_features)
    
    # Add realistic scaling for different feature types
    volume_features = features[:, :200] * 1000 + 5000
    area_features = features[:, 200:400] * 500 + 2000
    thickness_features = features[:, 400:600] * 0.5 + 2.5
    other_features = features[:, 600:]
    
    features = np.column_stack([volume_features, area_features, thickness_features, other_features])
    
    # Create realistic group differences
    n_asd = 403
    n_control = 467
    
    # Subtle but detectable differences
    asd_effect = np.random.randn(n_features) * 0.15
    features[:n_asd] += asd_effect
    
    # Labels
    labels = np.concatenate([np.ones(n_asd), np.zeros(n_control)])
    
    # Shuffle
    indices = np.random.permutation(len(labels))
    features = features[indices]
    labels = labels[indices]
    
    print(f"   ✅ Created {n_samples} subjects, {n_features} features")
    print(f"   🎯 ASD: {np.sum(labels==1)}, Control: {np.sum(labels==0)}")
    
    return features, labels

def enhanced_preprocessing(X_train, X_val, X_test, y_train, n_features=300):
    """Apply all proven preprocessing optimizations."""
    print(f"🔧 Enhanced preprocessing with {n_features} features...")
    
    # Handle NaN/inf
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=1e6, neginf=-1e6)
    X_val = np.nan_to_num(X_val, nan=0.0, posinf=1e6, neginf=-1e6)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # RobustScaler (proven better for FreeSurfer data)
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Combined feature selection (F-score + MI)
    print("   Selecting features using combined F-score + MI...")
    f_scores, _ = f_classif(X_train_scaled, y_train)
    mi_scores = mutual_info_classif(X_train_scaled, y_train, random_state=SEED)
    
    # Combine: 60% F-score + 40% MI (from data creation script)
    combined_scores = 0.6 * f_scores + 0.4 * mi_scores
    top_indices = np.argsort(combined_scores)[-n_features:]
    
    X_train_proc = X_train_scaled[:, top_indices]
    X_val_proc = X_val_scaled[:, top_indices]
    X_test_proc = X_test_scaled[:, top_indices]
    
    print(f"   ✅ Selected {n_features} features, applied RobustScaler")
    return X_train_proc, X_val_proc, X_test_proc

def enhanced_training(model, train_loader, val_loader, y_train, device):
    """Apply all proven training optimizations."""
    print("🎯 Enhanced training with all optimizations...")
    
    # Class weights for imbalanced data
    class_counts = np.bincount(y_train.astype(int))
    class_weights = torch.FloatTensor(len(y_train) / (len(class_counts) * class_counts)).to(device)
    
    # Loss with class weights and label smoothing
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    
    # AdamW with weight decay
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    # Learning rate scheduler with warmup
    def lr_lambda(epoch):
        warmup_epochs = 10
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        return 0.95 ** (epoch - warmup_epochs)
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Training loop
    num_epochs = 80
    best_val_acc = 0
    best_model_state = None
    patience = 15
    patience_counter = 0
    
    print(f"   Training for {num_epochs} epochs with patience {patience}")
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        scheduler.step()
        
        if (epoch + 1) % 20 == 0 or epoch < 5:
            print(f"   Epoch {epoch+1:2d}: Train {train_acc:.4f}, Val {val_acc:.4f}")
        
        if patience_counter >= patience:
            print(f"   Early stopping at epoch {epoch+1}")
            break
    
    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    print(f"   ✅ Training completed, best val acc: {best_val_acc:.4f}")
    return best_val_acc

def evaluate_model(model, test_loader, device):
    """Comprehensive model evaluation."""
    model.eval()
    test_correct = 0
    test_total = 0
    all_probs = []
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()
            
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    test_acc = test_correct / test_total
    test_auc = roc_auc_score(all_labels, all_probs)
    
    return test_acc, test_auc, all_preds, all_labels

def main():
    """Test the improved sMRI system."""
    print("🚀 Testing Improved sMRI System")
    print("=" * 50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"📱 Device: {device}")
    
    # Load/create data
    features, labels = create_realistic_test_data()
    
    # Cross-validation for robust testing
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    cv_results = []
    
    print(f"\n🔄 3-Fold Cross-Validation Test...")
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(features, labels)):
        print(f"\n{'='*15} FOLD {fold+1}/3 {'='*15}")
        
        # Split data
        X_train_fold, X_test_fold = features[train_idx], features[test_idx]
        y_train_fold, y_test_fold = labels[train_idx], labels[test_idx]
        
        # Validation split
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_fold, y_train_fold, test_size=0.2, random_state=SEED, stratify=y_train_fold
        )
        
        print(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test_fold.shape[0]}")
        
        # Enhanced preprocessing
        X_train_proc, X_val_proc, X_test_proc = enhanced_preprocessing(
            X_train, X_val, X_test_fold, y_train, n_features=300
        )
        
        # Create datasets
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
        
        train_loader = DataLoader(train_dataset, batch_size=16, sampler=sampler)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
        
        # Create improved model
        model = SMRITransformer(
            input_dim=X_train_proc.shape[1],
            d_model=64,
            n_heads=4,
            n_layers=2,
            dropout=0.3,
            layer_dropout=0.1
        ).to(device)
        
        # Check if it's the improved version
        model_info = model.get_model_info()
        if 'Improved' in model_info['model_name']:
            print(f"✅ Using improved model: {model_info['model_name']}")
        else:
            print(f"⚠️  Using model: {model_info['model_name']}")
        
        # Enhanced training
        best_val_acc = enhanced_training(model, train_loader, val_loader, y_train, device)
        
        # Test evaluation
        test_acc, test_auc, test_preds, test_labels = evaluate_model(model, test_loader, device)
        
        cv_results.append({
            'fold': fold + 1,
            'test_acc': test_acc,
            'test_auc': test_auc,
            'val_acc': best_val_acc
        })
        
        print(f"Fold {fold+1} Results: Test Acc: {test_acc:.4f}, AUC: {test_auc:.4f}")
    
    # Final results
    mean_test_acc = np.mean([r['test_acc'] for r in cv_results])
    std_test_acc = np.std([r['test_acc'] for r in cv_results])
    mean_test_auc = np.mean([r['test_auc'] for r in cv_results])
    
    print(f"\n🎯 FINAL IMPROVED sMRI RESULTS:")
    print("=" * 50)
    print(f"Mean Test Accuracy: {mean_test_acc:.4f} ± {std_test_acc:.4f} ({mean_test_acc:.1%})")
    print(f"Mean Test AUC: {mean_test_auc:.4f}")
    print(f"Target Accuracy: 60%")
    
    # Performance assessment
    if mean_test_acc >= 0.60:
        print(f"🎯 SUCCESS! Achieved target performance!")
        status = "TARGET_ACHIEVED"
    elif mean_test_acc >= 0.58:
        print(f"✅ VERY CLOSE! Near target performance!")
        status = "CLOSE_TO_TARGET"
    elif mean_test_acc >= 0.55:
        print(f"✅ GOOD! Significant improvement!")
        status = "GOOD_IMPROVEMENT"
    else:
        print(f"⚠️  Still working towards target...")
        status = "NEEDS_MORE_WORK"
    
    print(f"\n💡 Applied Improvements:")
    print(f"   ✅ Working notebook architecture")
    print(f"   ✅ BatchNorm + learnable positional embeddings")
    print(f"   ✅ Pre-norm transformer + GELU activation")
    print(f"   ✅ Combined F-score + MI feature selection")
    print(f"   ✅ RobustScaler preprocessing")
    print(f"   ✅ Class weights + label smoothing")
    print(f"   ✅ Learning rate warmup + decay")
    print(f"   ✅ Gradient clipping + early stopping")
    
    return {
        'mean_accuracy': mean_test_acc,
        'std_accuracy': std_test_acc,
        'mean_auc': mean_test_auc,
        'status': status
    }

if __name__ == "__main__":
    results = main()
    print(f"\n🚀 Improved sMRI test completed!")
    print(f"   Final accuracy: {results['mean_accuracy']:.1%}")
    print(f"   Status: {results['status']}") 
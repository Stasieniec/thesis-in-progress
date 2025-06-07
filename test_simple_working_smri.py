#!/usr/bin/env python3
"""
Simple test of working notebook sMRI architecture.
Focus on architecture comparison without complex config dependencies.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# Set seeds
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

from models.smri_transformer_working import WorkingNotebookSMRITransformer
from models.smri_transformer import SMRITransformer
from data import SMRIDataset

def load_smri_data():
    """Load sMRI data from the processed files."""
    try:
        # Try to load from the processed data
        base_path = Path("data")
        features_path = base_path / "processed_smri_features.npy"
        labels_path = base_path / "processed_smri_labels.npy"
        
        if features_path.exists() and labels_path.exists():
            features = np.load(features_path)
            labels = np.load(labels_path)
            print(f"✅ Loaded {len(labels)} subjects from processed files")
            return features, labels
        else:
            print("❌ Processed files not found, creating dummy data for testing...")
            # Create realistic dummy data for architecture testing
            n_samples = 870  # Realistic ABIDE size
            n_features = 694  # Realistic FreeSurfer feature count
            
            # Create realistic features (FreeSurfer-like data)
            features = np.random.randn(n_samples, n_features)
            # Add some realistic scaling and outliers
            features = features * np.random.uniform(0.5, 2.0, (1, n_features))
            features += np.random.uniform(-1, 1, (1, n_features))
            
            # Create realistic imbalanced labels (ASD vs Control)
            n_asd = 403  # Realistic ABIDE ASD count
            n_control = 467  # Realistic ABIDE Control count
            labels = np.concatenate([np.ones(n_asd), np.zeros(n_control)])
            
            # Shuffle
            indices = np.random.permutation(len(labels))
            features = features[indices]
            labels = labels[indices]
            
            print(f"✅ Created dummy data: {len(labels)} subjects, {features.shape[1]} features")
            return features, labels
            
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None

def preprocess_data(X_train, X_val, X_test, y_train, n_features=300):
    """Preprocess data exactly as in working notebook."""
    print(f"🔧 Preprocessing with {n_features} features...")
    
    # Handle NaN/inf values
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=1e6, neginf=-1e6)
    X_val = np.nan_to_num(X_val, nan=0.0, posinf=1e6, neginf=-1e6)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # RobustScaler (from working notebook)
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Feature selection (from working notebook)
    feature_selector = SelectKBest(score_func=f_classif, k=n_features)
    X_train_processed = feature_selector.fit_transform(X_train_scaled, y_train)
    X_val_processed = feature_selector.transform(X_val_scaled)
    X_test_processed = feature_selector.transform(X_test_scaled)
    
    print(f"✅ Selected {n_features} features, applied RobustScaler")
    return X_train_processed, X_val_processed, X_test_processed

def train_and_evaluate_model(model, train_loader, val_loader, test_loader, y_train, model_name, device):
    """Train and evaluate a model."""
    print(f"\n🎯 Training {model_name}...")
    
    # Move model to device
    model = model.to(device)
    
    # Calculate class weights
    class_counts = np.bincount(y_train.astype(int))
    class_weights = torch.FloatTensor(len(y_train) / (len(class_counts) * class_counts)).to(device)
    
    # Setup training
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    # Simple learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
    
    # Training loop
    num_epochs = 50  # Reduced for quick testing
    best_val_acc = 0
    best_model_state = None
    
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
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
        
        scheduler.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:2d}: Train {train_acc:.4f}, Val {val_acc:.4f}")
    
    # Load best model for testing
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    # Test evaluation
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
    
    print(f"  Final Results - Test Acc: {test_acc:.4f}, Test AUC: {test_auc:.4f}")
    
    return test_acc, test_auc, all_preds, all_labels

def main():
    """Main comparison test."""
    print("🔬 sMRI Architecture Comparison Test")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"📱 Device: {device}")
    
    # Load data
    print(f"\n📁 Loading sMRI data...")
    features, labels = load_smri_data()
    
    if features is None:
        print("❌ Could not load data. Exiting.")
        return
    
    print(f"📊 Data: {features.shape[0]} subjects, {features.shape[1]} features")
    print(f"🎯 Classes: ASD={np.sum(labels==1)}, Control={np.sum(labels==0)}")
    
    # Split data (same as working notebook: 70/15/15)
    X_temp, X_test, y_temp, y_test = train_test_split(
        features, labels, test_size=0.15, random_state=SEED, stratify=labels
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.15, random_state=SEED, stratify=y_temp
    )
    
    print(f"\n📊 Data Split:")
    print(f"   Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
    
    # Preprocess data
    X_train_proc, X_val_proc, X_test_proc = preprocess_data(
        X_train, X_val, X_test, y_train, n_features=300
    )
    
    # Create datasets and data loaders
    batch_size = 16
    
    train_dataset = SMRIDataset(X_train_proc, y_train, augment=True, noise_factor=0.005)
    val_dataset = SMRIDataset(X_val_proc, y_val, augment=False)
    test_dataset = SMRIDataset(X_test_proc, y_test, augment=False)
    
    # Weighted sampling for class imbalance
    class_counts = np.bincount(y_train.astype(int))
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[y_train.astype(int)]
    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Test both architectures
    input_dim = X_train_proc.shape[1]
    
    print(f"\n🧠 Architecture Comparison:")
    print(f"   Input dimension: {input_dim}")
    
    # Model 1: Current architecture
    print(f"\n1️⃣ Current SMRITransformer...")
    current_model = SMRITransformer(
        input_dim=input_dim,
        d_model=64,
        n_heads=4,
        n_layers=2,
        dropout=0.3
    )
    current_info = current_model.get_model_info()
    print(f"   Parameters: {current_info['total_params']:,}")
    
    current_acc, current_auc, _, _ = train_and_evaluate_model(
        current_model, train_loader, val_loader, test_loader, y_train, 
        "Current sMRI", device
    )
    
    # Model 2: Working notebook architecture
    print(f"\n2️⃣ Working Notebook SMRITransformer...")
    working_model = WorkingNotebookSMRITransformer(
        input_dim=input_dim,
        d_model=64,
        n_heads=4,
        n_layers=2,
        dropout=0.3,
        layer_dropout=0.1
    )
    working_info = working_model.get_model_info()
    print(f"   Parameters: {working_info['total_params']:,}")
    
    working_acc, working_auc, _, _ = train_and_evaluate_model(
        working_model, train_loader, val_loader, test_loader, y_train,
        "Working Notebook sMRI", device
    )
    
    # Results comparison
    print(f"\n📊 FINAL COMPARISON:")
    print("=" * 60)
    print(f"{'Architecture':<25} {'Accuracy':<10} {'AUC':<10} {'Target':<10}")
    print("-" * 60)
    print(f"{'Current sMRI':<25} {current_acc:<10.4f} {current_auc:<10.4f} {'~55%':<10}")
    print(f"{'Working Notebook':<25} {working_acc:<10.4f} {working_auc:<10.4f} {'~60%':<10}")
    print("-" * 60)
    
    improvement = working_acc - current_acc
    print(f"Improvement: {improvement:+.4f} ({improvement*100:+.1f}%)")
    
    # Analysis
    print(f"\n💡 Analysis:")
    if working_acc >= 0.60:
        print(f"   🎯 SUCCESS! Working notebook architecture achieved target!")
    elif working_acc > current_acc + 0.02:
        print(f"   ✅ Working notebook architecture shows clear improvement!")
    elif working_acc > current_acc:
        print(f"   ↗️  Working notebook architecture shows modest improvement")
    else:
        print(f"   ➡️  Architectures perform similarly")
    
    print(f"\n🔍 Key Differences in Working Notebook Architecture:")
    print(f"   • BatchNorm in input projection")
    print(f"   • Learnable positional embeddings")
    print(f"   • Pre-norm transformer layers")
    print(f"   • GELU activation")
    print(f"   • Layer dropout")
    print(f"   • Sophisticated weight initialization")
    print(f"   • Residual-like classification head")
    
    return {
        'current_acc': current_acc,
        'working_acc': working_acc,
        'improvement': improvement
    }

if __name__ == "__main__":
    results = main()
    print(f"\n🚀 Architecture comparison completed!")
    if results:
        print(f"   Working notebook improvement: {results['improvement']*100:+.1f}%") 
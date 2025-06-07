#!/usr/bin/env python3
"""
Test sMRI using EXACT working notebook architecture and training strategy.
This should achieve the 60% accuracy from the working notebook.
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
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import json

from config import get_config
from data import SMRIDataProcessor
from data import SMRIDataset
from models.smri_transformer_working import WorkingNotebookSMRITransformer
from training.utils import set_seed, EarlyStopping, calculate_class_weights

class WorkingNotebookTrainer:
    """Exact training strategy from working notebook."""
    
    def __init__(self, model, device, num_epochs=200, learning_rate=1e-3, 
                 weight_decay=1e-4, warmup_epochs=10):
        self.model = model.to(device)
        self.device = device
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        
    def train(self, train_loader, val_loader, y_train):
        """Train exactly as in working notebook."""
        # Calculate class weights for imbalanced dataset
        class_weights = calculate_class_weights(y_train, self.device)
        print(f"Class weights: {class_weights}")
        
        # Loss function with class weights and label smoothing (from notebook)
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
        
        # Optimizer with weight decay (from notebook)
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999)
        )
        
        # Learning rate scheduler with warmup (from notebook)
        def lr_lambda(epoch):
            if epoch < self.warmup_epochs:
                return (epoch + 1) / self.warmup_epochs
            return 0.95 ** (epoch - self.warmup_epochs)
        
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        # Early stopping (from notebook)
        early_stopping = EarlyStopping(patience=20, min_delta=0.001)
        
        # Training history
        history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [], 'lr': []
        }
        
        best_val_acc = 0
        print(f"Starting training with working notebook strategy...")
        print("-" * 80)
        
        for epoch in range(self.num_epochs):
            # Training epoch
            train_loss, train_acc = self._train_epoch(
                train_loader, criterion, optimizer
            )
            
            # Validation epoch
            val_loss, val_acc, _, _, _ = self._evaluate_epoch(
                val_loader, criterion
            )
            
            # Update learning rate
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            
            # Save history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['lr'].append(current_lr)
            
            # Print progress (every 10 epochs or first 10)
            if (epoch + 1) % 10 == 0 or epoch < 10:
                print(f'Epoch [{epoch+1:3d}/{self.num_epochs}] '
                      f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | '
                      f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} | '
                      f'LR: {current_lr:.6f}')
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.best_state = self.model.state_dict().copy()
            
            # Early stopping check
            if early_stopping(val_loss, self.model, Path('temp_model.pth')):
                print(f"\nEarly stopping at epoch {epoch+1}")
                print(f"Best validation accuracy: {best_val_acc:.4f}")
                break
        
        # Load best model
        if hasattr(self, 'best_state'):
            self.model.load_state_dict(self.best_state)
        
        print("-" * 80)
        print(f"Training completed! Best validation accuracy: {best_val_acc:.4f}")
        
        return history
    
    def _train_epoch(self, dataloader, criterion, optimizer):
        """Train for one epoch."""
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
            
            # Gradient clipping for stability (from notebook)
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

def test_working_notebook_smri():
    """Test sMRI using exact working notebook approach."""
    print("🔬 Working Notebook sMRI Test")
    print("=" * 50)
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Get config
    config = get_config('smri')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"📊 Working Notebook Configuration:")
    print(f"   Device: {device}")
    print(f"   Feature selection: 300 features (from notebook)")
    print(f"   Scaler: RobustScaler (from notebook)")
    print(f"   Batch size: 16 (from notebook)")
    print(f"   Epochs: 200 (from notebook)")
    print(f"   Learning rate: 1e-3 (from notebook)")
    
    # Load data
    print(f"\n📁 Loading sMRI data...")
    processor = SMRIDataProcessor(
        data_path=config.smri_data_path,
        feature_selection_k=None,  # We'll do it manually
        scaler_type='robust'
    )
    
    # Load raw features
    features, labels, subject_ids = processor.process_all_subjects(
        phenotypic_file=config.phenotypic_file
    )
    
    print(f"✅ Loaded {len(labels)} subjects")
    print(f"📊 Original features: {features.shape[1]}")
    print(f"🎯 Class distribution: ASD={np.sum(labels)}, Control={len(labels) - np.sum(labels)}")
    
    # Exact data split as in notebook (70/15/15)
    X_temp, X_test, y_temp, y_test = train_test_split(
        features, labels, test_size=0.15, random_state=42, stratify=labels
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.15, random_state=42, stratify=y_temp
    )
    
    print(f"\n📊 Working Notebook Data Split:")
    print(f"   Train: {X_train.shape[0]} ({np.sum(y_train==1)} ASD, {np.sum(y_train==0)} Control)")
    print(f"   Val: {X_val.shape[0]} ({np.sum(y_val==1)} ASD, {np.sum(y_val==0)} Control)")
    print(f"   Test: {X_test.shape[0]} ({np.sum(y_test==1)} ASD, {np.sum(y_test==0)} Control)")
    
    # Exact preprocessing as in notebook
    print(f"\n🔧 Applying working notebook preprocessing...")
    
    # Handle NaN/inf values
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=1e6, neginf=-1e6)
    X_val = np.nan_to_num(X_val, nan=0.0, posinf=1e6, neginf=-1e6)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # RobustScaler (from notebook)
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Feature selection: Top 300 features (from notebook)
    feature_selector = SelectKBest(score_func=f_classif, k=300)
    X_train_processed = feature_selector.fit_transform(X_train_scaled, y_train)
    X_val_processed = feature_selector.transform(X_val_scaled)
    X_test_processed = feature_selector.transform(X_test_scaled)
    
    print(f"✅ Selected top 300 features (from {features.shape[1]})")
    print(f"✅ Applied RobustScaler")
    print(f"✅ Final feature dimension: {X_train_processed.shape[1]}")
    
    # Create datasets with working notebook parameters
    train_dataset = SMRIDataset(
        X_train_processed, y_train,
        augment=True,
        noise_factor=0.005  # From notebook
    )
    val_dataset = SMRIDataset(X_val_processed, y_val, augment=False)
    test_dataset = SMRIDataset(X_test_processed, y_test, augment=False)
    
    # Create data loaders with exact notebook settings
    batch_size = 16  # From notebook
    
    # Weighted sampling for class imbalance (from notebook)
    class_counts = np.bincount(y_train)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[y_train]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=sampler, drop_last=True
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"\n🧠 Creating Working Notebook Model...")
    
    # Create model with exact notebook architecture
    model = WorkingNotebookSMRITransformer(
        input_dim=X_train_processed.shape[1],
        d_model=64,    # From notebook
        n_heads=4,     # From notebook
        n_layers=2,    # From notebook
        dropout=0.3,   # From notebook
        layer_dropout=0.1  # From notebook
    )
    
    model_info = model.get_model_info()
    print(f"   Name: {model_info['model_name']}")
    print(f"   Parameters: {model_info['total_params']:,}")
    print(f"   Input dim: {model_info['input_dim']}")
    
    # Train with exact notebook strategy
    print(f"\n🎯 Training with Working Notebook Strategy...")
    trainer = WorkingNotebookTrainer(
        model=model,
        device=device,
        num_epochs=200,      # From notebook
        learning_rate=1e-3,  # From notebook
        weight_decay=1e-4,   # From notebook
        warmup_epochs=10     # From notebook
    )
    
    # Train the model
    history = trainer.train(train_loader, val_loader, y_train)
    
    # Final evaluation
    print(f"\n📊 Final Evaluation...")
    _, test_acc, test_preds, test_labels, test_probs = trainer._evaluate_epoch(
        test_loader, nn.CrossEntropyLoss()
    )
    
    # Calculate comprehensive metrics
    test_auc = roc_auc_score(test_labels, test_probs)
    
    print(f"\n🎉 Working Notebook sMRI Results:")
    print(f"   Test Accuracy: {test_acc:.4f} ({test_acc:.1%})")
    print(f"   Test AUC: {test_auc:.4f}")
    
    # Compare with target
    target_acc = 0.60
    print(f"\n📈 Performance Comparison:")
    print(f"   Target (notebook): {target_acc:.1%}")
    print(f"   Achieved result: {test_acc:.1%}")
    print(f"   Difference: {test_acc - target_acc:+.1%}")
    
    if test_acc >= target_acc:
        print(f"   🎯 SUCCESS! Matched/exceeded notebook performance!")
    elif test_acc >= 0.58:
        print(f"   ✅ VERY GOOD! Close to notebook performance!")
    elif test_acc >= 0.56:
        print(f"   ✅ GOOD! Reasonable performance!")
    else:
        print(f"   ⚠️  Still below target...")
    
    # Detailed classification report
    print(f"\n📋 Detailed Results:")
    print(classification_report(test_labels, test_preds, target_names=['Control', 'ASD']))
    
    # Save results
    results = {
        'test_accuracy': float(test_acc),
        'test_auc': float(test_auc),
        'target_accuracy': target_acc,
        'model_params': model_info['total_params'],
        'training_epochs': len(history['train_loss']),
        'final_train_acc': float(history['train_acc'][-1]),
        'final_val_acc': float(history['val_acc'][-1]),
        'approach': 'exact_working_notebook_replication'
    }
    
    with open('working_notebook_smri_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💡 Key Insights:")
    print(f"   • Used exact notebook architecture (BatchNorm, GELU, pre-norm)")
    print(f"   • Applied exact preprocessing (RobustScaler, 300 features)")
    print(f"   • Used exact training strategy (warmup, class weights, label smoothing)")
    print(f"   • Replicated data augmentation (noise_factor=0.005)")
    print(f"   • Applied gradient clipping and early stopping")
    
    return results

if __name__ == "__main__":
    results = test_working_notebook_smri()
    print(f"\n🚀 Working notebook sMRI test completed!")
    print(f"   Final accuracy: {results['test_accuracy']:.1%}")
    print(f"   Target was: {results['target_accuracy']:.1%}") 
"""Advanced trainer for ABIDE experiments."""

import os
import warnings
from typing import Dict, List, Any, Optional
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR
from torch.amp import GradScaler, autocast
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from tqdm import tqdm

from training.utils import EarlyStopping, calculate_class_weights


class Trainer:
    """Advanced trainer with mixed precision, gradient clipping, and comprehensive logging."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        config: Any,
        model_type: str = 'single'  # 'single', 'multimodal'
    ):
        """
        Initialize trainer.
        
        Args:
            model: Model to train
            device: Device to use for training
            config: Configuration object
            model_type: Type of model ('single' or 'multimodal')
        """
        self.model = model.to(device)
        self.device = device
        self.config = config
        self.model_type = model_type

        # Optimizer (AdamW as recommended)
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

        # Learning rate scheduler
        self.scheduler = ExponentialLR(self.optimizer, gamma=0.97)

        # Loss function with class weights if needed
        self.criterion = nn.CrossEntropyLoss()

        # Mixed precision training
        self.scaler = GradScaler('cuda') if config.use_mixed_precision else None

        # Metrics tracking
        self.history = defaultdict(list)

    def _setup_class_weights(self, y_train: np.ndarray):
        """Setup class weights for imbalanced datasets."""
        # Always use class weights for sMRI as it's critical for performance
        class_weights = calculate_class_weights(y_train, self.device)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
        print(f"Using class weights: {class_weights} with label smoothing=0.1")

    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc='Training', leave=False)
        
        for batch_idx, batch_data in enumerate(pbar):
            # Handle different input formats
            if self.model_type == 'multimodal':
                fmri, smri, targets = batch_data
                fmri = fmri.to(self.device)
                smri = smri.to(self.device)
                targets = targets.to(self.device)
                inputs = (fmri, smri)
            else:
                inputs, targets = batch_data
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

            # Mixed precision training
            if self.scaler:
                with autocast(device_type='cuda'):
                    if self.model_type == 'multimodal':
                        outputs = self.model(*inputs)
                    else:
                        outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)

                # Backward pass with gradient scaling
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()

                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    max_norm=self.config.gradient_clip_norm
                )

                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard training without mixed precision
                if self.model_type == 'multimodal':
                    outputs = self.model(*inputs)
                else:
                    outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    max_norm=self.config.gradient_clip_norm
                )
                self.optimizer.step()

            # Metrics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{100.*correct/total:.2f}%"
            })

        return {
            'loss': total_loss / len(train_loader),
            'accuracy': correct / total
        }

    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_targets = []
        all_probs = []

        with torch.no_grad():
            for batch_data in tqdm(val_loader, desc='Validation', leave=False):
                # Handle different input formats
                if self.model_type == 'multimodal':
                    fmri, smri, targets = batch_data
                    fmri = fmri.to(self.device)
                    smri = smri.to(self.device)
                    targets = targets.to(self.device)
                    inputs = (fmri, smri)
                else:
                    inputs, targets = batch_data
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)

                if self.scaler:
                    with autocast(device_type='cuda'):
                        if self.model_type == 'multimodal':
                            outputs = self.model(*inputs)
                        else:
                            outputs = self.model(inputs)
                        loss = self.criterion(outputs, targets)
                else:
                    if self.model_type == 'multimodal':
                        outputs = self.model(*inputs)
                    else:
                        outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)

                total_loss += loss.item()
                probs = F.softmax(outputs, dim=1)

                all_probs.append(probs.cpu().numpy())
                all_preds.append(outputs.argmax(1).cpu().numpy())
                all_targets.append(targets.cpu().numpy())

        # Concatenate all batches
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        all_probs = np.concatenate(all_probs)

        # Calculate metrics
        metrics = {
            'loss': total_loss / len(val_loader),
            'accuracy': accuracy_score(all_targets, all_preds),
            'balanced_accuracy': balanced_accuracy_score(all_targets, all_preds),
        }

        # Add AUC if binary classification
        if len(np.unique(all_targets)) == 2:
            metrics['auc'] = roc_auc_score(all_targets, all_probs[:, 1])
        else:
            metrics['auc'] = 0.5

        return metrics

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
        checkpoint_path: Path,
        y_train: Optional[np.ndarray] = None
    ) -> Dict[str, List[float]]:
        """
        Full training loop with early stopping.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Maximum number of epochs
            checkpoint_path: Path to save best model
            y_train: Training labels for class weight calculation
            
        Returns:
            Training history dictionary
        """
        # Setup class weights if needed
        if y_train is not None:
            self._setup_class_weights(y_train)

        # Early stopping
        early_stopping = EarlyStopping(
            patience=self.config.early_stop_patience,
            mode='min'
        )

        # Warmup scheduler
        warmup_scheduler = None
        if hasattr(self.config, 'warmup_epochs') and self.config.warmup_epochs > 0:
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=self.config.warmup_epochs
            )

        best_val_acc = 0
        
        for epoch in range(1, num_epochs + 1):
            # Training
            train_metrics = self.train_epoch(train_loader)

            # Validation
            val_metrics = self.validate(val_loader)

            # Learning rate scheduling
            if warmup_scheduler and epoch <= self.config.warmup_epochs:
                warmup_scheduler.step()
            else:
                self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]['lr']

            # Logging
            if epoch % self.config.log_every == 0 or epoch <= 10:
                print(f"\nEpoch {epoch}/{num_epochs}")
                print(f"Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}")
                print(f"Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, "
                      f"Balanced Acc: {val_metrics['balanced_accuracy']:.4f}, AUC: {val_metrics['auc']:.4f}")
                print(f"LR: {current_lr:.6f}")

            # Track history
            for key, value in train_metrics.items():
                self.history[f'train_{key}'].append(value)
            for key, value in val_metrics.items():
                self.history[f'val_{key}'].append(value)
            self.history['lr'].append(current_lr)

            # Track best validation accuracy
            if val_metrics['accuracy'] > best_val_acc:
                best_val_acc = val_metrics['accuracy']

            # Early stopping
            early_stopping(val_metrics['loss'], self.model, checkpoint_path)
            if early_stopping.early_stop:
                print(f"\n⏹️ Early stopping triggered at epoch {epoch}")
                print(f"Best validation accuracy: {best_val_acc:.4f}")
                break

        # Load best model
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        return dict(self.history)

    def evaluate_final(self, test_loader: DataLoader) -> Dict[str, float]:
        """Final evaluation on test set."""
        return self.validate(test_loader)

    def get_model_summary(self) -> Dict[str, Any]:
        """Get model summary information."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        summary = {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024),
            'device': str(self.device),
            'mixed_precision': self.scaler is not None
        }
        
        # Add model-specific info if available
        if hasattr(self.model, 'get_model_info'):
            summary.update(self.model.get_model_info())
            
        return summary 
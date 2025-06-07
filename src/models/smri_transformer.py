"""Improved sMRI Transformer for structural neuroimaging features."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union


class SMRITransformer(nn.Module):
    """
    sMRI Transformer based on working notebook architecture.
    Processes sMRI as single feature vector, NOT as sequence with CLS tokens.
    """
    
    def __init__(
        self,
        input_dim: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.3,
        layer_dropout: float = 0.1,
        num_classes: int = 2
    ):
        """
        Initialize sMRI transformer based on working notebook.
        
        Args:
            input_dim: Input feature dimension
            d_model: Model dimension
            n_heads: Number of attention heads
            n_layers: Number of transformer layers
            dropout: Dropout probability
            layer_dropout: Layer dropout probability
            num_classes: Number of output classes
        """
        super().__init__()

        self.input_dim = input_dim
        self.d_model = d_model

        # Input projection with batch normalization (from working notebook)
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Learnable positional encoding for single sequence position (from working notebook)
        self.pos_embedding = nn.Parameter(torch.randn(1, 1, d_model) * 0.1)

        # Transformer encoder with layer dropout (from working notebook)
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        encoder_layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 2,  # Smaller feedforward to reduce overfitting
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-norm for better training stability
        )
        self.transformer = TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            enable_nested_tensor=False
        )

        # Layer dropout
        self.layer_dropout = nn.Dropout(layer_dropout)

        # Classification head with residual connection (from working notebook)
        self.pre_classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.classifier = nn.Linear(d_model // 2, num_classes)

        # Weight initialization
        self._init_weights()

    def _init_weights(self):
        """Initialize weights properly (from working notebook)."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(
        self, 
        x: torch.Tensor, 
        return_attention: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, list]]:
        """
        Forward pass - based on working notebook architecture.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            return_attention: Whether to return attention weights (not implemented)
            
        Returns:
            logits: Classification logits (batch_size, num_classes)
        """
        batch_size = x.size(0)

        # Project to d_model dimensions
        x = self.input_projection(x)

        # Add sequence dimension and positional embedding (from working notebook)
        x = x.unsqueeze(1)  # (batch_size, 1, d_model)
        x = x + self.pos_embedding

        # Pass through transformer
        x = self.transformer(x)

        # Apply layer dropout
        x = self.layer_dropout(x)

        # Global pooling (from working notebook) - squeeze the sequence dimension
        x = x.squeeze(1)  # (batch_size, d_model)

        # Classification (from working notebook)
        features = self.pre_classifier(x)
        logits = self.classifier(features)

        if return_attention:
            # Placeholder - attention extraction not implemented
            return logits, []
        return logits

    def get_model_info(self) -> dict:
        """Get model architecture information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': 'SMRITransformer_WorkingNotebook',
            'input_dim': self.input_dim,
            'd_model': self.d_model,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024)  # Assuming float32
        } 
"""Improved sMRI Transformer for structural neuroimaging features."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union


class SMRITransformer(nn.Module):
    """
    Improved sMRI Transformer for autism classification using structural features.
    Uses CLS token approach consistent with fMRI implementation.
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
        Initialize sMRI transformer.
        
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

        # Input projection with better scaling (from working notebook)
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Scaling factor for better training stability
        self.scale = math.sqrt(d_model)

        # Learnable [CLS] token (consistent with fMRI approach)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Learnable positional encoding
        self.pos_embedding = nn.Parameter(torch.randn(1, 2, d_model) * 0.1)  # For CLS + 1 feature token

        # Transformer encoder layers with pre-norm
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_model * 2,  # Smaller feedforward to reduce overfitting
                dropout=dropout,
                activation='gelu',
                batch_first=True,
                norm_first=True  # Pre-norm for better training stability
            )
            for _ in range(n_layers)
        ])

        # Layer dropout
        self.layer_dropout = nn.Dropout(layer_dropout)

        # Classification head (uses only CLS token)
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )

        # Weight initialization
        self._init_weights()

    def _init_weights(self):
        """Initialize weights properly."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(
        self, 
        x: torch.Tensor, 
        return_attention: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, list]]:
        """
        Forward pass through sMRI transformer.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            return_attention: Whether to return attention weights
            
        Returns:
            logits: Classification logits (batch_size, num_classes)
            attentions: Optional list of attention weights
        """
        batch_size = x.size(0)

        # Project to d_model dimensions with scaling (from working notebook)
        x = self.input_projection(x) / self.scale

        # Add sequence dimension and reshape for transformer
        x = x.unsqueeze(1)  # (batch_size, 1, d_model)

        # Add [CLS] token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (batch_size, 2, d_model)

        # Add positional embeddings
        x = x + self.pos_embedding

        # Pass through transformer layers
        attentions = []
        for layer in self.layers:
            if return_attention:
                # Extract attention from transformer layer
                # Note: This is a simplified attention extraction
                x_old = x
                x = layer(x)
                # Store a placeholder for attention (would need custom layer for true attention)
                attentions.append(None)
            else:
                x = layer(x)

        # Apply layer dropout
        x = self.layer_dropout(x)

        # Extract [CLS] token for classification
        cls_output = x[:, 0]  # (batch_size, d_model)

        # Classification
        logits = self.classifier(cls_output)

        if return_attention:
            return logits, attentions
        return logits

    def get_model_info(self) -> dict:
        """Get model architecture information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': 'SMRITransformer',
            'input_dim': self.input_dim,
            'd_model': self.d_model,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024)  # Assuming float32
        } 
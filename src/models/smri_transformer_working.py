"""
sMRI Transformer based on the exact working notebook architecture.
This model achieved ~60% accuracy in the working notebook.
"""

import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class WorkingNotebookSMRITransformer(nn.Module):
    """
    Exact reproduction of the sMRI Transformer from working notebook.
    This architecture achieved ~60% accuracy.
    """
    
    def __init__(self, input_dim, d_model=64, n_heads=4, n_layers=2,
                 dropout=0.3, layer_dropout=0.1):
        super(WorkingNotebookSMRITransformer, self).__init__()

        self.input_dim = input_dim
        self.d_model = d_model

        # Input projection with batch normalization (from working notebook)
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Learnable positional encoding (from working notebook)
        self.pos_embedding = nn.Parameter(torch.randn(1, 1, d_model) * 0.1)

        # Transformer encoder with pre-norm and GELU (from working notebook)
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

        # Layer dropout (from working notebook)
        self.layer_dropout = nn.Dropout(layer_dropout)

        # Classification head with residual connection (from working notebook)
        self.pre_classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.classifier = nn.Linear(d_model // 2, 2)

        # Weight initialization (from working notebook)
        self._init_weights()

    def _init_weights(self):
        """Initialize weights exactly as in working notebook."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        """Forward pass exactly as in working notebook."""
        batch_size = x.size(0)

        # Project to d_model dimensions
        x = self.input_projection(x)

        # Add sequence dimension and positional embedding
        x = x.unsqueeze(1)  # (batch_size, 1, d_model)
        x = x + self.pos_embedding

        # Pass through transformer
        x = self.transformer(x)

        # Apply layer dropout
        x = self.layer_dropout(x)

        # Global pooling (remove sequence dimension)
        x = x.squeeze(1)  # (batch_size, d_model)

        # Classification with residual-like structure
        features = self.pre_classifier(x)
        logits = self.classifier(features)

        return logits

    def get_model_info(self):
        """Get model information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': 'WorkingNotebook_sMRI_Transformer',
            'input_dim': self.input_dim,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'd_model': self.d_model
        } 